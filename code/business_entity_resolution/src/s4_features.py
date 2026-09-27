import os
import sys
import time
import re
import argparse
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein, LCSseq

_NON_ALNUM = re.compile(r"[\W_]+")
_LONGNUM = re.compile(r"\d{5,}")


_STREET_ABBR = {"rue": "r", "boulevard": "bd", "blvd": "bd", "avenue": "av", "ave": "av", "road": "rd",
                "street": "st", "place": "pl", "chemin": "ch", "route": "rte"}
_PUNCT = re.compile(r"[^\w]+", re.UNICODE)


def street_key(raw_address) -> str:
    """
    Street key: the first comma-separated segment of the raw address with a number followed by a word,
    lower-cased, punctuation -> space, leading zeros stripped from numbers, common street words shortened.
    Segments made of numbers only (postal codes) are skipped. '' if none.
    """
    if not isinstance(raw_address, str) or not raw_address:
        return ""
    for seg in raw_address.split(","):
        toks = _PUNCT.sub(" ", seg.lower()).split()
        if not toks or all(t.isdigit() for t in toks):
            continue
        if not any(any(ch.isdigit() for ch in t) and toks[i + 1].isalpha() for i, t in enumerate(toks[:-1])):
            continue
        out = []
        for t in toks:
            if t.isdigit():
                t = t.lstrip("0") or "0"
            out.append(_STREET_ABBR.get(t, t))
        return " ".join(out)
    return ""


def _legal_forms():
    from maps import LEGAL_SUFFIXES, NAME_ABBREVIATIONS
    extra = {"ei", "eirl", "scop", "sca", "scs", "sel", "selarl", "selas", "gie", "gmbh", "plc"}
    return set(LEGAL_SUFFIXES) | extra, dict(NAME_ABBREVIATIONS)


_LEGAL_SET, _NAME_ABBR = _legal_forms()


def legal_forms(raw_name, legal) -> str:
    """Sorted legal-form tokens: normalised 'legal' plus forms found in raw_name with dots removed (S.A.R.L. -> sarl)."""
    forms = set(str(legal).split()) if isinstance(legal, str) and legal.strip() else set()
    if isinstance(raw_name, str) and raw_name:
        for t in _PUNCT.sub(" ", raw_name.lower().replace(".", "")).split():
            t = _NAME_ABBR.get(t, t)
            for tt in t.split():
                if tt in _LEGAL_SET:
                    forms.add(tt)
    return " ".join(sorted(forms))


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """street_key from raw_address and legal2 from raw_name (+ legal); drops the raw columns."""
    if "raw_address" in df.columns:
        df["street_key"] = [street_key(x) for x in df["raw_address"].values]
        df = df.drop(columns=["raw_address"])
    if "raw_name" in df.columns:
        df["legal2"] = [legal_forms(r, l) for r, l in zip(df["raw_name"].values,
                                                           df["legal"].astype(str).values if "legal" in df.columns else [""] * len(df))]
        df = df.drop(columns=["raw_name"])
    return df


class KeyFreq:
    """
    log1p count of records sharing (country, key) within one split, over the given sources. key is a norm-table
    column ('addr_norm') or 'street_key' (derived from raw_address). Empty keys count 0.
    """

    def __init__(self, cache_dir: str, split: str, sources, key: str):
        self.key = key
        hashes = []
        col = "raw_address" if key == "street_key" else key
        for src in sources:
            pf = pq.ParquetFile(os.path.join(cache_dir, f"norm_{split}_{src}.parquet"))
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["country", col]).to_pandas()
                vals = [street_key(x) for x in t[col].values] if key == "street_key" else t[col].fillna("").astype(str).values
                vals = np.asarray(vals, dtype=object)
                ok = vals != ""
                if ok.any():
                    hashes.append(NameFreq._hash(t["country"].values[ok], vals[ok]))
                del t
        self.keys, self.counts = np.unique(np.concatenate(hashes), return_counts=True)

    def log_freq(self, country, vals) -> np.ndarray:
        vals = np.asarray(pd.Series(vals).fillna("").astype(str).values, dtype=object)
        h = NameFreq._hash(country, vals)
        pos = np.clip(np.searchsorted(self.keys, h), 0, len(self.keys) - 1)
        cnt = np.where((self.keys[pos] == h) & (vals != ""), self.counts[pos], 0)
        return np.log1p(cnt).astype(np.float32)


def relative_features(ctx_df: pd.DataFrame, ctx_support: np.ndarray, comb=None) -> dict:
    """
    Per-S1 relative versions of absolute scores over the S1's FULL candidate list (context rows):
    x_rel = x / max over the list (0 if max <= 0), x_z = z-score within the list (0 if std 0), x_rk = rank desc.
    """
    s1 = ctx_df["s1_id"].values
    cols = {
        "emb_score": ctx_df["emb_score"].values.astype(np.float32),
        "tfidf_score": (ctx_df["tfidf_score"].values if "tfidf_score" in ctx_df.columns
                        else np.zeros(len(ctx_df))).astype(np.float32),
        "support": np.asarray(ctx_support, dtype=np.float32),
    }
    if comb is not None:
        cc = comb.cos(_arrow_ids_to_int(pa.array(s1, type=pa.string())),
                      _arrow_ids_to_int(pa.array(ctx_df["cand_id"].values, type=pa.string())))
        cols["comb_cos"] = np.nan_to_num(cc, nan=0.0).astype(np.float32)
    else:
        cols["comb_cos"] = np.zeros(len(ctx_df), dtype=np.float32)
    codes = pd.factorize(s1)[0]
    out = {}
    for name, v in cols.items():
        g = pd.Series(v).groupby(codes)
        mx = g.transform("max").values
        mu = g.transform("mean").values
        sd = g.transform("std", ddof=0).values
        out[f"{name}_rel"] = np.where(mx > 0, v / np.where(mx > 0, mx, 1), 0).astype(np.float32)
        out[f"{name}_z"] = np.where(sd > 1e-9, (v - mu) / np.where(sd > 1e-9, sd, 1), 0).astype(np.float32)
        out[f"{name}_rk"] = g.rank(method="min", ascending=False).values.astype(np.float32)
    return out


def _initials(core: str) -> str:
    return "".join(t[0] for t in core.split() if t)


def _acronym_match(a: str, b: str) -> bool:
    """True if the initials of one core_name (>= 2 tokens) equal the other core_name with spaces removed."""
    ia, ib = _initials(a), _initials(b)
    return (len(ia) >= 2 and ia == b.replace(" ", "")) or (len(ib) >= 2 and ib == a.replace(" ", ""))

import config

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _rss_mb() -> float:
    """Returns current process RSS in MB, or -1 if psutil unavailable."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    return -1.0


# Max candidate rows per feature sub-batch inside one chunk. Sub-batches always hold
# whole S1 candidate lists, so per-S1 context features are exact.
SUB_BATCH_MAX_ROWS = 1_000_000


def _free_memory():
    """Runs GC and returns freed heap pages (glibc + Arrow pool) to the OS."""
    import gc
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _s1_sub_batches(s1_series: pd.Series, max_rows: int):
    """
    Groups rows into sub-batches of at most max_rows rows without splitting any S1
    (an S1 with more than max_rows candidates gets a batch of its own).
    S1 are packed greedily in order of first appearance.
    Returns: list of int64 row-index arrays (ascending within each batch).
    """
    codes, _ = pd.factorize(s1_series, sort=False)
    counts = np.bincount(codes)
    batch_of_s1 = np.empty(len(counts), dtype=np.int64)
    cur, acc = 0, 0
    for i, c in enumerate(counts):
        if acc > 0 and acc + c > max_rows:
            cur += 1
            acc = 0
        batch_of_s1[i] = cur
        acc += c
    row_batch = batch_of_s1[codes]
    order = np.argsort(row_batch, kind='stable')
    bounds = np.searchsorted(row_batch[order], np.arange(cur + 2))
    return [order[bounds[b]:bounds[b + 1]] for b in range(cur + 1)]


def _id_to_int(series: pd.Series, validate: bool = True) -> pd.Series:
    r"""
    Converts 'S1-12345', 'S2-12345', 'S3-12345' style IDs to int64.
    Encoding: source_digit × 10^12 + numeric_part
      S1-12345 → 1_000_000_012_345
      S2-12345 → 2_000_000_012_345
      S3-12345 → 3_000_000_012_345
    Guarantees no collision between S1/S2/S3 because numeric_part < 10^12.

    Hard assertions (per chunk):
      1. Every raw ID matches ^S[123]-\d+$
      2. Numeric part < 10^12
      3. No two distinct raw IDs map to the same int64 (raw nunique == enc nunique)
    Fails loudly with AssertionError if any check is violated.
    """
    if series.empty:
        return pd.Series([], dtype=np.int64)

    s = series.astype(str)

    if validate:
        valid_mask = s.str.match(r'^S[123]-\d+$')
        if not valid_mask.all():
            bad = s[~valid_mask]
            raise AssertionError(
                f"_id_to_int assertion failed: {len(bad)} raw IDs do not match ^S[123]-\\d+$. "
                f"First bad ID: {bad.iloc[0]!r}"
            )

    prefix = s.str[1].map({'1': 1_000_000_000_000, '2': 2_000_000_000_000, '3': 3_000_000_000_000})
    numeric = s.str.split('-', n=1).str[1].astype(np.int64)

    if validate:
        if not (numeric < 1_000_000_000_000).all():
            big = numeric[numeric >= 1_000_000_000_000]
            raise AssertionError(
                f"_id_to_int assertion failed: {len(big)} numeric parts >= 10^12. "
                f"First offender: numeric={big.iloc[0]}"
            )

    encoded = prefix.astype(np.int64) + numeric

    if validate:
        n_raw = series.nunique()
        n_enc = encoded.nunique()
        if n_raw != n_enc:
            raise AssertionError(
                f"_id_to_int assertion failed: collision detected! "
                f"Raw nunique ({n_raw}) != encoded nunique ({n_enc}) for this chunk."
            )

    return encoded


def _int_to_id(ints):
    """
    Converts int64 encoded IDs back to 'S1-12345', 'S2-12345', 'S3-12345' style strings.
    """
    if isinstance(ints, (pd.Series, np.ndarray)):
        arr = np.asarray(ints, dtype=np.int64)
    else:
        arr = np.array(list(ints), dtype=np.int64)
    if len(arr) == 0:
        return set() if isinstance(ints, set) else []
    src = arr // 1_000_000_000_000
    num = arr % 1_000_000_000_000
    res = [f"S{s}-{n}" for s, n in zip(src, num)]
    return set(res) if isinstance(ints, set) else res


def parse_args():
    """
    Parses command line arguments for feature engineering.
    Returns: argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(description="Step 6: Pairwise Feature Engineering")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--laptop-test", action="store_true", help="Use small laptop test pool in cache/laptop_test/")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing feature parquet files")
    parser.add_argument("--count-only", action="store_true", help="Print number of feature rows and unique IDs, then exit")
    parser.add_argument("--max-chunks", type=int, default=None, help="Process only the first N feature chunks")
    parser.add_argument("--sub-batch-rows", type=int, default=SUB_BATCH_MAX_ROWS,
                        help="Max candidate rows per sub-batch inside a chunk (whole S1 groups only)")
    return parser.parse_args()


def get_address_token_df(cache_dir: str, split: str):
    """
    Computes or loads per-country document frequencies of address tokens (len >= 5) across S1+S2+S3.
    Uses a memory-light streaming batch reader and caches results to cache/addr_df_{split}.parquet.
    Returns: dict mapping country string to dict of {token: doc_count}.
    """
    addr_df_path = os.path.join(cache_dir, f"addr_df_{split}.parquet")

    if os.path.exists(addr_df_path):
        print(f"Loading cached address token document frequencies from {addr_df_path}...", flush=True)
        t0 = time.time()
        df_saved = pd.read_parquet(addr_df_path)
        df_tokens = {}
        for c, grp in df_saved.groupby('country'):
            df_tokens[c] = dict(zip(grp['token'], grp['doc_freq']))
        print(f"  Loaded {len(df_saved):,} address tokens across {len(df_tokens)} countries "
              f"in {time.time() - t0:.2f}s (RSS {_rss_mb():.0f} MB)", flush=True)
        return df_tokens

    print(f"Computing address token document frequencies across S1+S2+S3 for split '{split}' (memory-light streaming)...", flush=True)
    t0 = time.time()
    counts = defaultdict(Counter)

    for src in ["source1", "source2", "source3"]:
        paths_to_try = [
            os.path.join(cache_dir, f"norm_{split}_{src}.parquet"),
            os.path.join(config.CACHE_DIR, f"norm_{split}_{src}.parquet"),
        ]
        # Only for non-test splits or fallback if standard norm_{src}.parquet exists
        if split != "test":
            paths_to_try.append(os.path.join(cache_dir, f"norm_{src}.parquet"))

        target_path = None
        for p in paths_to_try:
            if os.path.exists(p):
                target_path = p
                break
        if not target_path:
            raise FileNotFoundError(
                f"Cannot compute address token DF for split '{split}': missing normalized table for {src}. "
                f"Checked: {paths_to_try}. Never falling back to another split."
            )

        pf = pq.ParquetFile(target_path)
        for batch in pf.iter_batches(batch_size=500_000, columns=['country', 'addr_norm']):
            b_df = batch.to_pandas()
            c_arr = b_df['country'].values
            a_arr = b_df['addr_norm'].values
            for c, a in zip(c_arr, a_arr):
                if a and isinstance(a, str):
                    toks = set(t for t in a.split() if len(t) >= 5)
                    ctr = counts[c]
                    for t in toks:
                        ctr[t] += 1

    records = []
    for c, ctr in counts.items():
        for tok, cnt in ctr.items():
            records.append((c, tok, cnt))
    df_out = pd.DataFrame(records, columns=['country', 'token', 'doc_freq'])
    df_out['doc_freq'] = df_out['doc_freq'].astype(np.int32)
    save_path = os.path.join(cache_dir, f"addr_df_{split}.parquet")
    df_out.to_parquet(save_path, index=False)
    print(f"  Saved {len(df_out):,} address tokens to {save_path} in {time.time() - t0:.2f}s "
          f"(RSS {_rss_mb():.0f} MB)", flush=True)

    df_tokens = {}
    for c, grp in df_out.groupby('country'):
        df_tokens[c] = dict(zip(grp['token'], grp['doc_freq']))
    return df_tokens


def extract_rare3_set(addr, country, df_tokens):
    """
    Extracts up to 3 rarest address tokens (len >= 5) based on country document frequency.
    Returns: frozenset of rarest token strings.
    """
    if not isinstance(addr, str) or not addr.strip():
        return frozenset()
    toks = list(set(t for t in addr.split() if len(t) >= 5))
    if not toks:
        return frozenset()
    ctr = df_tokens.get(country, {})
    toks.sort(key=lambda t: (ctr.get(t, 0), t))
    return frozenset(toks[:3])


NEEDED_NORM_COLS = [
    'entity_id', 'country', 'name_full', 'core_name', 'name_skel',
    'addr_norm', 'legal', 'name_a', 'name_b', 'name_aka_a', 'name_aka_b',
    'house_no', 'zip_pin', 'state_code', 'num_tokens', 'house_cands', 'raw_name', 'raw_address'
]


def load_norm_table(cache_dir, split, source, needed_ids=None, columns=None):
    """
    Loads normalized table for a source with fallback paths across train and test splits,
    restricted to only the needed entity IDs and columns.
    Low-cardinality string columns ('country', 'legal', 'state_code') are converted
    to 'category' dtype to minimize memory footprint.
    Returns: pandas.DataFrame indexed by entity_id.
    """
    paths_to_try = [
        os.path.join(cache_dir, f"norm_{split}_{source}.parquet"),
        os.path.join(cache_dir, f"norm_train_{source}.parquet"),
        os.path.join(cache_dir, f"norm_{source}.parquet"),
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            cols_to_read = columns
            if cols_to_read is not None:
                if "entity_id" not in cols_to_read:
                    cols_to_read = ["entity_id"] + list(cols_to_read)
            df = pd.read_parquet(p, columns=cols_to_read)
            if needed_ids is not None:
                df = df[df["entity_id"].isin(needed_ids)]
            df = add_derived_columns(df)
            for cat_col in ['country', 'legal', 'state_code']:
                if cat_col in df.columns:
                    df[cat_col] = df[cat_col].astype('category')
            return df.set_index("entity_id")
    raise FileNotFoundError(f"Could not find normalized table for {source} in {cache_dir}")


def load_candidate_embeddings(cache_dir, split, needed_cand_ids=None):
    """
    Loads candidate main embeddings for Source 2 and Source 3 into a stacked float16 matrix.
    Uses mmap_mode='r' to read directly from disk into a pre-allocated float16 array,
    avoiding intermediate float32 copies and np.vstack memory duplication.
    If needed_cand_ids is provided, restricts to only those candidate IDs.
    Returns: tuple of (all_cand_emb, id_to_row_map).
    """
    s2_m_p = os.path.join(cache_dir, f"emb_{split}_source2.npy")
    if not os.path.exists(s2_m_p):
        s2_m_p = os.path.join(cache_dir, "emb_train_source2.npy")
    s2_id_p = os.path.join(cache_dir, f"ids_{split}_source2.npy")
    if not os.path.exists(s2_id_p):
        s2_id_p = os.path.join(cache_dir, "ids_train_source2.npy")

    s3_m_p = os.path.join(cache_dir, f"emb_{split}_source3.npy")
    if not os.path.exists(s3_m_p):
        s3_m_p = os.path.join(cache_dir, "emb_train_source3.npy")
    s3_id_p = os.path.join(cache_dir, f"ids_{split}_source3.npy")
    if not os.path.exists(s3_id_p):
        s3_id_p = os.path.join(cache_dir, "ids_train_source3.npy")

    s2_ids = np.load(s2_id_p, allow_pickle=True)
    s2_mmap = np.load(s2_m_p, mmap_mode='r')
    if needed_cand_ids is not None:
        s2_mask = pd.Series(s2_ids).isin(needed_cand_ids).values
        s2_ids = s2_ids[s2_mask]
        s2_indices = np.where(s2_mask)[0]
    else:
        s2_indices = slice(None)

    s3_ids = np.load(s3_id_p, allow_pickle=True)
    s3_mmap = np.load(s3_m_p, mmap_mode='r')
    if needed_cand_ids is not None:
        s3_mask = pd.Series(s3_ids).isin(needed_cand_ids).values
        s3_ids = s3_ids[s3_mask]
        s3_indices = np.where(s3_mask)[0]
    else:
        s3_indices = slice(None)

    n_s2 = len(s2_ids)
    n_s3 = len(s3_ids)
    n_total = n_s2 + n_s3
    dim = s2_mmap.shape[1]

    # Pre-allocate single float16 matrix directly (half memory of float32, no np.vstack duplication)
    all_cand_emb = np.empty((n_total, dim), dtype=np.float16)
    all_cand_emb[:n_s2] = s2_mmap[s2_indices].astype(np.float16)
    all_cand_emb[n_s2:] = s3_mmap[s3_indices].astype(np.float16)

    del s2_mmap, s3_mmap
    import gc; gc.collect()

    id_to_row = {cid: idx for idx, cid in enumerate(s2_ids)}
    for idx, cid in enumerate(s3_ids):
        id_to_row[cid] = n_s2 + idx

    return all_cand_emb, id_to_row


class CombEmbeddings:
    """
    Combined name+address embeddings (cache/embc_{split}_source{1,2,3}.npy, L2-normalised float16) kept
    memory-mapped; one sorted int64 ID index over all sources. cos() gathers only the rows a batch needs.
    """

    def __init__(self, cache_dir: str, split: str):
        ids_all, src_all, row_all = [], [], []
        self.mmaps = []
        for si, src in enumerate(("source1", "source2", "source3")):
            ep = os.path.join(cache_dir, f"embc_{split}_{src}.npy")
            ip = os.path.join(cache_dir, f"idsc_{split}_{src}.npy")
            if not (os.path.exists(ep) and os.path.exists(ip)):
                raise FileNotFoundError(f"combined embeddings missing: {ep} / {ip}")
            self.mmaps.append(np.load(ep, mmap_mode="r"))
            raw = pa.array(np.load(ip, allow_pickle=True).tolist(), type=pa.string())
            ids_all.append(_arrow_ids_to_int(raw))
            src_all.append(np.full(len(raw), si, dtype=np.int8))
            row_all.append(np.arange(len(raw), dtype=np.int64))
        ids = np.concatenate(ids_all)
        order = np.argsort(ids, kind="stable")
        self.ids = ids[order]
        self.src = np.concatenate(src_all)[order]
        self.row = np.concatenate(row_all)[order]

    def _vecs(self, ids_int: np.ndarray):
        pos = np.clip(np.searchsorted(self.ids, ids_int), 0, len(self.ids) - 1)
        ok = self.ids[pos] == ids_int
        out = np.zeros((len(ids_int), self.mmaps[0].shape[1]), dtype=np.float32)
        for si, mm in enumerate(self.mmaps):
            sel = np.flatnonzero(ok & (self.src[pos] == si))
            if len(sel):
                uniq, inv = np.unique(self.row[pos[sel]], return_inverse=True)
                out[sel] = np.asarray(mm[uniq], dtype=np.float32)[inv]
        return out, ok

    def cos(self, s1_int: np.ndarray, cand_int: np.ndarray, block: int = 200_000) -> np.ndarray:
        """Dot product of the stored combined embeddings per pair; NaN if either side has none."""
        res = np.full(len(s1_int), np.nan, dtype=np.float32)
        for b in range(0, len(s1_int), block):
            a, oka = self._vecs(s1_int[b:b + block])
            c, okc = self._vecs(cand_int[b:b + block])
            d = np.einsum("ij,ij->i", a, c).astype(np.float32)
            d[~(oka & okc)] = np.nan
            res[b:b + block] = d
        return res


class NameFreq:
    """
    How many records (S1 + S2 + S3 of one split) share a (country, core_name) key. Keys are 64-bit hashes kept
    as a sorted array with counts (no Python dict), built once per split.
    """

    def __init__(self, cache_dir: str, split: str):
        hashes = []
        for src in ("source1", "source2", "source3"):
            t = pq.read_table(os.path.join(cache_dir, f"norm_{split}_{src}.parquet"), columns=["country", "core_name"])
            hashes.append(self._hash(t["country"].to_numpy(zero_copy_only=False), t["core_name"].to_numpy(zero_copy_only=False)))
            del t
        self.keys, self.counts = np.unique(np.concatenate(hashes), return_counts=True)

    @staticmethod
    def _hash(country, core):
        key = pd.Series(country).astype(str).str.cat(pd.Series(core).fillna("").astype(str), sep="\x1f")
        return pd.util.hash_array(key.to_numpy(dtype=object))

    def log_freq(self, country, core) -> np.ndarray:
        h = self._hash(country, core)
        pos = np.clip(np.searchsorted(self.keys, h), 0, len(self.keys) - 1)
        cnt = np.where(self.keys[pos] == h, self.counts[pos], 0)
        return np.log1p(cnt).astype(np.float32)


def load_name_map(cache_dir: str) -> dict:
    """Channel K token dictionary (cache/indic_dict.json); empty dict if missing."""
    p = os.path.join(cache_dir, "indic_dict.json")
    if not os.path.exists(p):
        return {}
    import json
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _arrow_ids_to_int(col) -> np.ndarray:
    """'S2-12345' strings (arrow) -> int64 source_digit * 10^12 + number (same encoding as _id_to_int)."""
    import pyarrow.compute as pc
    digit = pc.cast(pc.utf8_slice_codeunits(col, 1, 2), "int64")
    num = pc.cast(pc.utf8_slice_codeunits(col, 3), "int64")
    return pc.add(pc.multiply(digit, 1_000_000_000_000), num).to_numpy()


def compute_support_feature(chunk_df, all_cand_emb, id_to_row):
    """
    Computes max cosine similarity to other top-5 candidates (by emb_score) for each S1.
    Processes in small blocks of S1 entities to avoid allocating a multi-gigabyte vector slice.
    Returns: numpy.ndarray of float32 support scores aligned with chunk_df rows.
    """
    n_rows = len(chunk_df)
    if n_rows == 0:
        return np.array([], dtype=np.float32)

    # Sort candidates per S1 by emb_score descending to guarantee top-5 order
    orig_indices = np.arange(n_rows)
    df_sorted = chunk_df[['s1_id', 'cand_id', 'emb_score']].assign(orig_idx=orig_indices).sort_values(
        ['s1_id', 'emb_score'], ascending=[True, False], kind='stable'
    )

    s1_sorted = df_sorted['s1_id'].values
    c_sorted = df_sorted['cand_id'].values
    sorted_orig_idx = df_sorted['orig_idx'].values
    del df_sorted

    c_rows = np.array([id_to_row.get(cid, 0) for cid in c_sorted], dtype=np.int32)
    del c_sorted

    unique_s1, start_indices, counts = np.unique(s1_sorted, return_index=True, return_counts=True)
    del s1_sorted
    sorted_support = np.zeros(n_rows, dtype=np.float32)
    n_s1 = len(unique_s1)

    block_size = 2000  # Process 2000 S1 entities per block (~80,000 candidate rows, ~60 MB)
    for b in range(0, n_s1, block_size):
        b_end = min(b + block_size, n_s1)
        b_start_row = start_indices[b]
        b_end_row = start_indices[b_end - 1] + counts[b_end - 1]

        b_c_rows = c_rows[b_start_row:b_end_row]
        b_vecs = all_cand_emb[b_c_rows].astype(np.float32)

        curr_offset = 0
        for s_i in range(b, b_end):
            cnt = counts[s_i]
            if cnt > 1:
                k = min(cnt, 5)
                E = b_vecs[curr_offset : curr_offset + cnt]
                T = E[:k]
                M = np.dot(E, T.T)
                for i in range(k):
                    M[i, i] = -999.0
                sorted_support[start_indices[s_i] : start_indices[s_i] + cnt] = np.max(M, axis=1)
            curr_offset += cnt

    support = np.zeros(n_rows, dtype=np.float32)
    support[sorted_orig_idx] = sorted_support
    return support


def compute_context_features(context_df, all_cand_emb, cand_id_map):
    """
    Computes per-S1 context features on the FULL candidate list of each S1.

    Per-S1 context features (depend on other candidates of the same S1):
      - gap_to_best:  max(emb_score for this S1) - emb_score of this pair
      - n_cands:      number of candidates for this S1
      - support:      max cosine similarity to top-5 candidates (by emb_score)

    These must be computed BEFORE filtering competitor S1 to only shared pairs,
    because the full 40-candidate context changes their values.

    Returns: (gap_to_best, n_cands, support) as float32 numpy arrays aligned with context_df.
    """
    emb_score = context_df['emb_score'].values.astype(np.float32)
    s1_best_score = context_df.groupby('s1_id')['emb_score'].transform('max').values.astype(np.float32)
    gap_to_best = s1_best_score - emb_score
    n_cands = context_df.groupby('s1_id')['cand_id'].transform('count').values.astype(np.float32)
    support = compute_support_feature(context_df, all_cand_emb, cand_id_map)
    return gap_to_best, n_cands, support


def compute_chunk_features(chunk_df, s1_df, cands_df, all_cand_emb, cand_id_map,
                           global_reverse_rank, df_tokens,
                           ctx_gap_to_best, ctx_n_cands, ctx_support, comb=None, name_freq=None, name_map=None,
                           rel=None, key_freqs=None):
    """
    Extracts all 31 features and rule_score for a candidate chunk without row-wise loops.
    Token sets and lengths are computed locally only for rows in this chunk.

    Per-S1 context features (gap_to_best, n_cands, support) are passed in pre-computed
    from compute_context_features which operates on the FULL candidate list per S1.

    Returns: pandas.DataFrame containing entity IDs, label, rule_score, and ordered config.FEATURES.
    """
    n_pairs = len(chunk_df)
    s1_ids = chunk_df['s1_id'].values
    cand_ids = chunk_df['cand_id'].values

    s1_sub = s1_df.loc[s1_ids]
    cand_sub = cands_df.loc[cand_ids]

    # 1. Embedding & Rank features
    emb_score = chunk_df['emb_score'].values.astype(np.float32)
    emb_rank = chunk_df['emb_rank'].values.astype(np.float32)

    # 2. Vectorized Name Similarities via RapidFuzz cpdist
    s1_nf = s1_sub['name_full'].tolist()
    c_nf = cand_sub['name_full'].tolist()
    s1_cn = s1_sub['core_name'].tolist()
    c_cn = cand_sub['core_name'].tolist()
    s1_sk = s1_sub['name_skel'].tolist()
    c_sk = cand_sub['name_skel'].tolist()
    s1_ad = s1_sub['addr_norm'].tolist()
    c_ad = cand_sub['addr_norm'].tolist()

    name_token_sort = process.cpdist(s1_nf, c_nf, scorer=fuzz.token_sort_ratio, workers=-1).astype(np.float32)
    name_token_set = process.cpdist(s1_nf, c_nf, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    core_ratio = process.cpdist(s1_cn, c_cn, scorer=fuzz.ratio, workers=-1).astype(np.float32)
    core_partial = process.cpdist(s1_cn, c_cn, scorer=fuzz.partial_ratio, workers=-1).astype(np.float32)

    skel_ratio = process.cpdist(s1_sk, c_sk, scorer=fuzz.ratio, workers=-1).astype(np.float32)
    empty_sk = (np.array(s1_sk) == '') | (np.array(c_sk) == '')
    skel_ratio[empty_sk] = np.nan

    # 3. name_jaccard over per-chunk token sets (cached per unique entity in chunk for speed)
    chunk_s1_unique = chunk_df['s1_id'].unique()
    chunk_c_unique = chunk_df['cand_id'].unique()

    s1_tok_map = {eid: frozenset(str(x).split()) for eid, x in zip(chunk_s1_unique, s1_df.loc[chunk_s1_unique, 'name_full'])}
    c_tok_map = {cid: frozenset(str(x).split()) for cid, x in zip(chunk_c_unique, cands_df.loc[chunk_c_unique, 'name_full'])}
    s1_tsets = [s1_tok_map[eid] for eid in s1_ids]
    c_tsets = [c_tok_map[cid] for cid in cand_ids]
    name_jaccard = np.array([
        len(a & b) / len(a | b) if (a or b) else 0.0
        for a, b in zip(s1_tsets, c_tsets)
    ], dtype=np.float32)
    del s1_tok_map, c_tok_map, s1_tsets, c_tsets

    # 4. legal_match: 1 if present and equal, 0 if present and different, -1 if missing
    s1_leg = s1_sub['legal'].to_numpy()
    c_leg = cand_sub['legal'].to_numpy()
    leg_miss = (s1_leg == '') | (c_leg == '')
    legal_match = np.where(leg_miss, -1, np.where(s1_leg == c_leg, 1, 0)).astype(np.float32)

    # 5. dba_max & aka_max
    dba_mask = (cand_sub['name_a'].values != '') | (s1_sub['name_a'].values != '')
    dba_max = np.full(n_pairs, np.nan, dtype=np.float32)
    if dba_mask.any():
        s_a = process.cpdist(s1_sub.loc[dba_mask, 'name_full'].tolist(), cand_sub.loc[dba_mask, 'name_a'].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
        s_b = process.cpdist(s1_sub.loc[dba_mask, 'name_full'].tolist(), cand_sub.loc[dba_mask, 'name_b'].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
        dba_max[dba_mask] = np.maximum(s_a, s_b)

    aka_mask = (cand_sub['name_aka_a'].values != '') | (s1_sub['name_aka_a'].values != '')
    aka_max = np.full(n_pairs, np.nan, dtype=np.float32)
    if aka_mask.any():
        s_aka_a = process.cpdist(s1_sub.loc[aka_mask, 'name_full'].tolist(), cand_sub.loc[aka_mask, 'name_aka_a'].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
        s_aka_b = process.cpdist(s1_sub.loc[aka_mask, 'name_full'].tolist(), cand_sub.loc[aka_mask, 'name_aka_b'].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
        aka_max[aka_mask] = np.maximum(s_aka_a, s_aka_b)

    # 6. len_diff on core_name
    s1_core_lens = s1_sub['core_name'].str.len().astype(np.int32).values
    c_core_lens = cand_sub['core_name'].str.len().astype(np.int32).values
    len_diff = np.abs(s1_core_lens - c_core_lens).astype(np.float32)

    # 7. addr_token_set
    addr_token_set = process.cpdist(s1_ad, c_ad, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    empty_ad = (np.array(s1_ad) == '') | (np.array(c_ad) == '')
    addr_token_set[empty_ad] = np.nan

    # 8. house_match, zip_match, state_match: 1 same, 0 different, -1 missing
    s1_h = s1_sub['house_no'].values
    c_h = cand_sub['house_no'].values
    h_miss = (s1_h == '') | (c_h == '')
    house_match = np.where(h_miss, -1, np.where(s1_h == c_h, 1, 0)).astype(np.float32)

    s1_z = s1_sub['zip_pin'].values
    c_z = cand_sub['zip_pin'].values
    z_miss = (s1_z == '') | (c_z == '')
    zip_match = np.where(z_miss, -1, np.where(s1_z == c_z, 1, 0)).astype(np.float32)

    s1_st = s1_sub['state_code'].to_numpy()
    c_st = cand_sub['state_code'].to_numpy()
    st_miss = (s1_st == '') | (c_st == '')
    state_match = np.where(st_miss, -1, np.where(s1_st == c_st, 1, 0)).astype(np.float32)

    # 9. house_cand_match: 1 if any candidate matches, 0 if both exist but differ, -1 if missing
    s1_hc_map = {eid: (frozenset(str(x).split(';')) if str(x).strip() else frozenset()) for eid, x in zip(chunk_s1_unique, s1_df.loc[chunk_s1_unique, 'house_cands'])}
    c_hc_map = {cid: (frozenset(str(x).split(';')) if str(x).strip() else frozenset()) for cid, x in zip(chunk_c_unique, cands_df.loc[chunk_c_unique, 'house_cands'])}
    s1_hc = [s1_hc_map[eid] for eid in s1_ids]
    c_hc = [c_hc_map[cid] for cid in cand_ids]
    house_cand_match = np.array([
        -1 if (not a or not b) else (1 if bool(a & b) else 0)
        for a, b in zip(s1_hc, c_hc)
    ], dtype=np.float32)
    del s1_hc_map, c_hc_map, s1_hc, c_hc

    # 10. num_jaccard: Jaccard on numeric tokens (NaN if either side has no numbers)
    s1_num_map = {eid: (frozenset(str(x).split()) if str(x).strip() else frozenset()) for eid, x in zip(chunk_s1_unique, s1_df.loc[chunk_s1_unique, 'num_tokens'])}
    c_num_map = {cid: (frozenset(str(x).split()) if str(x).strip() else frozenset()) for cid, x in zip(chunk_c_unique, cands_df.loc[chunk_c_unique, 'num_tokens'])}
    s1_num = [s1_num_map[eid] for eid in s1_ids]
    c_num = [c_num_map[cid] for cid in cand_ids]
    num_jaccard = np.array([
        np.nan if (not a or not b) else (len(a & b) / len(a | b))
        for a, b in zip(s1_num, c_num)
    ], dtype=np.float32)
    del s1_num_map, c_num_map, s1_num, c_num

    # 11. rare_tok_overlap: count of shared tokens among each side's 3 rarest address tokens
    s1_r3_map = {eid: extract_rare3_set(addr, str(ctry), df_tokens) for eid, addr, ctry in zip(chunk_s1_unique, s1_df.loc[chunk_s1_unique, 'addr_norm'], s1_df.loc[chunk_s1_unique, 'country'])}
    c_r3_map = {cid: extract_rare3_set(addr, str(ctry), df_tokens) for cid, addr, ctry in zip(chunk_c_unique, cands_df.loc[chunk_c_unique, 'addr_norm'], cands_df.loc[chunk_c_unique, 'country'])}
    s1_r3 = [s1_r3_map[eid] for eid in s1_ids]
    c_r3 = [c_r3_map[cid] for cid in cand_ids]
    rare_tok_overlap = np.array([
        len(a & b) for a, b in zip(s1_r3, c_r3)
    ], dtype=np.float32)
    del s1_r3_map, c_r3_map, s1_r3, c_r3

    # 11b. name_freq: log1p(#records of the split sharing (country, core_name)), for S1 and candidate
    if name_freq is not None:
        name_freq_s1 = name_freq.log_freq(s1_sub['country'].astype(str).values, s1_sub['core_name'].values)
        name_freq_cand = name_freq.log_freq(cand_sub['country'].astype(str).values, cand_sub['core_name'].values)
    else:
        name_freq_s1 = np.full(n_pairs, np.nan, dtype=np.float32)
        name_freq_cand = np.full(n_pairs, np.nan, dtype=np.float32)

    # 11d. core_name string similarities (0..1 normalised for JW / Levenshtein / LCSseq, 0..100 for ratio)
    s1_cn_s = [str(x) for x in s1_cn]
    c_cn_s = [str(x) for x in c_cn]
    jw_core = process.cpdist(s1_cn_s, c_cn_s, scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)
    lev_core = process.cpdist(s1_cn_s, c_cn_s, scorer=Levenshtein.normalized_similarity, workers=-1).astype(np.float32)
    lcs_core = process.cpdist(s1_cn_s, c_cn_s, scorer=LCSseq.normalized_similarity, workers=-1).astype(np.float32)
    s1_cmp = {eid: _NON_ALNUM.sub("", str(x)) for eid, x in zip(chunk_s1_unique, s1_df.loc[chunk_s1_unique, 'core_name'])}
    c_cmp = {cid: _NON_ALNUM.sub("", str(x)) for cid, x in zip(chunk_c_unique, cands_df.loc[chunk_c_unique, 'core_name'])}
    compact_ratio = process.cpdist([s1_cmp[e] for e in s1_ids], [c_cmp[c] for c in cand_ids],
                                   scorer=fuzz.ratio, workers=-1).astype(np.float32)
    del s1_cmp, c_cmp
    initials_match = np.fromiter((_acronym_match(a, b) for a, b in zip(s1_cn_s, c_cn_s)), dtype=np.float32, count=n_pairs)
    first_tok_match = np.fromiter(((a.split()[:1] == b.split()[:1]) and bool(a.split()) for a, b in zip(s1_cn_s, c_cn_s)),
                                  dtype=np.float32, count=n_pairs)
    del s1_cn_s, c_cn_s

    # 11e. longnum_match: both addresses share a digit run of length >= 5 (1), none shared (0), either has none (-1)
    s1_ln = {eid: frozenset(_LONGNUM.findall(str(x))) for eid, x in zip(chunk_s1_unique, s1_df.loc[chunk_s1_unique, 'addr_norm'])}
    c_ln = {cid: frozenset(_LONGNUM.findall(str(x))) for cid, x in zip(chunk_c_unique, cands_df.loc[chunk_c_unique, 'addr_norm'])}
    longnum_match = np.fromiter(((-1.0 if not (a := s1_ln[e]) or not (b := c_ln[c]) else float(bool(a & b)))
                                 for e, c in zip(s1_ids, cand_ids)), dtype=np.float32, count=n_pairs)
    del s1_ln, c_ln

    # 11c. dict_name_ratio: token_sort_ratio(S1 name_full, candidate name_full after the Channel K token map)
    if name_map:
        c_map = {cid: " ".join(name_map.get(t, t) for t in str(x).split())
                 for cid, x in zip(chunk_c_unique, cands_df.loc[chunk_c_unique, 'name_full'])}
        c_dict_names = [c_map[cid] for cid in cand_ids]
        del c_map
    else:
        c_dict_names = c_nf
    dict_name_ratio = process.cpdist(s1_nf, c_dict_names, scorer=fuzz.token_sort_ratio, workers=-1).astype(np.float32)
    del c_dict_names

    # 11f. Feature pack 2: street key equality/ratio, legal-form conflict, address / street-key frequencies
    s1_k = s1_sub['street_key'].values if 'street_key' in s1_sub.columns else np.full(n_pairs, "", dtype=object)
    c_k = cand_sub['street_key'].values if 'street_key' in cand_sub.columns else np.full(n_pairs, "", dtype=object)
    k_missing = (s1_k == "") | (c_k == "")
    k_eq = np.where(k_missing, -1, (s1_k == c_k).astype(np.int8)).astype(np.float32)
    k_ratio = process.cpdist(list(s1_k), list(c_k), scorer=fuzz.ratio, workers=-1).astype(np.float32)
    k_ratio[k_missing] = -1.0
    s1_lg = s1_sub['legal2'].values if 'legal2' in s1_sub.columns else np.full(n_pairs, "", dtype=object)
    c_lg = cand_sub['legal2'].values if 'legal2' in cand_sub.columns else np.full(n_pairs, "", dtype=object)
    legal_conflict = np.fromiter(((-1.0 if not a or not b else (0.0 if set(a.split()) & set(b.split()) else 1.0))
                                  for a, b in zip(s1_lg, c_lg)), dtype=np.float32, count=n_pairs)
    if key_freqs:
        s1_ctry = s1_sub['country'].astype(str).values
        c_ctry = cand_sub['country'].astype(str).values
        a_freq_s1 = key_freqs["a_s1"].log_freq(s1_ctry, s1_sub['addr_norm'].values)
        a_freq_cand = key_freqs["a_pool"].log_freq(c_ctry, cand_sub['addr_norm'].values)
        k_freq_s1 = key_freqs["k_s1"].log_freq(s1_ctry, s1_k)
        k_freq_cand = key_freqs["k_pool"].log_freq(c_ctry, c_k)
    else:
        a_freq_s1 = a_freq_cand = k_freq_s1 = k_freq_cand = np.zeros(n_pairs, dtype=np.float32)

    # 12. addr_missing_any: 1 if address empty on either side, else 0
    addr_missing_any = ((np.array(s1_ad) == '') | (np.array(c_ad) == '')).astype(np.float32)
    del s1_nf, c_nf, s1_cn, c_cn, s1_sk, c_sk, s1_ad, c_ad, s1_sub, cand_sub

    # 13. Channel flags and cand_source
    cand_source = chunk_df['cand_source'].values.astype(np.float32)
    ch_emb = chunk_df['ch_emb'].values.astype(np.float32)
    ch_addr = chunk_df['ch_addr'].values.astype(np.float32)
    ch_skel = chunk_df['ch_skel'].values.astype(np.float32)
    ch_rare = chunk_df.get('ch_rare', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_rev = chunk_df.get('ch_rev', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_rerank = chunk_df.get('ch_rerank', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_k1 = chunk_df.get('ch_k1', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_k3 = chunk_df.get('ch_k3', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_k5 = chunk_df.get('ch_k5', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_keyx = (ch_k1 + ch_k3 + ch_k5).astype(np.float32)
    ch_comb = chunk_df.get('ch_comb', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_tfidf = chunk_df.get('ch_tfidf', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    tfidf_rank = chunk_df.get('tfidf_rank', pd.Series(99, index=chunk_df.index)).values.astype(np.float32)
    tfidf_score = chunk_df.get('tfidf_score', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_revtf = chunk_df.get('ch_revtf', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    revtf_rank = chunk_df.get('revtf_rank', pd.Series(99, index=chunk_df.index)).values.astype(np.float32)
    ch_dict = chunk_df.get('ch_dict', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    n_channels = (ch_emb + ch_addr + ch_skel + ch_rare + ch_rev + ch_rerank + ch_k1 + ch_k3 + ch_k5 + ch_comb
                  + ch_tfidf + ch_revtf + ch_dict).astype(np.float32)
    # comb_cos: dot of the stored combined name+address embeddings (NaN when unavailable)
    if comb is not None:
        comb_cos = comb.cos(_arrow_ids_to_int(pa.array(s1_ids, type=pa.string())),
                            _arrow_ids_to_int(pa.array(cand_ids, type=pa.string())))
    else:
        comb_cos = np.full(n_pairs, np.nan, dtype=np.float32)

    # 14. Per-S1 context features: precomputed on FULL candidate list per S1
    gap_to_best = ctx_gap_to_best
    n_cands = ctx_n_cands

    # 15. reverse_rank (passed from global split calculation)
    reverse_rank = global_reverse_rank.astype(np.float32)

    # 16. support (precomputed on FULL candidate list per S1)
    support = ctx_support

    # 17. rule_score: 0.5*max(name_token_sort, core_ratio) + 0.3*addr_token_set + 20*(house_match==1)
    rule_score = (
        0.5 * np.maximum(name_token_sort, core_ratio)
        + 0.3 * np.nan_to_num(addr_token_set, nan=0.0)
        + 20.0 * (house_match == 1.0)
    ).astype(np.float32)

    feats_dict = {
        's1_id': s1_ids,
        'cand_id': cand_ids,
    }
    if 'label' in chunk_df.columns:
        feats_dict['label'] = chunk_df['label'].values.astype(np.int32)
    if 'is_competitor' in chunk_df.columns:
        feats_dict['is_competitor'] = chunk_df['is_competitor'].values.astype(np.int8)
    else:
        feats_dict['is_competitor'] = np.zeros(len(s1_ids), dtype=np.int8)

    feats_dict['rule_score'] = rule_score

    # Add features strictly in order of config.FEATURES
    computed_map = {
        'emb_score': emb_score, 'emb_rank': emb_rank, 'name_token_sort': name_token_sort,
        'name_token_set': name_token_set, 'core_ratio': core_ratio, 'core_partial': core_partial,
        'skel_ratio': skel_ratio, 'name_jaccard': name_jaccard, 'legal_match': legal_match,
        'dba_max': dba_max, 'aka_max': aka_max, 'len_diff': len_diff,
        'addr_token_set': addr_token_set, 'house_match': house_match, 'house_cand_match': house_cand_match,
        'num_jaccard': num_jaccard, 'rare_tok_overlap': rare_tok_overlap, 'zip_match': zip_match,
        'state_match': state_match, 'addr_missing_any': addr_missing_any, 'cand_source': cand_source,
        'ch_emb': ch_emb, 'ch_addr': ch_addr, 'ch_skel': ch_skel, 'ch_rare': ch_rare, 'ch_rev': ch_rev,
        'ch_rerank': ch_rerank, 'ch_keyx': ch_keyx, 'ch_comb': ch_comb,
        'ch_tfidf': ch_tfidf, 'tfidf_rank': tfidf_rank, 'tfidf_score': tfidf_score,
        'ch_revtf': ch_revtf, 'revtf_rank': revtf_rank, 'ch_dict': ch_dict, 'comb_cos': comb_cos,
        'name_freq_s1': name_freq_s1, 'name_freq_cand': name_freq_cand, 'dict_name_ratio': dict_name_ratio,
        'jw_core': jw_core, 'lev_core': lev_core, 'lcs_core': lcs_core, 'compact_ratio': compact_ratio,
        'k_eq': k_eq, 'k_ratio': k_ratio, 'legal_conflict': legal_conflict,
        'a_freq_s1': a_freq_s1, 'a_freq_cand': a_freq_cand, 'k_freq_s1': k_freq_s1, 'k_freq_cand': k_freq_cand,
        'initials_match': initials_match, 'first_tok_match': first_tok_match, 'longnum_match': longnum_match,
        'n_channels': n_channels, 'gap_to_best': gap_to_best, 'n_cands': n_cands,
        'reverse_rank': reverse_rank, 'support': support
    }

    if rel:
        computed_map.update(rel)
    # raw emb_score is kept as a column (decision-layer tie-breaks in s6/s7) even when it is not a model feature
    feats_dict['emb_score'] = emb_score
    for col in config.FEATURES:
        feats_dict[col] = computed_map[col]

    return pd.DataFrame(feats_dict)


def compute_global_reverse_ranks(cache_dir, split, chunk_files, sampled_s1_ids=None, val_s1_ids=None):
    """
    Computes global reverse rank per cand_id across all candidates of the split.
    Reads only (s1_id, cand_id, emb_score) per chunk with integer IDs to keep peak RAM
    well below 15 GB even for 9-crore-row train candidate tables.

    Also identifies feature rows to keep:
      - All candidate pairs for sampled S1
      - For competitor S1 (s1 not in sampled_s1), ONLY candidate pairs where cand_id in val_cand_ids.

    Returns:
      (chunk_rr_list, chunk_keep_mask_list, chunk_is_comp_list, needed_s1_ids, needed_cand_ids, counts_info)
    """
    print("Computing global reverse_rank across all candidate chunks...", flush=True)
    t0 = time.time()

    # --- Pass 1: read 3 columns only, convert IDs to int64, track chunk lengths ---
    int_chunk_dfs = []
    chunk_lens = []
    for ci, cf in enumerate(chunk_files):
        df = pd.read_parquet(cf, columns=['s1_id', 'cand_id', 'emb_score'])
        chunk_lens.append(len(df))
        df['s1_int'] = _id_to_int(df['s1_id'], validate=True)
        df['cid_int'] = _id_to_int(df['cand_id'], validate=True)
        df = df.drop(columns=['s1_id', 'cand_id'])
        int_chunk_dfs.append(df)
        rss = _rss_mb()
        print(f"  [reverse_rank] Read chunk {ci + 1}/{len(chunk_files)}: {len(df):,} rows  "
              f"(RSS {rss:.0f} MB)", flush=True)

    # --- Concatenate int64 frames and rank globally ---
    print(f"  Concatenating {len(int_chunk_dfs)} int64 frames ({sum(chunk_lens):,} rows)...", flush=True)
    all_int = pd.concat(int_chunk_dfs, ignore_index=True)
    del int_chunk_dfs
    import gc; gc.collect()
    rss = _rss_mb()
    print(f"  All pairs loaded: {len(all_int):,} rows  (RSS {rss:.0f} MB)", flush=True)

    all_int['reverse_rank'] = (
        all_int.groupby('cid_int')['emb_score']
        .rank(ascending=False, method='min')
        .astype(np.float32)
    )
    print(f"  reverse_rank computed in {time.time() - t0:.2f}s  (RSS {_rss_mb():.0f} MB)", flush=True)

    # --- Identify kept rows (sampled vs competitor) ---
    if sampled_s1_ids is not None:
        sampled_int_ids = set(_id_to_int(pd.Series(list(sampled_s1_ids))).values)
        is_sampled = all_int['s1_int'].isin(sampled_int_ids)

        if val_s1_ids is not None and len(val_s1_ids) > 0:
            val_int_ids = set(_id_to_int(pd.Series(list(val_s1_ids))).values)
            val_cid_mask = all_int['s1_int'].isin(val_int_ids)
            val_cid_ints = set(all_int.loc[val_cid_mask, 'cid_int'].values)
        else:
            val_cid_ints = set()

        is_comp = (~is_sampled) & (all_int['cid_int'].isin(val_cid_ints))
        keep_mask = is_sampled | is_comp
    else:
        # Test split or keep all
        keep_mask = pd.Series(True, index=all_int.index)
        is_comp = pd.Series(False, index=all_int.index)
        is_sampled = keep_mask

    # Calculate count statistics
    n_total_cands = len(all_int)
    n_sampled_rows = int(is_sampled.sum())
    n_comp_rows = int(is_comp.sum())
    n_kept_rows = int(keep_mask.sum())

    unique_sampled_s1_ints = set(all_int.loc[is_sampled, 's1_int'].unique())
    unique_comp_s1_ints = set(all_int.loc[is_comp, 's1_int'].unique())
    unique_s1_ints = unique_sampled_s1_ints | unique_comp_s1_ints

    kept_cid_ints = all_int.loc[keep_mask, 'cid_int'].values
    unique_cand_ints = set(np.unique(kept_cid_ints))

    # --- Add top-5 candidate IDs of each competitor S1 to needed set ---
    # Support computation needs embeddings for the top-5 candidates per S1.
    # For competitor S1, these top-5 might not be in the kept pairs.
    n_comp_top5_added = 0
    if unique_comp_s1_ints:
        comp_rows = all_int[all_int['s1_int'].isin(unique_comp_s1_ints)]
        top5_per_comp = (comp_rows
                         .sort_values(['s1_int', 'emb_score'], ascending=[True, False], kind='stable')
                         .groupby('s1_int')
                         .head(5))
        comp_top5_cids = set(top5_per_comp['cid_int'].values)
        n_comp_top5_added = len(comp_top5_cids - unique_cand_ints)
        unique_cand_ints |= comp_top5_cids
        del comp_rows, top5_per_comp, comp_top5_cids
        print(f"  Added {n_comp_top5_added:,} top-5 competitor cand IDs to needed set "
              f"(total: {len(unique_cand_ints):,})", flush=True)

    s2_cid_ints = {cid for cid in unique_cand_ints if 2_000_000_000_000 <= cid < 3_000_000_000_000}
    s3_cid_ints = {cid for cid in unique_cand_ints if cid >= 3_000_000_000_000}

    counts_info = {
        'total_candidate_rows': n_total_cands,
        'sampled_feature_rows': n_sampled_rows,
        'competitor_feature_rows': n_comp_rows,
        'total_feature_rows': n_kept_rows,
        'unique_sampled_s1': len(unique_sampled_s1_ints),
        'unique_competitor_s1': len(unique_comp_s1_ints),
        'unique_total_s1': len(unique_s1_ints),
        'unique_s2_cands': len(s2_cid_ints),
        'unique_s3_cands': len(s3_cid_ints),
        'unique_total_cands': len(unique_cand_ints),
        'comp_top5_cands_added': n_comp_top5_added,
    }

    needed_s1_ids = _int_to_id(unique_s1_ints)
    needed_cand_ids = _int_to_id(unique_cand_ints)

    # --- Slice back per chunk ---
    chunk_rr_list = []
    chunk_keep_mask_list = []
    chunk_is_comp_list = []
    offset = 0
    all_rr = all_int['reverse_rank'].values
    all_km = keep_mask.values
    all_ic = is_comp.values

    for clen in chunk_lens:
        chunk_rr_list.append(all_rr[offset : offset + clen].copy())
        chunk_keep_mask_list.append(all_km[offset : offset + clen].copy())
        chunk_is_comp_list.append(all_ic[offset : offset + clen].copy())
        offset += clen

    del all_int, all_rr, all_km, all_ic, kept_cid_ints
    import gc; gc.collect()

    print(f"Global reverse_rank & filtering done in {time.time() - t0:.2f}s  "
          f"(RSS {_rss_mb():.0f} MB)", flush=True)
    return chunk_rr_list, chunk_keep_mask_list, chunk_is_comp_list, needed_s1_ids, needed_cand_ids, counts_info


def print_acceptance_report(chunk_files, cache_dir, split, elapsed_time):
    """
    Computes summary statistics streaming chunk-by-chunk without loading all chunks
    into memory at once, then prints the acceptance verification table and runtime statistics.
    Returns: None.
    """
    target_paths = []
    for idx, cf in enumerate(chunk_files):
        chunk_suffix = os.path.basename(cf).replace(f"cands_{split}_", "").replace(".parquet", "")
        out_p = os.path.join(cache_dir, f"feats_{split}_{chunk_suffix}.parquet")
        if os.path.exists(out_p):
            target_paths.append(out_p)

    if not target_paths:
        print("No feature rows produced.")
        return

    n_pairs = 0
    n_comp_pairs = 0
    comp_s1_set = set()
    has_labels = False
    has_comp = False

    f_min = {f: float('inf') for f in config.FEATURES}
    f_max = {f: float('-inf') for f in config.FEATURES}
    f_sum = {f: 0.0 for f in config.FEATURES}
    f_nan = {f: 0 for f in config.FEATURES}
    f_pos_sum = {f: 0.0 for f in config.FEATURES}
    f_pos_cnt = {f: 0 for f in config.FEATURES}
    f_neg_sum = {f: 0.0 for f in config.FEATURES}
    f_neg_cnt = {f: 0 for f in config.FEATURES}

    rs_min = float('inf')
    rs_max = float('-inf')
    rs_sum = 0.0
    rs_pos_sum = 0.0
    rs_pos_cnt = 0
    rs_neg_sum = 0.0
    rs_neg_cnt = 0

    for tp in target_paths:
        df_chunk = pd.read_parquet(tp)
        clen = len(df_chunk)
        if clen == 0:
            continue
        n_pairs += clen

        labels = df_chunk['label'].values if 'label' in df_chunk.columns else None
        if labels is not None:
            has_labels = True

        if 'is_competitor' in df_chunk.columns:
            has_comp = True
            c_mask = df_chunk['is_competitor'] == 1
            n_comp_pairs += int(c_mask.sum())
            if c_mask.any():
                comp_s1_set.update(df_chunk.loc[c_mask, 's1_id'].unique())

        rs = df_chunk['rule_score'].values.astype(np.float32)
        rs_min = min(rs_min, float(rs.min()))
        rs_max = max(rs_max, float(rs.max()))
        rs_sum += float(rs.sum())
        if labels is not None:
            rs_pos_sum += float(rs[labels == 1].sum())
            rs_pos_cnt += int((labels == 1).sum())
            rs_neg_sum += float(rs[labels == 0].sum())
            rs_neg_cnt += int((labels == 0).sum())

        for fname in config.FEATURES:
            vals = df_chunk[fname].values.astype(np.float32)
            nan_m = np.isnan(vals)
            f_nan[fname] += int(nan_m.sum())
            valid = ~nan_m
            if valid.any():
                v_valid = vals[valid]
                f_min[fname] = min(f_min[fname], float(v_valid.min()))
                f_max[fname] = max(f_max[fname], float(v_valid.max()))
                f_sum[fname] += float(v_valid.sum())
                if labels is not None:
                    pos_m = (labels == 1) & valid
                    neg_m = (labels == 0) & valid
                    f_pos_sum[fname] += float(vals[pos_m].sum())
                    f_pos_cnt[fname] += int(pos_m.sum())
                    f_neg_sum[fname] += float(vals[neg_m].sum())
                    f_neg_cnt[fname] += int(neg_m.sum())

        del df_chunk

    print("\n" + "=" * 90)
    print(f"STEP 6 FEATURE VERIFICATION REPORT ({n_pairs:,} candidate pairs)")
    print("=" * 90)
    if has_comp:
        print(f"Competitor S1: {len(comp_s1_set):,} entities ({n_comp_pairs:,} candidate pairs)")
        print("-" * 90)

    print(f"{'Feature':<20} | {'Min':>7} | {'Max':>7} | {'Mean':>8} | {'% NaN':>6} | {'Pos Mean':>9} | {'Neg Mean':>9}")
    print("-" * 90)

    for fname in config.FEATURES:
        fmin = f_min[fname] if f_min[fname] != float('inf') else float('nan')
        fmax = f_max[fname] if f_max[fname] != float('-inf') else float('nan')
        valid_cnt = n_pairs - f_nan[fname]
        fmean = (f_sum[fname] / valid_cnt) if valid_cnt > 0 else float('nan')
        pct_nan = (f_nan[fname] / n_pairs) * 100 if n_pairs > 0 else 0.0

        if has_labels:
            pos_mean = (f_pos_sum[fname] / f_pos_cnt[fname]) if f_pos_cnt[fname] > 0 else float('nan')
            neg_mean = (f_neg_sum[fname] / f_neg_cnt[fname]) if f_neg_cnt[fname] > 0 else float('nan')
            print(f"{fname:<20} | {fmin:7.2f} | {fmax:7.2f} | {fmean:8.2f} | {pct_nan:5.1f}% | {pos_mean:9.2f} | {neg_mean:9.2f}")
        else:
            print(f"{fname:<20} | {fmin:7.2f} | {fmax:7.2f} | {fmean:8.2f} | {pct_nan:5.1f}% | {'N/A':>9} | {'N/A':>9}")

    print("-" * 90)
    rs_mean = rs_sum / n_pairs if n_pairs > 0 else float('nan')
    if has_labels:
        rs_pos_mean = rs_pos_sum / rs_pos_cnt if rs_pos_cnt > 0 else float('nan')
        rs_neg_mean = rs_neg_sum / rs_neg_cnt if rs_neg_cnt > 0 else float('nan')
        print(f"{'rule_score':<20} | {rs_min:7.2f} | {rs_max:7.2f} | {rs_mean:8.2f} | {'0.0%':>6} | {rs_pos_mean:9.2f} | {rs_neg_mean:9.2f}")
    else:
        print(f"{'rule_score':<20} | {rs_min:7.2f} | {rs_max:7.2f} | {rs_mean:8.2f} | {'0.0%':>6} | {'N/A':>9} | {'N/A':>9}")
    print("=" * 90)

    # Runtime and extrapolation
    time_per_1m = (elapsed_time / n_pairs) * 1_000_000 if n_pairs > 0 else 0
    print(f"\nTiming:")
    print(f"  Processed {n_pairs:,} candidate pairs in {elapsed_time:.2f}s")
    print(f"  Laptop CPU rate: {time_per_1m:.2f}s per 1M candidate pairs")

    est_train_sample = (16_000_000 / 1_000_000) * time_per_1m / 60.0
    est_test_8m = (8_000_000 / 1_000_000) * time_per_1m / 60.0
    est_test_70m = (70_000_000 / 1_000_000) * time_per_1m / 60.0
    print(f"  Estimated EC2 runtime (8 cores):")
    print(f"    - Full Train candidate pairs (~16M pairs): ~{est_train_sample:.1f} minutes")
    print(f"    - Test candidate pairs (capped ~8M pairs): ~{est_test_8m:.1f} minutes")
    print(f"    - Test candidate pairs (uncapped ~70M pairs): ~{est_test_70m:.1f} minutes")


def main():
    """
    Main orchestration function for Step 6 pairwise feature engineering.
    Returns: None.
    """
    args = parse_args()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(config.CACHE_DIR, "laptop_test") if args.laptop_test else config.CACHE_DIR

    t_start = time.time()
    print(f"=== Step 6: Pairwise Feature Engineering ===")
    print(f"Split: {args.split}")
    print(f"Cache Directory: {cache_dir}")
    print(f"Total features configured: {len(config.FEATURES)}")
    print(f"Initial RSS: {_rss_mb():.0f} MB", flush=True)

    # 0. For train split, load split.parquet to determine which S1 get features.
    sampled_s1_ids = None  # None means "keep all" (test split)
    val_s1_ids = None
    if args.split == "train":
        split_path = os.path.join(cache_dir, "split.parquet")
        if os.path.exists(split_path):
            split_df = pd.read_parquet(split_path)
            sampled_s1_ids = set(split_df['s1_id'].values)
            val_s1_ids = set(split_df.loc[split_df['fold'] == 'val', 's1_id'].values)
            print(f"Loaded split.parquet: {len(sampled_s1_ids):,} sampled S1 ({len(val_s1_ids):,} val S1) for feature extraction. "
                  f"(RSS {_rss_mb():.0f} MB)", flush=True)
        else:
            print("WARNING: split.parquet not found; computing features for ALL S1.")

    # 1. Locate candidate files
    chunk_pattern_prefix = f"cands_{args.split}_chunk_"
    raw_files = [f for f in os.listdir(cache_dir) if f.startswith(chunk_pattern_prefix) and f.endswith(".parquet")]
    chunk_files = [
        os.path.join(cache_dir, f)
        for f in sorted(
            raw_files,
            key=lambda x: int(re.search(r"chunk_(\d+)", x).group(1)) if re.search(r"chunk_(\d+)", x) else x
        )
    ]

    if not chunk_files:
        raise FileNotFoundError(
            f"No candidate chunk files (cands_{args.split}_chunk_*.parquet) found in '{cache_dir}'. "
            f"Run s3_block.py --split {args.split} first. "
            f"The old single-file fallback (cands_{args.split}.parquet) has been removed "
            f"because it required a 9-crore-row concat that caused OOM on EC2."
        )

    print(f"Found {len(chunk_files)} candidate chunk file(s).")

    # 2. Compute global reverse ranks and identify kept feature rows & needed IDs
    chunk_rr_list, chunk_keep_masks, chunk_is_comps, needed_s1_ids, needed_cand_ids, counts_info = (
        compute_global_reverse_ranks(cache_dir, args.split, chunk_files, sampled_s1_ids, val_s1_ids)
    )

    # 3. Print count report
    print("\n" + "=" * 80)
    print("STEP 6 FEATURE ROWS & ENTITY ID COUNT REPORT")
    print("=" * 80)
    print(f"Total Candidate Pairs in Chunks : {counts_info['total_candidate_rows']:,}")
    print(f"Sampled Feature Rows            : {counts_info['sampled_feature_rows']:,}")
    print(f"Competitor Feature Rows         : {counts_info['competitor_feature_rows']:,}")
    print(f"Total Feature Rows Needed       : {counts_info['total_feature_rows']:,}")
    print("-" * 80)
    print(f"Unique Sampled S1 Entities      : {counts_info['unique_sampled_s1']:,}")
    print(f"Unique Competitor S1 Entities   : {counts_info['unique_competitor_s1']:,}")
    print(f"Total Unique S1 Entities Needed : {counts_info['unique_total_s1']:,}")
    print("-" * 80)
    print(f"Unique S2 Candidate Entities    : {counts_info['unique_s2_cands']:,}")
    print(f"Unique S3 Candidate Entities    : {counts_info['unique_s3_cands']:,}")
    print(f"Total Unique Candidates Needed  : {counts_info['unique_total_cands']:,}")
    print("=" * 80 + "\n", flush=True)

    if args.count_only:
        print(f"--count-only flag passed: exiting after count report. Elapsed: {time.time() - t_start:.2f}s (RSS {_rss_mb():.0f} MB)")
        return

    # 4. Check resume status
    pending_chunks = []
    for idx, cf in enumerate(chunk_files):
        chunk_suffix = os.path.basename(cf).replace(f"cands_{args.split}_", "").replace(".parquet", "")
        out_name = f"feats_{args.split}_{chunk_suffix}.parquet"
        out_p = os.path.join(cache_dir, out_name)

        if os.path.exists(out_p) and not args.force:
            print(f"Chunk {idx + 1}/{len(chunk_files)} ({out_name}) already processed, skipping.")
        else:
            pending_chunks.append((idx, cf, out_p))

    if not pending_chunks:
        print("\nAll feature chunks already computed. Loading results for acceptance checks...")
        print_acceptance_report(chunk_files, cache_dir, args.split, elapsed_time=0.0)
        return

    if args.max_chunks is not None:
        print(f"--max-chunks set to {args.max_chunks}: limiting processing to {args.max_chunks} chunk(s).", flush=True)
        pending_chunks = pending_chunks[:args.max_chunks]

    # 5. Address document frequencies (memory-light streaming / cached parquet)
    t_df_start = time.time()
    df_tokens = get_address_token_df(cache_dir, args.split)
    print(f"Address document frequencies ready in {time.time() - t_df_start:.2f}s (RSS {_rss_mb():.0f} MB)", flush=True)

    # 6. Load normalized entity tables restricted to only needed IDs and feature columns
    t_norm = time.time()
    print("\nLoading restricted normalized tables (only needed IDs & columns)...", flush=True)
    s1_norm = load_norm_table(cache_dir, args.split, "source1", needed_ids=needed_s1_ids, columns=NEEDED_NORM_COLS)
    print(f"  Loaded Source 1: {len(s1_norm):,} rows (RSS {_rss_mb():.0f} MB)", flush=True)

    s2_norm = load_norm_table(cache_dir, args.split, "source2", needed_ids=needed_cand_ids, columns=NEEDED_NORM_COLS)
    print(f"  Loaded Source 2: {len(s2_norm):,} rows (RSS {_rss_mb():.0f} MB)", flush=True)

    s3_norm = load_norm_table(cache_dir, args.split, "source3", needed_ids=needed_cand_ids, columns=NEEDED_NORM_COLS)
    print(f"  Loaded Source 3: {len(s3_norm):,} rows (RSS {_rss_mb():.0f} MB)", flush=True)

    cands_norm = pd.concat([s2_norm, s3_norm])
    del s2_norm, s3_norm
    import gc; gc.collect()
    print(f"Restricted normalized tables ready in {time.time() - t_norm:.2f}s "
          f"(total candidate records: {len(cands_norm):,}) (RSS {_rss_mb():.0f} MB)", flush=True)

    # 7. Load candidate embeddings restricted to needed candidate IDs
    t_emb = time.time()
    print("\nLoading restricted candidate embeddings...", flush=True)
    all_cand_emb, cand_id_map = load_candidate_embeddings(cache_dir, args.split, needed_cand_ids=needed_cand_ids)
    print(f"Candidate embeddings ready: {len(all_cand_emb):,} vectors in {time.time() - t_emb:.2f}s "
          f"(RSS {_rss_mb():.0f} MB)", flush=True)

    # 7b. Combined embeddings for comb_cos (memory-mapped; only the ID index is held in RAM)
    t_comb = time.time()
    try:
        comb = CombEmbeddings(cache_dir, args.split)
        print(f"Combined embeddings index ready: {len(comb.ids):,} IDs in {time.time() - t_comb:.1f}s "
              f"(RSS {_rss_mb():.0f} MB)", flush=True)
    except FileNotFoundError as e:
        comb = None
        print(f"WARNING: {e}; comb_cos will be NaN", flush=True)

    # 7c. Name frequency index and Channel K token map (name_freq, dict_name_ratio)
    t_nf = time.time()
    name_freq = NameFreq(cache_dir, args.split)
    key_freqs = {
        "a_s1": KeyFreq(cache_dir, args.split, ["source1"], "addr_norm"),
        "a_pool": KeyFreq(cache_dir, args.split, ["source2", "source3"], "addr_norm"),
        "k_s1": KeyFreq(cache_dir, args.split, ["source1"], "street_key"),
        "k_pool": KeyFreq(cache_dir, args.split, ["source2", "source3"], "street_key"),
    }
    print(f"Key frequency indexes: " + ", ".join(f"{k} {len(v.keys):,}" for k, v in key_freqs.items()), flush=True)
    name_map = load_name_map(cache_dir)
    print(f"Name frequency index: {len(name_freq.keys):,} (country, core_name) keys; token map {len(name_map):,} entries "
          f"in {time.time() - t_nf:.1f}s (RSS {_rss_mb():.0f} MB)", flush=True)

    # Baseline memory breakdown
    s1_mem = s1_norm.memory_usage(deep=True).sum() / 1_048_576
    cands_mem = cands_norm.memory_usage(deep=True).sum() / 1_048_576
    emb_mem = all_cand_emb.nbytes / 1_048_576
    id_map_mem = sys.getsizeof(cand_id_map) / 1_048_576
    rr_mem = sum(arr.nbytes for arr in chunk_rr_list if arr is not None) / 1_048_576
    mask_mem = sum(arr.nbytes for arr in chunk_keep_masks if arr is not None) / 1_048_576
    comp_mem = sum(arr.nbytes for arr in chunk_is_comps if arr is not None) / 1_048_576
    df_mem = sum(sys.getsizeof(v) for v in df_tokens.values()) / 1_048_576
    total_baseline_mem = s1_mem + cands_mem + emb_mem + id_map_mem + rr_mem + mask_mem + comp_mem + df_mem

    print("\n" + "=" * 80)
    print("STEP 6 BASELINE IN-MEMORY OBJECT SIZES")
    print("=" * 80)
    print(f"Source 1 Table (s1_norm)       : {s1_mem:8.1f} MB ({len(s1_norm):,} rows)")
    print(f"Candidate Table (cands_norm)   : {cands_mem:8.1f} MB ({len(cands_norm):,} rows)")
    print(f"Candidate Embeddings (float16) : {emb_mem:8.1f} MB ({len(all_cand_emb):,} vectors, {all_cand_emb.dtype})")
    print(f"Candidate ID Map (dict)        : {id_map_mem:8.1f} MB ({len(cand_id_map):,} entries)")
    print(f"Reverse Rank & Chunk Masks     : {rr_mem + mask_mem + comp_mem:8.1f} MB ({len(chunk_files)} chunks)")
    print(f"Address Document Frequencies   : {df_mem:8.1f} MB ({sum(len(v) for v in df_tokens.values()):,} tokens)")
    print("-" * 80)
    print(f"Total Baseline In-Memory Size  : {total_baseline_mem:8.1f} MB ({total_baseline_mem/1024:.2f} GB)")
    print(f"Current Process RSS            : {_rss_mb():8.1f} MB")
    print("=" * 80 + "\n", flush=True)

    # 8. Process each pending chunk
    #    Per-S1 context features (gap_to_best, n_cands, support) must be computed
    #    on the FULL candidate list of each S1 BEFORE filtering competitor S1 down
    #    to only the shared pairs. This is because gap_to_best depends on the best
    #    emb_score across all 40 candidates, n_cands should be 40, and support
    #    depends on the top-5 candidates by emb_score.
    t_feat_start = time.time()
    for idx, cf, out_p in pending_chunks:
        cf_basename = os.path.basename(cf)
        print(f"\nProcessing chunk {idx + 1}/{len(chunk_files)}: {cf_basename}...", flush=True)
        t_ch = time.time()
        chunk_full = pd.read_parquet(cf)
        chunk_rr = chunk_rr_list[idx]
        keep_mask = chunk_keep_masks[idx]
        is_comp = chunk_is_comps[idx]

        n_before = len(chunk_full)

        # Identify S1 IDs that have at least one kept row
        kept_s1_set = set(chunk_full.loc[keep_mask, 's1_id'].unique())

        if not kept_s1_set:
            print(f"  No sampled/competitor S1 in this chunk, saving empty frame to {out_p}.")
            pd.DataFrame(columns=['s1_id', 'cand_id']).to_parquet(out_p, index=False)
            chunk_rr_list[idx] = None
            chunk_keep_masks[idx] = None
            chunk_is_comps[idx] = None
            continue

        # Context: ALL rows for S1 IDs that have any kept row
        # This gives us the full candidate list for competitor S1
        context_mask = chunk_full['s1_id'].isin(kept_s1_set).values
        context_df = chunk_full[context_mask].reset_index(drop=True)
        ctx_keep = keep_mask[context_mask]
        ctx_rr = chunk_rr[context_mask]
        ctx_comp = is_comp[context_mask]
        n_context = len(context_df)
        del chunk_full
        chunk_rr_list[idx] = None
        chunk_keep_masks[idx] = None
        chunk_is_comps[idx] = None

        # Position of each kept row in the unbatched kept order (= old output row order)
        kept_pos = np.cumsum(ctx_keep) - 1
        n_kept = int(ctx_keep.sum())
        n_comp = int(ctx_comp[ctx_keep].sum())
        print(f"  Context: {n_before:,} -> {n_context:,} rows (full S1 lists), "
              f"kept: {n_kept:,} (sampled: {n_kept - n_comp:,}, competitor: {n_comp:,})", flush=True)

        # Sub-batches of whole S1 groups: context features (gap_to_best, n_cands, support)
        # only depend on the S1's own candidate list, so they are exact per sub-batch.
        batches = _s1_sub_batches(context_df['s1_id'], args.sub_batch_rows)
        part_tables = []
        part_pos = []
        for bi, rows in enumerate(batches):
            t_sb = time.time()
            sub_ctx = context_df.iloc[rows].reset_index(drop=True)
            ctx_gap, ctx_nc, ctx_sup = compute_context_features(sub_ctx, all_cand_emb, cand_id_map)
            ctx_rel = relative_features(sub_ctx, ctx_sup, comb)
            sub_keep = ctx_keep[rows]
            if sub_keep.any():
                kept_df = sub_ctx[sub_keep].reset_index(drop=True)
                kept_df['is_competitor'] = ctx_comp[rows][sub_keep].astype(np.int8)
                feats_df = compute_chunk_features(
                    kept_df, s1_norm, cands_norm, all_cand_emb, cand_id_map,
                    ctx_rr[rows][sub_keep], df_tokens,
                    ctx_gap[sub_keep], ctx_nc[sub_keep], ctx_sup[sub_keep], comb=None,
                    name_freq=name_freq, name_map=name_map,
                    rel={k: v[sub_keep] for k, v in ctx_rel.items()}, key_freqs=key_freqs
                )
                part_tables.append(pa.Table.from_pandas(feats_df, preserve_index=False))
                part_pos.append(kept_pos[rows][sub_keep])
                n_sb_kept = len(kept_df)
                del kept_df, feats_df
            else:
                n_sb_kept = 0
            del sub_ctx, ctx_gap, ctx_nc, ctx_sup, sub_keep, ctx_rel
            _free_memory()
            print(f"    Sub-batch {bi + 1}/{len(batches)}: {len(rows):,} context rows, "
                  f"{n_sb_kept:,} feature rows in {time.time() - t_sb:.2f}s  (RSS {_rss_mb():.0f} MB)", flush=True)

        del context_df, ctx_keep, ctx_rr, ctx_comp, kept_pos, batches
        _free_memory()

        # Reassemble in the original kept-row order and write
        feats_tbl = pa.concat_tables(part_tables)
        del part_tables
        pos = np.concatenate(part_pos)
        del part_pos
        if not np.all(pos[1:] > pos[:-1]):
            feats_tbl = feats_tbl.take(pa.array(np.argsort(pos, kind='stable')))
        n_feats = feats_tbl.num_rows
        pq.write_table(feats_tbl, out_p)
        del feats_tbl, pos

        rss_aft = _rss_mb()
        print(f"Saved chunk {idx + 1}/{len(chunk_files)} features ({n_feats:,} pairs) to {out_p} "
              f"in {time.time() - t_ch:.2f}s  (RSS {rss_aft:.0f} MB)", flush=True)

        del context_mask
        _free_memory()

    t_feat_total = time.time() - t_feat_start

    # 9. Summary report and validation (streaming chunk-by-chunk, zero accumulation)
    print_acceptance_report(chunk_files, cache_dir, args.split, elapsed_time=t_feat_total)

    print(f"\nStep 6 finished in {time.time() - t_start:.2f}s. Final RSS: {_rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()