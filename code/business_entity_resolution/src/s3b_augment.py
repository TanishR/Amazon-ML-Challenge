#!/usr/bin/env python3
"""
Candidate Augmentation (Stage 3b): Adds high-precision candidate pairs on top of the
existing candidate pairs in cands_{split}_chunk_*.parquet to resolve recall bottlenecks.

Channels:
  Channel F (Wide Name Retrieval + Address Re-ranking, GPU):
    - Top 200 nearest neighbors by name embedding similarity (main + alt max, like Channel A).
    - Re-ranked on GPU using int32 address token ID arrays (numeric tokens + alpha tokens len >= 4).
    - Scored by sum of IDF of shared tokens; tie-break by emb_score.
    - Top 10 per source (not already in existing candidates) kept.
  Channel G (Address Keys):
    - Keys K5, K3, K1 within country and per source.
    - Block size cap <= 50 (larger blocks dropped).
    - At most 10 per S1 per key, prioritized by emb_score.
  Channel H (Combined Name + Address Embeddings, Optional):
    - Active if cache/embc_{split}_{source}.npy exists for both S1 and target source.
    - Top 20 nearest neighbors by combined embedding similarity (GPU chunked, score matrix <= 3 GB).
    - Sets ch_comb = 1.
  Channel I (char TF-IDF, Optional):
    - Active if cache/tfidf_{split}_{source}.parquet exists (built by s3c_tfidf.py, top-10 per S1 and source).
    - Applied after the F/G/H merge: rows already in the chunk that TF-IDF found get ch_tfidf = 1 and their
      tfidf_rank; TF-IDF pairs not yet in the chunk are added, at most 10 per S1 (by rank, then score),
      on top of the existing rows. tfidf_rank = 99 for rows outside the TF-IDF top-10.
    - Idempotency: a chunk that already has ch_comb but no ch_tfidf gets only Channel I.

Merge Rules:
  - If Channel H active: extra cap is 30 per S1.
    Priority: number of new channels, then ch_comb, then F score, then emb_score.
  - If Channel H not active: extra cap is 20 per S1.
    Priority: number of new channels, then F score, then emb_score.
  - Never remove or alter existing rows.
  - Columns: ch_rerank, ch_k1, ch_k3, ch_k5, and ch_comb (if H active).

Memory Optimization (Peak RSS < 18 GB):
  - Embeddings loaded via mmap_mode='r' (float16).
  - IDs kept as int64 via _id_to_int.
  - Address tokens stored as compact (N, 24) int32 numpy arrays (no Python lists/sets).
  - Intermediates explicitly freed with gc.collect() between channels and chunks.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from config import CAND_CAP, EMB_DIM, K_PER_SOURCE


def _rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


# ==============================================================================
# INT64 ID ENCODING / DECODING
# ==============================================================================

def _id_to_int_single(eid: str) -> int:
    s = str(eid).strip()
    prefix = int(s[1]) * 1_000_000_000_000
    num = int(s.split("-", 1)[1])
    return prefix + num


def _id_to_int(series: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(series, pd.Series):
        s = series.astype(str)
    else:
        s = pd.Series(series).astype(str)
    prefix = s.str[1].map({"1": 1_000_000_000_000, "2": 2_000_000_000_000, "3": 3_000_000_000_000}).astype(np.int64)
    numeric = s.str.split("-", n=1).str[1].astype(np.int64)
    return (prefix + numeric).values


def _int_to_id(arr: np.ndarray | list[int]) -> list[str]:
    a = np.asarray(arr, dtype=np.int64)
    src = a // 1_000_000_000_000
    num = a % 1_000_000_000_000
    return [f"S{s}-{n}" for s, n in zip(src, num)]


def _arrow_id_to_int(col) -> np.ndarray:
    """'S2-12345' arrow string column -> int64 (same encoding as _id_to_int), without Python strings."""
    digit = pc.cast(pc.utf8_slice_codeunits(col, 1, 2), "int64")
    num = pc.cast(pc.utf8_slice_codeunits(col, 3), "int64")
    return pc.add(pc.multiply(digit, 1_000_000_000_000), num).to_numpy()


# ==============================================================================
# CHANNEL I: CHAR TF-IDF PAIRS (from s3c_tfidf.py)
# ==============================================================================

TFIDF_MAX_NEW = 10
TFIDF_RANK_MISSING = 99


def load_tfidf_pairs(cache_dir: str, split: str) -> Optional[dict[str, np.ndarray]]:
    """Loads cache/tfidf_{split}_{source}.parquet for source2/3 as int64-encoded arrays sorted by S1."""
    s1s, cands, scores, ranks = [], [], [], []
    for source in ("source2", "source3"):
        p = os.path.join(cache_dir, f"tfidf_{split}_{source}.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p, columns=["s1_id", "cand_id", "tfidf_score", "tfidf_rank"])
        s1s.append(_arrow_id_to_int(t["s1_id"]))
        cands.append(_arrow_id_to_int(t["cand_id"]))
        scores.append(t["tfidf_score"].to_numpy().astype(np.float32))
        ranks.append(t["tfidf_rank"].to_numpy().astype(np.int8))
        print(f"Loaded Channel I pairs from {p}: {t.num_rows:,} (RSS: {_rss_mb():.1f} MB)")
        del t
    if not s1s:
        return None
    s1 = np.concatenate(s1s)
    order = np.argsort(s1, kind="stable")
    return {"s1": s1[order], "cand": np.concatenate(cands)[order],
            "score": np.concatenate(scores)[order], "rank": np.concatenate(ranks)[order]}


def _rows_for_s1(sorted_s1: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Indices of all entries of sorted_s1 whose value is in wanted."""
    u = np.unique(wanted)
    lo = np.searchsorted(sorted_s1, u, "left")
    hi = np.searchsorted(sorted_s1, u, "right")
    lens = hi - lo
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(lo - (np.cumsum(lens) - lens), lens)
    return np.arange(total, dtype=np.int64) + starts


def apply_channel_i(
    aug_df: pd.DataFrame,
    tfidf: dict[str, np.ndarray],
    emb_scorer,
    gt_pairs_int: set,
    split: str,
    max_new: int = TFIDF_MAX_NEW,
) -> tuple[pd.DataFrame, int, int]:
    """
    Marks rows TF-IDF found (ch_tfidf = 1, tfidf_rank) and appends TF-IDF pairs that are not yet in the
    chunk, at most max_new per S1 (lowest rank first, then highest score).
    Returns (augmented DataFrame, number of existing rows marked, number of new rows).
    """
    s1_int = _id_to_int(aug_df["s1_id"])
    c_int = _id_to_int(aug_df["cand_id"])
    sel = _rows_for_s1(tfidf["s1"], s1_int)
    T = pd.DataFrame({"s1": tfidf["s1"][sel], "cand": tfidf["cand"][sel],
                      "score": tfidf["score"][sel], "rank": tfidf["rank"][sel]})
    keys = pd.DataFrame({"s1": s1_int, "cand": c_int, "row": np.arange(len(aug_df), dtype=np.int64)})
    m = T.merge(keys, on=["s1", "cand"], how="left")
    hit = m["row"].notna().values

    ch = np.zeros(len(aug_df), dtype=np.int8)
    rk = np.full(len(aug_df), TFIDF_RANK_MISSING, dtype=np.int8)
    rows = m.loc[hit, "row"].values.astype(np.int64)
    ch[rows] = 1
    rk[rows] = m.loc[hit, "rank"].values
    aug_df = aug_df.copy()
    aug_df["ch_tfidf"] = ch
    aug_df["tfidf_rank"] = rk
    n_marked = int(len(np.unique(rows)))

    new = (m.loc[~hit, ["s1", "cand", "score", "rank"]]
           .sort_values(["s1", "rank", "score"], ascending=[True, True, False], kind="stable")
           .groupby("s1", sort=False).head(max_new))
    if len(new) == 0:
        return aug_df, n_marked, 0

    ns1 = new["s1"].values.astype(np.int64)
    nc = new["cand"].values.astype(np.int64)
    is_s2 = (nc // 1_000_000_000_000) == 2
    cols: dict[str, Any] = {
        "s1_id": _int_to_id(ns1),
        "cand_id": _int_to_id(nc),
        "cand_source": np.where(is_s2, 0, 1),
        "emb_score": emb_scorer(ns1, nc, is_s2),
        "emb_rank": np.float32(999.0),
        "ch_tfidf": np.int8(1),
        "tfidf_rank": new["rank"].values.astype(np.int8),
    }
    for c in ("ch_emb", "ch_addr", "ch_skel", "ch_rare", "ch_rev", "ch_rerank", "ch_k1", "ch_k3", "ch_k5", "ch_comb"):
        if c in aug_df.columns:
            cols[c] = np.int8(0)
    if "label" in aug_df.columns:
        cols["label"] = np.fromiter(((int(a), int(b)) in gt_pairs_int for a, b in zip(ns1, nc)), dtype=bool, count=len(ns1))
    missing = set(aug_df.columns) - set(cols)
    if missing:
        raise RuntimeError(f"Channel I: no value for chunk columns {sorted(missing)}")
    new_df = pd.DataFrame({c: cols[c] for c in aug_df.columns}, index=range(len(ns1)))
    for c in aug_df.columns:
        new_df[c] = new_df[c].astype(aug_df[c].dtype)
    return pd.concat([aug_df, new_df], ignore_index=True), n_marked, len(new_df)


FGH_FLAGS = ("ch_rerank", "ch_k1", "ch_k3", "ch_k5", "ch_comb")
BASE_FLAGS = ("ch_emb", "ch_addr", "ch_skel", "ch_rare", "ch_rev")


def _val_stage_rows(df: pd.DataFrame, val_s1: set) -> pd.DataFrame:
    """
    Val-S1 rows of a chunk as int64 (s1, cand) with the stage that produced them:
      0 = original S3 row, 1 = added by F/G/H, 2 = added by Channel I (only ch_tfidf set).
    """
    d = df[df["s1_id"].isin(val_s1)]
    z = np.zeros(len(d), dtype=bool)
    fgh = np.zeros(len(d), dtype=bool)
    for c in FGH_FLAGS:
        if c in d.columns:
            fgh |= d[c].values > 0
    base = z.copy()
    for c in BASE_FLAGS:
        if c in d.columns:
            base |= d[c].values > 0
    i_new = (d["ch_tfidf"].values > 0) & ~fgh & ~base if "ch_tfidf" in d.columns else z
    stage = np.where(i_new, 2, np.where(fgh, 1, 0)).astype(np.int8)
    return pd.DataFrame({"s1": _id_to_int(d["s1_id"]), "cand": _id_to_int(d["cand_id"]), "stage": stage})


# ==============================================================================
# CANDIDATE KEYS K1, K3, K5
# ==============================================================================

def extract_k1(country: str, num_tokens_str: str) -> set[str]:
    """K1: (country, two consecutive numeric tokens in the address, each of length >= 2)."""
    if not country or not num_tokens_str or num_tokens_str == "nan":
        return set()
    nums = [t for t in num_tokens_str.split() if len(t) >= 2]
    keys = set()
    for i in range(len(nums) - 1):
        keys.add(f"K1_{country}_{nums[i]}_{nums[i+1]}")
    return keys


def extract_k3(country: str, num_tokens_str: str) -> set[str]:
    """K3: (country, sorted set of all numeric address tokens)."""
    if not country or not num_tokens_str or num_tokens_str == "nan":
        return set()
    nums = sorted(set(num_tokens_str.split()))
    if not nums:
        return set()
    return {f"K3_{country}_" + "_".join(nums)}


def extract_k5(country: str, num_tokens_str: str, addr_norm_str: str, doc_freqs: dict[tuple[str, str], int]) -> set[str]:
    """K5: (country, any numeric token of length >= 3, rarest alphabetic address token of length >= 5)."""
    if not country or not num_tokens_str or num_tokens_str == "nan":
        return set()
    nums_ge3 = [t for t in set(num_tokens_str.split()) if len(t) >= 3]
    if not nums_ge3:
        return set()
    alpha_toks = [t for t in set(str(addr_norm_str).split()) if t.isalpha() and len(t) >= 5]
    if not alpha_toks:
        return set()
    rarest = min(alpha_toks, key=lambda t: (doc_freqs.get((country, t), 0), t))
    keys = set()
    for n in nums_ge3:
        keys.add(f"K5_{country}_{n}_{rarest}")
    return keys


# ==============================================================================
# TOKENIZATION & IDF COMPUTATION FOR CHANNEL F
# ==============================================================================

def tokenize_address_f(addr_str: str) -> list[str]:
    """Numeric tokens + alpha tokens of length >= 4, deduplicated in order of appearance."""
    if not addr_str or not isinstance(addr_str, str) or addr_str == "nan":
        return []
    toks = []
    seen = set()
    for t in addr_str.split():
        if t.isdigit() or (t.isalpha() and len(t) >= 4):
            if t not in seen:
                seen.add(t)
                toks.append(t)
    return toks


def load_doc_freqs(cache_dir: str, split: str) -> dict[tuple[str, str], int]:
    """Loads cache/addr_df_{split}.parquet, computing it if missing."""
    path = os.path.join(cache_dir, f"addr_df_{split}.parquet")
    if not os.path.exists(path):
        from s4_features import get_address_token_df
        print(f"Document frequencies file '{path}' missing. Computing for split '{split}'...", flush=True)
        try:
            get_address_token_df(cache_dir, split)
        except Exception as e:
            print(f"Warning: Failed to compute {path}: {e}")

    doc_freqs: dict[tuple[str, str], int] = {}
    if os.path.exists(path):
        print(f"Loading document frequencies from {path}...")
        df_p = pd.read_parquet(path)
        for ctry, tok, df_val in zip(df_p["country"], df_p["token"], df_p["doc_freq"]):
            doc_freqs[(str(ctry), str(tok))] = int(df_val)
        print(f"  Loaded {len(doc_freqs):,} document frequencies (RSS: {_rss_mb():.1f} MB)")
    else:
        print(f"Warning: {path} not found. IDF scores will use default frequencies.")
    return doc_freqs


# ==============================================================================
# EMBEDDINGS LOADING HELPERS (MMAP)
# ==============================================================================

def load_embeddings_mmap(
    cache_dir: str, split: str, source: str
) -> tuple[np.ndarray, dict[int, int], np.ndarray, np.ndarray, dict[int, int], np.ndarray]:
    """Loads main and alt name embeddings via mmap with int64 IDs."""
    m_p = os.path.join(cache_dir, f"emb_{split}_{source}.npy")
    m_id_p = os.path.join(cache_dir, f"ids_{split}_{source}.npy")

    if not os.path.exists(m_p) or not os.path.exists(m_id_p):
        return (
            np.zeros((0, EMB_DIM), dtype=np.float16),
            {},
            np.array([], dtype=np.int64),
            np.zeros((0, EMB_DIM), dtype=np.float16),
            {},
            np.array([], dtype=np.int64),
        )

    main_emb = np.load(m_p, mmap_mode="r")
    main_ids_raw = np.load(m_id_p, allow_pickle=True)
    main_ids_int = _id_to_int(main_ids_raw)
    main_map = {int(eid_int): idx for idx, eid_int in enumerate(main_ids_int)}

    a_p = os.path.join(cache_dir, f"emb_{split}_{source}_alt.npy")
    a_id_p = os.path.join(cache_dir, f"ids_{split}_{source}_alt.npy")

    if os.path.exists(a_p) and os.path.exists(a_id_p):
        alt_emb = np.load(a_p, mmap_mode="r")
        alt_ids_raw = np.load(a_id_p, allow_pickle=True)
        alt_ids_int = _id_to_int(alt_ids_raw)
        alt_map = {int(eid_int): idx for idx, eid_int in enumerate(alt_ids_int)}
    else:
        alt_emb = np.zeros((0, EMB_DIM), dtype=np.float16)
        alt_ids_int = np.array([], dtype=np.int64)
        alt_map = {}

    return main_emb, main_map, main_ids_int, alt_emb, alt_map, alt_ids_int


def load_combined_embeddings_mmap(
    cache_dir: str, split: str, source: str
) -> tuple[Optional[np.ndarray], dict[int, int]]:
    """Loads combined name+address embeddings via mmap with int64 IDs if available."""
    alias_map = {"source1": "s1", "source2": "s2", "source3": "s3"}
    s_alt = alias_map.get(source, source)

    c_paths = [
        (os.path.join(cache_dir, f"embc_{split}_{source}.npy"), os.path.join(cache_dir, f"idsc_{split}_{source}.npy")),
        (os.path.join(cache_dir, f"embc_{split}_{s_alt}.npy"), os.path.join(cache_dir, f"idsc_{split}_{s_alt}.npy")),
    ]

    for emb_p, ids_p in c_paths:
        if os.path.exists(emb_p) and os.path.exists(ids_p):
            embc = np.load(emb_p, mmap_mode="r")
            idsc_raw = np.load(ids_p, allow_pickle=True)
            idsc_int = _id_to_int(idsc_raw)
            comb_map = {int(eid_int): idx for idx, eid_int in enumerate(idsc_int)}
            print(f"Loaded Channel H combined embeddings for {source} from {emb_p} ({len(idsc_int):,} vectors)")
            return embc, comb_map

    return None, {}


def combined_embeddings_exist(cache_dir: str, split: str, source: str) -> bool:
    """True if load_combined_embeddings_mmap would find embc/idsc files for this source."""
    s_alt = {"source1": "s1", "source2": "s2", "source3": "s3"}.get(source, source)
    return any(
        os.path.exists(os.path.join(cache_dir, f"embc_{split}_{s}.npy")) and os.path.exists(os.path.join(cache_dir, f"idsc_{split}_{s}.npy"))
        for s in (source, s_alt)
    )


# ==============================================================================
# S2 & S3 TARGET INDEX PREPARATION
# ==============================================================================

class TargetSourceIndex:
    """Precomputed target index for a source (source2 or source3)."""

    def __init__(
        self,
        source_name: str,
        df: pd.DataFrame,
        main_emb: np.ndarray,
        main_map: dict[int, int],
        main_ids_int: np.ndarray,
        alt_emb: np.ndarray,
        alt_map: dict[int, int],
        alt_ids_int: np.ndarray,
        doc_freqs: dict[tuple[str, str], int],
        global_tok_to_id: dict[str, int],
        global_id_to_tok: list[str],
        max_pad_len: int = 24,
    ) -> None:
        self.source_name = source_name
        self.max_pad_len = max_pad_len
        self.main_emb = main_emb
        self.main_map = main_map
        self.main_ids_int = main_ids_int
        self.alt_emb = alt_emb
        self.alt_map = alt_map
        self.alt_ids_int = alt_ids_int

        # int64 entity IDs and metadata
        self.entity_ids_int = _id_to_int(df["entity_id"])
        self.countries = df["country"].astype(str).values
        self.id_to_row_idx = {int(eid): idx for idx, eid in enumerate(self.entity_ids_int)}

        # Country partitions (local indices)
        self.cand_indices_by_country: dict[str, list[int]] = defaultdict(list)
        for idx, c in enumerate(self.countries):
            c_str = c.strip()
            if c_str:
                self.cand_indices_by_country[c_str].append(idx)

        # Build compact int32 address token matrix for Channel F
        print(f"Building compact int32 token array for {source_name} Channel F...", flush=True)
        t_tok0 = time.time()
        ad_arr = df["addr_norm"].to_numpy() if "addr_norm" in df else np.array([""] * len(df))
        self.cand_token_matrix = np.zeros((len(df), max_pad_len), dtype=np.int32)

        for i, a_str in enumerate(ad_arr):
            toks = tokenize_address_f(str(a_str))[:max_pad_len]
            for pos, t in enumerate(toks):
                if t not in global_tok_to_id:
                    tid = len(global_id_to_tok)
                    global_tok_to_id[t] = tid
                    global_id_to_tok.append(t)
                self.cand_token_matrix[i, pos] = global_tok_to_id[t]

        print(f"  Token array shape {self.cand_token_matrix.shape} built in {time.time() - t_tok0:.1f}s | RSS: {_rss_mb():.1f} MB", flush=True)

        # Inverted index for Channel G (K5, K3, K1) with int64 IDs
        print(f"Building Channel G inverted indices for {source_name}...", flush=True)
        t0 = time.time()
        raw_k1: dict[str, list[int]] = defaultdict(list)
        raw_k3: dict[str, list[int]] = defaultdict(list)
        raw_k5: dict[str, list[int]] = defaultdict(list)

        num_arr = df["num_tokens"].to_numpy() if "num_tokens" in df else np.array([""] * len(df))

        for eid_int, ctry, num, ad in zip(self.entity_ids_int, self.countries, num_arr, ad_arr):
            c = ctry.strip()
            if not c:
                continue
            n_str = str(num).strip()
            a_str = str(ad).strip()

            eid_val = int(eid_int)
            for k in extract_k1(c, n_str):
                raw_k1[k].append(eid_val)
            for k in extract_k3(c, n_str):
                raw_k3[k].append(eid_val)
            for k in extract_k5(c, n_str, a_str, doc_freqs):
                raw_k5[k].append(eid_val)

        # Prune blocks larger than 50 and convert to compact int64 arrays
        self.k1_index = {k: np.array(v, dtype=np.int64) for k, v in raw_k1.items() if len(v) <= 50}
        self.k3_index = {k: np.array(v, dtype=np.int64) for k, v in raw_k3.items() if len(v) <= 50}
        self.k5_index = {k: np.array(v, dtype=np.int64) for k, v in raw_k5.items() if len(v) <= 50}
        print(f"  {source_name} Channel G indices built in {time.time() - t0:.1f}s | "
              f"K1={len(self.k1_index):,} (dropped {len(raw_k1) - len(self.k1_index):,}), "
              f"K3={len(self.k3_index):,} (dropped {len(raw_k3) - len(self.k3_index):,}), "
              f"K5={len(self.k5_index):,} (dropped {len(raw_k5) - len(self.k5_index):,}) | "
              f"RSS: {_rss_mb():.1f} MB", flush=True)

        del raw_k1, raw_k3, raw_k5, num_arr, ad_arr
        gc.collect()


# ==============================================================================
# SAFETY CHECK: BACKUP VERIFICATION
# ==============================================================================

def verify_backup_safety(cache_dir: str, split: str) -> list[str]:
    """Verifies that cache/cands_backup_{split}/ exists and contains all original chunk files."""
    backup_dir = os.path.join(cache_dir, f"cands_backup_{split}")
    if not os.path.isdir(backup_dir):
        raise RuntimeError(
            f"SAFETY CHECK FAILED: Required backup directory '{backup_dir}' does not exist! "
            f"You must create this backup containing all original chunk files before running augmentation."
        )

    chunk_pattern = os.path.join(cache_dir, f"cands_{split}_chunk_*.parquet")
    chunk_files = sorted(glob.glob(chunk_pattern))
    if not chunk_files:
        raise FileNotFoundError(f"No candidate chunk files found matching '{chunk_pattern}'.")

    backup_chunk_files = sorted(glob.glob(os.path.join(backup_dir, f"cands_{split}_chunk_*.parquet")))
    if len(backup_chunk_files) != len(chunk_files):
        raise RuntimeError(
            f"SAFETY CHECK FAILED: Backup directory '{backup_dir}' contains {len(backup_chunk_files)} chunk files, "
            f"but {len(chunk_files)} chunk files exist in '{cache_dir}'. All original chunk files must be backed up!"
        )

    for cf in chunk_files:
        bf = os.path.join(backup_dir, os.path.basename(cf))
        if not os.path.exists(bf) or os.path.getsize(bf) == 0:
            raise RuntimeError(f"SAFETY CHECK FAILED: Backup file '{bf}' is missing or empty!")

    print(f"Safety Check PASSED: Backup directory '{backup_dir}' verified ({len(backup_chunk_files)} chunk files).")
    return chunk_files


# ==============================================================================
# MAIN AUGMENTATION PIPELINE
# ==============================================================================

def run_augmentation(
    cache_dir: str,
    split: str,
    chunk_files: list[str],
    max_chunks: Optional[int] = None,
    batch_size: int = 5000,
    dry_run: bool = False,
) -> None:
    """Executes Stage 3b candidate augmentation chunk by chunk with minimal RSS."""
    total_start = time.time()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== Running Stage 3b Candidate Augmentation ({split.upper()}) ===")
    print(f"Compute Device: {device} | Initial RSS: {_rss_mb():.1f} MB")

    if max_chunks is not None and max_chunks > 0:
        chunk_files = chunk_files[:max_chunks]
        print(f"Limiting execution to first {len(chunk_files)} chunk(s).")

    # 0. Which channels does any chunk still need? (schema scan only) Heavy F/G/H setup is skipped when unused.
    channel_h_active_s2 = combined_embeddings_exist(cache_dir, split, "source1") and combined_embeddings_exist(cache_dir, split, "source2")
    channel_h_active_s3 = combined_embeddings_exist(cache_dir, split, "source1") and combined_embeddings_exist(cache_dir, split, "source3")
    channel_h_any_active = channel_h_active_s2 or channel_h_active_s3
    tfidf = load_tfidf_pairs(cache_dir, split)
    channel_i_active = tfidf is not None
    chunk_cols = {cf: set(pq.ParquetFile(cf).schema.names) for cf in chunk_files}
    need_fg = any("ch_rerank" not in c for c in chunk_cols.values())
    need_h = channel_h_any_active and any("ch_comb" not in c for c in chunk_cols.values())
    need_i = channel_i_active and any("ch_tfidf" not in c for c in chunk_cols.values())
    need_heavy = need_fg or need_h
    print(f"Channels needed by some chunk: F/G={need_fg}, H={need_h} (active={channel_h_any_active}), "
          f"I={need_i} (active={channel_i_active}) | dry_run={dry_run}")

    # 1. Load document frequencies
    doc_freqs = load_doc_freqs(cache_dir, split) if need_heavy else {}

    # Global token vocabulary for compact int32 token arrays (index 0 = padding)
    global_tok_to_id: dict[str, int] = {}
    global_id_to_tok: list[str] = ["<PAD>"]

    # 2. Load Source 2 & Source 3 normalized tables (only needed columns)
    print("\nLoading normalized tables for Source 2 and Source 3...")
    cols = ["entity_id", "country", "num_tokens", "addr_norm"]
    p_s2 = os.path.join(cache_dir, f"norm_{split}_source2.parquet")
    p_s3 = os.path.join(cache_dir, f"norm_{split}_source3.parquet")

    def _read_norm_cols(p: str) -> pd.DataFrame:
        if not os.path.exists(p):
            return pd.DataFrame(columns=cols)
        schema_names = pq.ParquetFile(p).schema.names
        load_cols = [c for c in cols if c in schema_names]
        return pd.read_parquet(p, columns=load_cols)

    if need_heavy:
        df_s2 = _read_norm_cols(p_s2)
        df_s3 = _read_norm_cols(p_s3)
        print(f"Loaded normalized records: S2={len(df_s2):,}, S3={len(df_s3):,} (RSS: {_rss_mb():.1f} MB)")
    else:
        print("  Skipped (only Channel I needed).")

    # 3. Load name embeddings for Source 2 & Source 3 via mmap
    print("Memory-mapping name embeddings for Source 2 and Source 3...")
    s2_m_emb, s2_m_map, s2_m_ids, s2_a_emb, s2_a_map, s2_a_ids = load_embeddings_mmap(cache_dir, split, "source2")
    s3_m_emb, s3_m_map, s3_m_ids, s3_a_emb, s3_a_map, s3_a_ids = load_embeddings_mmap(cache_dir, split, "source3")
    print(f"Name embeddings mapped (RSS: {_rss_mb():.1f} MB)")

    # 4. Check & Load Channel H combined embeddings if present
    print("Checking for Channel H combined embeddings...")
    s1_comb_emb, s1_comb_map = load_combined_embeddings_mmap(cache_dir, split, "source1") if need_h else (None, {})
    s2_comb_emb, s2_comb_map = load_combined_embeddings_mmap(cache_dir, split, "source2") if need_h else (None, {})
    s3_comb_emb, s3_comb_map = load_combined_embeddings_mmap(cache_dir, split, "source3") if need_h else (None, {})

    if channel_h_any_active:
        print(f"Channel H active (S2={channel_h_active_s2}, S3={channel_h_active_s3})! Extra candidate cap = 30.")
    else:
        print("Channel H combined embeddings not found. Extra candidate cap = 20.")

    # 5. Build Target Indices for S2 and S3
    idx_s2 = idx_s3 = None
    if need_heavy:
        idx_s2 = TargetSourceIndex(
            "source2", df_s2, s2_m_emb, s2_m_map, s2_m_ids, s2_a_emb, s2_a_map, s2_a_ids,
            doc_freqs, global_tok_to_id, global_id_to_tok
        )
        idx_s3 = TargetSourceIndex(
            "source3", df_s3, s3_m_emb, s3_m_map, s3_m_ids, s3_a_emb, s3_a_map, s3_a_ids,
            doc_freqs, global_tok_to_id, global_id_to_tok
        )
        del df_s2, df_s3
        gc.collect()
        print(f"Target indices built (RSS: {_rss_mb():.1f} MB)")

    # 6. Load Source 1 metadata and name embeddings via mmap
    print("\nLoading Source 1 metadata and embeddings...")
    p_s1 = os.path.join(cache_dir, f"norm_{split}_source1.parquet")
    s1_norm = _read_norm_cols(p_s1)
    s1_norm["s1_int"] = _id_to_int(s1_norm["entity_id"])
    s1_meta_by_int = dict(zip(s1_norm["s1_int"], zip(s1_norm["country"], s1_norm["num_tokens"], s1_norm["addr_norm"])))
    s1_m_emb, s1_m_map, s1_m_ids, s1_a_emb, s1_a_map, s1_a_ids = load_embeddings_mmap(cache_dir, split, "source1")
    del s1_norm
    gc.collect()
    print(f"Loaded Source 1: {len(s1_meta_by_int):,} records (RSS: {_rss_mb():.1f} MB)")

    # 7. Load Ground Truth for label assignment if train
    gt_pairs_int: set[tuple[int, int]] = set()
    if split == "train":
        gt_path = os.path.join(cache_dir, "gt_long.parquet")
        if os.path.exists(gt_path):
            gt_df = pd.read_parquet(gt_path, columns=["s1_id", "match_id"])
            s1_i = _id_to_int(gt_df["s1_id"])
            m_i = _id_to_int(gt_df["match_id"])
            gt_pairs_int = set(zip(s1_i, m_i))
            del gt_df, s1_i, m_i
            print(f"Loaded ground truth: {len(gt_pairs_int):,} pairs")

    val_s1_set: set[str] = set()
    val_frames: list[pd.DataFrame] = []
    split_path = os.path.join(cache_dir, "split.parquet")
    if split == "train" and gt_pairs_int and os.path.exists(split_path):
        sp_df = pd.read_parquet(split_path)
        val_s1_set = set(sp_df.loc[sp_df["fold"] == "val", "s1_id"].astype(str))
        del sp_df

    # Pairwise embedding dot product helper
    def _calc_emb_score(s1_int: int, cand_int: int, is_s2: bool) -> float:
        cm_emb, cm_map, ca_emb, ca_map = (s2_m_emb, s2_m_map, s2_a_emb, s2_a_map) if is_s2 else (s3_m_emb, s3_m_map, s3_a_emb, s3_a_map)
        if s1_int not in s1_m_map or cand_int not in cm_map:
            return 0.0
        v1_m = s1_m_emb[s1_m_map[s1_int]].astype(np.float32)
        v2_m = cm_emb[cm_map[cand_int]].astype(np.float32)
        best_s = float(np.dot(v1_m, v2_m))

        has_s1_a = s1_int in s1_a_map
        has_c_a = cand_int in ca_map
        if has_c_a:
            v2_a = ca_emb[ca_map[cand_int]].astype(np.float32)
            s_ma = float(np.dot(v1_m, v2_a))
            if s_ma > best_s:
                best_s = s_ma
        if has_s1_a:
            v1_a = s1_a_emb[s1_a_map[s1_int]].astype(np.float32)
            s_am = float(np.dot(v1_a, v2_m))
            if s_am > best_s:
                best_s = s_am
            if has_c_a:
                s_aa = float(np.dot(v1_a, v2_a))
                if s_aa > best_s:
                    best_s = s_aa
        return best_s

    def _calc_emb_scores_batch(s1_arr: np.ndarray, cand_arr: np.ndarray, is_s2_arr: np.ndarray) -> np.ndarray:
        """Vectorised _calc_emb_score: max dot over main/alt name embeddings, 0 if a main vector is missing."""
        out = np.zeros(len(s1_arr), dtype=np.float32)

        def _idx(mp, arr):
            return np.fromiter((mp.get(int(x), -1) for x in arr), dtype=np.int64, count=len(arr))

        def _dot(A, ia, B, ib):
            res = np.empty(len(ia), dtype=np.float32)
            for b in range(0, len(ia), 50_000):
                va = np.asarray(A[ia[b:b + 50_000]], dtype=np.float32)
                vb = np.asarray(B[ib[b:b + 50_000]], dtype=np.float32)
                res[b:b + 50_000] = np.einsum("ij,ij->i", va, vb)
            return res

        for is_s2 in (True, False):
            sel = np.flatnonzero(is_s2_arr == is_s2)
            if len(sel) == 0:
                continue
            cm_emb, cm_map, ca_emb, ca_map = (s2_m_emb, s2_m_map, s2_a_emb, s2_a_map) if is_s2 else (s3_m_emb, s3_m_map, s3_a_emb, s3_a_map)
            s1s, cs = s1_arr[sel], cand_arr[sel]
            m1, m2 = _idx(s1_m_map, s1s), _idx(cm_map, cs)
            a1, a2 = _idx(s1_a_map, s1s), _idx(ca_map, cs)
            ok = (m1 >= 0) & (m2 >= 0)
            best = np.zeros(len(sel), dtype=np.float32)
            if ok.any():
                best[ok] = _dot(s1_m_emb, m1[ok], cm_emb, m2[ok])
            for A, ia, B, ib, mask in (
                (s1_m_emb, m1, ca_emb, a2, ok & (a2 >= 0)),
                (s1_a_emb, a1, cm_emb, m2, ok & (a1 >= 0)),
                (s1_a_emb, a1, ca_emb, a2, ok & (a1 >= 0) & (a2 >= 0)),
            ):
                if mask.any():
                    best[mask] = np.maximum(best[mask], _dot(A, ia[mask], B, ib[mask]))
            out[sel] = best
        return out

    # 8. Process each chunk
    print(f"\nProcessing {len(chunk_files)} chunk(s)...")
    for ch_idx, chunk_path in enumerate(chunk_files):
        t_ch_start = time.time()
        print(f"\n--- Chunk {ch_idx + 1}/{len(chunk_files)}: {chunk_path} ---", flush=True)

        pf = pq.ParquetFile(chunk_path)
        has_rerank = "ch_rerank" in pf.schema.names
        has_comb = "ch_comb" in pf.schema.names

        has_tfidf = "ch_tfidf" in pf.schema.names

        # Each channel runs only if its marker column is missing (and, for H/I, its inputs exist):
        #   no ch_rerank -> F + G;  no ch_comb (H active) -> H;  no ch_tfidf (I active) -> I.
        # So a chunk that has ch_comb but no ch_tfidf gets only Channel I.
        run_fg = not has_rerank
        run_h = channel_h_any_active and (not has_comb)
        run_i = channel_i_active and (not has_tfidf)
        extra_cap = 30 if channel_h_any_active else 20

        if not (run_fg or run_h or run_i):
            print(f"  Nothing to do (ch_rerank={has_rerank}, ch_comb={has_comb}, ch_tfidf={has_tfidf}), skipping (idempotent).")
            if val_s1_set:
                val_frames.append(_val_stage_rows(pd.read_parquet(chunk_path), val_s1_set))
            continue
        print(f"  Channels to run: {'F+G ' if run_fg else ''}{'H ' if run_h else ''}{'I' if run_i else ''}", flush=True)

        chunk_df = pd.read_parquet(chunk_path)
        existing_rows_count = len(chunk_df)

        s1_ints = _id_to_int(chunk_df["s1_id"])
        cand_ints = _id_to_int(chunk_df["cand_id"])
        chunk_s1_ids_order = pd.unique(chunk_df["s1_id"])
        chunk_s1_ints_unique = _id_to_int(chunk_s1_ids_order)
        print(f"  Existing rows: {existing_rows_count:,} across {len(chunk_s1_ints_unique):,} unique S1s (RSS: {_rss_mb():.1f} MB)")

        # Map existing candidates and existing extra count per S1
        existing_cands_map: dict[int, set[int]] = defaultdict(set)
        existing_extra_count: dict[int, int] = defaultdict(int)

        if not (run_fg or run_h):
            pass  # Channel I only: F/G/H bookkeeping not needed
        elif has_rerank:
            rr_arr = chunk_df["ch_rerank"].values
            k1_arr = chunk_df["ch_k1"].values if "ch_k1" in chunk_df.columns else np.zeros(len(chunk_df))
            k3_arr = chunk_df["ch_k3"].values if "ch_k3" in chunk_df.columns else np.zeros(len(chunk_df))
            k5_arr = chunk_df["ch_k5"].values if "ch_k5" in chunk_df.columns else np.zeros(len(chunk_df))
            for s1_i, cid_i, rrk, k1, k3, k5 in zip(s1_ints, cand_ints, rr_arr, k1_arr, k3_arr, k5_arr):
                existing_cands_map[int(s1_i)].add(int(cid_i))
                if rrk > 0 or k1 > 0 or k3 > 0 or k5 > 0:
                    existing_extra_count[int(s1_i)] += 1
        else:
            for s1_i, cid_i in zip(s1_ints, cand_ints):
                existing_cands_map[int(s1_i)].add(int(cid_i))

        # Group S1 IDs by country
        s1_by_country: dict[str, list[int]] = defaultdict(list)
        for sid_int in chunk_s1_ints_unique:
            meta = s1_meta_by_int.get(int(sid_int))
            c = str(meta[0]).strip() if meta else ""
            if c:
                s1_by_country[c].append(int(sid_int))

        # Candidate collectors for this chunk: s1_int -> cand_int -> candidate dict
        augmented_candidates: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)

        # ----------------------------------------------------------------------
        # CHANNEL F: Wide Retrieval (top 200) + Address Re-ranking (GPU)
        # ----------------------------------------------------------------------
        if run_fg:
            print(f"  Running Channel F (Wide 200 + GPU Re-ranking)...", flush=True)
            t_f0 = time.time()

            for ctry, s1_ids_c in s1_by_country.items():
                for target_idx, is_s2 in [(idx_s2, True), (idx_s3, False)]:
                    cand_pool_indices = target_idx.cand_indices_by_country.get(ctry, [])
                    if not cand_pool_indices or not s1_ids_c:
                        continue

                    cand_ids_pool = target_idx.entity_ids_int[cand_pool_indices]

                    # Stack target embeddings (main + alt)
                    c_m_indices = [target_idx.main_map[cid] for cid in cand_ids_pool if cid in target_idx.main_map]
                    X_main = target_idx.main_emb[c_m_indices]
                    X_main_cids = cand_ids_pool[[cid in target_idx.main_map for cid in cand_ids_pool]]

                    c_a_cids = [cid for cid in cand_ids_pool if cid in target_idx.alt_map]
                    if c_a_cids:
                        c_a_indices = [target_idx.alt_map[cid] for cid in c_a_cids]
                        X_alt = target_idx.alt_emb[c_a_indices]
                        X_alt_cids = np.array(c_a_cids, dtype=np.int64)
                        X_stacked = np.vstack([X_main, X_alt])
                        row_to_cid = np.concatenate([X_main_cids, X_alt_cids])
                    else:
                        X_stacked = X_main
                        row_to_cid = X_main_cids

                    N_index = len(X_stacked)
                    if N_index == 0:
                        continue

                    X_gpu = torch.from_numpy(X_stacked).to(device)
                    k_search = min(400, N_index)

                    # Build IDF weights for this country over vocabulary
                    max_docs_ctry = max(1, len(cand_pool_indices))

                    # Token arrays for S1 queries in this country
                    s1_token_matrix = np.zeros((len(s1_ids_c), 24), dtype=np.int32)
                    s1_weights_matrix = np.zeros((len(s1_ids_c), 24), dtype=np.float32)

                    for q_i, sid_int in enumerate(s1_ids_c):
                        meta = s1_meta_by_int.get(sid_int)
                        ad_s1 = meta[2] if meta else ""
                        toks_s1 = tokenize_address_f(str(ad_s1))[:24]
                        for t_pos, t_str in enumerate(toks_s1):
                            if t_str not in global_tok_to_id:
                                tid = len(global_id_to_tok)
                                global_tok_to_id[t_str] = tid
                                global_id_to_tok.append(t_str)
                            t_id = global_tok_to_id[t_str]
                            df_val = doc_freqs.get((ctry, t_str), 0)
                            idf_val = float(np.log(1.0 + (max_docs_ctry + 1.0) / (df_val + 1.0)))
                            s1_token_matrix[q_i, t_pos] = t_id
                            s1_weights_matrix[q_i, t_pos] = idf_val

                    # Slice candidate token matrix directly
                    cand_token_matrix = target_idx.cand_token_matrix[cand_pool_indices]
                    cand_id_to_pool_idx = {int(cid): idx for idx, cid in enumerate(cand_ids_pool)}

                    # Calculate query chunk size (score matrix <= 3 GB)
                    max_bytes = 3 * 1024 * 1024 * 1024  # 3 GB
                    bytes_per_query = N_index * 2
                    max_q_chunk = max(1, max_bytes // max(1, bytes_per_query))
                    q_chunk_size = min(max_q_chunk, 5000)
                    if batch_size:
                        q_chunk_size = min(q_chunk_size, batch_size)

                    # Process S1 queries in batches
                    for b_start in range(0, len(s1_ids_c), q_chunk_size):
                        b_end = min(b_start + q_chunk_size, len(s1_ids_c))
                        sub_sids = s1_ids_c[b_start:b_end]

                        # Query embeddings
                        sub_m_idx = [s1_m_map[sid] for sid in sub_sids if sid in s1_m_map]
                        if len(sub_m_idx) != len(sub_sids):
                            continue
                        Q_m = torch.from_numpy(s1_m_emb[sub_m_idx]).to(device)
                        S = Q_m @ X_gpu.T

                        has_alt_mask = np.array([sid in s1_a_map for sid in sub_sids])
                        if has_alt_mask.any():
                            sub_alt_sids = [sid for sid in sub_sids if sid in s1_a_map]
                            sub_alt_idx = [s1_a_map[sid] for sid in sub_alt_sids]
                            Q_a = torch.from_numpy(s1_a_emb[sub_alt_idx]).to(device)
                            S_a = Q_a @ X_gpu.T
                            S[has_alt_mask] = torch.maximum(S[has_alt_mask], S_a)
                            del Q_a, S_a

                        # Top 200 embedding scores
                        top_scores, top_indices = torch.topk(S, k=k_search, dim=1)
                        top_scores = top_scores.float().cpu().numpy()
                        top_indices = top_indices.cpu().numpy()
                        del S, Q_m

                        # Deduplicate candidates to 200 unique
                        B_curr = len(sub_sids)
                        top200_cand_ids: list[list[int]] = []
                        top200_cand_scores: list[list[float]] = []
                        top200_pool_indices: list[list[int]] = []

                        for b in range(B_curr):
                            b_cids = row_to_cid[top_indices[b]]
                            b_scs = top_scores[b]
                            seen_c: dict[int, float] = {}
                            for cid, sc in zip(b_cids, b_scs):
                                cid_val = int(cid)
                                if cid_val not in seen_c or sc > seen_c[cid_val]:
                                    seen_c[cid_val] = float(sc)
                                    if len(seen_c) >= 200:
                                        break
                            c_list = list(seen_c.keys())
                            top200_cand_ids.append(c_list)
                            top200_cand_scores.append(list(seen_c.values()))
                            top200_pool_indices.append([cand_id_to_pool_idx[c] for c in c_list])

                        # GPU address re-rank
                        max_c_len = max(len(cl) for cl in top200_cand_ids) if top200_cand_ids else 0
                        if max_c_len == 0:
                            continue

                        T_cand_sub = np.zeros((B_curr, max_c_len, 24), dtype=np.int32)
                        for b in range(B_curr):
                            for j, p_idx in enumerate(top200_pool_indices[b]):
                                T_cand_sub[b, j] = cand_token_matrix[p_idx]

                        T_cand_gpu = torch.from_numpy(T_cand_sub).to(device)
                        T_s1_gpu = torch.from_numpy(s1_token_matrix[b_start:b_end]).to(device)
                        W_s1_gpu = torch.from_numpy(s1_weights_matrix[b_start:b_end]).to(device)

                        has_tok = (T_s1_gpu.unsqueeze(1).unsqueeze(-1) == T_cand_gpu.unsqueeze(2)).any(dim=-1)
                        has_tok = has_tok & (T_s1_gpu.unsqueeze(1) != 0)
                        rerank_scores = (has_tok.float() * W_s1_gpu.unsqueeze(1)).sum(dim=-1).cpu().numpy()

                        del T_cand_gpu, T_s1_gpu, W_s1_gpu, has_tok, T_cand_sub

                        # Select top 10 per S1 not already in candidates
                        for b, sid in enumerate(sub_sids):
                            cids_b = top200_cand_ids[b]
                            emb_scs_b = top200_cand_scores[b]
                            rr_scs_b = rerank_scores[b]
                            existing_set = existing_cands_map[sid]

                            ranked_order = sorted(
                                range(len(cids_b)),
                                key=lambda j: (rr_scs_b[j], emb_scs_b[j]),
                                reverse=True,
                            )

                            kept_count = 0
                            for j in ranked_order:
                                cid = cids_b[j]
                                if cid in existing_set:
                                    continue

                                c_dict = augmented_candidates[sid].setdefault(cid, {
                                    "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0, "ch_comb": 0,
                                    "f_score": 0.0, "emb_score": emb_scs_b[j], "is_s2": is_s2,
                                })
                                c_dict["ch_rerank"] = 1
                                c_dict["f_score"] = max(c_dict["f_score"], float(rr_scs_b[j]))
                                c_dict["emb_score"] = max(c_dict["emb_score"], float(emb_scs_b[j]))

                                kept_count += 1
                                if kept_count >= 10:
                                    break

                        del top200_cand_ids, top200_cand_scores, top200_pool_indices, rerank_scores

                    del X_gpu, cand_token_matrix, s1_token_matrix, s1_weights_matrix
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            print(f"  Channel F completed in {time.time() - t_f0:.1f}s | RSS: {_rss_mb():.1f} MB", flush=True)

        # ----------------------------------------------------------------------
        # CHANNEL G: Address Keys K5, K3, K1
        # ----------------------------------------------------------------------
        if run_fg:
            print(f"  Running Channel G (Address Keys K5, K3, K1)...", flush=True)
            t_g0 = time.time()
            for sid in chunk_s1_ints_unique:
                meta = s1_meta_by_int.get(int(sid))
                if not meta:
                    continue
                c = str(meta[0]).strip()
                num_str = str(meta[1]).strip()
                addr_str = str(meta[2]).strip()
                existing_set = existing_cands_map[int(sid)]

                for target_idx, is_s2 in [(idx_s2, True), (idx_s3, False)]:
                    # Key K5
                    k5_keys = extract_k5(c, num_str, addr_str, doc_freqs)
                    cands_k5: set[int] = set()
                    for k in k5_keys:
                        if k in target_idx.k5_index:
                            cands_k5.update(target_idx.k5_index[k])
                    cands_k5.difference_update(existing_set)
                    if cands_k5:
                        scored_k5 = [(cid, _calc_emb_score(sid, cid, is_s2)) for cid in cands_k5]
                        scored_k5.sort(key=lambda x: x[1], reverse=True)
                        for cid, sc in scored_k5[:10]:
                            c_dict = augmented_candidates[sid].setdefault(cid, {
                                "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0, "ch_comb": 0,
                                "f_score": 0.0, "emb_score": sc, "is_s2": is_s2,
                            })
                            c_dict["ch_k5"] = 1
                            c_dict["emb_score"] = max(c_dict["emb_score"], sc)

                    # Key K3
                    k3_keys = extract_k3(c, num_str)
                    cands_k3: set[int] = set()
                    for k in k3_keys:
                        if k in target_idx.k3_index:
                            cands_k3.update(target_idx.k3_index[k])
                    cands_k3.difference_update(existing_set)
                    if cands_k3:
                        scored_k3 = [(cid, _calc_emb_score(sid, cid, is_s2)) for cid in cands_k3]
                        scored_k3.sort(key=lambda x: x[1], reverse=True)
                        for cid, sc in scored_k3[:10]:
                            c_dict = augmented_candidates[sid].setdefault(cid, {
                                "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0, "ch_comb": 0,
                                "f_score": 0.0, "emb_score": sc, "is_s2": is_s2,
                            })
                            c_dict["ch_k3"] = 1
                            c_dict["emb_score"] = max(c_dict["emb_score"], sc)

                    # Key K1
                    k1_keys = extract_k1(c, num_str)
                    cands_k1: set[int] = set()
                    for k in k1_keys:
                        if k in target_idx.k1_index:
                            cands_k1.update(target_idx.k1_index[k])
                    cands_k1.difference_update(existing_set)
                    if cands_k1:
                        scored_k1 = [(cid, _calc_emb_score(sid, cid, is_s2)) for cid in cands_k1]
                        scored_k1.sort(key=lambda x: x[1], reverse=True)
                        for cid, sc in scored_k1[:10]:
                            c_dict = augmented_candidates[sid].setdefault(cid, {
                                "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0, "ch_comb": 0,
                                "f_score": 0.0, "emb_score": sc, "is_s2": is_s2,
                            })
                            c_dict["ch_k1"] = 1
                            c_dict["emb_score"] = max(c_dict["emb_score"], sc)

            print(f"  Channel G completed in {time.time() - t_g0:.1f}s | RSS: {_rss_mb():.1f} MB", flush=True)

        # ----------------------------------------------------------------------
        # CHANNEL H: Combined Name + Address Embeddings (GPU Top-20)
        # ----------------------------------------------------------------------
        if run_h:
            print(f"  Running Channel H (Combined Embeddings GPU Top-20)...", flush=True)
            t_h0 = time.time()
            for target_idx, is_s2, comb_emb, comb_map, active in [
                (idx_s2, True, s2_comb_emb, s2_comb_map, channel_h_active_s2),
                (idx_s3, False, s3_comb_emb, s3_comb_map, channel_h_active_s3),
            ]:
                if not active or comb_emb is None:
                    continue

                for ctry, s1_ids_c in s1_by_country.items():
                    cand_pool_indices = target_idx.cand_indices_by_country.get(ctry, [])
                    if not cand_pool_indices or not s1_ids_c:
                        continue

                    # Target candidate pool in this country that have combined embeddings
                    cand_ids_pool = target_idx.entity_ids_int[cand_pool_indices]
                    valid_c_mask = [cid in comb_map for cid in cand_ids_pool]
                    valid_cids = cand_ids_pool[valid_c_mask]
                    if len(valid_cids) == 0:
                        continue

                    target_emb_indices = [comb_map[cid] for cid in valid_cids]
                    X_comb = comb_emb[target_emb_indices]
                    N_comb = len(X_comb)

                    # S1 queries that have combined embeddings
                    valid_s1 = [sid for sid in s1_ids_c if sid in s1_comb_map]
                    if not valid_s1:
                        continue

                    X_comb_gpu = torch.from_numpy(X_comb).to(device)
                    k_search = min(20, N_comb)

                    max_bytes = 3 * 1024 * 1024 * 1024
                    bytes_per_query = N_comb * 2
                    max_q_chunk = max(1, max_bytes // max(1, bytes_per_query))
                    q_chunk_size = min(max_q_chunk, 5000)
                    if batch_size:
                        q_chunk_size = min(q_chunk_size, batch_size)

                    for b_start in range(0, len(valid_s1), q_chunk_size):
                        b_end = min(b_start + q_chunk_size, len(valid_s1))
                        sub_sids = valid_s1[b_start:b_end]

                        sub_s1_indices = [s1_comb_map[sid] for sid in sub_sids]
                        Q_comb = torch.from_numpy(s1_comb_emb[sub_s1_indices]).to(device)

                        S_comb = Q_comb @ X_comb_gpu.T
                        top_scs, top_idxs = torch.topk(S_comb, k=k_search, dim=1)
                        top_scs = top_scs.float().cpu().numpy()
                        top_idxs = top_idxs.cpu().numpy()
                        del S_comb, Q_comb

                        for b, sid in enumerate(sub_sids):
                            c_indices = top_idxs[b]
                            existing_set = existing_cands_map[sid]
                            for c_idx in c_indices:
                                cid = int(valid_cids[c_idx])
                                if cid in existing_set:
                                    continue
                                sc_emb = _calc_emb_score(sid, cid, is_s2)
                                c_dict = augmented_candidates[sid].setdefault(cid, {
                                    "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0, "ch_comb": 0,
                                    "f_score": 0.0, "emb_score": sc_emb, "is_s2": is_s2,
                                })
                                c_dict["ch_comb"] = 1
                                c_dict["emb_score"] = max(c_dict["emb_score"], sc_emb)

                    del X_comb_gpu
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            print(f"  Channel H completed in {time.time() - t_h0:.1f}s | RSS: {_rss_mb():.1f} MB", flush=True)

        # ----------------------------------------------------------------------
        # MERGE & ATOMIC WRITE CHUNK
        # ----------------------------------------------------------------------
        print(f"  Merging new candidates (cap {extra_cap} per S1)...", flush=True)
        new_rows: list[dict[str, Any]] = []

        for sid_int in chunk_s1_ints_unique:
            sid = int(sid_int)
            cands_dict = augmented_candidates.get(sid, {})
            if not cands_dict:
                continue

            # Remaining budget if chunk already had F/G extra rows
            budget = max(0, extra_cap - existing_extra_count.get(sid, 0))
            if budget <= 0:
                continue

            # Priority: number of new channels, then ch_comb, then f_score, then emb_score
            sorted_new_cands = sorted(
                cands_dict.items(),
                key=lambda item: (
                    item[1]["ch_rerank"] + item[1]["ch_k1"] + item[1]["ch_k3"] + item[1]["ch_k5"] + item[1].get("ch_comb", 0),
                    item[1].get("ch_comb", 0),
                    item[1]["f_score"],
                    item[1]["emb_score"],
                ),
                reverse=True,
            )

            s1_str = _int_to_id([sid])[0]
            for cid, info in sorted_new_cands[:budget]:
                is_s2 = info["is_s2"]
                cid_str = _int_to_id([cid])[0]
                row_dict: dict[str, Any] = {
                    "s1_id": s1_str,
                    "cand_id": cid_str,
                    "cand_source": 0 if is_s2 else 1,
                    "emb_score": np.float32(info["emb_score"]),
                    "emb_rank": np.float32(999.0),
                    "ch_emb": np.int8(0),
                    "ch_addr": np.int8(0),
                    "ch_skel": np.int8(0),
                    "ch_rare": np.int8(0),
                    "ch_rev": np.int8(0),
                    "ch_rerank": np.int8(info["ch_rerank"]),
                    "ch_k1": np.int8(info["ch_k1"]),
                    "ch_k3": np.int8(info["ch_k3"]),
                    "ch_k5": np.int8(info["ch_k5"]),
                }
                if channel_h_any_active:
                    row_dict["ch_comb"] = np.int8(info.get("ch_comb", 0))
                if "ch_tfidf" in chunk_df.columns:
                    row_dict["ch_tfidf"] = np.int8(0)
                    row_dict["tfidf_rank"] = np.int8(TFIDF_RANK_MISSING)

                if split == "train":
                    row_dict["label"] = np.int32(1 if (sid, cid) in gt_pairs_int else 0)
                new_rows.append(row_dict)

        # Update existing rows
        if "ch_rerank" not in chunk_df.columns:
            chunk_df["ch_rerank"] = np.int8(0)
            chunk_df["ch_k1"] = np.int8(0)
            chunk_df["ch_k3"] = np.int8(0)
            chunk_df["ch_k5"] = np.int8(0)
        if channel_h_any_active and "ch_comb" not in chunk_df.columns:
            chunk_df["ch_comb"] = np.int8(0)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            for col in chunk_df.columns:
                if col in new_df.columns:
                    new_df[col] = new_df[col].astype(chunk_df[col].dtype)
            augmented_df = pd.concat([chunk_df, new_df], ignore_index=True)
        else:
            augmented_df = chunk_df

        # CHANNEL I: char TF-IDF pairs, applied after the F/G/H merge
        n_i_new = 0
        if run_i:
            t_i0 = time.time()
            augmented_df, n_i_marked, n_i_new = apply_channel_i(
                augmented_df, tfidf, _calc_emb_scores_batch, gt_pairs_int, split
            )
            print(f"  Channel I: marked {n_i_marked:,} existing rows ch_tfidf=1, added {n_i_new:,} new rows "
                  f"(avg +{n_i_new / max(1, len(chunk_s1_ints_unique)):.2f}/S1) in {time.time() - t_i0:.1f}s | "
                  f"RSS: {_rss_mb():.1f} MB", flush=True)

        if val_s1_set:
            val_frames.append(_val_stage_rows(augmented_df, val_s1_set))

        if dry_run:
            print("  DRY RUN: chunk not written.")
        else:
            # Atomic write back to chunk_path
            tmp_out = chunk_path + ".tmp"
            augmented_df.to_parquet(tmp_out, index=False)
            os.replace(tmp_out, chunk_path)

        added_cnt = len(new_rows) + n_i_new
        avg_extra = added_cnt / max(1, len(chunk_s1_ints_unique))
        t_ch_elapsed = time.time() - t_ch_start
        print(f"  Chunk {ch_idx + 1} {'Computed' if dry_run else 'Saved'}: {len(augmented_df):,} total rows "
              f"(+{added_cnt:,} new rows, avg +{avg_extra:.2f}/S1) in {t_ch_elapsed:.1f}s | "
              f"RSS: {_rss_mb():.1f} MB", flush=True)

        del chunk_df, augmented_df, augmented_candidates, new_rows
        gc.collect()

    print(f"\nAll chunk processing completed in {time.time() - total_start:.1f}s. Final RSS: {_rss_mb():.1f} MB")

    # 9. Validation Recall Report (train only)
    if split == "train" and val_frames:
        run_validation_report(cache_dir, val_frames, s1_meta_by_int)


# ==============================================================================
# VALIDATION RECALL REPORT (BEFORE vs AFTER)
# ==============================================================================

def run_validation_report(
    cache_dir: str,
    val_frames: list[pd.DataFrame],
    s1_meta_by_int: dict[int, tuple[Any, Any, Any]],
) -> None:
    """
    Val pair recall by stage (original S3 rows, + F/G/H, + Channel I), overall, per country, per source and
    per country x source (countries taken from the data), plus entity recall and candidates per S1.
    """
    print("\n" + "=" * 100)
    print("                 VAL BLOCKING RECALL REPORT (S3 original -> + F/G/H -> + Channel I)")
    print("=" * 100)
    rows = pd.concat(val_frames, ignore_index=True)
    first = rows.groupby(["s1", "cand"], sort=False)["stage"].min().reset_index()
    val_s1 = np.unique(rows["s1"].values)
    n_val_s1 = len(val_s1)

    gt = pd.read_parquet(os.path.join(cache_dir, "gt_long.parquet"), columns=["s1_id", "match_id"])
    g = pd.DataFrame({"s1": _id_to_int(gt["s1_id"]), "cand": _id_to_int(gt["match_id"])})
    del gt
    g = g[np.isin(g["s1"].values, val_s1)].reset_index(drop=True)
    g["country"] = [str(s1_meta_by_int.get(int(x), ("",))[0]).strip() or "?" for x in g["s1"].values]
    g["source"] = np.where(g["cand"].values // 1_000_000_000_000 == 2, "S2", "S3")
    g = g.merge(first, on=["s1", "cand"], how="left")
    g["stage"] = g["stage"].fillna(9).astype(np.int8)
    print(f"Val S1 in processed chunks: {n_val_s1:,} | val GT pairs: {len(g):,} across {g['s1'].nunique():,} S1")
    if len(g) == 0:
        return

    stages = [(0, "S3 original"), (1, "+ F/G/H"), (2, "+ Channel I")]
    groups = [("Overall", np.ones(len(g), dtype=bool))]
    countries = sorted(g["country"].unique())
    groups += [(f"  {c}", (g["country"] == c).values) for c in countries]
    groups += [(f"  {s}", (g["source"] == s).values) for s in ("S2", "S3")]
    groups += [(f"  {c} {s}", ((g["country"] == c) & (g["source"] == s)).values) for c in countries for s in ("S2", "S3")]

    hdr = f"{'Pair recall':<18} | {'GT pairs':>9} | " + " | ".join(f"{n:>13}" for _, n in stages) + f" | {'Delta I':>8}"
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    st = g["stage"].values
    for name, mask in groups:
        n = int(mask.sum())
        if n == 0:
            continue
        pct = [100.0 * int(((st <= s) & mask).sum()) / n for s, _ in stages]
        print(f"{name:<18} | {n:>9,} | " + " | ".join(f"{p:>12.2f}%" for p in pct) + f" | {pct[2] - pct[1]:>+7.2f}%")
    print("-" * len(hdr))
    ent_stage = g.groupby("s1")["stage"].max().values
    ent = [100.0 * (ent_stage <= s).mean() for s, _ in stages]
    print(f"{'Entity recall':<18} | {len(ent_stage):>9,} | " + " | ".join(f"{p:>12.2f}%" for p in ent) + f" | {ent[2] - ent[1]:>+7.2f}%")
    fst = first["stage"].values
    cps = [int((fst <= s).sum()) / max(1, n_val_s1) for s, _ in stages]
    print(f"{'Cands per S1':<18} | {'':>9} | " + " | ".join(f"{c:>13.1f}" for c in cps) + f" | {cps[2] - cps[1]:>+8.1f}")
    print("=" * 100 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3b Candidate Augmentation.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory.")
    parser.add_argument("--laptop-test", action="store_true", help="Use cache/laptop_test.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximum number of chunks to process.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for GPU operations.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute everything and print the recall report, but write no chunk files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.laptop_test:
        cache_dir = os.path.join(root_dir, "cache", "laptop_test")
    else:
        cache_dir = os.path.join(root_dir, "cache")

    chunk_files = verify_backup_safety(cache_dir, args.split)

    run_augmentation(
        cache_dir=cache_dir,
        split=args.split,
        chunk_files=chunk_files,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
