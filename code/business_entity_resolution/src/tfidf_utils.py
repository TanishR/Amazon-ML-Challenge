"""
Memory-light char TF-IDF + GPU top-k search, shared by s3c_tfidf.py and benchmarks.

TF-IDF matches sklearn TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
dtype=float32, max_features=M) fit on the concatenation of the fit documents, but is built in two
streaming passes so that CPU RAM stays at about one batch:
  pass 1: per-batch CountVectorizer -> global term counts (for max_features) and document frequencies
  pass 2: CountVectorizer(vocabulary) -> 1 + log(tf) -> * idf -> L2 row normalisation,
          each batch uploaded to the GPU as CSR pieces and concatenated there.
idf = ln((1 + n_docs) / (1 + df)) + 1 (smooth_idf=True), as in sklearn.
"""
import os
import time

import numpy as np
import psutil
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ANALYZER_KW = dict(analyzer="char_wb", ngram_range=(3, 3), lowercase=True)


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 2**30


def avail_gb() -> float:
    return psutil.virtual_memory().available / 2**30


def wait_for_memory(min_avail_gb: float, where: str = "", poll_s: int = 30) -> None:
    """Blocks while system available RAM is below min_avail_gb."""
    if avail_gb() >= min_avail_gb:
        return
    t0 = time.time()
    while avail_gb() < min_avail_gb:
        print(f"  PAUSED{' at ' + where if where else ''}: available RAM {avail_gb():.1f} GB < {min_avail_gb} GB "
              f"(waiting {time.time() - t0:.0f}s)", flush=True)
        time.sleep(poll_s)
    print(f"  RESUMED after {time.time() - t0:.0f}s (available {avail_gb():.1f} GB)", flush=True)


def make_texts(df, variant: str = "na", name_map: dict = None) -> list:
    """
    variant 'nsa' (Channel I default): name_full + ' ' + name_skel + ' ' + addr_norm.
    variant 'na': name_full + ' ' + addr_norm.
    char_wb n-grams are built per word, so word order does not change the vectors.
    """
    name = df["name_full"].fillna("").astype(str)
    if name_map:        # Channel K: token dictionary applied to a copy of the name (name_full itself is never changed)
        name = name.map(lambda s: " ".join(name_map.get(tok, tok) for tok in s.split()))
    addr = df["addr_norm"].fillna("").astype(str)
    if variant == "na":
        return (name + " " + addr).tolist()
    if variant == "nsa":
        return (name + " " + df["name_skel"].fillna("").astype(str) + " " + addr).tolist()
    raise ValueError(f"unknown text variant {variant!r}")


class GpuCsr:
    """Row-major CSR matrix held as torch tensors on the GPU (int32 indices when they fit)."""

    def __init__(self, crow, col, val, n_rows, n_cols):
        self.crow, self.col, self.val = crow, col, val
        self.shape = (n_rows, n_cols)

    @property
    def nnz(self):
        return int(self.val.numel())

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.crow, self.col, self.val))

    def to_sparse(self, dtype=torch.float32):
        return torch.sparse_csr_tensor(self.crow, self.col, self.val.to(dtype), size=self.shape)

    def dense_rows(self, s, e, dtype=torch.float32):
        """Dense [e - s, n_cols] block of rows s..e-1."""
        c0, c1 = int(self.crow[s]), int(self.crow[e])
        counts = (self.crow[s + 1:e + 1] - self.crow[s:e]).long()
        rows = torch.repeat_interleave(torch.arange(e - s, device=self.val.device), counts)
        out = torch.zeros((e - s, self.shape[1]), dtype=dtype, device=self.val.device)
        out[rows, self.col[c0:c1].long()] = self.val[c0:c1].to(dtype)
        return out


def _batches(docs, batch):
    for s in range(0, len(docs), batch):
        yield docs[s:s + batch]


def build_tfidf_gpu(fit_docs, transform_docs, max_features=300_000, batch=200_000, device="cuda",
                    min_avail_gb=None, log=print):
    """
    fit_docs: list of document lists whose concatenation is the TF-IDF fit corpus.
    transform_docs: list of document lists to vectorise with the fitted model.
    Returns (list of GpuCsr aligned with transform_docs, info dict).
    """
    t0 = time.time()
    term_id = {}
    tf_tot = np.zeros(1 << 16, dtype=np.int64)
    df_tot = np.zeros(1 << 16, dtype=np.int64)
    n_docs = 0
    for docs in fit_docs:
        for b in _batches(docs, batch):
            if min_avail_gb:
                wait_for_memory(min_avail_gb, "tfidf pass 1")
            n_docs += len(b)
            cv = CountVectorizer(**ANALYZER_KW, dtype=np.int64)
            try:
                X = cv.fit_transform(b)
            except ValueError:          # batch without any trigram
                continue
            terms = cv.get_feature_names_out()
            gids = np.fromiter((term_id.setdefault(t, len(term_id)) for t in terms), dtype=np.int64, count=len(terms))
            if len(term_id) > len(tf_tot):
                new = max(len(term_id), 2 * len(tf_tot))
                tf_tot = np.concatenate([tf_tot, np.zeros(new - len(tf_tot), np.int64)])
                df_tot = np.concatenate([df_tot, np.zeros(new - len(df_tot), np.int64)])
            tf_tot[gids] += np.asarray(X.sum(axis=0)).ravel()
            df_tot[gids] += np.bincount(X.indices, minlength=X.shape[1])
            del X, cv
    n_terms = len(term_id)
    tf_tot, df_tot = tf_tot[:n_terms], df_tot[:n_terms]
    terms_arr = np.empty(n_terms, dtype=object)
    for t, i in term_id.items():
        terms_arr[i] = t
    del term_id
    # Alphabetical vocabulary; max_features keeps the most frequent terms exactly like sklearn's
    # _limit_features: (-tfs).argsort()[:limit] (default sort kind) over alphabetically ordered columns.
    alpha = np.argsort(terms_arr)
    keep = alpha
    if n_terms > max_features:
        keep = alpha[np.sort((-tf_tot[alpha]).argsort()[:max_features])]
    vocab = {t: i for i, t in enumerate(terms_arr[keep])}
    df_kept = df_tot[keep].astype(np.float32)
    idf = (np.log(np.float32(n_docs + 1) / (df_kept + np.float32(1))) + np.float32(1)).astype(np.float32)
    t1 = time.time()
    log(f"    TF-IDF pass 1: {n_docs:,} docs, {n_terms:,} terms -> vocab {len(vocab):,} in {t1 - t0:.1f}s "
        f"(RSS {rss_gb():.2f} GB)")

    idx_dtype = torch.int32
    outs = []
    for docs in transform_docs:
        crow_parts, col_parts, val_parts = [], [], []
        offset, n_rows = 0, 0
        for b in _batches(docs, batch):
            if min_avail_gb:
                wait_for_memory(min_avail_gb, "tfidf pass 2")
            cv = CountVectorizer(**ANALYZER_KW, vocabulary=vocab, dtype=np.float32)
            X = cv.transform(b)
            X.sort_indices()
            np.log(X.data, out=X.data)
            X.data += 1.0
            X.data *= idf[X.indices]
            X = normalize(X, norm="l2", copy=False)
            crow_parts.append(torch.from_numpy(X.indptr[1:].astype(np.int64) + offset).to(device))
            col_parts.append(torch.from_numpy(X.indices.astype(np.int32)).to(device))
            val_parts.append(torch.from_numpy(X.data.astype(np.float32)).to(device))
            offset += X.nnz
            n_rows += X.shape[0]
            del X, cv
        crow = torch.cat([torch.zeros(1, dtype=torch.int64, device=device)] + crow_parts)
        col = torch.cat(col_parts) if col_parts else torch.zeros(0, dtype=torch.int32, device=device)
        val = torch.cat(val_parts) if val_parts else torch.zeros(0, dtype=torch.float32, device=device)
        del crow_parts, col_parts, val_parts
        if offset < 2**31 - 1:
            crow = crow.to(idx_dtype)
        else:
            col = col.to(torch.int64)
        outs.append(GpuCsr(crow, col, val, n_rows, len(vocab)))
    info = {"n_docs": n_docs, "n_terms": n_terms, "vocab": len(vocab),
            "pass1_s": t1 - t0, "pass2_s": time.time() - t1}
    log(f"    TF-IDF pass 2: {sum(o.shape[0] for o in outs):,} rows, nnz {sum(o.nnz for o in outs):,} in "
        f"{info['pass2_s']:.1f}s | GPU {torch.cuda.memory_allocated() / 2**30:.2f} GB | RSS {rss_gb():.2f} GB")
    return outs, info


def gpu_topk(pool: GpuCsr, queries: GpuCsr, k: int, chunk: int, fp16: bool = False, progress=None):
    """
    Exact top-k cosine (dot of L2-normalised rows) of every query row against every pool row.
    scores = pool_csr @ query_dense.T per query chunk, topk over the pool dimension.
    Returns (idx int64 [n_q, k], score float32 [n_q, k]) numpy arrays, sorted by score desc.
    progress: optional callable(done_queries) called after each chunk.
    """
    dt = torch.float16 if fp16 else torch.float32
    P = pool.to_sparse(dt)
    k = min(k, pool.shape[0])
    n_q = queries.shape[0]
    out_i = np.empty((n_q, k), dtype=np.int64)
    out_s = np.empty((n_q, k), dtype=np.float32)
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        B = queries.dense_rows(s, e, dt).t().contiguous()    # [V, chunk]
        scores = P @ B                                        # [pool, chunk]
        v, i = torch.topk(scores, k, dim=0)
        out_i[s:e] = i.t().cpu().numpy()
        out_s[s:e] = v.t().float().cpu().numpy()
        del B, scores, v, i
        if progress:
            progress(e)
    del P
    return out_i, out_s


def auto_chunk(pool_rows: int, fp16: bool, budget_gb: float = 2.0, cap: int = 256) -> int:
    """Largest query chunk whose [pool, chunk] score buffer fits budget_gb (topk makes one more copy)."""
    per_q = pool_rows * (2 if fp16 else 4)
    return int(max(1, min(cap, budget_gb * 1e9 // per_q)))
