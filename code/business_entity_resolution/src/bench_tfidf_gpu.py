"""
Benchmark: GPU top-k search over a char TF-IDF pool (India S3) vs exact CPU search.

Pool: first --pool-rows India rows of norm_train_source3 (file order).
Queries: first --n-queries India val-fold S1 (split.parquet order).
Text: name_full + " " + addr_norm; TfidfVectorizer(char_wb, (3,3), sublinear_tf, float32, l2) fit on the pool.
GPU: pool as torch sparse CSR; per query chunk: scores = pool_csr @ query_dense.T, topk(k, dim=0).
CPU: same product with scipy (pool_csr @ query_dense.T), exact top-k via argpartition.
Reports q/s per chunk size, GPU memory, peak RSS, recall@k on GT pairs whose S3 is in the pool,
and GPU vs CPU agreement.
"""
import argparse
import os
import resource
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


def rss_gb():
    return psutil.Process().memory_info().rss / 2**30


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def guard(min_avail_gb, where):
    avail = psutil.virtual_memory().available / 2**30
    if avail < min_avail_gb:
        raise SystemExit(f"ABORT at {where}: system available RAM {avail:.1f} GB < {min_avail_gb} GB")
    return avail


def parse_args():
    p = argparse.ArgumentParser(description="GPU char TF-IDF top-k benchmark")
    p.add_argument("--pool-rows", type=int, default=1_000_000)
    p.add_argument("--n-queries", type=int, default=5_000)
    p.add_argument("--country", type=str, default="India")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--chunks", type=str, default="128,256,512,1000")
    p.add_argument("--result-budget-gb", type=float, default=4.0, help="Max bytes of the pool x chunk score buffer (1e9 bytes/GB)")
    p.add_argument("--cpu-chunk", type=int, default=128)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--min-avail-gb", type=float, default=7.0)
    p.add_argument("--data-root", type=str, default=os.path.expanduser("~/Attempt_1"),
                   help="Read-only root holding cache/ and dataset/")
    return p.parse_args()


def load_pool(cache_dir, country, n_rows):
    pf = pq.ParquetFile(os.path.join(cache_dir, "norm_train_source3.parquet"))
    parts, n = [], 0
    for b in pf.iter_batches(batch_size=500_000, columns=["entity_id", "country", "name_full", "addr_norm"]):
        t = pa.Table.from_batches([b])
        t = t.filter(pc.equal(t.column("country"), country)).select(["entity_id", "name_full", "addr_norm"])
        parts.append(t.slice(0, n_rows - n))
        n += parts[-1].num_rows
        if n >= n_rows:
            break
    return pa.concat_tables(parts).to_pandas()


def load_queries(cache_dir, country, n_queries):
    sp = pq.read_table(os.path.join(cache_dir, "split.parquet"))
    val = pc.filter(sp.column("s1_id"), pc.equal(sp.column("fold"), "val"))
    n1 = pq.read_table(os.path.join(cache_dir, "norm_train_source1.parquet"),
                       columns=["entity_id", "country", "name_full", "addr_norm"])
    n1 = n1.filter(pc.and_(pc.equal(n1.column("country"), country), pc.is_in(n1.column("entity_id"), val)))
    q = n1.select(["entity_id", "name_full", "addr_norm"]).to_pandas().set_index("entity_id")
    val_order = [s for s in val.to_pylist() if s in q.index][:n_queries]
    return q.loc[val_order].reset_index()


def load_gt(gt_path, query_ids, pool_ids):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt = gt[gt["source1_entity_id"].isin(set(query_ids))]
    pos = {e: i for i, e in enumerate(pool_ids)}
    qpos = {e: i for i, e in enumerate(query_ids)}
    pairs, n_s3_total = [], 0
    for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"].fillna("")):
        for c in m.split(","):
            if c.startswith("S3-"):
                n_s3_total += 1
                if c in pos:
                    pairs.append((qpos[s1], pos[c]))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2), n_s3_total


def recall(topk_idx, pairs):
    hit = sum(1 for qi, pi in pairs if pi in topk_idx[qi])
    return hit, len(pairs)


def gpu_search(pool_t, Q, k, chunk, dev):
    """Returns (top-k indices [n_q, k] int64, top-k scores [n_q, k] float32) for all queries."""
    n_q = Q.shape[0]
    out_i = np.empty((n_q, k), dtype=np.int64)
    out_s = np.empty((n_q, k), dtype=np.float32)
    for s in range(0, n_q, chunk):
        qc = Q[s:s + chunk]
        q_sp = torch.sparse_csr_tensor(
            torch.from_numpy(qc.indptr.astype(np.int64)), torch.from_numpy(qc.indices.astype(np.int64)),
            torch.from_numpy(qc.data), size=qc.shape).to(dev)
        q_dense_t = q_sp.to_dense().t().contiguous()      # [V, chunk]
        scores = pool_t @ q_dense_t                        # [pool, chunk]
        vals, idx = torch.topk(scores, k, dim=0)           # [k, chunk], sorted desc
        out_i[s:s + chunk] = idx.t().cpu().numpy()
        out_s[s:s + chunk] = vals.t().cpu().numpy()
        del q_sp, q_dense_t, scores, vals, idx
    return out_i, out_s


def cpu_search(pool, Q, k, chunk, min_avail):
    n_q = Q.shape[0]
    out_i = np.empty((n_q, k), dtype=np.int64)
    out_s = np.empty((n_q, k), dtype=np.float32)
    for s in range(0, n_q, chunk):
        if (s // chunk) % 10 == 0:
            guard(min_avail, f"cpu chunk {s}")
        qd = Q[s:s + chunk].toarray().T                    # [V, chunk]
        scores = np.asarray(pool @ qd, dtype=np.float32)   # [pool, chunk]
        part = np.argpartition(-scores, k - 1, axis=0)[:k]  # [k, chunk]
        ps = np.take_along_axis(scores, part, axis=0)
        order = np.argsort(-ps, axis=0, kind="stable")
        out_i[s:s + chunk] = np.take_along_axis(part, order, axis=0).T
        out_s[s:s + chunk] = np.take_along_axis(ps, order, axis=0).T
        del qd, scores, part, ps, order
    return out_i, out_s


def main():
    args = parse_args()
    cache = os.path.join(args.data_root, "cache")
    gt_path = os.path.join(args.data_root, "dataset/train/train_ground_truth.tsv")
    k = args.k
    print(f"available RAM {guard(args.min_avail_gb, 'start'):.1f} GB | RSS {rss_gb():.2f} GB", flush=True)

    t0 = time.time()
    pool_df = load_pool(cache, args.country, args.pool_rows)
    q_df = load_queries(cache, args.country, args.n_queries)
    pairs, n_s3_total = load_gt(gt_path, q_df["entity_id"].tolist(), pool_df["entity_id"].tolist())
    print(f"pool {len(pool_df):,} {args.country} S3 rows | queries {len(q_df):,} {args.country} val S1 | "
          f"GT S3 pairs for queries: {n_s3_total:,}, in pool: {len(pairs):,} | {time.time() - t0:.1f}s | "
          f"RSS {rss_gb():.2f} GB", flush=True)

    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32)
    pool = vec.fit_transform((pool_df["name_full"].fillna("") + " " + pool_df["addr_norm"].fillna("")).tolist())
    Q = vec.transform((q_df["name_full"].fillna("") + " " + q_df["addr_norm"].fillna("")).tolist())
    del pool_df
    pool.sort_indices()
    Q.sort_indices()
    print(f"TF-IDF fit+transform {time.time() - t0:.1f}s | vocab {len(vec.vocabulary_):,} | pool nnz {pool.nnz:,} "
          f"({pool.nnz / pool.shape[0]:.1f}/row) | query nnz/row {Q.nnz / Q.shape[0]:.1f} | RSS {rss_gb():.2f} GB",
          flush=True)
    vec.vocabulary_ = None
    guard(args.min_avail_gb, "after tfidf")

    # ---- GPU ----
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    pool_t = torch.sparse_csr_tensor(
        torch.from_numpy(pool.indptr.astype(np.int64)), torch.from_numpy(pool.indices.astype(np.int64)),
        torch.from_numpy(pool.data), size=pool.shape).to(dev)
    pool_gpu_gb = torch.cuda.memory_allocated() / 2**30
    max_chunk = int(args.result_budget_gb * 1e9 // (4 * pool.shape[0]))
    chunks = [c for c in (int(x) for x in args.chunks.split(",")) if c <= max_chunk] or [max_chunk]
    print(f"\nGPU pool CSR resident: {pool_gpu_gb:.2f} GB | max chunk for {args.result_budget_gb:g} GB result "
          f"buffer: {max_chunk} | testing chunks {chunks}", flush=True)
    print(f"{'chunk':>6} {'result buf GB':>13} {'q/s (best of %d)' % args.reps:>17} {'peak GPU alloc GB':>17} {'peak RSS GB':>11}")
    gpu_res = {}
    for c in chunks:
        gpu_search(pool_t, Q[:c], k, c, dev)               # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        best = float("inf")
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t = time.time()
            gi, gs = gpu_search(pool_t, Q, k, c, dev)
            torch.cuda.synchronize()
            best = min(best, time.time() - t)
        gpu_res[c] = (gi, gs)
        print(f"{c:>6} {4 * pool.shape[0] * c / 1e9:>13.2f} {Q.shape[0] / best:>17,.0f} "
              f"{torch.cuda.max_memory_allocated() / 2**30:>17.2f} {peak_rss_gb():>11.2f}", flush=True)
        guard(args.min_avail_gb, f"gpu chunk {c}")
    del pool_t
    torch.cuda.empty_cache()

    # ---- CPU exact ----
    t = time.time()
    ci, cs = cpu_search(pool, Q, k, args.cpu_chunk, args.min_avail_gb)
    cpu_t = time.time() - t
    print(f"\nCPU exact (scipy, chunk {args.cpu_chunk}): {Q.shape[0] / cpu_t:,.1f} q/s | peak RSS {peak_rss_gb():.2f} GB",
          flush=True)

    # ---- Agreement + recall ----
    c_hit, n = recall([set(r) for r in ci], pairs)
    print(f"\nrecall@{k} (GT S3 pairs with S3 in pool, n={n:,}):  CPU exact {c_hit:,}/{n:,} = {100 * c_hit / max(n, 1):.2f}%")
    for c, (gi, gs) in gpu_res.items():
        g_hit, _ = recall([set(r) for r in gi], pairs)
        same_set = np.array([set(a) == set(b) for a, b in zip(gi, ci)])
        max_score_diff = float(np.abs(gs - cs).max())
        # set differences allowed only where the k-th score is tied within float32 tolerance
        tie_ok = all(abs(gs[q, -1] - cs[q, -1]) < 1e-5 and
                     np.all(np.abs(np.sort(gs[q]) - np.sort(cs[q])) < 1e-5)
                     for q in np.flatnonzero(~same_set))
        print(f"  GPU chunk {c:>5}: recall {g_hit:,}/{n:,} = {100 * g_hit / max(n, 1):.2f}% | identical top-{k} sets "
              f"{same_set.sum():,}/{len(same_set):,} | max |score diff| {max_score_diff:.2e} | "
              f"non-identical sets explained by ties: {tie_ok}")
    print(f"\npeak RSS {peak_rss_gb():.2f} GB | available RAM now {psutil.virtual_memory().available / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
