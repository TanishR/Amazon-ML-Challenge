"""
Variant benchmark for Channel I (char TF-IDF): text variants x fp32/fp16 SpMM x query chunk size.
Pool: all --country rows of norm_train_{source}; fit corpus: all --country S1 + pool (as in s3c_tfidf.py).
Queries: first --n-queries val-fold S1 of --country (split.parquet order).
Reports recall@k of GT pairs (gt_long) from the queries to --source, q/s, GPU memory, and top-k set
agreement between fp16 and fp32.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from tfidf_utils import auto_chunk, build_tfidf_gpu, gpu_topk, make_texts, rss_gb, wait_for_memory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=os.path.expanduser("~/Attempt_1/cache"))
    p.add_argument("--country", default="India")
    p.add_argument("--source", default="source3")
    p.add_argument("--n-queries", type=int, default=5000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--variants", default="na,nsa")
    p.add_argument("--chunks", default="64,128,256")
    p.add_argument("--max-features", type=int, default=300_000)
    p.add_argument("--min-avail-gb", type=float, default=6.5)
    return p.parse_args()


def main():
    a = parse_args()
    cols = ["entity_id", "country", "name_full", "addr_norm", "name_skel"]
    t = pq.read_table(os.path.join(a.cache_dir, "norm_train_source1.parquet"), columns=cols)
    s1 = t.filter(pc.equal(t["country"], a.country)).to_pandas()
    t = pq.read_table(os.path.join(a.cache_dir, f"norm_train_{a.source}.parquet"), columns=cols)
    pool = t.filter(pc.equal(t["country"], a.country)).to_pandas()
    del t
    split = pd.read_parquet(os.path.join(a.cache_dir, "split.parquet"))
    s1_set = set(s1["entity_id"])
    qids = [s for s in split.loc[split["fold"] == "val", "s1_id"] if s in s1_set][:a.n_queries]
    qdf = s1.set_index("entity_id").loc[qids].reset_index()
    gt = pd.read_parquet(os.path.join(a.cache_dir, "gt_long.parquet"), columns=["s1_id", "match_id"])
    prefix = "S2-" if a.source == "source2" else "S3-"
    gt = gt[gt["s1_id"].isin(set(qids)) & gt["match_id"].str.startswith(prefix)]
    ppos = {e: i for i, e in enumerate(pool["entity_id"])}
    qpos = {e: i for i, e in enumerate(qids)}
    pairs = [(qpos[s], ppos[m]) for s, m in zip(gt["s1_id"], gt["match_id"]) if m in ppos]
    print(f"{a.country} {a.source}: pool {len(pool):,} | fit S1 {len(s1):,} | queries {len(qids):,} | "
          f"GT pairs {len(gt):,} (in pool {len(pairs):,}) | RSS {rss_gb():.2f} GB", flush=True)

    results = {}
    for variant in a.variants.split(","):
        wait_for_memory(a.min_avail_gb, "variant start")
        print(f"\n=== variant {variant} ===", flush=True)
        (Gq, Gp), info = build_tfidf_gpu([make_texts(s1, variant), make_texts(pool, variant)],
                                         [make_texts(qdf, variant), make_texts(pool, variant)],
                                         max_features=a.max_features, min_avail_gb=a.min_avail_gb)
        print(f"  pool on GPU: {Gp.nbytes() / 2**30:.2f} GB (nnz {Gp.nnz:,}, {Gp.nnz / Gp.shape[0]:.1f}/row)")
        for fp16 in (False, True):
            chunks = sorted({min(int(c), auto_chunk(Gp.shape[0], fp16, 2.0, 1024)) for c in a.chunks.split(",")})
            for c in chunks:
                gpu_topk(Gp, Gq, a.k, c, fp16)                     # warm-up
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                idx, sc = gpu_topk(Gp, Gq, a.k, c, fp16)
                torch.cuda.synchronize()
                dt = time.time() - t0
                tops = [set(r) for r in idx]
                hit = sum(1 for qi, pi in pairs if pi in tops[qi])
                results[(variant, fp16, c)] = idx
                print(f"  {'fp16' if fp16 else 'fp32'} chunk {c:>4}: {len(qids) / dt:>7,.0f} q/s | recall@{a.k} "
                      f"{hit:,}/{len(pairs):,} = {100 * hit / max(1, len(pairs)):.2f}% | peak GPU "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB | RSS {rss_gb():.2f} GB", flush=True)
        c32 = max(c for (v, f, c) in results if v == variant and not f)
        c16 = max(c for (v, f, c) in results if v == variant and f)
        same = np.mean([set(x) == set(y) for x, y in zip(results[(variant, False, c32)], results[(variant, True, c16)])])
        print(f"  fp16 vs fp32 identical top-{a.k} sets: {100 * same:.2f}% of queries")
        del Gq, Gp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
