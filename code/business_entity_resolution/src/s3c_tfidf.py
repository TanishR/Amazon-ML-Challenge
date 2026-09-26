#!/usr/bin/env python3
"""
Stage 3c: Channel I candidate retrieval with char TF-IDF (GPU top-k).

For every (country, source) present in the split (countries read from the data, never hardcoded):
  - fit char TF-IDF (char_wb, (3,3), sublinear_tf, float32, max_features) on that split's S1 records of the
    country + that source's records of the country; text = name_full + " " + name_skel + " " + addr_norm
  - GPU top-k search of every S1 of the country against the source's records of the country
Output: cache/tfidf_{split}_{source}.parquet with s1_id, cand_id, tfidf_score (float32), tfidf_rank (int8, 1 = best).
Resume: each (country, source) part is saved to cache/tfidf_parts/ and skipped when present.
Memory: TF-IDF is built in two streaming passes and held on the GPU (tfidf_utils); the run pauses whenever
system available RAM drops below --min-avail-gb.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from tfidf_utils import auto_chunk, build_tfidf_gpu, gpu_topk, make_texts, rss_gb, wait_for_memory

SOURCES = ["source2", "source3"]
NORM_COLS = ["entity_id", "country", "name_full", "name_skel", "addr_norm"]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3c: Channel I char TF-IDF GPU top-k candidates")
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--cache-dir", default=None, help="Cache with norm_{split}_source*.parquet (default: config.CACHE_DIR)")
    p.add_argument("--out-dir", default=None, help="Where tfidf_{split}_{source}.parquet and tfidf_parts/ go (default: cache dir)")
    p.add_argument("--mode", default="fwd", choices=["fwd", "rev", "dict"],
                   help="fwd: S1 -> S2/S3 top-k (Channel I); rev: S2/S3 -> S1 top-k (Channel J); "
                        "dict: fwd with dictionary-mapped S2/S3 names (Channel K)")
    p.add_argument("--dict", default=None, help="dict mode: token map JSON (default cache/indic_dict.json)")
    p.add_argument("--k", type=int, default=None, help="top-k per query (default: fwd/dict 10, rev 5)")
    p.add_argument("--max-features", type=int, default=300_000)
    p.add_argument("--text", default="nsa", choices=["nsa", "na"])
    p.add_argument("--fp32", action="store_true", help="fp32 SpMM (default fp16: ~1.5x faster, same recall@10 +-1 pair)")
    p.add_argument("--chunk", type=int, default=128, help="Queries per GPU SpMM (capped so the score buffer <= 2 GB)")
    p.add_argument("--limit", type=int, default=None, help="TEST: max S1 queries per (country, source) part")
    p.add_argument("--pool-limit", type=int, default=None, help="TEST: max source records per country")
    p.add_argument("--s1-list", default=None, help="TEST: parquet with column s1_id; only these S1 are fit and searched")
    p.add_argument("--min-avail-gb", type=float, default=6.5, help="Pause while system available RAM is below this")
    return p.parse_args()


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _read_norm(cache_dir, split, source):
    return pq.read_table(os.path.join(cache_dir, f"norm_{split}_{source}.parquet"), columns=NORM_COLS)


MODES = {
    # mode: output file prefix, column names, default k, which side is queried
    "fwd": dict(prefix="tfidf", cols=("s1_id", "cand_id", "tfidf_score", "tfidf_rank"), k=10, query="s1"),
    "rev": dict(prefix="tfidf_rev", cols=("cand_id", "s1_id", "score", "rank"), k=5, query="src"),
    "dict": dict(prefix="tfidf_dict", cols=("s1_id", "cand_id", "score", "rank"), k=10, query="s1"),
}


def main():
    a = parse_args()
    if a.cache_dir is None:
        import config
        a.cache_dir = config.CACHE_DIR
    out_dir = a.out_dir or a.cache_dir
    if (a.limit or a.pool_limit or a.s1_list) and os.path.realpath(out_dir) == os.path.realpath(a.cache_dir):
        raise SystemExit("--limit/--pool-limit/--s1-list are test options: pass --out-dir outside the cache dir")
    mode = MODES[a.mode]
    k_req = a.k or mode["k"]
    name_map = None
    if a.mode == "dict":
        dict_path = a.dict or os.path.join(a.cache_dir, "indic_dict.json")
        with open(dict_path, encoding="utf-8") as f:
            name_map = json.load(f)
        print(f"Loaded name dictionary {dict_path}: {len(name_map):,} token mappings", flush=True)
    parts_dir = os.path.join(out_dir, "tfidf_parts")
    os.makedirs(parts_dir, exist_ok=True)
    fp16 = not a.fp32
    dev = "cuda"
    t_start = time.time()
    print(f"=== Stage 3c: char TF-IDF mode={a.mode} ({a.split}) === text={a.text} k={k_req} "
          f"{'fp16' if fp16 else 'fp32'} max_features={a.max_features:,} limit={a.limit} pool_limit={a.pool_limit}")
    print(f"cache: {a.cache_dir} | out: {out_dir} | RSS {rss_gb():.2f} GB", flush=True)

    s1_tbl = _read_norm(a.cache_dir, a.split, "source1")
    if a.s1_list:
        s1_tbl = s1_tbl.filter(pc.is_in(s1_tbl["entity_id"], pq.read_table(a.s1_list, columns=["s1_id"])["s1_id"]))
    s1_countries = sorted(c for c in pc.unique(s1_tbl["country"]).to_pylist() if c and str(c).strip())
    print(f"S1: {s1_tbl.num_rows:,} rows, countries: {s1_countries}", flush=True)

    # Plan every (source, country) part; --limit caps the query side, --pool-limit the indexed side
    plan = []
    for source in SOURCES:
        final_p = os.path.join(out_dir, f"{mode['prefix']}_{a.split}_{source}.parquet")
        src_tbl = _read_norm(a.cache_dir, a.split, source)
        src_counts = dict(zip(*[x.to_pylist() for x in pc.value_counts(src_tbl["country"]).flatten()]))
        del src_tbl
        for c in s1_countries:
            n_src = src_counts.get(c, 0)
            if n_src == 0:
                print(f"  skip {source}/{c}: no {source} records", flush=True)
                continue
            n_s1 = int(pc.sum(pc.equal(s1_tbl["country"], c)).as_py())
            n_q, n_idx = (n_s1, n_src) if mode["query"] == "s1" else (n_src, n_s1)
            n_q = min(n_q, a.limit) if a.limit else n_q
            n_idx = min(n_idx, a.pool_limit) if a.pool_limit else n_idx
            tag = "" if a.mode == "fwd" else f"{a.mode}_"
            part_p = os.path.join(parts_dir, f"{tag}{a.split}_{source}_{_safe(c)}.parquet")
            plan.append(dict(source=source, country=c, n_q=n_q, n_pool=n_idx, part=part_p, final=final_p,
                             done=os.path.exists(part_p) or os.path.exists(final_p)))
    total_work = sum(p["n_q"] * p["n_pool"] for p in plan if not p["done"])
    print("Plan:")
    for p in plan:
        print(f"  {p['source']:8} {p['country']:10} queries {p['n_q']:>10,} index {p['n_pool']:>10,} "
              f"{'DONE (resume skip)' if p['done'] else ''}")
    print(flush=True)

    work_done = 0
    t_work = time.time()
    cur_source = None
    src_tbl = None
    for p in plan:
        if p["done"]:
            continue
        wait_for_memory(a.min_avail_gb, f"{p['source']}/{p['country']} start")
        t_part = time.time()
        if cur_source != p["source"]:
            src_tbl = _read_norm(a.cache_dir, a.split, p["source"])
            cur_source = p["source"]
        c = p["country"]
        s1_c = s1_tbl.filter(pc.equal(s1_tbl["country"], c)).to_pandas()
        src_c = src_tbl.filter(pc.equal(src_tbl["country"], c)).to_pandas()
        if mode["query"] == "s1":
            q_df, idx_df = s1_c, src_c
        else:
            q_df, idx_df = src_c, s1_c
        if a.pool_limit:
            idx_df = idx_df.iloc[:a.pool_limit]
        if a.limit:
            q_df = q_df.iloc[:a.limit]
        if a.limit or a.pool_limit:            # test mode: fit on the limited subsets only
            s1_c, src_c = (q_df, idx_df) if mode["query"] == "s1" else (idx_df, q_df)
        print(f"--- {p['source']} / {c}: fit {len(s1_c):,} S1 + {len(src_c):,} {p['source']} | "
              f"queries {len(q_df):,} ({'S1' if mode['query'] == 's1' else p['source']}) | index {len(idx_df):,} | "
              f"RSS {rss_gb():.2f} GB", flush=True)

        s1_txt = make_texts(s1_c, a.text)
        src_txt = make_texts(src_c, a.text, name_map)
        if mode["query"] == "s1":
            q_txt = s1_txt if len(q_df) == len(s1_c) else make_texts(q_df, a.text)
            idx_txt = src_txt
        else:
            q_txt = src_txt if len(q_df) == len(src_c) else make_texts(q_df, a.text, name_map)
            idx_txt = s1_txt
        (Gq, Gp), _ = build_tfidf_gpu([s1_txt, src_txt], [q_txt, idx_txt],
                                      max_features=a.max_features, device=dev, min_avail_gb=a.min_avail_gb,
                                      log=lambda m: print(m, flush=True))
        del s1_txt, src_txt, q_txt, idx_txt
        chunk = min(a.chunk, auto_chunk(Gp.shape[0], fp16, 2.0, 1024))
        n_q = Gq.shape[0]
        t_s = time.time()
        last = [t_s]

        def progress(done):
            now = time.time()
            if now - last[0] < 60 and done < n_q:
                return
            last[0] = now
            qps = done / max(1e-9, now - t_s)
            part_eta = (n_q - done) / max(qps, 1e-9)
            wd = work_done + done * Gp.shape[0]
            rate = wd / max(1e-9, now - t_work)          # query x index-row per second, overall
            all_eta = (total_work - wd) / max(rate, 1e-9)
            print(f"    {p['source']}/{c}: {done:,}/{n_q:,} queries | {qps:,.0f} q/s | part ETA {part_eta / 60:.1f} min | "
                  f"split ETA {all_eta / 60:.1f} min | RSS {rss_gb():.2f} GB | GPU "
                  f"{torch.cuda.memory_allocated() / 2**30:.2f} GB (peak {torch.cuda.max_memory_allocated() / 2**30:.2f})",
                  flush=True)
            wait_for_memory(a.min_avail_gb, f"{p['source']}/{c} search")

        idx, sc = gpu_topk(Gp, Gq, k_req, chunk, fp16, progress)
        k = idx.shape[1]
        idx_ids = idx_df["entity_id"].to_numpy()
        keep = sc.ravel() > 0
        c_q, c_idx, c_score, c_rank = mode["cols"]
        part = pa.table({
            c_q: pa.array(np.repeat(q_df["entity_id"].to_numpy(), k)[keep]),
            c_idx: pa.array(idx_ids[idx.ravel()][keep]),
            c_score: pa.array(sc.ravel()[keep].astype(np.float32)),
            c_rank: pa.array(np.tile(np.arange(1, k + 1, dtype=np.int8), n_q)[keep]),
        })
        tmp = p["part"] + ".tmp"
        pq.write_table(part, tmp)
        os.replace(tmp, p["part"])
        work_done += n_q * Gp.shape[0]
        print(f"  saved {p['part']}: {part.num_rows:,} pairs ({part.num_rows / max(1, n_q):.2f}/query) | "
              f"{n_q / max(1e-9, time.time() - t_s):,.0f} q/s search | part {time.time() - t_part:.0f}s | "
              f"RSS {rss_gb():.2f} GB", flush=True)
        del Gq, Gp, idx, sc, part, s1_c, src_c, q_df, idx_df
        torch.cuda.empty_cache()

    # Assemble {prefix}_{split}_{source}.parquet from its parts
    for source in SOURCES:
        src_plan = [p for p in plan if p["source"] == source]
        final_p = os.path.join(out_dir, f"{mode['prefix']}_{a.split}_{source}.parquet")
        if not src_plan or os.path.exists(final_p):
            continue
        missing = [p["part"] for p in src_plan if not os.path.exists(p["part"])]
        if missing:
            print(f"NOT assembling {final_p}: missing parts {missing}", flush=True)
            continue
        tbl = pa.concat_tables([pq.read_table(p["part"]) for p in src_plan])
        pq.write_table(tbl, final_p + ".tmp")
        os.replace(final_p + ".tmp", final_p)
        print(f"Wrote {final_p}: {tbl.num_rows:,} pairs, {len(pc.unique(tbl['s1_id'])):,} S1, "
              f"{len(pc.unique(tbl['cand_id'])):,} {source} records", flush=True)
        del tbl

    print(f"\nStage 3c mode={a.mode} ({a.split}) finished in {(time.time() - t_start) / 60:.1f} min | RSS {rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
