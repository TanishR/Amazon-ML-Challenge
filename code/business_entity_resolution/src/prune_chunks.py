#!/usr/bin/env python3
"""
Two-stage blocking: a small FILTER model on cheap blocking-stage signals that already exist in
cands_{split}_chunk_*.parquet (+ the J/K scores from tfidf_rev_* / tfidf_dict_*), no stage-4 features.
Per S1 only the top-K candidates by filter score are kept.

  train   fit LightGBM (<= 100 trees) on train-fold S1 rows, labels from cache/gt_long.parquet
  eval    val S1 rows: % of GT pairs present in the unpruned candidates that survive top-K, cands/S1
  apply   prune cands_{split}_chunk_*.parquet IN PLACE to the top-K per S1 (requires the full backup dir)

Filter features per row: emb_score, emb_rank, ch_* flags, cand_source, tfidf_score, tfidf_rank, revtf_rank,
revtf_score, dict_rank, dict_score, n_flags, per-S1 ranks of emb/tfidf/revtf/dict scores, gap to the S1's best
emb_score, and the S1's candidate count.
"""
import argparse
import json
import os
import re
import threading
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config

FLAGS = ["ch_emb", "ch_addr", "ch_skel", "ch_rare", "ch_rev", "ch_rerank", "ch_k1", "ch_k3", "ch_k5",
         "ch_comb", "ch_tfidf", "ch_revtf", "ch_dict"]
BASE = ["emb_score", "emb_rank", "cand_source", "tfidf_score", "tfidf_rank", "revtf_rank"] + FLAGS
FEATURES = BASE + ["revtf_score", "dict_rank", "dict_score", "n_flags", "r_emb", "r_tfidf", "r_revtf", "r_dict",
                   "gap_emb", "n_cands"]
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_child_samples": 200,
          "bagging_fraction": 0.5, "bagging_freq": 1, "n_jobs": 8, "verbose": -1, "seed": 42}
N_TREES = 100


def rss_gb():
    return psutil.Process().memory_info().rss / 2**30


def start_watchdog(min_avail_gb):
    def run():
        while True:
            a = psutil.virtual_memory().available / 2**30
            if a < min_avail_gb:
                print(f"\nABORT: system available RAM {a:.1f} GB < {min_avail_gb} GB (RSS {rss_gb():.1f} GB)", flush=True)
                os._exit(3)
            time.sleep(3)
    threading.Thread(target=run, daemon=True).start()


def ids_to_int(col) -> np.ndarray:
    col = pa.array(col) if not isinstance(col, (pa.Array, pa.ChunkedArray)) else col
    digit = pc.cast(pc.utf8_slice_codeunits(col, 1, 2), "int64")
    num = pc.cast(pc.utf8_slice_codeunits(col, 3), "int64")
    return pc.add(pc.multiply(digit, 1_000_000_000_000), num).to_numpy()


def chunk_files(cache_dir, split):
    fs = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if re.match(rf"cands_{split}_chunk_\d+\.parquet$", f)]
    return sorted(fs, key=lambda x: int(re.search(r"chunk_(\d+)", os.path.basename(x)).group(1)))


def load_pairs(cache_dir, split, prefix):
    """(s1, cand, score, rank) int64/float32/int16 arrays of {prefix}_{split}_source{2,3}.parquet, sorted by S1."""
    parts = []
    for src in ("source2", "source3"):
        p = os.path.join(cache_dir, f"{prefix}_{split}_{src}.parquet")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        t = pq.read_table(p, columns=["s1_id", "cand_id", "score", "rank"])
        parts.append((ids_to_int(t["s1_id"]), ids_to_int(t["cand_id"]), t["score"].to_numpy().astype(np.float32),
                      t["rank"].to_numpy().astype(np.int16)))
    s1 = np.concatenate([x[0] for x in parts])
    o = np.argsort(s1, kind="stable")
    return tuple(np.concatenate([x[i] for x in parts])[o] for i in range(4))


def _pairs_for(pairs, s1_wanted):
    s1s = pairs[0]
    u = np.unique(s1_wanted)
    lo, hi = np.searchsorted(s1s, u, "left"), np.searchsorted(s1s, u, "right")
    lens = hi - lo
    idx = np.arange(int(lens.sum())) + np.repeat(lo - (np.cumsum(lens) - lens), lens)
    return pd.DataFrame({"s1": pairs[0][idx], "cand": pairs[1][idx], "score": pairs[2][idx], "rank": pairs[3][idx]})


def _group_rank(s1, v):
    """1-based descending rank of v within each S1 (ties share the minimum rank)."""
    return pd.Series(v).groupby(s1).rank(method="min", ascending=False).values.astype(np.float32)


def build_features(d: pd.DataFrame, pj, pk) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Filter feature frame for chunk rows d; returns (X, s1_int, cand_int)."""
    s1 = ids_to_int(pa.array(d["s1_id"].values, type=pa.string()))
    cand = ids_to_int(pa.array(d["cand_id"].values, type=pa.string()))
    keys = pd.DataFrame({"s1": s1, "cand": cand, "row": np.arange(len(d))})
    X = pd.DataFrame({c: d[c].values.astype(np.float32) for c in BASE})
    for name, pairs, rank_col in (("revtf", pj, None), ("dict", pk, "dict_rank")):
        m = _pairs_for(pairs, s1).merge(keys, on=["s1", "cand"], how="inner")
        sc = np.zeros(len(d), dtype=np.float32)
        sc[m["row"].values] = m["score"].values
        X[f"{name}_score"] = sc
        if rank_col:
            rk = np.full(len(d), 99, dtype=np.float32)
            rk[m["row"].values] = m["rank"].values
            X[rank_col] = rk
    X["n_flags"] = X[FLAGS].sum(axis=1).astype(np.float32)
    X["r_emb"] = _group_rank(s1, X["emb_score"].values)
    X["r_tfidf"] = _group_rank(s1, X["tfidf_score"].values)
    X["r_revtf"] = _group_rank(s1, X["revtf_score"].values)
    X["r_dict"] = _group_rank(s1, X["dict_score"].values)
    X["gap_emb"] = (pd.Series(X["emb_score"].values).groupby(s1).transform("max").values - X["emb_score"].values).astype(np.float32)
    X["n_cands"] = pd.Series(s1).groupby(s1).transform("size").values.astype(np.float32)
    return X[FEATURES], s1, cand


def topk_keep(s1, score, emb_score, k):
    order = np.lexsort((-emb_score, -score, s1))
    s1s = s1[order]
    start = np.r_[0, np.flatnonzero(s1s[1:] != s1s[:-1]) + 1]
    rank_sorted = np.arange(len(s1s)) - np.repeat(start, np.diff(np.r_[start, len(s1s)])) + 1
    rank = np.empty(len(s1), dtype=np.int32)
    rank[order] = rank_sorted
    return rank <= k, rank


def cmd_train_eval(a):
    start_watchdog(a.min_avail_gb)
    split = pd.read_parquet(os.path.join(a.cache_dir, "split.parquet"))
    tr_all = np.sort(split.loc[split.fold == "train", "s1_id"].unique())
    rng = np.random.default_rng(42)
    train_s1 = set(tr_all[rng.choice(len(tr_all), int(len(tr_all) * a.train_frac), replace=False)])
    print(f"filter trains on {len(train_s1):,} of {len(tr_all):,} train-fold S1 ({a.train_frac:.0%}, seed 42)", flush=True)
    val_s1 = set(split.loc[split.fold == "val", "s1_id"])
    gt = pd.read_parquet(os.path.join(a.cache_dir, "gt_long.parquet"), columns=["s1_id", "match_id"])
    gt_i = pd.DataFrame({"s1": ids_to_int(pa.array(gt.s1_id.values, type=pa.string())),
                         "cand": ids_to_int(pa.array(gt.match_id.values, type=pa.string())), "y": np.int8(1)})
    n_gt_val = int(gt.s1_id.isin(val_s1).sum())
    del gt
    pj = load_pairs(a.cache_dir, "train", "tfidf_rev")
    pk = load_pairs(a.cache_dir, "train", "tfidf_dict")
    print(f"J pairs {len(pj[0]):,} | K pairs {len(pk[0]):,} | RSS {rss_gb():.2f} GB", flush=True)
    cols = ["s1_id", "cand_id"] + BASE
    Xtr, ytr, Xva, yva, s1va, embva = [], [], [], [], [], []
    files = chunk_files(a.cache_dir, "train")[: a.max_chunks]
    for i, fp in enumerate(files):
        d = pd.read_parquet(fp, columns=cols)
        if "ch_revtf" not in d.columns or "ch_dict" not in d.columns:
            raise SystemExit(f"{fp} has no J/K columns yet")
        d = d[d.s1_id.isin(train_s1 | val_s1)].reset_index(drop=True)
        X, s1, cand = build_features(d, pj, pk)
        y = pd.DataFrame({"s1": s1, "cand": cand}).merge(gt_i, on=["s1", "cand"], how="left")["y"].fillna(0).values.astype(np.int8)
        tr = d.s1_id.isin(train_s1).values
        Xtr.append(X.values[tr]); ytr.append(y[tr])
        Xva.append(X.values[~tr]); yva.append(y[~tr]); s1va.append(s1[~tr]); embva.append(X["emb_score"].values[~tr])
        del d, X
        print(f"  {os.path.basename(fp)}: train rows {int(tr.sum()):,}, val rows {int((~tr).sum()):,} | RSS {rss_gb():.2f} GB", flush=True)
    Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
    t0 = time.time()
    booster = lgb.train(PARAMS, lgb.Dataset(Xtr, label=ytr, feature_name=FEATURES), num_boost_round=N_TREES)
    print(f"filter trained on {len(Xtr):,} train-fold rows (pos {int(ytr.sum()):,}) in {time.time() - t0:.0f}s", flush=True)
    del Xtr, ytr
    os.makedirs(os.path.dirname(a.model), exist_ok=True)
    booster.save_model(a.model)
    imp = booster.feature_importance("gain")
    print("top gain:", ", ".join(f"{f} {100 * g / imp.sum():.1f}%" for f, g in sorted(zip(FEATURES, imp), key=lambda x: -x[1])[:8]))

    Xva, yva, s1va, embva = np.vstack(Xva), np.concatenate(yva), np.concatenate(s1va), np.concatenate(embva)
    score = booster.predict(Xva).astype(np.float32)
    n_s1 = len(np.unique(s1va))
    pos_unpruned = int(yva.sum())
    print(f"\nVAL: {n_s1:,} S1 | unpruned candidates {len(yva):,} ({len(yva) / n_s1:.1f}/S1) | GT pairs in unpruned "
          f"{pos_unpruned:,} of {n_gt_val:,} val GT pairs ({100 * pos_unpruned / n_gt_val:.2f}% blocking recall)")
    print(f"{'K':>4} | {'GT retained (vs unpruned)':>26} | {'vs all val GT':>13} | {'cands/S1':>8}")
    rows = []
    for k in [int(x) for x in a.ks.split(",")]:
        keep, _ = topk_keep(s1va, score, embva, k)
        ret = int(yva[keep].sum())
        rows.append({"k": k, "retained_pct": 100 * ret / pos_unpruned, "recall_all_pct": 100 * ret / n_gt_val,
                     "cands_per_s1": keep.sum() / n_s1})
        print(f"{k:>4} | {100 * ret / pos_unpruned:>25.2f}% | {100 * ret / n_gt_val:>12.2f}% | {keep.sum() / n_s1:>8.2f}")
    ok = [r for r in rows if r["retained_pct"] >= a.target]
    chosen = min(ok, key=lambda r: r["k"]) if ok else None
    print(f"\nchosen K (smallest with >= {a.target}% retained): {chosen['k'] if chosen else 'NONE'}")
    with open(os.path.join(os.path.dirname(a.model), "filter_eval.json"), "w") as f:
        json.dump({"rows": rows, "chosen": chosen, "features": FEATURES}, f, indent=2)


def cmd_apply(a):
    start_watchdog(a.min_avail_gb)
    files = chunk_files(a.cache_dir, a.split)
    backup = os.path.join(a.cache_dir, f"cands_JK_{a.split}")
    if not os.path.isdir(backup) or len([f for f in os.listdir(backup) if f.endswith(".parquet")]) != len(files):
        raise SystemExit(f"refusing to prune in place: backup {backup} missing or file count differs from {len(files)}")
    booster = lgb.Booster(model_file=a.model)
    pj = load_pairs(a.cache_dir, a.split, "tfidf_rev")
    pk = load_pairs(a.cache_dir, a.split, "tfidf_dict")
    tot_in = tot_out = tot_s1 = 0
    for fp in files[: a.max_chunks]:
        t0 = time.time()
        d = pd.read_parquet(fp)
        X, s1, _ = build_features(d, pj, pk)
        score = booster.predict(X.values).astype(np.float32)
        keep, _ = topk_keep(s1, score, X["emb_score"].values, a.k)
        out = d[keep].reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), fp + ".tmp")
        os.replace(fp + ".tmp", fp)
        n_s1 = len(np.unique(s1))
        tot_in += len(d); tot_out += len(out); tot_s1 += n_s1
        print(f"  {os.path.basename(fp)}: {len(d):,} -> {len(out):,} ({len(out) / n_s1:.2f}/S1) in {time.time() - t0:.0f}s "
              f"| RSS {rss_gb():.2f} GB", flush=True)
        del d, X, out
    print(f"pruned {a.split} in place: {tot_in:,} -> {tot_out:,} pairs over {tot_s1:,} S1 ({tot_out / max(1, tot_s1):.2f}/S1)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train-eval", "apply"])
    p.add_argument("--cache-dir", default=config.CACHE_DIR)
    p.add_argument("--model", default=os.path.join(config.CACHE_DIR, "prune", "filter_model.txt"))
    p.add_argument("--ks", default="10,15,20,25,30")
    p.add_argument("--target", type=float, default=99.5)
    p.add_argument("--train-frac", type=float, default=0.25, help="fraction of train-fold S1 used to fit the filter")
    p.add_argument("--split", default="train")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--min-avail-gb", type=float, default=6.5)
    a = p.parse_args()
    if a.cmd == "apply" and not a.k:
        raise SystemExit("--k required for apply")
    {"train-eval": cmd_train_eval, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
