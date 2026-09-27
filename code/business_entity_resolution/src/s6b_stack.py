#!/usr/bin/env python3
"""
Stage 6b: coherence stacking (experiment E6, logs/e6_coherence_stacking.py) on top of the stage-5 model.

Features (22, exactly as in E6), for candidate pairs with prob >= 0.05:
  prob, rank, prob_T, gap_to_T, num_gt_05, is_top1, name_sim_T, addr_sim_T, house_match_T, num_jaccard_T,
  same_src_T, n/a/h/j _other_mean/_max (vs the S1's other candidates with prob > 0.5),
  best_competing_prob, prob_minus_comp, other_gt_03 (competition over every S1 in the frame).

  val   : 2-fold cross-fit (split by val S1) -> out-of-fold stacked probs for val rows; competitor rows are also
          stacked (mean of both fold models). t_top1 / t_extra / margin / exclusivity re-tuned with s6_tune's
          grid on val + competitor rows. A final stacker is fit on all val S1.
  test  : final stacker on all test pairs (competition features over all test S1), decide() with the tuned
          parameters -> <out-dir>/test_predictions.parquet (+ thresholds.json) for s8_write.
"""
import argparse
import json
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from rapidfuzz import fuzz

import config
from decide import decide
from metrics import build_gold_map, macro_f05
from s4_features import _id_to_int
from s6_tune import run_grid_search
from s7_predict import _int_to_id

FEATS = ["prob", "rank", "prob_T", "gap_to_T", "num_gt_05",
         "is_top1", "name_sim_T", "addr_sim_T", "house_match_T", "num_jaccard_T", "same_src_T",
         "n_other_mean", "n_other_max", "a_other_mean", "a_other_max", "h_other_mean", "h_other_max",
         "j_other_mean", "j_other_max", "best_competing_prob", "prob_minus_comp", "other_gt_03"]
PARAMS = {"objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt", "n_estimators": 100,
          "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31, "verbosity": -1,
          "n_jobs": min(8, os.cpu_count() or 8), "random_state": 42}
MIN_PROB = 0.05


def rss_gb():
    return psutil.Process().memory_info().rss / 2**30


def load_meta(cache_dir, split, cand_ids):
    """name_full / addr_norm / house_no / num_tokens for the given S2/S3 candidate int IDs."""
    cols = ["entity_id", "name_full", "addr_norm", "house_no", "num_tokens"]
    wanted = set(int(c) for c in cand_ids)
    name, addr, house, nums = {}, {}, {}, {}
    for src in ("source2", "source3"):
        pf = pq.ParquetFile(os.path.join(cache_dir, f"norm_{split}_{src}.parquet"))
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=cols).to_pandas()
            ids = _id_to_int(t["entity_id"], validate=False).values
            m = np.fromiter((int(x) in wanted for x in ids), dtype=bool, count=len(ids))
            for cid, nf, ad, hn, nt in zip(ids[m], t["name_full"].values[m], t["addr_norm"].values[m],
                                           t["house_no"].values[m], t["num_tokens"].values[m]):
                cid = int(cid)
                name[cid] = nf if isinstance(nf, str) else ""
                addr[cid] = ad if isinstance(ad, str) else ""
                house[cid] = hn if isinstance(hn, str) else ""
                nums[cid] = set(str(nt).split()) if isinstance(nt, str) else set()
            del t
    return name, addr, house, nums


def build_features(df: pd.DataFrame, meta):
    """
    df: s1_id, cand_id (int64), prob, emb_score [+ is_competitor]. Competition features use ALL rows of df.
    Returns (sub, X) for rows with prob >= MIN_PROB, sub sorted by S1, prob desc, emb_score desc, cand_id.
    """
    name, addr, house, nums = meta
    gt03 = df.loc[df["prob"] > 0.30, "cand_id"].value_counts().to_dict()
    g = df.loc[df["prob"] > 0.05, ["cand_id", "s1_id", "prob"]].sort_values(["cand_id", "prob"], ascending=[True, False])
    c_ids, s_ids, p_vals = g["cand_id"].values, g["s1_id"].values, g["prob"].values.astype(np.float32)
    del g
    st_idx = np.r_[0, np.flatnonzero(c_ids[1:] != c_ids[:-1]) + 1]
    en_idx = np.r_[st_idx[1:], len(c_ids)]
    top1_s1 = dict(zip(c_ids[st_idx], s_ids[st_idx]))
    top1_p = dict(zip(c_ids[st_idx], p_vals[st_idx]))
    top2_p = {c: (float(p_vals[s + 1]) if e - s > 1 else 0.0) for c, s, e in zip(c_ids[st_idx], st_idx, en_idx)}
    del c_ids, s_ids, p_vals

    sub = df[df["prob"] >= MIN_PROB].sort_values(["s1_id", "prob", "emb_score", "cand_id"],
                                                 ascending=[True, False, False, True]).reset_index(drop=True)
    n = len(sub)
    s1v, cv, pv = sub["s1_id"].values, sub["cand_id"].values, sub["prob"].values.astype(np.float32)
    F = {f: np.zeros(n, dtype=np.float32) for f in FEATS}
    F["prob"][:] = pv

    def hmatch(a, b):
        return -1.0 if (a == "" or b == "") else (1.0 if a == b else 0.0)

    def jac(a, b):
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    starts = np.r_[0, np.flatnonzero(s1v[1:] != s1v[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    for st, en in zip(starts, ends):
        cs, ps = cv[st:en], pv[st:en]
        T = int(cs[0]); Tp = ps[0]
        Tn, Ta, Th, Tj, Ts = name.get(T, ""), addr.get(T, ""), house.get(T, ""), nums.get(T, set()), T // 10**12
        hi = [int(c) for c in cs[ps > 0.5]]
        for off in range(en - st):
            i = st + off
            c = int(cs[off])
            F["rank"][i] = off
            F["prob_T"][i] = Tp
            F["gap_to_T"][i] = Tp - ps[off]
            F["num_gt_05"][i] = len(hi)
            cn, ca, ch, cj = name.get(c, ""), addr.get(c, ""), house.get(c, ""), nums.get(c, set())
            if off == 0:
                F["is_top1"][i] = 1.0
            else:
                F["name_sim_T"][i] = fuzz.token_sort_ratio(cn, Tn)
                F["addr_sim_T"][i] = fuzz.token_set_ratio(ca, Ta)
                F["house_match_T"][i] = hmatch(ch, Th)
                F["num_jaccard_T"][i] = jac(cj, Tj)
                F["same_src_T"][i] = 1.0 if c // 10**12 == Ts else 0.0
            others = [o for o in hi if o != c]
            if others:
                ns = [fuzz.token_sort_ratio(cn, name.get(o, "")) for o in others]
                as_ = [fuzz.token_set_ratio(ca, addr.get(o, "")) for o in others]
                hs = [hmatch(ch, house.get(o, "")) for o in others]
                js = [jac(cj, nums.get(o, set())) for o in others]
                F["n_other_mean"][i], F["n_other_max"][i] = np.mean(ns), max(ns)
                F["a_other_mean"][i], F["a_other_max"][i] = np.mean(as_), max(as_)
                F["h_other_mean"][i], F["h_other_max"][i] = np.mean(hs), max(hs)
                F["j_other_mean"][i], F["j_other_max"][i] = np.mean(js), max(js)
    for i in range(n):
        c, s, p = int(cv[i]), s1v[i], pv[i]
        cnt = gt03.get(c, 0)
        F["other_gt_03"][i] = cnt - 1 if p > 0.30 else cnt
        F["best_competing_prob"][i] = top2_p.get(c, 0.0) if top1_s1.get(c, -1) == s else top1_p.get(c, 0.0)
    F["prob_minus_comp"] = pv - F["best_competing_prob"]
    return sub, np.column_stack([F[f] for f in FEATS]).astype(np.float32)


def stage6_f05(log_path):
    try:
        m = re.findall(r"Tuned LightGBM:\s+Val Macro F0.5 = ([0-9.]+)", open(log_path).read())
        return float(m[-1]) if m else float("nan")
    except OSError:
        return float("nan")


def cmd_val(a):
    t0 = time.time()
    split = pd.read_parquet(os.path.join(a.cache_dir, "split.parquet"))
    val_s1 = sorted(split.loc[split.fold == "val", "s1_id"].unique())
    val_s1_split_order = split.loc[split.fold == "val", "s1_id"].tolist()
    probs = pd.read_parquet(a.val_probs)
    print(f"val_probs: {len(probs):,} rows (competitor {int((probs.is_competitor == 1).sum()):,}) | RSS {rss_gb():.2f} GB", flush=True)
    meta = load_meta(a.cache_dir, "train", probs.loc[probs.prob >= MIN_PROB, "cand_id"].unique())
    sub, X = build_features(probs, meta)
    del meta
    gt = pd.read_parquet(os.path.join(a.cache_dir, "gt_long.parquet"))
    gt_set = set(zip(_id_to_int(gt.s1_id, validate=False).values, _id_to_int(gt.match_id, validate=False).values))
    gold = build_gold_map(gt, val_s1)
    del gt
    is_val = sub["is_competitor"].values == 0
    y = np.fromiter(((int(s), int(c)) in gt_set for s, c in zip(sub.s1_id.values, sub.cand_id.values)), dtype=np.int8, count=len(sub))
    print(f"features: {len(sub):,} rows with prob >= {MIN_PROB} (val {int(is_val.sum()):,}, pos {int(y[is_val].sum()):,}) "
          f"in {time.time() - t0:.0f}s | RSS {rss_gb():.2f} GB", flush=True)

    # 2-fold cross-fit strictly by val S1 (template: first / second half of the split order)
    vint = _id_to_int(pd.Series(val_s1_split_order), validate=False).values
    fold0 = set(vint[: len(vint) // 2].tolist())
    in0 = np.fromiter((int(s) in fold0 for s in sub.s1_id.values), dtype=bool, count=len(sub)) & is_val
    in1 = is_val & ~in0
    stacked = np.zeros(len(sub), dtype=np.float32)
    m0 = lgb.LGBMClassifier(**PARAMS).fit(X[in0], y[in0])
    m1 = lgb.LGBMClassifier(**PARAMS).fit(X[in1], y[in1])
    stacked[in1] = m0.predict_proba(X[in1])[:, 1]
    stacked[in0] = m1.predict_proba(X[in0])[:, 1]
    comp = ~is_val
    stacked[comp] = 0.5 * (m0.predict_proba(X[comp])[:, 1] + m1.predict_proba(X[comp])[:, 1])
    imp = (m0.feature_importances_ + m1.feature_importances_) / 2
    print("stacker importance:", ", ".join(f"{f} {v:.0f}" for f, v in sorted(zip(FEATS, imp), key=lambda x: -x[1])[:8]), flush=True)

    val_s1_int = _id_to_int(pd.Series(val_s1), validate=True).tolist()
    gold_int = {si: (set(_id_to_int(pd.Series(list(gold[ss])), validate=True).tolist()) if gold[ss] else set())
                for ss, si in zip(val_s1, val_s1_int)}
    st_df = sub[["s1_id", "cand_id", "emb_score", "is_competitor"]].assign(prob=stacked)
    res = run_grid_search(st_df, gold_int, val_s1_int)
    ue, m, t1, te, best = res[0]
    preds = decide(st_df, t_top1=t1, t_extra=te, margin=m, use_exclusivity=ue, all_s1_ids=val_s1_int)
    chk = macro_f05(preds, gold_int)
    tp = sum(len(set(p) & gold_int[s]) for s, p in preds.items())
    n_pred = sum(len(p) for p in preds.values())
    n_gold = sum(len(g) for g in gold_int.values())
    base = stage6_f05(a.stage6_log)
    print(f"\nVAL (out-of-fold stacked): tuned macro F0.5 {best:.4f} (verified {chk:.4f}) | precision {tp / max(1, n_pred):.4f} "
          f"| recall {tp / max(1, n_gold):.4f} | excl={ue} margin={m:.2f} t_top1={t1:.2f} t_extra={te:.2f}")
    print(f"VAL base (stage 6, same pruned candidates): {base:.4f}  ->  delta {best - base:+.4f}", flush=True)

    final = lgb.LGBMClassifier(**PARAMS).fit(X[is_val], y[is_val])
    os.makedirs(a.out_dir, exist_ok=True)
    final.booster_.save_model(os.path.join(a.out_dir, "stacker.txt"))
    thr = {"t_top1": t1, "t_extra": te, "margin": m, "use_exclusivity": bool(ue),
           "val_macro_f05_stacked": best, "val_macro_f05_base": base, "source": "s6b_stack.py"}
    with open(os.path.join(a.out_dir, "thresholds.json"), "w") as f:
        json.dump(thr, f, indent=2)
    print(f"saved stacker + thresholds -> {a.out_dir} | {time.time() - t0:.0f}s | RSS {rss_gb():.2f} GB")


def cmd_test(a):
    t0 = time.time()
    thr = json.load(open(os.path.join(a.out_dir, "thresholds.json")))
    booster = lgb.Booster(model_file=os.path.join(a.out_dir, "stacker.txt"))
    probs = pd.read_parquet(a.test_probs)
    print(f"test_probs: {len(probs):,} rows over {probs.s1_id.nunique():,} S1 | RSS {rss_gb():.2f} GB", flush=True)
    meta = load_meta(a.cache_dir, "test", probs.loc[probs.prob >= MIN_PROB, "cand_id"].unique())
    sub, X = build_features(probs, meta)
    del meta, probs
    sub["prob"] = booster.predict(X).astype(np.float32)
    sub["is_competitor"] = np.int8(0)
    print(f"stacked {len(sub):,} test pairs (prob >= {MIN_PROB}) in {time.time() - t0:.0f}s | RSS {rss_gb():.2f} GB", flush=True)
    preds = decide(sub, t_top1=thr["t_top1"], t_extra=thr["t_extra"], margin=thr["margin"],
                   use_exclusivity=thr["use_exclusivity"], all_s1_ids=None)
    out = pd.DataFrame([{"s1_id": _int_to_id(s), "matched_ids": ",".join(_int_to_id(c) for c in cs)} for s, cs in preds.items()])
    out.to_parquet(os.path.join(a.out_dir, "test_predictions.parquet"), index=False)
    n_match = sum(len(c) for c in preds.values())
    print(f"test predictions: {len(out):,} S1 with a decision, {n_match:,} matches, "
          f"{sum(1 for c in preds.values() if c):,} S1 non-empty -> {a.out_dir}/test_predictions.parquet")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["val", "test"])
    p.add_argument("--cache-dir", default=config.CACHE_DIR)
    p.add_argument("--val-probs", default=os.path.join(config.CACHE_DIR, "val_probs_v1.parquet"))
    p.add_argument("--test-probs", default=os.path.join(config.CACHE_DIR, "test_probs.parquet"))
    p.add_argument("--stage6-log", default=os.path.join(os.path.dirname(config.CACHE_DIR), "logs", "stage_6.log"))
    p.add_argument("--out-dir", default=os.path.join(config.CACHE_DIR, "stack"))
    a = p.parse_args()
    {"val": cmd_val, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
