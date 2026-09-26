#!/usr/bin/env python3
"""
Channel K: learn an S2/S3-token -> S1-token name dictionary from TRAIN-FOLD S1 only.

For every (train-fold S1, GT S2/S3 match) pair, the name_full tokens are aligned:
  - tokens present on both sides are identity alignments (s -> s)
  - the remaining tokens are aligned by position when both sides have the same number (1..MAX_ALIGN)
For each S2/S3 token s: count(s -> t) over aligned pairs with t != s, and occ(s) = all aligned
occurrences of s (identity included). Keep s -> t when count >= --min-count and count / occ(s) >= --min-share.
Output: cache/indic_dict.json {s: t}. name_full is never modified; s3c_tfidf --mode dict applies the map
to a copy of the S2/S3 names.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

MAX_ALIGN = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out", default=None, help="default: <cache>/indic_dict.json")
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--min-share", type=float, default=0.6)
    return p.parse_args()


def names_for(cache, source, ids):
    t = pq.read_table(os.path.join(cache, f"norm_train_{source}.parquet"), columns=["entity_id", "name_full", "country"])
    t = t.filter(pc.is_in(t["entity_id"], pa_array(ids)))
    d = t.to_pandas()
    return dict(zip(d["entity_id"], d["name_full"].fillna(""))), dict(zip(d["entity_id"], d["country"].fillna("")))


def pa_array(x):
    import pyarrow as pa
    return pa.array(list(x))


def main():
    a = parse_args()
    if a.cache_dir is None:
        import config
        a.cache_dir = config.CACHE_DIR
    out = a.out or os.path.join(a.cache_dir, "indic_dict.json")
    t0 = time.time()
    split = pd.read_parquet(os.path.join(a.cache_dir, "split.parquet"))
    train_s1 = set(split.loc[split["fold"] == "train", "s1_id"])
    gt = pd.read_parquet(os.path.join(a.cache_dir, "gt_long.parquet"), columns=["s1_id", "match_id"])
    gt = gt[gt["s1_id"].isin(train_s1)]
    print(f"train-fold S1: {len(train_s1):,} | GT pairs: {len(gt):,} (S2 {gt.match_id.str.startswith('S2-').sum():,}, "
          f"S3 {gt.match_id.str.startswith('S3-').sum():,})", flush=True)

    s1_name, s1_ctry = names_for(a.cache_dir, "source1", set(gt["s1_id"]))
    src_name = {}
    for src, pre in (("source2", "S2-"), ("source3", "S3-")):
        n, _ = names_for(a.cache_dir, src, set(gt.loc[gt.match_id.str.startswith(pre), "match_id"]))
        src_name.update(n)

    pair_cnt = defaultdict(Counter)
    occ = Counter()
    ctry_of = defaultdict(Counter)
    n_aligned = n_pairs = 0
    for s1, m in zip(gt["s1_id"], gt["match_id"]):
        a_toks = s1_name.get(s1, "").split()
        b_toks = src_name.get(m, "").split()
        if not a_toks or not b_toks:
            continue
        n_pairs += 1
        common = set(a_toks) & set(b_toks)
        for t in b_toks:
            if t in common:
                occ[t] += 1
        ra = [t for t in a_toks if t not in common]
        rb = [t for t in b_toks if t not in common]
        if ra and len(ra) == len(rb) and len(ra) <= MAX_ALIGN:
            n_aligned += 1
            for s, t in zip(rb, ra):
                pair_cnt[s][t] += 1
                occ[s] += 1
                ctry_of[s][s1_ctry.get(s1, "")] += 1

    mapping, rows = {}, []
    for s, ctr in pair_cnt.items():
        t, c = ctr.most_common(1)[0]
        share = c / occ[s]
        if c >= a.min_count and share >= a.min_share:
            mapping[s] = t
            rows.append((s, t, c, occ[s], share, ctry_of[s].most_common(1)[0][0]))
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=0, sort_keys=True)
    os.replace(out + ".tmp", out)

    df = pd.DataFrame(rows, columns=["s2s3_token", "s1_token", "count", "occ", "share", "top_country"])
    df = df.sort_values("count", ascending=False)
    non_ascii = df["s2s3_token"].map(lambda x: not x.isascii()).sum()
    print(f"pairs with names: {n_pairs:,} | pairs with an aligned token difference: {n_aligned:,}")
    print(f"candidate mappings: {len(pair_cnt):,} | kept (count >= {a.min_count}, share >= {a.min_share}): {len(mapping):,} "
          f"(non-ASCII source tokens {non_ascii:,}) -> {out}")
    print("kept by country:", df["top_country"].value_counts().to_dict())
    print("\nTop 20 by count:")
    print(df.head(20).to_string(index=False, formatters={"share": "{:.2f}".format}))
    na = df[df["s2s3_token"].map(lambda x: not x.isascii())]
    if len(na):
        print("\nTop 10 non-ASCII:")
        print(na.head(10).to_string(index=False, formatters={"share": "{:.2f}".format}))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
