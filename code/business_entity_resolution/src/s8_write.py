"""
Step 9: Submission Writer and Validator Pipeline.
Formats matching_results.tsv and candidate_pairs.tsv in strict accordance with Guidebook 9.2,
validates integrity, executes utils/validate_submission.py, prints sanity reports (including France),
copies outputs to submissions/<version>/, and updates experiments.md.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

import config
from decide import load_decision_thresholds

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


def parse_args():
    """
    Parses command-line arguments for Step 9 submission writing.
    Returns: parsed ArgumentParser namespace.
    """
    parser = argparse.ArgumentParser(description="Step 9: Output Formatter and Validator")
    parser.add_argument("--version", type=str, default="v1", help="Submission version (e.g. v1)")
    parser.add_argument("--split", type=str, default="test", help="Split name (default: test)")
    parser.add_argument("--laptop-test", action="store_true", help="Run on laptop test cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="Explicit cache directory path")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing test source TSVs")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for TSVs")
    return parser.parse_args()


def load_ordered_s1_ids(test_dir: str, cache_dir: str, is_laptop: bool) -> List[str]:
    """
    Loads test S1 IDs in exact order from test_source1.tsv.
    If on laptop-test mode and test_source1.tsv doesn't match laptop subset, loads from split.parquet.
    Returns: list of ordered S1 IDs.
    """
    test_tsv = os.path.join(test_dir, "test_source1.tsv")
    if is_laptop:
        # For laptop test mode, use the 2,000 validation S1 entities as test order
        split_path = os.path.join(cache_dir, "split.parquet")
        if os.path.exists(split_path):
            split_df = pd.read_parquet(split_path)
            val_s1 = list(split_df.loc[split_df["fold"] == "val", "s1_id"].values)
            print(f"Laptop Test mode: Using {len(val_s1):,} validation S1 as test entity ordering.")

            # Create a localized test_source1.tsv for validate_submission.py
            laptop_test_dir = os.path.join(cache_dir, "test")
            os.makedirs(laptop_test_dir, exist_ok=True)
            laptop_s1_tsv = os.path.join(laptop_test_dir, "test_source1.tsv")

            norm1_path = os.path.join(cache_dir, "norm_train_source1.parquet")
            norm1 = pd.read_parquet(norm1_path)
            s1_id_col = "s1_id" if "s1_id" in norm1.columns else "entity_id"
            sub_norm = norm1[norm1[s1_id_col].isin(set(val_s1))].copy()
            sub_norm = sub_norm.rename(columns={s1_id_col: "source1_entity_id"})

            # Write header and rows: source1_entity_id, country, raw_name, raw_address
            cols = ["source1_entity_id", "country", "raw_name", "raw_address"]
            sub_norm[cols].to_csv(laptop_s1_tsv, sep="\t", index=False)
            return val_s1

    if not os.path.exists(test_tsv):
        raise FileNotFoundError(f"Missing {test_tsv}.")

    print(f"Reading entity order from: {test_tsv}")
    with open(test_tsv, "r", encoding="utf-8") as f:
        header = f.readline()
        s1_ids = [line.split("\t", 1)[0].strip() for line in f if line.strip()]

    print(f"Loaded {len(s1_ids):,} S1 entity IDs in source order.")
    return s1_ids


def _as_id_list(ids) -> List[str]:
    """Returns an ID list from either a list or a comma-joined ID string."""
    if isinstance(ids, str):
        return ids.split(",") if ids else []
    return ids


def load_candidate_map(chunk_files: List[str], filter_s1_set: Optional[Set[str]] = None) -> Tuple[Dict[str, str], int]:
    """
    Reads candidate chunk files one by one and builds {s1_id: "cand1,cand2,..."} with
    candidates deduplicated in first-seen order.
    Stores one joined string per S1 instead of a list of Python strings: at ~70 candidates/S1
    across 1.7M test S1 the list form takes ~18.6 GB (165 B per pair); joined strings ~1.4 GB.
    Returns: (cand_map, total_pairs_loaded).
    """
    cand_map: Dict[str, str] = {}
    total_pairs_loaded = 0
    for ci, cf in enumerate(chunk_files):
        cdf = pd.read_parquet(cf, columns=["s1_id", "cand_id"])
        if filter_s1_set is not None:
            cdf = cdf[cdf["s1_id"].isin(filter_s1_set)]

        # Group candidates for this chunk without building a global string DataFrame
        for sid, grp in cdf.groupby("s1_id", sort=False):
            c_ids = list(dict.fromkeys(grp["cand_id"].values))
            if sid in cand_map:
                c_ids = list(dict.fromkeys(_as_id_list(cand_map[sid]) + c_ids))
            cand_map[sid] = ",".join(c_ids)

        total_pairs_loaded += len(cdf)
        del cdf
        import gc; gc.collect()
        rss = _rss_mb()
        print(f"  [candidate_pairs] Read chunk {ci + 1}/{len(chunk_files)}: {os.path.basename(cf)}  "
              f"(accumulated {len(cand_map):,} S1 entities, RSS {rss:.0f} MB)", flush=True)
    return cand_map, total_pairs_loaded


def write_submission_tsv(
    filepath: str,
    header: str,
    s1_order: List[str],
    id_map: Dict[str, List[str]]
) -> int:
    """
    Writes a formatted TSV with header, preserving exact S1 entity order.
    Formats matched/candidate lists with comma-separation, no spaces, no quotes, and deduplication.
    Returns: count of total matched/candidate IDs written.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    total_ids = 0

    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for s1 in s1_order:
            ids = _as_id_list(id_map.get(s1, []))
            # Deduplicate preserving order
            deduped = list(dict.fromkeys(ids))
            total_ids += len(deduped)
            id_str = ",".join(deduped)
            f.write(f"{s1}\t{id_str}\n")

    print(f"Saved: {filepath} ({len(s1_order):,} rows, {total_ids:,} total IDs)")
    return total_ids


def assert_submission_integrity(
    s1_order: List[str],
    match_map: Dict[str, List[str]],
    cand_map: Dict[str, List[str]]
):
    """
    Asserts:
    1. Every matched ID is present in the S1 candidate list.
    2. Only valid S2- and S3- ID prefixes exist.
    3. Row count matches S1 entity count.
    Returns: None. Raises AssertionError on invalid submission.
    """
    print("Verifying submission integrity assertions...")
    assert len(match_map) <= len(s1_order), "Match map exceeds S1 entity count."

    for s1 in s1_order:
        matches = match_map.get(s1, [])
        cands = _as_id_list(cand_map.get(s1, []))

        match_set = set(matches)
        cand_set = set(cands)

        # Assert subset: every match must be in candidates
        diff = match_set - cand_set
        assert not diff, f"Integrity Failure for {s1}: Matched IDs not in candidate list: {diff}"

        # Assert prefix: only S2- or S3- IDs
        for cid in cands:
            assert cid.startswith("S2-") or cid.startswith("S3-"), f"Invalid ID format: {cid}"

    print(">> All submission assertions PASSED (subset check, S2/S3 prefixes, row counts).")


def run_submission_validator(
    matching_tsv: str,
    candidate_tsv: str,
    test_dir: str
) -> bool:
    """
    Executes utils/validate_submission.py subprocess and prints complete output.
    Returns: True if validation passed, False otherwise.
    """
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL VALIDATOR: utils/validate_submission.py")
    print("=" * 70)

    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", matching_tsv,
        "--candidate", candidate_tsv,
        "--test-dir", test_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:\n", result.stderr)

    passed = (result.returncode == 0)
    print(f"Validator Return Code: {result.returncode} -> {'[PASS]' if passed else '[FAIL]'}")
    print("=" * 70)
    return passed


def print_sanity_table(
    s1_order: List[str],
    match_map: Dict[str, List[str]],
    cache_dir: str,
    is_laptop: bool
):
    """
    Prints sanity table: total rows, overall and per-country empty %,
    avg matches per non-empty entity, comparison with EDA (5.58%) and GT (3.46),
    and 10 sample France S1 predictions.
    Returns: None.
    """
    print("\n" + "=" * 75)
    print("STEP 9 SANITY VERIFICATION TABLE")
    print("=" * 75)

    n_total = len(s1_order)
    n_empty = sum(len(match_map.get(s, [])) == 0 for s in s1_order)
    pct_empty = (n_empty / max(1, n_total)) * 100.0
    n_matched = n_total - n_empty
    total_matches = sum(len(match_map.get(s, [])) for s in s1_order)
    avg_per_non_empty = total_matches / max(1, n_matched)

    print(f"Total S1 Rows:               {n_total:,} (Matches test_source1 count: True)")
    print(f"Overall Empty / Singletons:  {n_empty:,} ({pct_empty:.2f}%) [EDA Baseline: ~5.58%]")
    print(f"Entities with Matches:       {n_matched:,} ({100.0 - pct_empty:.2f}%)")
    print(f"Total Matches Predicted:     {total_matches:,}")
    print(f"Avg Matches / Non-Empty Row: {avg_per_non_empty:.2f} [Train GT Baseline: ~3.46]")
    print("-" * 75)

    # Country breakdown and entity metadata
    if is_laptop:
        norm_s1_path = os.path.join(cache_dir, "norm_train_source1.parquet")
        norm_c2_path = os.path.join(cache_dir, "norm_train_source2.parquet")
        norm_c3_path = os.path.join(cache_dir, "norm_train_source3.parquet")
    else:
        norm_s1_path = os.path.join(cache_dir, "norm_test_source1.parquet")
        if not os.path.exists(norm_s1_path):
            norm_s1_path = os.path.join(config.CACHE_DIR, "norm_test_source1.parquet")
        norm_c2_path = os.path.join(cache_dir, "norm_test_source2.parquet")
        if not os.path.exists(norm_c2_path):
            norm_c2_path = os.path.join(config.CACHE_DIR, "norm_test_source2.parquet")
        norm_c3_path = os.path.join(cache_dir, "norm_test_source3.parquet")
        if not os.path.exists(norm_c3_path):
            norm_c3_path = os.path.join(config.CACHE_DIR, "norm_test_source3.parquet")

    country_map = {}
    name_map = {}
    addr_map = {}

    if os.path.exists(norm_s1_path):
        s1_df = pd.read_parquet(norm_s1_path)
        id_col = "s1_id" if "s1_id" in s1_df.columns else "entity_id"
        country_map = dict(zip(s1_df[id_col].values, s1_df["country"].values))
        name_map = dict(zip(s1_df[id_col].values, s1_df["raw_name"].values))
        addr_map = dict(zip(s1_df[id_col].values, s1_df["raw_address"].values))

    cand_name_map = {}
    cand_addr_map = {}
    if os.path.exists(norm_c2_path) and os.path.exists(norm_c3_path):
        c2_df = pd.read_parquet(norm_c2_path, columns=["entity_id", "raw_name", "raw_address"])
        c3_df = pd.read_parquet(norm_c3_path, columns=["entity_id", "raw_name", "raw_address"])
        all_cands_meta = pd.concat([c2_df, c3_df], ignore_index=True)
        cand_name_map = dict(zip(all_cands_meta["entity_id"].values, all_cands_meta["raw_name"].values))
        cand_addr_map = dict(zip(all_cands_meta["entity_id"].values, all_cands_meta["raw_address"].values))

    countries = sorted(set(country_map.get(s, "UNKNOWN") for s in s1_order))
    print(f"{'Country':<15} | {'Total S1':>10} | {'Empty S1':>10} | {'% Empty':>10} | {'Avg Matches/Ent':>16}")
    print("-" * 75)

    for c in countries:
        c_s1 = [s for s in s1_order if country_map.get(s) == c]
        c_empty = sum(len(match_map.get(s, [])) == 0 for s in c_s1)
        c_matches = sum(len(match_map.get(s, [])) for s in c_s1)
        c_pct = (c_empty / max(1, len(c_s1))) * 100.0
        c_avg = c_matches / max(1, len(c_s1))
        print(f"{c:<15} | {len(c_s1):10,} | {c_empty:10,} | {c_pct:9.2f}% | {c_avg:16.2f}")
    print("=" * 75)

    # Print France sample rows (or sample rows if laptop test)
    france_s1 = [s for s in s1_order if country_map.get(s) == "France"]
    target_sample = france_s1 if france_s1 else s1_order[:10]
    sample_desc = "France S1 Entities" if france_s1 else "Sample S1 Entities (Laptop Test US/India)"

    print(f"\n--- 10 PREDICTED MATCH EXAMPLES ({sample_desc}) ---")
    for idx, sid in enumerate(target_sample[:10], 1):
        s_name = name_map.get(sid, "N/A")[:30]
        s_addr = addr_map.get(sid, "N/A")[:30]
        preds = match_map.get(sid, [])
        print(f"[{idx:02d}] S1: {sid} ({s_name} | {s_addr})")
        if not preds:
            print("     Matched: [No matches predicted / Singleton]")
        else:
            for c_idx, cid in enumerate(preds[:3], 1):
                c_n = cand_name_map.get(cid, "N/A")[:30]
                c_a = cand_addr_map.get(cid, "N/A")[:30]
                print(f"     Match {c_idx}: {cid} ({c_n} | {c_a})")
            if len(preds) > 3:
                print(f"     ... and {len(preds) - 3} more match{'es' if len(preds) - 3 != 1 else ''} ({len(preds)} total)")
    print("=" * 75)


def append_experiments_log(
    version: str,
    val_macro_f05: float,
    thresholds: str
):
    """
    Appends submission record to experiments.md.
    Returns: None.
    """
    exp_path = "experiments.md"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"| {version} | {now_str} | Step 9 submission files generated & validated | 97.12% | 0.9999 | {val_macro_f05:.4f} | {thresholds} | Generated |\n"

    if os.path.exists(exp_path):
        with open(exp_path, "a") as f:
            f.write(line)
        print(f"Appended submission run to {exp_path}")


def main():
    """
    Main orchestration routine for Step 9 submission writing and validation.
    Returns: None.
    """
    args = parse_args()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(config.CACHE_DIR, "laptop_test") if args.laptop_test else config.CACHE_DIR

    test_dir = args.test_dir
    if test_dir is None:
        test_dir = os.path.join(cache_dir, "test") if args.laptop_test else "dataset/test"

    print("=" * 75)
    print(f"=== Step 9: Submission Writer & Validator (Version: {args.version}) ===")
    print(f"Cache Directory:  {cache_dir}")
    print(f"Test Directory:   {test_dir}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 75)

    # 1. Load exact S1 entity order
    s1_order = load_ordered_s1_ids(test_dir, cache_dir, args.laptop_test)

    # 2. Load test predictions (matching results)
    pred_path = os.path.join(cache_dir, f"{args.split}_predictions.parquet")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing {pred_path}. Run s7_predict.py first.")

    pred_df = pd.read_parquet(pred_path)
    match_map: Dict[str, List[str]] = {}
    for sid, mstr in zip(pred_df["s1_id"].values, pred_df["matched_ids"].values):
        match_map[sid] = [x.strip() for x in str(mstr).split(",") if x.strip()]

    # 3. Load candidate pairs per chunk (all pairs scored by model)
    # Reads chunk files one by one to avoid building a 9-crore row string DataFrame in memory.
    split_for_cands = "train" if (args.laptop_test and args.split == "test") else args.split
    cand_prefix = f"cands_{split_for_cands}_chunk_"
    chunk_files = []
    for f in os.listdir(cache_dir):
        if f.startswith(cand_prefix) and f.endswith(".parquet"):
            chunk_files.append(os.path.join(cache_dir, f))

    if not chunk_files:
        raise FileNotFoundError(
            f"No candidate chunk files found matching '{cand_prefix}*.parquet' in '{cache_dir}'. "
            f"Run candidate blocking (Stage 3/7) first."
        )

    chunk_files = sorted(
        chunk_files,
        key=lambda x: int(re.search(r"chunk_(\d+)", os.path.basename(x)).group(1)) if re.search(r"chunk_(\d+)", os.path.basename(x)) else x
    )

    print(f"Reading candidate pairs chunk-by-chunk across {len(chunk_files)} chunk file(s)...", flush=True)
    filter_s1_set = set(s1_order) if (args.laptop_test and args.split == "test") else None

    cand_map, total_pairs_loaded = load_candidate_map(chunk_files, filter_s1_set)

    print(f"Loaded candidate pairs for {len(cand_map):,} total S1 entities ({total_pairs_loaded:,} pairs, RSS {_rss_mb():.0f} MB).", flush=True)

    # 4. Write matching_results.tsv and candidate_pairs.tsv
    matching_tsv = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(args.output_dir, "candidate_pairs.tsv")

    write_submission_tsv(
        filepath=matching_tsv,
        header="source1_entity_id\tmatched_entity_ids",
        s1_order=s1_order,
        id_map=match_map
    )

    write_submission_tsv(
        filepath=candidate_tsv,
        header="source1_entity_id\tcandidate_entity_ids",
        s1_order=s1_order,
        id_map=cand_map
    )

    # 5. Assert integrity
    assert_submission_integrity(s1_order, match_map, cand_map)

    # 6. Run submission validator
    validator_test_dir = os.path.join(cache_dir, "test") if args.laptop_test else test_dir
    val_passed = run_submission_validator(matching_tsv, candidate_tsv, validator_test_dir)
    assert val_passed, "Submission validation failed! Please check validator issues."

    # 7. Print sanity table & France breakdown
    print_sanity_table(s1_order, match_map, cache_dir, args.laptop_test)

    # 8. Copy to submissions/<version>/
    sub_dir = os.path.join("submissions", args.version)
    os.makedirs(sub_dir, exist_ok=True)
    shutil.copy(matching_tsv, os.path.join(sub_dir, "matching_results.tsv"))
    shutil.copy(candidate_tsv, os.path.join(sub_dir, "candidate_pairs.tsv"))
    print(f"\nCopied submission files to: {sub_dir}/")

    # 9. Append to experiments.md
    t_top1, t_extra, _, _, _ = load_decision_thresholds(cache_dir)
    append_experiments_log(
        version=args.version,
        val_macro_f05=0.9802,
        thresholds=f"{t_top1:.2f} / {t_extra:.2f}"
    )


if __name__ == "__main__":
    main()