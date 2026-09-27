# Amazon ML Challenge 2026 - entity resolution (team Codebook)
Deadline: 27 Sept 2026 11:59 PM IST (18:29 UTC). Machine: g5.2xlarge, 31 GB RAM, A10G 22 GB, Python 3.13.
Metric: macro F0.5 per S1. Submission 1 public LB = 0.875 (val 0.901). After F+G augment: val 0.9257, pair recall 92.2%.

## Pipeline
Stages: bash code/business_entity_resolution/src/ec2_full_pipeline.sh --stage N --force (markers exist, so --force is REQUIRED).
Env: source /opt/pytorch/bin/activate; export AMLC_SAMPLE=none PYTHONUNBUFFERED=1 PYTHONPATH=code/business_entity_resolution/src
Running now in tmux "chain": s3b_augment.py train then test (F+G+H, ~70 cands/S1), then stages 4, 5, 6.
Stage 8 at 60+ cands/S1 OOMs (26.5 GB) unless s4_features.py processes each chunk in sub-batches.

## Hard rules
- NEVER delete or modify cache/cands_backup_train/, cache/cands_backup_test/, cache/emb*, cache/embc_*, cache/norm_*, submissions/.
- NEVER kill a running python process or touch tmux sessions chain/emb/misc without asking me first.
- Only ONE heavy job (>10 GB RAM) at a time. Check free -g before starting anything.
- Long jobs in tmux with: set -o pipefail; ... 2>&1 | tee logs/<name>.log
- git pull before editing, commit + push after each fix. Never commit keys or data.
- Test every change on EC2 with a small --max-chunks/--limit run first. Show real output; never claim PASS without running it.
- Use per-split address DF (cache/addr_df_{split}.parquet). Never hardcode countries. Country is never a feature.
- name_full/raw_name are frozen (embeddings depend on them).
