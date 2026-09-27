# Business Entity Resolution Pipeline: End-to-End Reproduction Guide

This directory contains the complete source code, configuration, and dependencies to reproduce the final entity resolution results (`output/matching_results.tsv` and `output/candidate_pairs.tsv`) from raw input data on an AWS EC2 `g5.2xlarge` instance.

---

## 1. Environment & Requirements

- **Operating System:** Ubuntu 22.04 LTS (AWS EC2) or compatible Linux
- **Hardware Specification:** AWS `g5.2xlarge` instance (8 vCPUs, 31 GB RAM, 1× NVIDIA A10G GPU with 24 GB VRAM)
- **Python Version:** Python 3.13
- **CUDA:** CUDA 12.1+ / PyTorch with CUDA acceleration

### Environment Activation & Dependency Installation

Run from the repository root:

```bash
# 1. Create and activate a clean virtual environment using Python 3.13
python3 -m venv amlc_env
source amlc_env/bin/activate

# 2. Upgrade pip and install pinned requirements
pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 2. Data Placement & Model Weight Download

### Raw Data Placement

Ensure raw competition dataset files are placed in the repository root under `dataset/`:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### Pretrained Neural Model Download

The pipeline uses `Qwen/Qwen3-Embedding-0.6B` (Apache 2.0 license, 0.6B parameters). Download and cache the model weights into the local Hugging Face cache prior to running the offline pipeline:

```bash
python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
"
```

*Note: Once cached, the entire pipeline executes 100% offline with zero network connectivity.*

---

## 3. End-to-End Execution Sequence

Execute all pipeline stages from the project root in the exact sequence outlined below. All intermediate files are cached in `cache/` and final submission files are written to `output/`.

### Stage 1: Data Normalization (`s1_normalize.py`)
Normalizes names and addresses across all splits and sources:
- Multilingual transliteration using `anyascii` (handling native Indian scripts: Hindi, Tamil, Telugu, Kannada, Bengali, etc.)
- Strips legal entity suffixes while tracking entity types
- Extracts consonant skeletons (`name_skel`) and address tokens
- Operates dynamically on open-set countries (`India`, `US`, and unobserved `France`)

```bash
python3 code/business_entity_resolution/src/s1_normalize.py
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 2: Dense Name Embeddings (`s2_embed.py`)
Encodes normalized core names into 256-dimensional L2-normalized `float16` embeddings using `Qwen3-Embedding-0.6B` on GPU with chunked batching.

```bash
python3 code/business_entity_resolution/src/s2_embed.py
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 3: Combined Name + Address Embeddings (`embed_combined_all.sh` / `s2b_embed_combined.py`)
Encodes joint text representations (`name_full + " " + addr_norm`) into joint dense embeddings (Channel H) to capture holistic entity identity.

```bash
bash code/business_entity_resolution/src/embed_combined_all.sh
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 4: Primary Multi-Channel Blocking (`s3_block.py`)
Generates primary candidate pools across 5 complementary blocking channels:
- **Channel A:** GPU cosine top-20 nearest neighbors from name embeddings
- **Channel B:** Address inverted index (block size cap: 50)
- **Channel C:** Consonant skeleton index (`name_skel`)
- **Channel D (Reverse Embedding Search):** Each S2/S3 record queries the S1 index of its country and takes its top-3 S1
- **Channel E:** Rare address token overlap (IDF-based index)

```bash
python3 code/business_entity_resolution/src/s3_block.py --split train
python3 code/business_entity_resolution/src/s3_block.py --split test
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 5: Candidate Augmentation (`s3b_augment.py`)
Augments candidate pools to resolve high-density duplicates and address-specific matches:
- **Channel F:** Wide name retrieval (top 200) + fast GPU address token re-ranking
- **Channel G:** Compound numeric address inverted keys (K1, K3, K5)
- **Channel H:** Combined name+address neural embedding cosine retrieval

```bash
python3 code/business_entity_resolution/src/s3b_augment.py --split train
python3 code/business_entity_resolution/src/s3b_augment.py --split test
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 6: Transliteration Dictionary & TF-IDF Retrieval (`build_indic_dict.py` & `s3c_tfidf.py`)
1. **Transliteration Dictionary:** Learns token-to-token mappings strictly from train-fold matches (725 mappings, count $\ge 3$, share $\ge 60\%$; e.g. `prodkts` $\to$ `products`, `lojistiks` $\to$ `logistics`).
2. **TF-IDF Channels (`s3c_tfidf.py`):**
   - **Channel I:** Character 3-gram TF-IDF (`char_wb`, ngram (3,3), top-10) on `name_full + " " + name_skel + " " + addr_norm`
   - **Channel J:** Reverse char TF-IDF (each S2/S3 record queries country S1 index, keeps top-5 S1; at most +10 per S1)
   - **Channel K:** TF-IDF on names normalized by the learned transliteration dictionary

```bash
python3 code/business_entity_resolution/src/build_indic_dict.py
python3 code/business_entity_resolution/src/s3c_tfidf.py --split train
python3 code/business_entity_resolution/src/s3c_tfidf.py --split test
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 7: Augment Integration for Channels I, J, K (`s3b_augment.py`)
Merges candidate pools from Channels I, J, and K into the master candidate chunks for both train and test splits, reaching **98.90%** unpruned pair recall on validation.

```bash
python3 code/business_entity_resolution/src/s3b_augment.py --split train --channels I,J,K
python3 code/business_entity_resolution/src/s3b_augment.py --split test --channels I,J,K
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 8: Two-Stage Blocking Filter / Prune Chunks (`prune_chunks.py`)
Applies a fast, lightweight LightGBM filter on cheap blocking-stage signals (channel presence flags, embedding and TF-IDF scores, and ranks) to keep the top $K=15$ candidates per S1 prior to feature computation:
- Enforces exactly **15.0 candidates per S1** on the test set (**25,988,160 candidate pairs**).
- Preserves **98.41%** validation pair recall.
- Outputs the filtered candidate set directly to `output/candidate_pairs.tsv`.
- Downstream feature computation and inference evaluate only this pruned candidate set.

```bash
python3 code/business_entity_resolution/src/prune_chunks.py --split train --top-k 15
python3 code/business_entity_resolution/src/prune_chunks.py --split test --top-k 15
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 9: Feature Engineering (`s4_features.py`)
Extracts the complete 51 pairwise feature pack across pruned candidate chunks:
- Core string and token similarities: `jw_core`, `lev_core`, `lcs_core`, `compact_ratio`
- Structural indicators: `initials_match`, `first_tok_match`, `longnum_match`
- Corpus frequency signals: `name_freq_s1`, `name_freq_cand`
- Transliteration dictionary agreement: `dict_name_ratio`
- Comprehensive address, house number, PIN/ZIP, state, and channel presence features.

```bash
python3 code/business_entity_resolution/src/s4_features.py --split train
python3 code/business_entity_resolution/src/s4_features.py --split test
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 10: Primary Classifier Training (`s5_train.py`)
Trains the LightGBM gradient boosted decision tree pairwise matcher:
- Training uses the 300,000 train-fold S1 candidate pairs (~1.2 crore pairs).
- The 100,000 val-fold S1 candidate pairs are used for validation and early stopping.
- Exact hyperparameters:
  ```python
  params = {
      "objective": "binary",
      "metric": "binary_logloss",
      "learning_rate": 0.05,
      "num_leaves": 63,
      "min_child_samples": 100,
      "feature_fraction": 0.8,
      "bagging_fraction": 0.8,
      "bagging_freq": 1,
      "n_jobs": min(8, os.cpu_count() or 8),
      "verbose": -1,
      "seed": 42,
  }
  ```
  Trained with `num_boost_round=2000` and early stopping at `stopping_rounds=100`.

```bash
python3 code/business_entity_resolution/src/s5_train.py
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 11: Stage 6b Coherence Stacking & Threshold Tuning (`s6b_stacking.py` / `s6_tune.py`)
Trains a second-stage LightGBM meta-model cross-fitted on validation S1 entities:
- Uses 22 features: base probability, candidate rank, gap to top-1, candidate coherence with the top-1 and other confident ($\text{prob} > 0.5$) candidates, and competitor pressure features.
- Applies exclusivity with `margin = 0.10`.
- Tunes decision thresholds to optimal values: $t_{\text{top1}} = 0.46$, $t_{\text{extra}} = 0.68$.
- Pushes validation macro F0.5 from 0.9642 to **0.9698** (precision **0.991**, recall **0.941**).

```bash
python3 code/business_entity_resolution/src/s6b_stacking.py
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 12: Test Inference & Stacking (`s7_predict.py`)
Performs batch model inference across test candidate chunks, followed by Stage 6b stacking transformation to generate final calibrated match probabilities.

```bash
python3 code/business_entity_resolution/src/s7_predict.py
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 13: Exclusivity Resolution & Output Writing (`s8_write.py`)
Applies exclusivity (margin 0.10) and tuned thresholds ($t_{\text{top1}} = 0.46, t_{\text{extra}} = 0.68$) to produce final submission files:
- Each S2/S3 record is kept only for the S1 with the highest probability (tie-break: higher `emb_score`, then smaller `s1_id`), dropping candidates where $(p_{\text{top1}} - p_{\text{top2}}) < 0.10$.
- Per S1: predicts empty list if top-1 probability $< 0.46$, else top-1 plus all extra candidates with probability $\ge 0.68$.
- Outputs:
  1. `output/matching_results.tsv`: Final predicted entity matches
  2. `output/candidate_pairs.tsv`: Pruned candidate pairs pool from Stage 8 (25,988,160 pairs)

```bash
python3 code/business_entity_resolution/src/s8_write.py
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

### Stage 14: Submission Output Validation (`utils/validate_submission.py`)
Validates generated submission files against all competition formatting and integrity checks:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
*Expected Runtime: [[TODO: runtime from EC2 logs]]*

---

## 4. Expected Total Runtime Summary (AWS g5.2xlarge)

| Stage | Script / Command | Description | Expected Runtime |
| :--- | :--- | :--- | :---: |
| **Stage 1** | `s1_normalize.py` | Multilingual text cleaning & normalization | [[TODO: runtime from EC2 logs]] |
| **Stage 2** | `s2_embed.py` | Qwen3 name embeddings (256d fp16) | [[TODO: runtime from EC2 logs]] |
| **Stage 3** | `embed_combined_all.sh` | Qwen3 combined name+address embeddings | [[TODO: runtime from EC2 logs]] |
| **Stage 4** | `s3_block.py` | Primary 5-channel blocking (Channels A–E) | [[TODO: runtime from EC2 logs]] |
| **Stage 5** | `s3b_augment.py` | Augmentation (Channels F, G, H) | [[TODO: runtime from EC2 logs]] |
| **Stage 6** | `build_indic_dict.py` & `s3c_tfidf.py` | Translit dictionary & TF-IDF (Channels I, J, K) | [[TODO: runtime from EC2 logs]] |
| **Stage 7** | `s3b_augment.py` (I, J, K) | Merge TF-IDF candidate pools | [[TODO: runtime from EC2 logs]] |
| **Stage 8** | `prune_chunks.py` | Two-stage blocking filter (top-K=15 per S1, 25.98M test pairs) | [[TODO: runtime from EC2 logs]] |
| **Stage 9** | `s4_features.py` | Pairwise feature extraction (51 features) | [[TODO: runtime from EC2 logs]] |
| **Stage 10** | `s5_train.py` | LightGBM pairwise matcher training | [[TODO: runtime from EC2 logs]] |
| **Stage 11** | `s6b_stacking.py` | Stage 6b coherence stacking & threshold optimization | [[TODO: runtime from EC2 logs]] |
| **Stage 12** | `s7_predict.py` | Test candidate scoring & stacking inference | [[TODO: runtime from EC2 logs]] |
| **Stage 13** | `s8_write.py` | Exclusivity (margin 0.10) & output generation ($t_1=0.46, t_e=0.68$) | [[TODO: runtime from EC2 logs]] |
| **Stage 14** | `validate_submission.py` | Integrity and formatting verification | [[TODO: runtime from EC2 logs]] |
| **Total** | **Full Pipeline** | **End-to-End from raw TSVs to final outputs** | **[[TODO: total runtime from EC2 logs]]** |
