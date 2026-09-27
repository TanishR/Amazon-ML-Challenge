# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Codebook  
**Team Members:** Naveen Kumar Modi, Tanish Ranjan, Anurag Gupta  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present a high-performance, strictly offline Entity Resolution pipeline combining multi-channel hybrid blocking (dense neural embeddings, inverted address indices, consonant skeletons, character n-gram TF-IDF, and learned transliteration normalization), a two-stage candidate pruning filter, a 51-feature LightGBM gradient-boosted pairwise classifier, and a second-stage coherence stacking meta-model with an F0.5-optimized decision layer. The candidate generation achieves **98.90%** unpruned pair recall on validation (**98.41%** after top-$K=15$ candidate pruning), producing exactly **15.0 candidates per S1** on test (**25,988,160 candidate pairs**). 

The pipeline drives the validation macro F0.5 score from **0.9010** (baseline, public LB 0.875) $\to$ **0.9257** (+F, G) $\to$ **0.9387** (+H) $\to$ **0.9433** (+I, public LB 0.9264) $\to$ **0.9642** ($K=15$ filter + 51-feature pack + J, K) $\to$ **0.9698** (+ coherence stacking), with a final precision of **0.991** and recall of **0.941**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across Source 1 (reference), Source 2, and Source 3 revealed several distinct challenges:
- **Transliteration & Multi-Script Variance:** Heavy native script representation in India (Hindi, Tamil, Telugu, Kannada, Bengali, Malayalam) with diverse phonetic Romanization and schwa-dropping (e.g. *Sharma* vs *Sharm*, *Gupta* vs *Gupt*, *prodkts* vs *products*).
- **High-Density Name Duplicates:** In Source 3, hundreds of entities share near-identical names (median 20th-best name embedding similarity is 0.95–1.00), making name-only matching ambiguous without address disambiguation.
- **Address Sparsity & Landmark Variations:** Over 98% of Indian records lack standardized PIN/ZIP codes, relying instead on local landmarks, ward numbers, or unstructured municipal phrasing.
- **Zero-Shot Test Country Generalization:** The test dataset introduces France, which is completely absent from the training set (`US` and `India`). The pipeline must not hardcode country tokens and must generalize zero-shot.

### 2.2 Solution Strategy

**Approach Type:** Multi-Channel Hybrid Blocking (A–K) + Two-Stage Candidate Pruning ($K=15$) + 51-Feature Pairwise GBDT + Coherence Stacking (Stage 6b) + Exclusivity Decision Layer  
**Core Innovation:** 
1. **11 Complementary Blocking Channels (A–K):** Unifying dense neural representation (Qwen3-Embedding-0.6B), sparse character 3-gram TF-IDF, reverse nearest-neighbor indexing, multi-key inverted indexing (numeric tokens, rare address words, consonant skeletons), and a learned transliteration normalization dictionary.
2. **Two-Stage Candidate Pruning Filter:** A lightweight LightGBM classifier operating on cheap blocking signals filters candidates down to the top $K=15$ per S1 before heavy feature extraction, preserving **98.41%** recall while enforcing a strict budget of 15.0 candidates per S1 (25,988,160 total test pairs).
3. **51 Pairwise Feature Pack:** Rich lexical similarities (`jw_core`, `lev_core`, `lcs_core`, `compact_ratio`), token overlap, structural identity (`initials_match`, `first_tok_match`, `longnum_match`), corpus frequency signals (`name_freq_s1`, `name_freq_cand`), transliteration dictionary match (`dict_name_ratio`), address and PIN consistency, and channel flags.
4. **Stage 6b Coherence Stacking & Decision Layer:** A small cross-fitted LightGBM evaluates intra-cluster coherence between extra candidates and top candidates $T$ alongside competing S1 pressures, followed by exclusivity resolution (margin 0.10) and tuned thresholds ($t_{\text{top1}} = 0.46, t_{\text{extra}} = 0.68$).

---

## 3. Candidate Generation (Blocking)

To achieve maximum recall while keeping downstream feature computation compact, we implement an 11-channel blocking architecture followed by a two-stage filter:

- **Primary Blocking Channels (`s3_block.py`):**
  - **Channel A (Dense Name Embeddings):** Cosine top-20 nearest neighbors using Qwen3-Embedding-0.6B (256d fp16, L2-normalized).
  - **Channel B (Address Inverted Key):** Match on normalized address keys with block cap of 50.
  - **Channel C (Consonant Skeleton Key):** Match on `name_skel` (consonant skeleton of core business name tokens).
  - **Channel D (Reverse Embedding Search):** Each S2/S3 record queries the S1 index of its country using main + alt embeddings and takes its top-3 S1.
  - **Channel E (Rare Address Token Overlap):** Token frequency inverted index matching on low-document-frequency address tokens.

- **Candidate Augmentation Channels (`s3b_augment.py` & `s3c_tfidf.py`):**
  - **Channel F (Wide Name Retrieval + GPU Address Re-Rank):** Retrieves top-200 candidates by name embedding per query, then evaluates tokenized address overlap on GPU to rescue address-disambiguated duplicates.
  - **Channel G (Compound Address Keys):**
    - **K1:** Country + two consecutive numeric tokens (length $\ge 2$).
    - **K3:** Country + sorted set of all numeric address tokens.
    - **K5:** Country + numeric token (length $\ge 3$) + rarest alpha address token (length $\ge 5$).
  - **Channel H (Combined Name + Address Neural Embeddings):** Dense cosine retrieval using joint representations (`name_full + " " + addr_norm`).
  - **Channel I (Character 3-Gram TF-IDF):** GPU sparse (CSR) × dense matrix product with top-10, text = `name_full + " " + name_skel + " " + addr_norm`.
  - **Channel J (Reverse Character TF-IDF):** Each S2/S3 record queries the S1 index of its country and keeps its top-5 S1; at most +10 per S1.
  - **Channel K (Transliteration Dictionary TF-IDF):** Character TF-IDF on names normalized by a transliteration dictionary learned strictly from train-fold matches (725 mappings, count $\ge 3$, share $\ge 60\%$; e.g. `prodkts` $\to$ `products`, `lojistiks` $\to$ `logistics`, `teknoloji` $\to$ `technology`).

- **Two-Stage Blocking Filter (`prune_chunks.py`):**
  - A fast LightGBM filter trained on cheap blocking-stage signals (channel presence flags, embedding and TF-IDF scores, and candidate ranks) scores candidates and retains the top $K=15$ candidates per S1 prior to feature computation.
  - Test set candidate generation: exactly **15.0 candidates per S1**, yielding **25,988,160 candidate pairs** in `output/candidate_pairs.tsv`.
  - Pair recall remains exceptionally high at **98.41%** on validation after pruning.

- **Validation Set Blocking Recall Progression:**

| Blocking Stage | Channels Included | Val Pair Recall |
| :--- | :--- | :---: |
| **Primary Baseline** | Channels A–E | 87.16% |
| **Augmentation 1** | Channels A–G (+F, G) | 92.20% |
| **Augmentation 2** | Channels A–H (+H) | 95.41% |
| **Augmentation 3** | Channels A–I (+I) | 97.53% |
| **Full Candidate Pool** | Channels A–K (+J, K, unpruned) | **98.90%** |
| **Pruned Final Pool** | Channels A–K (after top-$K=15$ filter) | **98.41%** |

---

## 4. Matching Model

### 4.1 Feature Engineering (51 Pairwise Features)
The matching model evaluates candidate pairs using 51 pairwise features:

1. **Embedding & Rank Features:**
   - `emb_score`: Cosine similarity of Qwen3 name embeddings
   - `emb_rank`: Candidate rank in Channel A (1..20, or 999 if not from Channel A)
2. **Core & Lexical Name Similarity Features (Expanded Feature Pack):**
   - `name_token_sort`: RapidFuzz `token_sort_ratio` on `name_full`
   - `name_token_set`: RapidFuzz `token_set_ratio` on `name_full`
   - `core_ratio`: Levenshtein ratio on core name tokens
   - `core_partial`: Partial ratio on core name tokens
   - `skel_ratio`: Similarity on consonant skeleton of core name tokens
   - `name_jaccard`: Character 3-gram Jaccard similarity on core names
   - `legal_match`: Compatibility of legal form suffix (1 if match, 0 if different, 0.5 if neutral/missing)
   - `dba_max`: Maximum similarity across alternative trade names (`name_a`, `name_b`)
   - `aka_max`: Maximum similarity across aka name parts (`name_aka_a`, `name_aka_b`)
   - `len_diff`: Absolute difference in character length of core names
   - `jw_core`: Jaro-Winkler similarity on core business name tokens
   - `lev_core`: Normalized Levenshtein distance on core business name tokens
   - `lcs_core`: Longest common substring ratio on core names
   - `compact_ratio`: String similarity after removing all whitespaces and punctuation
   - `initials_match`: Binary match between extracted name initials / acronyms
   - `first_tok_match`: Exact match indicator for the first significant name token
   - `dict_name_ratio`: Token similarity after applying learned transliteration dictionary
   - `name_freq_s1`: Document frequency of S1 name in the reference corpus
   - `name_freq_cand`: Document frequency of candidate name in candidate corpus
3. **Address & Numeric Consistency Features:**
   - `addr_token_set`: RapidFuzz `token_set_ratio` on `addr_norm`
   - `house_match`: House number match ternary indicator (1 = match, 0 = mismatch, -1 = missing)
   - `house_cand_match`: Match against extracted house candidate tokens
   - `num_jaccard`: Jaccard similarity of extracted numeric token sets
   - `longnum_match`: Exact match on long numeric identifiers (phone, tax, or registration sequences)
   - `rare_tok_overlap`: IDF-weighted overlap of rare address tokens
   - `zip_match`: PIN / ZIP code match ternary indicator (1 = match, 0 = mismatch, -1 = missing)
   - `state_match`: State code match indicator (1 = match, 0 = mismatch, -1 = missing)
   - `addr_missing_any`: 1 if either entity's address is missing/empty, else 0
4. **Channel & Source Flags:**
   - `cand_source`: Indicator for candidate source (0 for S2, 1 for S3)
   - `ch_emb`: Retrieved by Channel A (name embedding top-20)
   - `ch_addr`: Retrieved by Channel B (address key)
   - `ch_skel`: Retrieved by Channel C (consonant skeleton key)
   - `ch_rare`: Retrieved by Channel E (rare address tokens)
   - `ch_rev`: Retrieved by Channel D (reverse embedding search)
   - `ch_rerank`: Retrieved by Channel F (wide name retrieval + address re-rank)
   - `ch_keyx`: Retrieved by Channel G (compound address keys K1, K3, K5)
   - `ch_comb`: Retrieved by Channel H (combined name+address embedding)
   - `ch_tfidf`: Retrieved by Channel I (character TF-IDF)
   - `ch_rev_tfidf`: Retrieved by Channel J (reverse character TF-IDF)
   - `ch_dict_tfidf`: Retrieved by Channel K (transliteration dictionary TF-IDF)
   - `n_channels`: Total count of blocking channels that generated this candidate pair
5. **Context & Competition Features:**
   - `gap_to_best`: Score difference between this candidate and the top candidate for S1
   - `n_cands`: Total candidate count for this S1 entity
   - `reverse_rank`: Rank of this S1 in candidate's reverse nearest neighbor list
   - `support`: Number of shared candidate tokens between S1 and candidate

### 4.2 Primary Matcher Training Setup
- **Model Type:** LightGBM Binary Classifier (`lgb.train`)
- **Hyperparameters (from `s5_train.py`):**
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
- **Data Splits:**
  - Training uses the 300,000 train-fold S1 candidate pairs (~1.2 crore pairs).
  - The 100,000 val-fold S1 candidate pairs are used for validation and early stopping.

### 4.3 Stage 6b Coherence Stacking Meta-Model
Diagnostic analysis revealed that false positive extras and threshold-clipped extras represent the largest error categories. To address this, Stage 6b cross-fits a second-level LightGBM meta-model on validation S1 entities:
- **Architecture:** 2-fold cross-fitting, split strictly by S1 entity (`n_estimators=100`, `learning_rate=0.05`, `max_depth=5`, `num_leaves=31`).
- **Stacking Features (22):**
  - Base pair probability `prob`, intra-S1 candidate `rank`, probability of top candidate `prob_T`, gap to top candidate `gap_to_T`, and count of candidates with prob > 0.5 (`num_gt_05`).
  - Coherence with top candidate $T$: `is_top1`, `name_sim_T` (RapidFuzz `token_sort_ratio`), `addr_sim_T` (`token_set_ratio`), `house_match_T`, `num_jaccard_T`, `same_src_T`.
  - Coherence with other confident candidates ($\text{prob} > 0.5$): mean and max for `name_sim`, `addr_sim`, `house_match`, `num_jaccard`.
  - Competition features: `best_competing_prob` (highest probability among competing S1s for this candidate), `prob_minus_comp`, and `other_gt_03` (count of competing S1s with prob > 0.3).

### 4.4 Decision Layer & Exclusivity Resolution
Entity resolution evaluation uses **macro F0.5**, heavily penalizing false positive merges (weighting precision 2× over recall) while awarding a full 1.0 for correctly identified singletons:
- **Exclusivity Constraint (`decide.py`):**
  - Each S2/S3 record is kept only for the S1 with the highest stacked probability.
  - **Tie-Break Order:** Higher `emb_score` (if available), then smaller `s1_id` (alphabetically / numerically ascending).
  - **Margin Check:** Uses `margin = 0.10`. If `(prob_top1 - prob_top2) < 0.10`, drops the candidate from all S1s to eliminate ambiguous merges.
- **Per-S1 Thresholding:**
  - If top-1 probability $< t_{\text{top1}} = 0.46$, predicts an empty list (singleton).
  - Else, predicts top-1 candidate plus all extra candidates with probability $\ge t_{\text{extra}} = 0.68$.

---

## 5. Results & Error Analysis

### 5.1 Validation Macro F0.5 Progression

| Milestone / Experiment | Changes Introduced | Val Macro F0.5 | Public LB F0.5 | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Primary Channels A–E, base features | 0.9010 | 0.8750 | — | — |
| **Augmentation 1** | + Channels F, G (wide name + GPU addr re-rank, compound keys) | 0.9257 | — | — | — |
| **Augmentation 2** | + Channel H (combined name+address neural embeddings) | 0.9387 | — | 0.9790 | 0.8853 |
| **Augmentation 3** | + Channel I (character 3-gram TF-IDF) | 0.9433 | 0.9264 | — | — |
| **Pruning + Features** | $K=15$ filter + 51-feature pack + Channels J, K (translit dict) | 0.9642 | — | — | — |
| **Stage 6b Stacking** | **Coherence Stacking + Exclusivity (margin 0.10, $t_{\text{top1}}=0.46, t_{\text{extra}}=0.68$)** | **0.9698** | — | **0.991** | **0.941** |

### 5.2 Decision-Layer Exploration & Diagnostics
- **Error Diagnostics on Baseline Decision Layer:**
  - Bucket 5 (non-top-1 extras lost with prob $< t_{\text{extra}}$) accounted for 4.71% of all ground truth pairs (16,322 pairs).
  - Wrong extra predictions accounted for 89.7% of all false positive errors (5,914 pairs).
- **Ablation Studies on Decision Layer:**
  1. *Global Bipartite Matching:* $\Delta = -0.0012$ (0.9376). Bounded degree caps truncated valid multi-matches; unconstrained matching without margin dropping admitted false positives.
  2. *Second-Chance Reassignment:* $\Delta = -0.0008$ (0.9379). Recovered minor recall (+58 TP) but hurt precision, lowering macro F0.5.
  3. *Source-Specific Thresholds:* $\Delta = +0.0001$ (0.9388). Marginal difference did not justify complexity.
  4. *Coherence Stacking (Adopted):* $\Delta = +0.0056 \to +0.0064$ alone on the base model, and $\mathbf{+0.0056}$ on top of the 51-feature model ($0.9642 \to \mathbf{0.9698}$). Slashed Bucket 5 losses by >35% and wrong extras by >16% while pushing precision to **0.991** and recall to **0.941**.

---

## 6. Compliance & Academic Integrity

- **Strictly Offline Execution (No External APIs):** The entire pipeline executes 100% offline. Zero network calls, zero external web APIs, zero external databases, and zero geocoding services are used during preprocessing, training, or inference.
- **Permissive Open-Source Licensing:**
  - **Qwen3-Embedding-0.6B:** Apache 2.0 license (0.6B parameters, well within the 8B competition limit).
  - **LightGBM:** MIT license.
  - **AnyAscii:** ISC license.
  - **RapidFuzz / Scikit-Learn / PyTorch / SciPy / sparse_dot_topn:** MIT / BSD licenses.
- **Dynamic Country Generalization (Zero Hardcoding):**
  - `country` is never hardcoded and is never used as an input feature for the matching models.
  - Test entities from `France` are natively processed through the exact same multilingual normalization, embedding, blocking, feature extraction, and inference pipeline as `India` and `US`.

---

## 7. Conclusion

By pairing dense multilingual neural embeddings with multi-key compound address indexing, sparse character TF-IDF, learned transliteration normalization, a two-stage candidate pruning filter ($K=15$), a comprehensive 51-feature pairwise classifier, and Stage 6b coherence stacking, our pipeline achieves **98.41%** candidate pair recall on validation while maintaining a compact **15.0 candidates per S1** test pool. The decision layer pushes final validation macro F0.5 to **0.9698** (precision **0.991**, recall **0.941**) with full zero-shot generalization to unobserved countries.

---

## Appendix: Pipeline Structure & Execution Order

The complete solution is structured in `code/business_entity_resolution/`:
```text
code/business_entity_resolution/
├── README.md                 # End-to-end reproduction guide with exact run order
├── requirements.txt          # Pinned dependency environment
└── src/
    ├── s1_normalize.py       # Multilingual text normalization & anyascii transliteration
    ├── s2_embed.py           # Qwen3-Embedding-0.6B name embeddings (256d fp16)
    ├── embed_combined_all.sh # Combined name+address embeddings runner (s2b_embed_combined.py)
    ├── s3_block.py           # Primary multi-channel blocking (Channels A–E)
    ├── s3b_augment.py        # Candidate augmentation (Channels F, G, H & I, J, K)
    ├── build_indic_dict.py   # Learned transliteration dictionary from train matches
    ├── s3c_tfidf.py          # Character TF-IDF retrieval (Channels I, J, K)
    ├── prune_chunks.py       # Two-stage blocking filter (top-K=15 per S1)
    ├── s4_features.py        # 51 pairwise feature extraction
    ├── s5_train.py           # Primary LightGBM pairwise classifier training
    ├── s6_tune.py            # Macro F0.5 decision threshold optimization
    ├── s6b_stacking.py       # Stage 6b coherence stacking meta-model
    ├── s7_predict.py         # LightGBM inference on test candidate pairs
    ├── s8_write.py           # Exclusivity resolution & final TSV generation
    └── utils/validate_submission.py # Official submission validation script
```
All outputs are generated into `output/matching_results.tsv` and `output/candidate_pairs.tsv` strictly satisfying all competition formatting specifications.
