import os
import pandas as pd

# Root path detection
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(CODE_DIR))

DATA_DIR = os.path.join(ROOT_DIR, "dataset")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# Ensure cache and output dirs exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample configuration
AMLC_SAMPLE = os.environ.get("AMLC_SAMPLE", "50000").lower()
SAMPLE = None if AMLC_SAMPLE == "none" else int(AMLC_SAMPLE)
SAMPLE_BLOCK_TEST = 5000

SEED = 42

# Blocking and Decision parameters (from DECISIONS.md)
K_PER_SOURCE = 20
MAX_ADDR_CANDS = 20
MAX_SKEL_CANDS = 20
MAX_RARE_CANDS = 20
MAX_BLOCK_SIZE = 50
CAND_CAP = 40
USE_EXCLUSIVITY = True
BLOCK_BY_COUNTRY = True
EMB_DIM = 256
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"

T_TOP1 = 0.48
T_EXTRA = 0.68
EXCL_MARGIN = 0.05
FEATURES = [
    # Embedding & Rank
    "emb_score",
    "emb_rank",
    # Name Similarity
    "name_token_sort",
    "name_token_set",
    "core_ratio",
    "core_partial",
    "skel_ratio",
    "name_jaccard",
    "legal_match",
    "dba_max",
    "aka_max",
    "len_diff",
    # Address Similarity & Matching
    "addr_token_set",
    "house_match",
    "house_cand_match",
    "num_jaccard",
    "rare_tok_overlap",
    "zip_match",
    "state_match",
    "addr_missing_any",
    # Channel & Source Flags
    "cand_source",
    "ch_emb",
    "ch_addr",
    "ch_skel",
    "ch_rare",
    "ch_rev",
    "ch_rerank",
    "ch_keyx",
    "ch_comb",
    "ch_tfidf",
    "tfidf_rank",
    "n_channels",
    # Context Features
    "gap_to_best",
    "n_cands",
    "reverse_rank",
    "support",
]

def load_tsv(path):
    """
    Loads TSV file with correct quoting and dtype parameters.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=3
    )
