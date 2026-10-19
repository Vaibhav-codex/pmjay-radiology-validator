import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = Path(os.environ.get("CLAIMLENS_OUTPUT", ROOT / "outputs"))
UPLOAD_DIR = Path(os.environ.get("CLAIMLENS_UPLOADS", ROOT / "uploads"))
REGISTRY_PATH = Path(os.environ.get("CLAIMLENS_REGISTRY", ROOT / "data" / "registry.json"))
SAMPLE_DIR = ROOT / "sample_data"

STG_RULES_PATH = CONFIG_DIR / "stg_rules.yaml"
LEXICON_PATH = CONFIG_DIR / "clinical_lexicon.yaml"

# ---- decision thresholds (tune on labelled adjudication data) ----
T = dict(
    detect_conf=0.50,        # image finding counts as "present" at/above this
    weak_conf=0.35,          # below this an image finding is treated as "absent"
    min_side_px=512,         # below -> LOW_RESOLUTION
    min_dyn_range=70,        # p99-p1 grey levels; below -> LOW_CONTRAST
    min_sharpness=40.0,      # variance of Laplacian @512px; below -> BLURRY
    dup_hamming=6,           # dHash bit distance for "same image"
)

SEVERITY_RANK = {"unknown": 0, "mild": 1, "moderate": 2, "severe": 3}

# risk points per flag code (capped at 100 overall)
RISK_WEIGHTS = {
    "LATERALITY_MISMATCH": 35,
    "RECYCLED_IMAGE": 40,
    "EXAGGERATION": 30,
    "CONTRADICTION": 30,
    "UNSUPPORTED_FINDING": 25,
    "OMISSION": 20,
    "BODY_PART_MISMATCH": 25,
    "POOR_IMAGE_QUALITY": 8,
    "DUPLICATE_SUBMISSION": 5,
    "POSSIBLE_OVERSTATEMENT": 12,
    "NOT_VERIFIABLE": 3,
}
STG_FAIL_POINTS = 15
STG_WARN_POINTS = 4
