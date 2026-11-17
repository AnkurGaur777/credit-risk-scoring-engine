"""Central configuration: paths, seeds and modelling policy decisions."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("CR_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = Path(os.getenv("CR_DATA_RAW", ROOT / "data" / "raw"))
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = Path(os.getenv("CR_ARTIFACTS", ROOT / "artifacts"))
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

SEED = 42
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
DAYS_EMPLOYED_SENTINEL = 365243

# Split: 60 / 20 / 20, stratified on the target. Test set is touched once for final numbers.
SPLIT = (0.6, 0.2, 0.2)

# ---- Fairness / policy decisions (documented in the model card) -------------------------
# Gender is never a model input. Age-derived features are also excluded by default
# (set CR_ALLOW_AGE=1 to include them). Both are still used to AUDIT outcomes by group.
ALLOW_AGE = os.getenv("CR_ALLOW_AGE", "0") == "1"
SENSITIVE_COLUMNS = ["CODE_GENDER", "AGE_YEARS", "NAME_FAMILY_STATUS"]
ALWAYS_EXCLUDED = ["CODE_GENDER"]
AGE_DERIVED = ["DAYS_BIRTH", "AGE_YEARS", "EMPLOYED_TO_AGE_RATIO"]

# ---- Scorecard scaling ------------------------------------------------------------------
BASE_SCORE = 600
BASE_ODDS = 50  # good:bad odds at the base score
PDO = 20  # points to double the odds

# ---- Scorecard variable selection thresholds -------------------------------------------
IV_MIN = 0.02
IV_SUSPICIOUS = 0.5
CORR_MAX = 0.70


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, ARTIFACTS, REPORTS, FIGURES, DOCS):
        d.mkdir(parents=True, exist_ok=True)
