"""Loading, cleaning (sentinels, XNA, types), auditing and splitting the application table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

XNA_COLUMNS = ["CODE_GENDER", "ORGANIZATION_TYPE", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE"]
TABLES = [
    "application_train",
    "bureau",
    "previous_application",
    "installments_payments",
    "POS_CASH_balance",
    "credit_card_balance",
]


def read_table(name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Read a table from Parquet if present (fast), else from the Kaggle CSV."""
    raw_dir = Path(raw_dir or C.DATA_RAW)
    pq, csv = raw_dir / f"{name}.parquet", raw_dir / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"{name} not found in {raw_dir} (looked for .parquet and .csv)")


def has_table(name: str, raw_dir: Path | None = None) -> bool:
    raw_dir = Path(raw_dir or C.DATA_RAW)
    return (raw_dir / f"{name}.parquet").exists() or (raw_dir / f"{name}.csv").exists()


def csv_to_parquet(raw_dir: Path | None = None) -> list[str]:
    """One-time conversion of Kaggle CSVs to Parquet with downcast numerics."""
    raw_dir = Path(raw_dir or C.DATA_RAW)
    done = []
    for name in TABLES:
        csv = raw_dir / f"{name}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            for col in df.select_dtypes("float64").columns:
                df[col] = pd.to_numeric(df[col], downcast="float")
            for col in df.select_dtypes("int64").columns:
                if col not in (C.ID_COL, "SK_ID_PREV", "SK_ID_BUREAU"):
                    df[col] = pd.to_numeric(df[col], downcast="integer")
            df.to_parquet(raw_dir / f"{name}.parquet", index=False)
            done.append(name)
    return done


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning: sentinels -> NaN (+flag), XNA -> NaN, readable age. Safe to run before splitting."""
    out = df.copy()
    if "DAYS_EMPLOYED" in out:
        anom = out["DAYS_EMPLOYED"] == C.DAYS_EMPLOYED_SENTINEL
        out["DAYS_EMPLOYED_ANOM"] = anom.astype("int8")
        out.loc[anom, "DAYS_EMPLOYED"] = np.nan
    for col in XNA_COLUMNS:
        if col in out:
            out[col] = out[col].replace("XNA", np.nan)
    if "DAYS_BIRTH" in out:
        out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    for col in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"):
        if col in out:  # impossible values -> missing
            out.loc[out[col] <= 0, col] = np.nan
    return out


def split_data(df: pd.DataFrame, seed: int = C.SEED):
    """Stratified 60/20/20 split with a fixed seed. Returns train, valid, test."""
    tr, rest = train_test_split(df, test_size=1 - C.SPLIT[0], stratify=df[C.TARGET], random_state=seed)
    va, te = train_test_split(
        rest, test_size=C.SPLIT[2] / (C.SPLIT[1] + C.SPLIT[2]), stratify=rest[C.TARGET], random_state=seed
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def audit_application(df: pd.DataFrame) -> dict:
    """Numbers behind docs/data_audit.md."""
    miss = df.isna().mean().sort_values(ascending=False)
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    audit = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target_rate": float(df[C.TARGET].mean()),
        "imbalance_ratio": float((1 - df[C.TARGET].mean()) / df[C.TARGET].mean()),
        "id_unique": bool(df[C.ID_COL].is_unique),
        "duplicate_rows": int(df.drop(columns=[C.ID_COL]).duplicated().sum()),
        "missing_over_50pct": miss[miss > 0.5].round(3).to_dict(),
        "missing_top15": miss.head(15).round(3).to_dict(),
        "sentinel_days_employed": int(
            (df.get("DAYS_EMPLOYED", pd.Series(dtype=float)) == C.DAYS_EMPLOYED_SENTINEL).sum()
        ),
        "xna_counts": {c: int((df[c] == "XNA").sum()) for c in XNA_COLUMNS if c in df},
        "negative_income": int((df.get("AMT_INCOME_TOTAL", pd.Series(dtype=float)) <= 0).sum()),
        "age_range": [float((-df["DAYS_BIRTH"] / 365.25).min()), float((-df["DAYS_BIRTH"] / 365.25).max())]
        if "DAYS_BIRTH" in df
        else None,
        "cardinality": {c: int(df[c].nunique()) for c in cat_cols},
    }
    return audit
