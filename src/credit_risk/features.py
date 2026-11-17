"""Feature engineering: Stage A (application table) and Stage B (history tables aggregated per applicant).

Everything here is either stateless (ratios, flags) or fitted on TRAIN ONLY (CategoryEncoder),
which is how leakage is avoided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EPS = 1e-9
RARE_LABEL = "__RARE__"  # categories below min_share are grouped here (train-fitted)

# --- feature dictionary: name -> (how computed, why it should matter) -------------------------
FEATURE_DICTIONARY: dict[str, tuple[str, str]] = {
    "CREDIT_INCOME_RATIO": (
        "AMT_CREDIT / AMT_INCOME_TOTAL",
        "Larger loans relative to income are harder to repay.",
    ),
    "ANNUITY_INCOME_RATIO": (
        "AMT_ANNUITY / AMT_INCOME_TOTAL",
        "Payment burden: share of income consumed each period.",
    ),
    "CREDIT_GOODS_RATIO": (
        "AMT_CREDIT / AMT_GOODS_PRICE",
        "Above 1 means the borrower finances more than the goods cost (fees, add-ons).",
    ),
    "ANNUITY_CREDIT_RATIO": (
        "AMT_ANNUITY / AMT_CREDIT",
        "Proxy for loan term: high ratio = short term = heavy instalments.",
    ),
    "INCOME_PER_FAMILY_MEMBER": (
        "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS",
        "Disposable income is shared across the household.",
    ),
    "YEARS_EMPLOYED": ("-DAYS_EMPLOYED / 365.25 (sentinel -> NaN)", "Job stability."),
    "EMPLOYED_TO_AGE_RATIO": (
        "YEARS_EMPLOYED / AGE_YEARS",
        "Share of adult life in work. Age-derived: excluded by default policy.",
    ),
    "YEARS_REGISTRATION": ("-DAYS_REGISTRATION / 365.25", "Residential/registration stability."),
    "YEARS_ID_PUBLISH": ("-DAYS_ID_PUBLISH / 365.25", "Recently re-issued ID can signal instability."),
    "YEARS_PHONE_CHANGE": (
        "-DAYS_LAST_PHONE_CHANGE / 365.25",
        "Recent phone change is a mild instability signal.",
    ),
    "EXT_SOURCE_1_MISSING": ("EXT_SOURCE_1 is NaN", "Missing external score is informative, not random."),
    "EXT_SOURCE_3_MISSING": ("EXT_SOURCE_3 is NaN", "Missing external score is informative, not random."),
    "HOUSING_FEATURES_MISSING_SHARE": (
        "share of *_AVG housing columns that are NaN",
        "Proxy for how much is known about the home.",
    ),
    "EXT_SOURCE_MEAN": ("mean of available EXT_SOURCE_1..3", "Combined external credit opinion."),
    "EXT_SOURCE_MIN": ("min of available EXT_SOURCE_1..3", "Weakest external opinion."),
    "EXT_SOURCE_PROD": ("product of available EXT_SOURCE_1..3", "Interaction: all three must be good."),
    "EXT_SOURCE_WEIGHTED": (
        "weighted mean (1:2, 2:3, 3:4)",
        "Weights roughly follow individual predictive strength.",
    ),
    "DOCUMENTS_PROVIDED": ("sum of FLAG_DOCUMENT_* columns", "Documentation completeness."),
    "BUREAU_ENQUIRIES_TOTAL": ("sum of AMT_REQ_CREDIT_BUREAU_* columns", "Credit-seeking intensity."),
    "BUREAU_ENQUIRIES_RECENT": ("HOUR + DAY + WEEK + MON enquiries", "Recent credit hunger."),
    # Stage B
    "BUREAU_COUNT": ("number of past bureau credits", "Credit experience."),
    "BUREAU_ACTIVE_COUNT": ("number of Active bureau credits", "Current outstanding obligations."),
    "BUREAU_DEBT_SUM": ("sum AMT_CREDIT_SUM_DEBT", "Total debt at other lenders."),
    "BUREAU_OVERDUE_SUM": ("sum AMT_CREDIT_SUM_OVERDUE", "Overdue amounts at other lenders."),
    "BUREAU_MAX_DAYS_OVERDUE": ("max CREDIT_DAY_OVERDUE", "Worst past delinquency."),
    "BUREAU_DEBT_CREDIT_RATIO": ("sum debt / sum credit", "Bureau utilisation."),
    "BUREAU_AVG_CREDIT_AGE_YEARS": ("mean -DAYS_CREDIT / 365.25", "Length of credit history."),
    "PREV_COUNT": ("number of previous applications", "Application history."),
    "PREV_REFUSED_RATIO": (
        "refused / all previous applications",
        "Other lenders' (and our own) past rejections.",
    ),
    "PREV_APPROVED_COUNT": ("approved previous applications", "Successful borrowing history."),
    "PREV_AVG_CREDIT": ("mean AMT_CREDIT of approved", "Typical past loan size."),
    "PREV_DAYS_SINCE_LAST": ("min -DAYS_DECISION", "Time since last application."),
    "INST_LATE_SHARE": ("share of instalments paid after due date", "Core behavioural risk signal."),
    "INST_AVG_DAYS_LATE": ("mean max(0, entry - due)", "Severity of lateness."),
    "INST_PAYMENT_RATIO": ("sum paid / sum due", "Under-payment."),
    "POS_DPD_MONTHS": ("months with SK_DPD > 0", "Delinquency on POS/cash loans."),
    "POS_MAX_DPD": ("max SK_DPD", "Worst delinquency on POS/cash loans."),
    "CC_UTILISATION_MEAN": ("mean AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL", "Card utilisation."),
    "CC_DPD_MONTHS": ("months with SK_DPD > 0 on cards", "Delinquency on cards."),
    "CC_DRAWINGS_RATIO": ("mean drawings / limit", "Cash-drawing behaviour."),
}


def engineer_application(df: pd.DataFrame) -> pd.DataFrame:
    """Stage A features. Stateless; tolerant of missing raw columns (the API sends a subset)."""
    out = df.copy()

    def col(name):
        return out[name].astype(float) if name in out else pd.Series(np.nan, index=out.index)

    inc, cred, ann, goods = (
        col("AMT_INCOME_TOTAL"),
        col("AMT_CREDIT"),
        col("AMT_ANNUITY"),
        col("AMT_GOODS_PRICE"),
    )
    fam = col("CNT_FAM_MEMBERS").clip(lower=1)
    out["CREDIT_INCOME_RATIO"] = cred / (inc + EPS)
    out["ANNUITY_INCOME_RATIO"] = ann / (inc + EPS)
    out["CREDIT_GOODS_RATIO"] = cred / (goods + EPS)
    out["ANNUITY_CREDIT_RATIO"] = ann / (cred + EPS)
    out["INCOME_PER_FAMILY_MEMBER"] = inc / fam
    out["YEARS_EMPLOYED"] = -col("DAYS_EMPLOYED") / 365.25
    if "AGE_YEARS" in out or "DAYS_BIRTH" in out:
        age = out["AGE_YEARS"] if "AGE_YEARS" in out else -col("DAYS_BIRTH") / 365.25
        out["EMPLOYED_TO_AGE_RATIO"] = out["YEARS_EMPLOYED"] / age
    out["YEARS_REGISTRATION"] = -col("DAYS_REGISTRATION") / 365.25
    out["YEARS_ID_PUBLISH"] = -col("DAYS_ID_PUBLISH") / 365.25
    out["YEARS_PHONE_CHANGE"] = -col("DAYS_LAST_PHONE_CHANGE") / 365.25
    ext = pd.concat([col(f"EXT_SOURCE_{i}") for i in (1, 2, 3)], axis=1)
    ext.columns = ["e1", "e2", "e3"]
    out["EXT_SOURCE_1_MISSING"] = ext["e1"].isna().astype("int8")
    out["EXT_SOURCE_3_MISSING"] = ext["e3"].isna().astype("int8")
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1)
    out["EXT_SOURCE_PROD"] = ext.prod(axis=1, min_count=1)
    w = np.array([1.0, 2.0, 3.0])
    avail = ext.notna().to_numpy() * w
    out["EXT_SOURCE_WEIGHTED"] = (ext.fillna(0).to_numpy() * w).sum(axis=1) / np.where(
        avail.sum(axis=1) == 0, np.nan, avail.sum(axis=1)
    )
    housing = [c for c in out.columns if c.endswith("_AVG")]
    out["HOUSING_FEATURES_MISSING_SHARE"] = out[housing].isna().mean(axis=1) if housing else np.nan
    docs = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    out["DOCUMENTS_PROVIDED"] = out[docs].sum(axis=1) if docs else np.nan
    enq = [c for c in out.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    out["BUREAU_ENQUIRIES_TOTAL"] = out[enq].sum(axis=1, min_count=1) if enq else np.nan
    recent = [
        f"AMT_REQ_CREDIT_BUREAU_{k}"
        for k in ("HOUR", "DAY", "WEEK", "MON")
        if f"AMT_REQ_CREDIT_BUREAU_{k}" in out
    ]
    out["BUREAU_ENQUIRIES_RECENT"] = out[recent].sum(axis=1, min_count=1) if recent else np.nan
    for c in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        if c in out:
            out[c] = out[c].map({"Y": 1, "N": 0}).astype("float")
    return out


# ------------------------------------------------------------------------------------------
class CategoryEncoder:
    """Groups rare categories into 'Other' and fixes category sets, fitted on TRAIN only."""

    def __init__(self, min_share: float = 0.01):
        self.min_share = min_share
        self.categories_: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame, cols: list[str]) -> CategoryEncoder:
        for c in cols:
            share = df[c].astype("object").value_counts(normalize=True, dropna=True)
            keep = sorted(share[share >= self.min_share].index.astype(str))
            self.categories_[c] = keep + [RARE_LABEL]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, cats in self.categories_.items():
            s = out[c].astype("object").where(out[c].notna(), None)
            s = s.map(lambda v, cats=cats: None if v is None else (str(v) if str(v) in cats else RARE_LABEL))
            out[c] = pd.Categorical(s, categories=cats)
        return out


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) in ("string", "str", "category")
    ]


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns the models may use, after the fairness policy in config (gender always out, age by default)."""
    drop = {C.TARGET, C.ID_COL, *C.ALWAYS_EXCLUDED}
    if not C.ALLOW_AGE:
        drop |= set(C.AGE_DERIVED)
    return [c for c in df.columns if c not in drop and not c.startswith("_")]


# ------------------------------------------------------------------------------------------
# Stage B: aggregations of history tables (one row per SK_ID_CURR).
# Leakage guard: only records dated BEFORE the current application are used (relative days <= 0;
# monthly balances < 0). `before_application` is unit-tested.
def before_application(df: pd.DataFrame, col: str, strict: bool = False) -> pd.DataFrame:
    if col not in df:
        return df
    return df[df[col] < 0] if strict else df[df[col] <= 0]


def _flat(g: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    g.columns = [prefix + c for c in g.columns]
    return g


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    b = before_application(bureau, "DAYS_CREDIT").copy()
    b["_active"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["_age"] = -b["DAYS_CREDIT"] / 365.25
    g = b.groupby(C.ID_COL).agg(
        BUREAU_COUNT=("DAYS_CREDIT", "size"),
        BUREAU_ACTIVE_COUNT=("_active", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_MAX_DAYS_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_AVG_CREDIT_AGE_YEARS=("_age", "mean"),
    )
    g["BUREAU_DEBT_CREDIT_RATIO"] = g["BUREAU_DEBT_SUM"] / (g["BUREAU_CREDIT_SUM"] + EPS)
    return g.drop(columns="BUREAU_CREDIT_SUM")


def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    p = before_application(prev, "DAYS_DECISION").copy()
    p["_refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["_approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["_credit_appr"] = p["AMT_CREDIT"].where(p["_approved"] == 1)
    p["_since"] = -p["DAYS_DECISION"]
    g = p.groupby(C.ID_COL).agg(
        PREV_COUNT=("_refused", "size"),
        PREV_REFUSED_RATIO=("_refused", "mean"),
        PREV_APPROVED_COUNT=("_approved", "sum"),
        PREV_AVG_CREDIT=("_credit_appr", "mean"),
        PREV_DAYS_SINCE_LAST=("_since", "min"),
    )
    return g


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    i = before_application(inst, "DAYS_INSTALMENT").copy()
    i = i[i["DAYS_ENTRY_PAYMENT"] <= 0] if "DAYS_ENTRY_PAYMENT" in i else i
    i["_late_days"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["_late"] = (i["_late_days"] > 0).astype(int)
    g = i.groupby(C.ID_COL).agg(
        INST_LATE_SHARE=("_late", "mean"),
        INST_AVG_DAYS_LATE=("_late_days", "mean"),
        _paid=("AMT_PAYMENT", "sum"),
        _due=("AMT_INSTALMENT", "sum"),
    )
    g["INST_PAYMENT_RATIO"] = g["_paid"] / (g["_due"] + EPS)
    return g.drop(columns=["_paid", "_due"])


def aggregate_pos(pos: pd.DataFrame) -> pd.DataFrame:
    p = before_application(pos, "MONTHS_BALANCE", strict=True).copy()
    p["_dpd"] = (p["SK_DPD"] > 0).astype(int)
    return p.groupby(C.ID_COL).agg(POS_DPD_MONTHS=("_dpd", "sum"), POS_MAX_DPD=("SK_DPD", "max"))


def aggregate_cards(cc: pd.DataFrame) -> pd.DataFrame:
    c = before_application(cc, "MONTHS_BALANCE", strict=True).copy()
    lim = c["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    c["_util"] = c["AMT_BALANCE"] / lim
    c["_draw"] = c["AMT_DRAWINGS_CURRENT"] / lim
    c["_dpd"] = (c["SK_DPD"] > 0).astype(int)
    return c.groupby(C.ID_COL).agg(
        CC_UTILISATION_MEAN=("_util", "mean"),
        CC_DPD_MONTHS=("_dpd", "sum"),
        CC_DRAWINGS_RATIO=("_draw", "mean"),
    )


def build_stage_b(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate every available history table to one row per applicant (indexed by SK_ID_CURR)."""
    parts = []
    fns = {
        "bureau": aggregate_bureau,
        "previous_application": aggregate_previous,
        "installments_payments": aggregate_installments,
        "POS_CASH_balance": aggregate_pos,
        "credit_card_balance": aggregate_cards,
    }
    for name, fn in fns.items():
        if name in tables and tables[name] is not None:
            parts.append(fn(tables[name]))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def merge_stage_b(app: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return app
    out = app.merge(agg, how="left", left_on=C.ID_COL, right_index=True)
    # 'no history' is information: counts become 0, everything else stays NaN for the models to handle
    for c in ("BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "PREV_COUNT", "PREV_APPROVED_COUNT"):
        if c in out:
            out[c] = out[c].fillna(0)
    out["HAS_BUREAU_HISTORY"] = (out.get("BUREAU_COUNT", 0) > 0).astype("int8")
    out["HAS_INSTALMENT_HISTORY"] = (
        out["INST_LATE_SHARE"].notna().astype("int8") if "INST_LATE_SHARE" in out else 0
    )
    return out
