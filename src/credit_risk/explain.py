"""Global SHAP plots and per-applicant reason codes.

Per-row contributions come from LightGBM's native TreeSHAP (`pred_contrib=True`), identical to
shap.TreeExplainer for LightGBM and with no `shap` dependency at serving time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# feature -> customer-friendly sentence shown when this feature pushes risk UP
REASON_CODES: dict[str, str] = {
    "EXT_SOURCE_MEAN": "Low overall external credit score",
    "EXT_SOURCE_WEIGHTED": "Low overall external credit score",
    "EXT_SOURCE_MIN": "At least one external credit score is weak",
    "EXT_SOURCE_PROD": "External credit scores are weak when combined",
    "EXT_SOURCE_1": "Low external credit score (source 1)",
    "EXT_SOURCE_2": "Low external credit score (source 2)",
    "EXT_SOURCE_3": "Low external credit score (source 3)",
    "EXT_SOURCE_1_MISSING": "External credit score (source 1) not available",
    "EXT_SOURCE_3_MISSING": "External credit score (source 3) not available",
    "ANNUITY_INCOME_RATIO": "High instalment relative to income",
    "CREDIT_INCOME_RATIO": "Loan amount is high relative to income",
    "CREDIT_GOODS_RATIO": "Loan amount exceeds the price of the goods financed",
    "ANNUITY_CREDIT_RATIO": "Short loan term / heavy repayments",
    "AMT_ANNUITY": "High instalment amount",
    "AMT_CREDIT": "High loan amount",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "INCOME_PER_FAMILY_MEMBER": "Low income per household member",
    "AMT_INCOME_TOTAL": "Low income",
    "YEARS_EMPLOYED": "Short employment history",
    "DAYS_EMPLOYED": "Short employment history",
    "DAYS_EMPLOYED_ANOM": "Employment information not applicable or unavailable",
    "YEARS_ID_PUBLISH": "Identity document recently re-issued",
    "DAYS_ID_PUBLISH": "Identity document recently re-issued",
    "YEARS_REGISTRATION": "Recently changed registration details",
    "DAYS_REGISTRATION": "Recently changed registration details",
    "YEARS_PHONE_CHANGE": "Recent phone number change",
    "DAYS_LAST_PHONE_CHANGE": "Recent phone number change",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_HOUSING_TYPE": "Housing situation",
    "NAME_CONTRACT_TYPE": "Loan product type",
    "OCCUPATION_TYPE": "Occupation category",
    "ORGANIZATION_TYPE": "Employer type",
    "BUREAU_ENQUIRIES_TOTAL": "Many recent credit enquiries",
    "BUREAU_ENQUIRIES_RECENT": "Many very recent credit enquiries",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Many credit enquiries in the last year",
    "HOUSING_FEATURES_MISSING_SHARE": "Limited information about the property",
    "DOCUMENTS_PROVIDED": "Fewer supporting documents provided",
    "BUREAU_MAX_DAYS_OVERDUE": "Past overdue payments at other lenders",
    "BUREAU_OVERDUE_SUM": "Overdue amounts at other lenders",
    "BUREAU_DEBT_SUM": "High existing debt at other lenders",
    "BUREAU_DEBT_CREDIT_RATIO": "High utilisation of existing credit",
    "BUREAU_ACTIVE_COUNT": "Many active credit accounts",
    "PREV_REFUSED_RATIO": "Previous applications were refused",
    "INST_LATE_SHARE": "History of late instalment payments",
    "INST_AVG_DAYS_LATE": "Instalments paid late on average",
    "INST_PAYMENT_RATIO": "Past instalments under-paid",
    "POS_DPD_MONTHS": "Past delinquency on instalment loans",
    "POS_MAX_DPD": "Past delinquency on instalment loans",
    "CC_UTILISATION_MEAN": "High credit-card utilisation",
    "CC_DPD_MONTHS": "Past credit-card delinquency",
}


def contributions(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(values[n, f], base[n]) in log-odds space via LightGBM TreeSHAP."""
    c = model.predict(X, pred_contrib=True)
    c = np.asarray(c)
    return c[:, :-1], c[:, -1]


def reason_codes(
    values_row: np.ndarray, features: list[str], top_k: int = 4, min_contrib: float = 0.0
) -> list[dict]:
    """Top features pushing risk UP for one applicant, translated to plain English (de-duplicated sentences)."""
    order = np.argsort(-values_row)
    out, seen = [], set()
    for i in order:
        v = float(values_row[i])
        if v <= min_contrib or len(out) >= top_k:
            break
        text = REASON_CODES.get(features[i], f"Risk factor: {features[i].replace('_', ' ').title()}")
        if text in seen:
            continue
        seen.add(text)
        out.append({"feature": features[i], "reason": text, "impact_log_odds": round(v, 3)})
    return out


def global_importance(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _plot_frame(X: pd.DataFrame) -> pd.DataFrame:
    P = X.copy()
    for c in P.columns:
        if str(P[c].dtype) == "category":
            P[c] = P[c].cat.codes.replace(-1, np.nan)
    return P


def save_shap_plots(
    values: np.ndarray, X: pd.DataFrame, out_dir, top_dependence: int = 5, max_rows: int = 3000
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt
    import shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.random.default_rng(0).choice(len(X), size=min(max_rows, len(X)), replace=False)
    P, V = _plot_frame(X.iloc[idx]), values[idx]
    files = []
    plt.figure()
    shap.summary_plot(V, P, show=False, max_display=15)
    plt.tight_layout()
    f = out_dir / "shap_summary.png"
    plt.savefig(f, dpi=130, bbox_inches="tight")
    plt.close()
    files.append(str(f))
    top = global_importance(values, list(X.columns))["feature"].head(top_dependence)
    for feat in top:
        plt.figure()
        shap.dependence_plot(feat, V, P, interaction_index=None, show=False)
        plt.tight_layout()
        f = out_dir / f"shap_dependence_{feat}.png"
        plt.savefig(f, dpi=110, bbox_inches="tight")
        plt.close()
        files.append(str(f))
    return files
