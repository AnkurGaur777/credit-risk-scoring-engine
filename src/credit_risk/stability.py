"""Population Stability Index and the monitoring plan."""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_STABLE, PSI_SHIFT = 0.10, 0.25  # conventions, not laws


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI of a numeric variable; bins are quantiles of the EXPECTED (training) sample. Missing is its own bin."""
    e, a = pd.Series(expected).astype(float), pd.Series(actual).astype(float)
    edges = np.unique(np.quantile(e.dropna(), np.linspace(0, 1, bins + 1)[1:-1]))

    def dist(s):
        idx = np.where(s.isna(), len(edges) + 1, np.searchsorted(edges, s.fillna(0).to_numpy(), side="right"))
        return np.bincount(idx, minlength=len(edges) + 2) / max(len(s), 1)

    pe, pa = np.clip(dist(e), eps, None), np.clip(dist(a), eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_label(v: float) -> str:
    return "stable" if v < PSI_STABLE else "some shift" if v < PSI_SHIFT else "significant shift"


def psi_table(train: pd.DataFrame, other: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            v = psi(train[c], other[c])
            rows.append({"feature": c, "psi": v, "status": psi_label(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def monitoring_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "Score PSI (vs. training)",
                "Monthly",
                "<0.10 ok; 0.10-0.25 investigate; >0.25 escalate",
                "Model owner",
                "PSI > 0.25 two months running -> retrain review",
            ),
            (
                "Feature PSI (top 10 by SHAP)",
                "Monthly",
                "Same thresholds",
                "Model owner",
                "Check upstream data/pipeline first, then population change",
            ),
            (
                "Approval rate",
                "Weekly",
                "+/- 3 pts vs. 3-month average",
                "Credit risk team",
                "Check cut-off, channel mix, data feeds",
            ),
            (
                "Bad rate by origination cohort",
                "Monthly (labels mature in 6-12 months)",
                "Observed > predicted by 20% relative",
                "Risk analytics",
                "Re-calibrate; consider retrain",
            ),
            (
                "Calibration (predicted vs observed PD by decile)",
                "Quarterly",
                "ECE > 0.02 or Brier up > 5% relative",
                "Model owner",
                "Refit calibrator on recent matured cohorts",
            ),
            (
                "Ranking power (AUC / KS on matured cohorts)",
                "Quarterly",
                "AUC drop > 0.02 vs. development",
                "Model validation",
                "Retrain challenger, champion-challenger test",
            ),
            (
                "Fairness (approval & error rates by group)",
                "Quarterly",
                "Disparate-impact ratio < 0.8 or gap widening",
                "Compliance + risk",
                "Review features/proxies, document decision",
            ),
            (
                "Data quality (missing rates, sentinel counts)",
                "Daily",
                "Missing rate change > 5 pts",
                "Data engineering",
                "Block scoring or fall back to manual review",
            ),
        ],
        columns=["What to track", "Frequency", "Alert threshold", "Owner", "Action / retrain trigger"],
    )
