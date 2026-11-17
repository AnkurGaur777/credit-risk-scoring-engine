"""Reliability curves, ECE, Platt / isotonic calibrators (fit on VALIDATION, never on test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Quantile bins of predicted PD vs observed default rate."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    t = d.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()
    return t


def ece(y, p, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins)
    return float(np.sum(t["n"] / t["n"].sum() * np.abs(t["mean_pred"] - t["observed"])))


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        assert method in ("isotonic", "platt")
        self.method = method
        self.model = None

    def fit(self, p, y) -> Calibrator:
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(
                np.asarray(p), np.asarray(y)
            )
        else:
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), np.asarray(y))
        return self

    def transform(self, p) -> np.ndarray:
        if self.method == "isotonic":
            return np.asarray(self.model.predict(np.asarray(p)))
        return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def compare_calibrators(p_valid, y_valid, p_test, y_test) -> tuple[pd.DataFrame, dict[str, Calibrator]]:
    rows = [
        {
            "method": "raw",
            "brier": brier_score_loss(y_test, p_test),
            "ece": ece(y_test, p_test),
            "mean_pred": float(np.mean(p_test)),
        }
    ]
    cals = {}
    for m in ("platt", "isotonic"):
        c = Calibrator(m).fit(p_valid, y_valid)
        pt = c.transform(p_test)
        rows.append(
            {
                "method": m,
                "brier": brier_score_loss(y_test, pt),
                "ece": ece(y_test, pt),
                "mean_pred": float(pt.mean()),
            }
        )
        cals[m] = c
    return pd.DataFrame(rows), cals
