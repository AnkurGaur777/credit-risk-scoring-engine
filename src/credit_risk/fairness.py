"""Fairness audit by group (computed with plain pandas, no extra dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def age_band(age) -> pd.Series:
    return pd.cut(
        pd.Series(age),
        [0, 30, 40, 50, 60, 120],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
        right=False,
    ).astype(str)


def group_metrics(group, y, pd_hat, score, cutoff: float, min_n: int = 30) -> pd.DataFrame:
    """Per group: approval rate, bad rate among approved, error rates, mean score/PD.

    false-decline rate  = declined | actually good   (customer-facing harm)
    false-approve rate  = approved | actually bad    (lender-facing harm)
    """
    d = pd.DataFrame(
        {
            "g": np.asarray(group).astype(str),
            "y": np.asarray(y),
            "pd": np.asarray(pd_hat),
            "score": np.asarray(score),
        }
    )
    d["approve"] = d["pd"] <= cutoff
    rows = []
    for g, s in d.groupby("g"):
        if len(s) < min_n:
            continue
        good, bad = s[s["y"] == 0], s[s["y"] == 1]
        rows.append(
            {
                "group": g,
                "n": len(s),
                "default_rate": s["y"].mean(),
                "approval_rate": s["approve"].mean(),
                "bad_rate_among_approved": s.loc[s["approve"], "y"].mean() if s["approve"].any() else np.nan,
                "false_decline_rate": (~good["approve"]).mean() if len(good) else np.nan,
                "false_approve_rate": bad["approve"].mean() if len(bad) else np.nan,
                "mean_pd": s["pd"].mean(),
                "mean_score": s["score"].mean(),
            }
        )
    return pd.DataFrame(rows)


def gaps(t: pd.DataFrame) -> dict:
    """Max-min gaps and the disparate-impact ratio (min approval rate / max approval rate)."""
    if t.empty:
        return {}
    ar = t["approval_rate"]
    return {
        "approval_rate_gap": float(ar.max() - ar.min()),
        "disparate_impact_ratio": float(ar.min() / ar.max()) if ar.max() > 0 else float("nan"),
        "false_decline_gap": float(t["false_decline_rate"].max() - t["false_decline_rate"].min()),
        "false_approve_gap": float(t["false_approve_rate"].max() - t["false_approve_rate"].min()),
        "mean_score_gap": float(t["mean_score"].max() - t["mean_score"].min()),
        "four_fifths_rule_flag": bool(ar.min() / ar.max() < 0.8) if ar.max() > 0 else False,
    }
