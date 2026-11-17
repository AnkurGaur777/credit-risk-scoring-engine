"""Turn PDs into money: profit-maximising cut-off, baselines, score bands, sensitivity.

All economics are ILLUSTRATIVE assumptions, stated in a table and stress-tested.
Revenue on a good approved loan = margin_rate * AMT_CREDIT; loss on a defaulted approved loan = lgd * AMT_CREDIT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    margin_rate: float = 0.10  # interest margin over the loan life, as a share of the loan amount
    lgd: float = 0.70  # loss given default, share of loan amount lost
    cost_per_application: float = 0.0  # optional operating cost per approved loan

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    "Revenue on a repaid loan",
                    f"{self.margin_rate:.0%} of loan amount",
                    "Illustrative margin over the term",
                ),
                (
                    "Loss on a defaulted loan",
                    f"{self.lgd:.0%} of loan amount (LGD)",
                    "Illustrative; real LGD depends on collateral/recoveries",
                ),
                (
                    "Rejected good customer",
                    "Lost margin (no revenue)",
                    "Opportunity cost is implicit: rejected loans earn 0",
                ),
                ("Operating cost", f"{self.cost_per_application:,.0f} per approved loan", "Optional"),
            ],
            columns=["Assumption", "Value", "Note"],
        )


def loan_profit(y, amount, a: Assumptions) -> np.ndarray:
    y, amount = np.asarray(y), np.asarray(amount, float)
    return np.where(y == 0, a.margin_rate * amount, -a.lgd * amount) - a.cost_per_application


def profit_curve(pd_hat, y, amount, a: Assumptions) -> pd.DataFrame:
    """Approve the k safest applicants, for k = 0..n. Cumulative REALISED profit (uses true outcomes)."""
    order = np.argsort(pd_hat, kind="stable")
    prof = loan_profit(np.asarray(y)[order], np.asarray(amount)[order], a)
    cum = np.concatenate([[0.0], np.cumsum(prof)])
    n = len(order)
    pd_sorted = np.asarray(pd_hat)[order]
    cutoff = np.concatenate([[pd_sorted[0] - 1e-9], pd_sorted])
    return pd.DataFrame({"approval_rate": np.arange(n + 1) / n, "profit": cum, "pd_cutoff": cutoff})


def best_cutoff(pd_hat, y, amount, a: Assumptions) -> dict:
    c = profit_curve(pd_hat, y, amount, a)
    i = int(c["profit"].idxmax())
    return {
        "approval_rate": float(c.loc[i, "approval_rate"]),
        "profit": float(c.loc[i, "profit"]),
        "pd_cutoff": float(c.loc[i, "pd_cutoff"]),
    }


def profit_at_cutoff(pd_hat, y, amount, cutoff: float, a: Assumptions) -> dict:
    pd_hat, y = np.asarray(pd_hat), np.asarray(y)
    approve = pd_hat <= cutoff
    prof = loan_profit(y, amount, a)
    return {
        "profit": float(prof[approve].sum()),
        "approval_rate": float(approve.mean()),
        "bad_rate_approved": float(y[approve].mean()) if approve.any() else float("nan"),
    }


def baselines(pd_hat, y, amount, a: Assumptions, top_share: float = 0.70) -> dict:
    """Approve everyone, or the 'gut feel' rule: approve the top 70% by model score."""
    all_ = float(loan_profit(y, amount, a).sum())
    k = int(len(pd_hat) * top_share)
    order = np.argsort(pd_hat, kind="stable")[:k]
    gut = float(loan_profit(np.asarray(y)[order], np.asarray(amount)[order], a).sum())
    return {"approve_all": all_, f"approve_top_{int(top_share * 100)}pct": gut}


def sensitivity(pd_hat, y, amount, base: Assumptions, steps=(-0.2, 0.0, 0.2)) -> pd.DataFrame:
    rows = []
    for dl in steps:
        for dm in steps:
            a = Assumptions(base.margin_rate * (1 + dm), base.lgd * (1 + dl), base.cost_per_application)
            b = best_cutoff(pd_hat, y, amount, a)
            rows.append(
                {
                    "lgd_change": dl,
                    "margin_change": dm,
                    "lgd": a.lgd,
                    "margin": a.margin_rate,
                    "best_approval_rate": b["approval_rate"],
                    "best_pd_cutoff": b["pd_cutoff"],
                    "profit": b["profit"],
                }
            )
    return pd.DataFrame(rows)


def score_band_table(score, pd_hat, y, n_bands: int = 10) -> pd.DataFrame:
    """Deciles of score (band 1 = riskiest). Includes cumulative approval / bad rates and lift."""
    d = pd.DataFrame({"score": np.asarray(score), "pd": np.asarray(pd_hat), "y": np.asarray(y)})
    d["band"] = pd.qcut(d["score"].rank(method="first"), n_bands, labels=False) + 1
    g = (
        d.groupby("band")
        .agg(
            applicants=("y", "size"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            mean_pd=("pd", "mean"),
            bads=("y", "sum"),
            bad_rate=("y", "mean"),
        )
        .reset_index()
    )
    g["share"] = g["applicants"] / g["applicants"].sum()
    g = g.sort_values("band", ascending=False).reset_index(drop=True)  # safest first -> cumulative approval
    g["cum_approval_rate"] = g["share"].cumsum()
    g["cum_bad_rate"] = g["bads"].cumsum() / g["applicants"].cumsum()
    base = d["y"].mean()
    g["lift"] = g["bad_rate"] / base
    g["capture_rate_of_defaults"] = g["bads"] / d["y"].sum()
    return g.sort_values("band").reset_index(drop=True)


def assumptions_dict(a: Assumptions) -> dict:
    return asdict(a)
