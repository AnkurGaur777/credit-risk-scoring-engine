"""Figures saved as PNGs (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .calibration import reliability_table  # noqa: E402


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def roc_compare(y, preds: dict, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in preds.items():
        f, t, _ = roc_curve(y, p)
        ax.plot(f, t, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC on test set")
    ax.legend()
    return _save(fig, path)


def reliability(y, preds: dict, path, title="Reliability (test set)"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in preds.items():
        t = reliability_table(y, p, 10)
        ax.plot(t["mean_pred"], t["observed"], marker="o", label=name)
    m = max(0.05, max(float(np.max(reliability_table(y, p, 10)["mean_pred"])) for p in preds.values()))
    ax.plot([0, m], [0, m], "k--", lw=0.8, label="perfect")
    ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate", title=title)
    ax.legend()
    return _save(fig, path)


def profit_plot(curve, cutoff_rate, path, title="Expected profit vs approval rate (validation)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["approval_rate"], curve["profit"] / 1e6)
    ax.axvline(cutoff_rate, color="red", ls="--", label=f"best: approve {cutoff_rate:.0%}")
    ax.set(xlabel="Approval rate (safest first)", ylabel="Profit (millions, currency units)", title=title)
    ax.legend()
    return _save(fig, path)


def band_plot(bands, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bands["band"].astype(str), bands["bad_rate"] * 100)
    ax.set(xlabel="Score band (1 = riskiest)", ylabel="Bad rate %", title="Bad rate by score band (test)")
    return _save(fig, path)


def iv_plot(ivt, path, top=15):
    t = ivt.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(t["feature"], t["iv"])
    ax.set(xlabel="Information Value", title="Top features by IV (train)")
    return _save(fig, path)


def psi_plot(t, path, top=12):
    t = t.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(t["feature"], t["psi"])
    ax.axvline(0.1, color="orange", ls="--")
    ax.axvline(0.25, color="red", ls="--")
    ax.set(xlabel="PSI", title="Feature PSI (train vs test)")
    return _save(fig, path)


def missing_plot(miss, path, top=20):
    s = miss.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(s.index, s.values * 100)
    ax.set(xlabel="% missing", title="Missingness (top 20 columns)")
    return _save(fig, path)


def fairness_plot(tables: dict, path):
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4), squeeze=False)
    for ax, (name, t) in zip(axes[0], tables.items(), strict=True):
        ax.bar(t["group"].astype(str), t["approval_rate"] * 100)
        ax.set(title=f"Approval rate by {name}", ylabel="%")
        ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)
