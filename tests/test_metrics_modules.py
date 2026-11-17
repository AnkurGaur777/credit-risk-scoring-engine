import numpy as np
import pandas as pd

from credit_risk import business as B
from credit_risk import calibration as CAL
from credit_risk import explain as X
from credit_risk import fairness as FAIR
from credit_risk import models as M
from credit_risk import stability as ST


def test_psi_identical_vs_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert ST.psi(a, rng.normal(size=20000)) < 0.02
    assert ST.psi(a, rng.normal(1.5, 1, 20000)) > 0.25
    assert (
        ST.psi_label(0.05) == "stable"
        and ST.psi_label(0.15) == "some shift"
        and ST.psi_label(0.4) == "significant shift"
    )
    assert (
        ST.psi(pd.Series([1.0, np.nan, 2, 3, 4] * 100), pd.Series([1.0, 2, 3, 4, 5] * 100)) > 0
    )  # missing handled


def test_calibration_fixes_overprediction():
    rng = np.random.default_rng(1)
    true_p = rng.beta(1, 12, 20000)
    y = (rng.random(20000) < true_p).astype(int)
    over = np.clip(true_p * 3.5, 0, 0.99)  # like a class-weighted GBM: ranks fine, probabilities inflated
    va, te = slice(0, 10000), slice(10000, None)
    before = CAL.ece(y[te], over[te])
    for m in ("platt", "isotonic"):
        c = CAL.Calibrator(m).fit(over[va], y[va])
        assert CAL.ece(y[te], c.transform(over[te])) < before / 3


def test_metrics_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1])
    assert M.evaluate(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))["auc"] == 1.0
    assert M.ks_stat(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 1.0
    assert np.isclose(M.gini(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])), 1.0)


def test_profit_curve_and_cutoff():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 10, 5000)
    y = (rng.random(5000) < p).astype(int)
    amt = np.full(5000, 100_000.0)
    a = B.Assumptions(margin_rate=0.10, lgd=0.70)
    best = B.best_cutoff(p, y, amt, a)
    base = B.baselines(p, y, amt, a)
    assert best["profit"] >= base["approve_all"] and best["profit"] >= base["approve_top_70pct"]
    assert 0 < best["approval_rate"] <= 1
    manual = B.profit_at_cutoff(p, y, amt, best["pd_cutoff"], a)
    assert np.isclose(manual["profit"], best["profit"])
    sens = B.sensitivity(p, y, amt, a)
    assert len(sens) == 9
    harsher = sens[(sens.lgd_change == 0.2) & (sens.margin_change == -0.2)].best_approval_rate.iloc[0]
    softer = sens[(sens.lgd_change == -0.2) & (sens.margin_change == 0.2)].best_approval_rate.iloc[0]
    assert harsher <= softer  # higher loss / lower margin -> be stricter


def test_score_bands_capture_and_lift():
    rng = np.random.default_rng(3)
    p = rng.beta(1, 10, 10000)
    y = (rng.random(10000) < p).astype(int)
    bands = B.score_band_table(-p, p, y)
    assert bands["applicants"].sum() == 10000
    assert bands["bad_rate"].iloc[0] > bands["bad_rate"].iloc[-1]  # band 1 is the riskiest
    assert np.isclose(bands["capture_rate_of_defaults"].sum(), 1.0)


def test_fairness_group_metrics():
    g = ["A"] * 100 + ["B"] * 100
    pd_hat = np.r_[np.full(100, 0.05), np.full(100, 0.30)]
    y = np.r_[np.zeros(95), np.ones(5), np.r_[np.zeros(70), np.ones(30)]]
    t = FAIR.group_metrics(g, y, pd_hat, 600 - 100 * pd_hat, cutoff=0.1)
    assert (
        t.set_index("group").loc["A", "approval_rate"] == 1.0
        and t.set_index("group").loc["B", "approval_rate"] == 0.0
    )
    gp = FAIR.gaps(t)
    assert gp["approval_rate_gap"] == 1.0 and gp["four_fifths_rule_flag"] is True


def test_reason_codes_only_risk_increasing_and_deduplicated():
    feats = ["EXT_SOURCE_MEAN", "EXT_SOURCE_WEIGHTED", "AMT_CREDIT", "YEARS_EMPLOYED", "NAME_EDUCATION_TYPE"]
    vals = np.array([0.5, 0.4, -0.3, 0.2, 0.0])
    rc = X.reason_codes(vals, feats, top_k=4)
    assert [r["feature"] for r in rc] == [
        "EXT_SOURCE_MEAN",
        "YEARS_EMPLOYED",
    ]  # duplicate sentence + negatives dropped
    assert all(r["impact_log_odds"] > 0 for r in rc)
