import numpy as np
import pandas as pd

from credit_risk import config as C
from credit_risk import scorecard as S


def _data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c", "d"], n, p=[0.5, 0.3, 0.17, 0.03])
    p = 1 / (1 + np.exp(-(-2.5 + 0.45 * x + 0.5 * (cat == "c"))))
    return pd.DataFrame({"x": x, "cat": cat, "noise": rng.normal(size=n)}), pd.Series(
        (rng.random(n) < p).astype(int)
    )


def test_numeric_binning_is_monotone_and_woe_sign():
    X, y = _data()
    b = S.WoEBinner("numeric").fit(X["x"], y)
    rates = b.table_.loc[b.table_["bin"] != S.MISSING, "bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= -1e-12) or np.all(np.diff(rates) <= 1e-12)
    t = b.table_[b.table_["bin"] != S.MISSING]
    assert t["woe"].iloc[0] > t["woe"].iloc[-1]  # lowest-x bin is safest -> highest WoE (WoE = ln good/bad)
    assert b.iv_ > 0.1


def test_missing_gets_own_bin_and_rare_categories_group():
    X, y = _data()
    X.loc[:300, "x"] = np.nan
    b = S.WoEBinner("numeric").fit(X["x"], y)
    assert S.MISSING in b.table_["bin"].tolist()
    cb = S.WoEBinner("categorical", min_share=0.05).fit(X["cat"], y)
    assert S.OTHER in cb.table_["bin"].tolist() and "d" not in cb.table_["bin"].tolist()
    assert np.isfinite(cb.transform(pd.Series(["a", "zzz", None]))).all()  # unseen/missing never crash


def test_noise_variable_has_low_iv():
    X, y = _data()
    ivt = S.iv_table(S.fit_binners(X, y, ["x", "noise"]))
    assert ivt.set_index("feature").loc["noise", "iv"] < 0.02 < ivt.set_index("feature").loc["x", "iv"]


def test_score_scaling_conventions():
    X, y = _data()
    sc, _, rep = S.fit_scorecard(X, y, ["x", "cat", "noise"])
    assert all(c < 0 for c in rep["coefficients"].values())
    # at good:bad odds == BASE_ODDS the score equals BASE_SCORE; doubling the odds adds PDO points
    p_base = 1 / (1 + C.BASE_ODDS)
    assert np.isclose(S.pd_to_score(p_base), C.BASE_SCORE, atol=1e-6)
    p_double = 1 / (1 + 2 * C.BASE_ODDS)
    assert np.isclose(S.pd_to_score(p_double) - S.pd_to_score(p_base), C.PDO, atol=1e-6)
    # sum of points table entries reproduces the score
    pts = sc.points_table_
    row = X.iloc[[0]]
    total = 0.0
    for f in sc.features:
        idx = sc.binners[f].bin_index(row[f])[0]
        total += sc.binners[f].table_.index[idx] * 0 + pts[pts.variable == f].iloc[idx]["points"]
    assert abs(total - sc.score(row)[0]) < 0.6 * len(sc.features)  # rounding to 0.1 per bin


def test_scorecard_ranks_better_than_chance():
    from sklearn.metrics import roc_auc_score

    X, y = _data()
    sc, _, _ = S.fit_scorecard(X, y, ["x", "cat", "noise"])
    assert roc_auc_score(y, sc.predict_pd(X)) > 0.60
