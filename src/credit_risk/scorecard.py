"""Traditional scorecard: supervised monotonic binning -> WoE/IV -> logistic regression -> points.

Convention: bad = TARGET 1. WoE = ln(share of goods / share of bads) per bin, so HIGHER WoE = SAFER.
Hence a sensible logistic regression on WoE has NEGATIVE coefficients for P(bad).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as C

OTHER, MISSING = "__OTHER__", "__MISSING__"


def _woe_iv(good: np.ndarray, bad: np.ndarray, tot_good: float, tot_bad: float, smooth: float = 0.5):
    g = (good + smooth) / (tot_good + smooth * len(good))
    b = (bad + smooth) / (tot_bad + smooth * len(bad))
    woe = np.log(g / b)
    return woe, (g - b) * woe


class WoEBinner:
    """One variable. Numeric -> monotone bins (pool-adjacent-violators); categorical -> grouped categories."""

    def __init__(self, kind: str, n_prebins: int = 20, min_share: float = 0.05, monotone: str = "auto"):
        self.kind, self.n_prebins, self.min_share, self.monotone = kind, n_prebins, min_share, monotone
        self.edges_: np.ndarray | None = None
        self.cat_map_: dict[str, int] = {}
        self.table_: pd.DataFrame | None = None
        self.iv_: float = 0.0
        self.woe_: np.ndarray = np.array([])
        self.missing_idx_: int | None = None
        self.other_idx_: int | None = None

    # ---- fitting ------------------------------------------------------------------------
    def fit(self, x: pd.Series, y: pd.Series) -> WoEBinner:
        y = np.asarray(y).astype(int)
        n = len(y)
        miss = x.isna().to_numpy()
        if self.kind == "numeric":
            idx, labels, is_missing = self._fit_numeric(x.to_numpy(float), y, miss, n)
        else:
            idx, labels, is_missing = self._fit_categorical(x, y, miss, n)
        k = len(labels)
        good = np.bincount(idx, weights=(y == 0), minlength=k)
        bad = np.bincount(idx, weights=(y == 1), minlength=k)
        woe, iv = _woe_iv(good, bad, good.sum(), bad.sum())
        empty = (good + bad) == 0  # bin never seen in train (e.g. no missing values): neutral, not extreme
        woe, iv = np.where(empty, 0.0, woe), np.where(empty, 0.0, iv)
        self.woe_, self.iv_ = woe, float(iv.sum())
        self.table_ = pd.DataFrame(
            {
                "bin": labels,
                "count": (good + bad).astype(int),
                "share": (good + bad) / n,
                "bad": bad.astype(int),
                "bad_rate": bad / np.maximum(good + bad, 1),
                "woe": woe,
                "iv": iv,
            }
        )
        return self

    def _fit_numeric(self, v, y, miss, n):
        nn = ~miss
        uniq = np.unique(v[nn])
        if len(uniq) <= 1:
            edges = np.array([])
        else:
            qs = np.unique(np.quantile(v[nn], np.linspace(0, 1, self.n_prebins + 1)[1:-1]))
            edges = qs[(qs > uniq.min())]
        bins = np.searchsorted(edges, v[nn], side="right")
        k = len(edges) + 1
        good = np.bincount(bins, weights=(y[nn] == 0), minlength=k)
        bad = np.bincount(bins, weights=(y[nn] == 1), minlength=k)
        groups = [[i] for i in range(k)]  # each group = list of contiguous prebins
        gg, bb = list(good), list(bad)

        def rate(i):
            return bb[i] / max(gg[i] + bb[i], 1)

        sign = self.monotone
        if sign == "auto":
            r = [rate(i) for i in range(len(gg))]
            sign = (
                "increasing" if len(r) < 2 or np.corrcoef(np.arange(len(r)), r)[0, 1] >= 0 else "decreasing"
            )
        changed = True
        while changed and len(gg) > 1:  # pool adjacent violators
            changed = False
            for i in range(len(gg) - 1):
                bad_order = rate(i) >= rate(i + 1) if sign == "increasing" else rate(i) <= rate(i + 1)
                if bad_order:
                    self._merge(groups, gg, bb, i)
                    changed = True
                    break
        nn_total = nn.sum()
        while len(gg) > 1:  # enforce minimum bin size by merging the smallest into its closest neighbour
            sizes = [(gg[i] + bb[i]) / max(n, 1) for i in range(len(gg))]
            j = int(np.argmin(sizes))
            if sizes[j] >= self.min_share:
                break
            if j == 0:
                nb = 1
            elif j == len(gg) - 1:
                nb = j - 1
            else:
                nb = j - 1 if abs(rate(j) - rate(j - 1)) <= abs(rate(j) - rate(j + 1)) else j
            self._merge(groups, gg, bb, min(j, nb) if nb != j else j)
        # final edges = upper boundaries of each group except the last
        cut_idx = [grp[-1] for grp in groups[:-1]]
        self.edges_ = np.array([edges[i] for i in cut_idx]) if len(edges) else np.array([])
        # assign final bin index per row
        final = np.searchsorted(self.edges_, v, side="right")
        n_num = len(self.edges_) + 1
        final = np.where(miss, n_num, final)
        labels = self._numeric_labels()
        self.missing_idx_ = n_num
        labels.append(MISSING)
        _ = nn_total
        return final.astype(int), labels, miss

    @staticmethod
    def _merge(groups, gg, bb, i):
        groups[i] = groups[i] + groups[i + 1]
        gg[i] += gg[i + 1]
        bb[i] += bb[i + 1]
        del groups[i + 1], gg[i + 1], bb[i + 1]

    def _numeric_labels(self):
        e = self.edges_
        if len(e) == 0:
            return ["(-inf, inf)"]
        lab = [f"(-inf, {e[0]:.4g}]"]
        lab += [f"({e[i]:.4g}, {e[i + 1]:.4g}]" for i in range(len(e) - 1)]
        lab.append(f"({e[-1]:.4g}, inf)")
        return lab

    def _fit_categorical(self, x, y, miss, n):
        s = x.astype("object").where(~miss, MISSING).astype(str)
        df = pd.DataFrame({"c": s.to_numpy(), "y": y})
        agg = df.groupby("c")["y"].agg(["size", "sum"])
        share = agg["size"] / n
        small = share[(share < self.min_share) & (agg.index != MISSING)].index
        df["c"] = df["c"].where(~df["c"].isin(small), OTHER)
        agg = df.groupby("c")["y"].agg(["size", "sum"])
        agg["rate"] = agg["sum"] / agg["size"]
        order = agg.sort_values("rate").index.tolist()
        self.cat_map_ = {c: i for i, c in enumerate(order)}
        self.other_idx_ = self.cat_map_.get(OTHER)
        self.missing_idx_ = self.cat_map_.get(MISSING)
        idx = df["c"].map(self.cat_map_).to_numpy(int)
        return idx, order, miss

    # ---- transform ------------------------------------------------------------------------
    def bin_index(self, x: pd.Series) -> np.ndarray:
        miss = x.isna().to_numpy()
        if self.kind == "numeric":
            idx = np.searchsorted(self.edges_, x.to_numpy(float), side="right")
            return np.where(miss, len(self.edges_) + 1, idx).astype(int)
        s = x.astype("object").where(~miss, MISSING).astype(str)
        default = self.other_idx_ if self.other_idx_ is not None else self.missing_idx_
        return s.map(self.cat_map_).fillna(-1 if default is None else default).to_numpy(int)

    def transform(self, x: pd.Series) -> np.ndarray:
        idx = self.bin_index(x)
        woe = self.woe_
        return np.where(idx < 0, 0.0, woe[np.clip(idx, 0, len(woe) - 1)])


# ------------------------------------------------------------------------------------------
def fit_binners(train: pd.DataFrame, y: pd.Series, cols: list[str], **kw) -> dict[str, WoEBinner]:
    from .features import categorical_columns

    cats = set(categorical_columns(train[cols]))
    out = {}
    for c in cols:
        kind = "categorical" if c in cats else "numeric"
        if kind == "numeric" and train[c].nunique(dropna=True) < 2:
            continue
        out[c] = WoEBinner(kind, **kw).fit(train[c], y)
    return out


def iv_table(binners: dict[str, WoEBinner]) -> pd.DataFrame:
    def label(iv):
        return (
            "useless"
            if iv < 0.02
            else "weak"
            if iv < 0.1
            else "medium"
            if iv < 0.3
            else "strong"
            if iv < 0.5
            else "suspicious (check leakage)"
        )

    t = pd.DataFrame({"feature": list(binners), "iv": [b.iv_ for b in binners.values()]})
    t["strength"] = t["iv"].map(label)
    return t.sort_values("iv", ascending=False).reset_index(drop=True)


def woe_matrix(
    df: pd.DataFrame, binners: dict[str, WoEBinner], cols: list[str] | None = None
) -> pd.DataFrame:
    cols = cols or list(binners)
    return pd.DataFrame({c: binners[c].transform(df[c]) for c in cols}, index=df.index)


def drop_correlated(W: pd.DataFrame, iv: dict[str, float], thresh: float = C.CORR_MAX) -> list[str]:
    """Greedy: walk features by IV desc; keep one only if |corr| with all kept < thresh."""
    kept: list[str] = []
    corr = W.corr().abs()
    for f in sorted(W.columns, key=lambda c: -iv[c]):
        if all(corr.loc[f, k] < thresh for k in kept):
            kept.append(f)
    return kept


@dataclass
class Scorecard:
    binners: dict[str, WoEBinner]
    features: list[str]
    model: LogisticRegression
    base_score: float = C.BASE_SCORE
    base_odds: float = C.BASE_ODDS
    pdo: float = C.PDO
    points_table_: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def logit_bad(self, df: pd.DataFrame) -> np.ndarray:
        W = woe_matrix(df, self.binners, self.features).to_numpy()
        return self.model.intercept_[0] + W @ self.model.coef_[0]

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit_bad(df)))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Points = offset + factor * ln(good:bad odds)."""
        return self.offset + self.factor * (-self.logit_bad(df))

    def build_points_table(self) -> pd.DataFrame:
        k = len(self.features)
        a = self.model.intercept_[0]
        rows = []
        for f, beta in zip(self.features, self.model.coef_[0], strict=True):
            t = self.binners[f].table_
            for _, r in t.iterrows():
                pts = -(beta * r["woe"] + a / k) * self.factor + self.offset / k
                rows.append(
                    {
                        "variable": f,
                        "bin": r["bin"],
                        "share": r["share"],
                        "bad_rate": r["bad_rate"],
                        "woe": r["woe"],
                        "points": round(float(pts), 1),
                    }
                )
        self.points_table_ = pd.DataFrame(rows)
        return self.points_table_


def fit_scorecard(
    train: pd.DataFrame,
    y: pd.Series,
    candidate_cols: list[str],
    *,
    c_reg: float = 1.0,
    min_share: float = 0.05,
):
    """Full recipe (Step 6). Returns (Scorecard, iv_table, report dict)."""
    binners = fit_binners(train, y, candidate_cols, min_share=min_share)
    ivt = iv_table(binners)
    ivd = dict(zip(ivt["feature"], ivt["iv"], strict=True))
    selected = [f for f in ivt["feature"] if C.IV_MIN <= ivd[f] < C.IV_SUSPICIOUS]
    suspicious = [f for f in ivt["feature"] if ivd[f] >= C.IV_SUSPICIOUS]
    W = woe_matrix(train, binners, selected)
    selected = drop_correlated(W, ivd)
    dropped_sign: list[str] = []
    while True:  # drop features whose coefficient has the wrong sign (WoE must reduce risk)
        lr = LogisticRegression(C=c_reg, max_iter=1000)
        lr.fit(W[selected], y)
        bad = [f for f, b in zip(selected, lr.coef_[0], strict=True) if b > 0]
        if not bad:
            break
        worst = min(bad, key=lambda f: ivd[f])
        selected.remove(worst)
        dropped_sign.append(worst)
    sc = Scorecard(binners=binners, features=selected, model=lr)
    sc.build_points_table()
    report = {
        "n_candidates": len(candidate_cols),
        "n_after_iv": int(((ivt["iv"] >= C.IV_MIN) & (ivt["iv"] < C.IV_SUSPICIOUS)).sum()),
        "n_final": len(selected),
        "suspicious_iv": suspicious,
        "dropped_wrong_sign": dropped_sign,
        "coefficients": dict(zip(selected, map(float, lr.coef_[0]), strict=True)),
    }
    return sc, ivt, report


def pd_to_score(p, base_score=C.BASE_SCORE, base_odds=C.BASE_ODDS, pdo=C.PDO):
    """Map any PD (e.g. calibrated GBM) onto the same points scale as the scorecard."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    return base_score - factor * np.log(base_odds) + factor * np.log((1 - p) / p)
