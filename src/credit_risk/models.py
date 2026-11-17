"""Metrics, imbalance comparison, LightGBM / XGBoost training and Optuna tuning."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from . import config as C
from . import tracking

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning)


# ---- metrics -----------------------------------------------------------------------------
def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def evaluate(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_stat(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }


def bootstrap_ci(y, p, n_boot: int = 200, seed: int = C.SEED, metrics=("auc", "ks", "pr_auc")) -> dict:
    """95% bootstrap CI of ranking metrics on a (test) set."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        e = evaluate(y[i], p[i])
        for m in metrics:
            vals[m].append(e[m])
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in vals.items()}


# ---- matrices ----------------------------------------------------------------------------
def to_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Selects model columns; categoricals are already pandas 'category' (via CategoryEncoder)."""
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == "object" or str(X[c].dtype) in ("string", "str"):
            X[c] = X[c].astype("category")
    return X


# ---- LightGBM ----------------------------------------------------------------------------
DEFAULT_LGB = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    n_jobs=-1,
    random_state=C.SEED,
    verbose=-1,
)


def fit_lgbm(Xtr, ytr, Xva, yva, params: dict | None = None, class_weight: str | None = None):
    import lightgbm as lgb

    p = {**DEFAULT_LGB, **(params or {})}
    m = lgb.LGBMClassifier(class_weight=class_weight, **p)
    m.fit(
        Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    return m


def compare_imbalance(Xtr, ytr, Xva, yva, params: dict | None = None) -> pd.DataFrame:
    """Step 7: compare, don't assume. none vs class weights vs random under-sampling."""
    rows, models = [], {}
    for name in ("none", "class_weight", "undersample"):
        if name == "undersample":
            rng = np.random.default_rng(C.SEED)
            pos = np.flatnonzero(ytr.to_numpy() == 1)
            neg = np.flatnonzero(ytr.to_numpy() == 0)
            neg = rng.choice(neg, size=min(len(neg), len(pos) * 4), replace=False)
            idx = np.sort(np.concatenate([pos, neg]))
            m = fit_lgbm(Xtr.iloc[idx], ytr.iloc[idx], Xva, yva, params)
        else:
            m = fit_lgbm(Xtr, ytr, Xva, yva, params, "balanced" if name == "class_weight" else None)
        p = m.predict_proba(Xva)[:, 1]
        e = evaluate(yva, p)
        rows.append(
            {
                "strategy": name,
                **e,
                "best_iter": int(m.best_iteration_ or m.n_estimators_),
                "mean_pred": float(p.mean()),
            }
        )
        models[name] = m
    return pd.DataFrame(rows), models


def tune_lgbm(Xtr, ytr, Xva, yva, n_trials: int = 15, class_weight: str | None = None, seed: int = C.SEED):
    """Optuna search, every trial logged to MLflow as a nested run. Few trials on purpose (avoid overfitting valid)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 8, 64, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 300, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        )
        m = fit_lgbm(Xtr, ytr, Xva, yva, params, class_weight)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        with tracking.run(f"optuna_trial_{trial.number}", nested=True, params=params):
            tracking.log_metrics({"valid_auc": auc})
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


def fit_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb

    m = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        n_jobs=-1,
        random_state=C.SEED,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m
