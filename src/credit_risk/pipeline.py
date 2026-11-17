"""End-to-end pipeline: audit -> split -> scorecard -> GBM challenger -> calibration -> SHAP -> cut-off
-> fairness -> stability -> artifacts + reports.   `python -m credit_risk.pipeline --stage A|B`
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from . import __version__, synthetic, tracking
from . import business as B
from . import calibration as CAL
from . import config as C
from . import data as D
from . import explain as X
from . import fairness as FAIR
from . import features as F
from . import models as M
from . import plots as P
from . import scorecard as S
from . import stability as ST
from .utils import dump_json


def log(msg: str, quiet: bool = False):
    if not quiet:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tables(
    source: str, data_dir: Path | None, n_rows: int | None, stage: str
) -> dict[str, pd.DataFrame]:
    if source == "synthetic":
        d = Path(data_dir or C.ROOT / "data" / "synthetic")
        if n_rows or not (d / "application_train.parquet").exists():
            return synthetic.generate(n_rows or 20000)
        return {n: pd.read_parquet(d / f"{n}.parquet") for n in D.TABLES if (d / f"{n}.parquet").exists()}
    d = Path(data_dir or C.DATA_RAW)
    names = ["application_train"] + (D.TABLES[1:] if stage == "B" else [])
    return {n: D.read_table(n, d) for n in names if D.has_table(n, d)}


def choose_calibrator(p_valid, y_valid, seed: int = C.SEED) -> str:
    """Pick Platt vs isotonic by 2-fold cross-fitting INSIDE the validation set (test never touched)."""
    p_valid, y_valid = np.asarray(p_valid), np.asarray(y_valid)
    score = {"platt": [], "isotonic": []}
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(p_valid, y_valid):
        for m in score:
            c = CAL.Calibrator(m).fit(p_valid[tr], y_valid[tr])
            score[m].append(float(np.mean((c.transform(p_valid[te]) - y_valid[te]) ** 2)))
    return min(score, key=lambda m: np.mean(score[m]))


def run(
    stage: str = "A",
    source: str = "synthetic",
    data_dir: Path | None = None,
    n_rows: int | None = None,
    n_trials: int = 12,
    reports_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    use_mlflow: bool = True,
    n_boot: int = 200,
    quiet: bool = False,
    make_plots: bool = True,
) -> dict:
    t0 = time.time()
    reports_dir = Path(reports_dir or C.REPORTS)
    artifacts_dir = Path(artifacts_dir or C.ARTIFACTS)
    fig_dir = reports_dir / "figures"
    for d in (reports_dir, fig_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)
    tag = f"_{stage}"
    mlf = use_mlflow and tracking.setup(f"credit-risk-stage-{stage}")
    res: dict = {"stage": stage, "source": source, "version": __version__, "n_trials": n_trials}

    # ---- 1. load + audit ------------------------------------------------------------------
    log(f"loading tables ({source}, stage {stage})", quiet)
    tables = load_tables(source, data_dir, n_rows, stage)
    raw_app = tables["application_train"]
    res["audit"] = D.audit_application(raw_app)
    app = F.engineer_application(D.clean_application(raw_app))
    if stage == "B":
        agg = F.build_stage_b(tables)
        app = F.merge_stage_b(app, agg)
        res["stage_b_tables"] = [k for k in tables if k != "application_train"]
    res["audit"]["feature_dictionary_size"] = len(F.FEATURE_DICTIONARY)
    if make_plots:
        P.missing_plot(raw_app.isna().mean().sort_values(ascending=False), fig_dir / f"missingness{tag}.png")

    # ---- 2. split (stratified 60/20/20, fixed seed) + train-only encoders ----------------------
    train, valid, test = D.split_data(app)
    feats = [c for c in F.model_feature_columns(train) if train[c].notna().any()]
    cat_cols = F.categorical_columns(train[feats])
    enc = F.CategoryEncoder().fit(train, cat_cols)
    tr, va, te = (enc.transform(d) for d in (train, valid, test))
    ytr, yva, yte = tr[C.TARGET], va[C.TARGET], te[C.TARGET]
    res["split"] = {
        "train": len(tr),
        "valid": len(va),
        "test": len(te),
        "default_rate": [float(ytr.mean()), float(yva.mean()), float(yte.mean())],
    }
    res["n_features"] = len(feats)
    res["excluded_by_policy"] = sorted(
        set(C.ALWAYS_EXCLUDED) | (set() if C.ALLOW_AGE else set(C.AGE_DERIVED))
    )
    log(f"{len(feats)} model features; split {len(tr)}/{len(va)}/{len(te)}", quiet)

    # ---- 3. scorecard ---------------------------------------------------------------------
    log("fitting WoE scorecard", quiet)
    with tracking.run(f"scorecard_v1_{stage}", params={"stage": stage}):
        sc, ivt, sc_report = S.fit_scorecard(tr, ytr, feats)
        sc_valid, sc_test = sc.predict_pd(va), sc.predict_pd(te)
        res["scorecard"] = {
            "report": sc_report,
            "valid": M.evaluate(yva, sc_valid),
            "test": M.evaluate(yte, sc_test),
            "base_score": sc.base_score,
            "base_odds": sc.base_odds,
            "pdo": sc.pdo,
        }
        tracking.log_metrics(res["scorecard"]["valid"], "valid_")
    ivt.to_csv(reports_dir / f"iv_table{tag}.csv", index=False)
    sc.points_table_.to_csv(reports_dir / f"scorecard_points{tag}.csv", index=False)

    # ---- 4. GBM challenger ----------------------------------------------------------------
    log("LightGBM: imbalance comparison", quiet)
    Xtr, Xva, Xte = (M.to_matrix(d, feats) for d in (tr, va, te))
    imb, imb_models = M.compare_imbalance(Xtr, ytr, Xva, yva)
    res["imbalance_comparison"] = imb
    imb.to_csv(reports_dir / f"imbalance_comparison{tag}.csv", index=False)
    best_strategy = imb.sort_values("auc", ascending=False).iloc[0]["strategy"]
    cw = "balanced" if best_strategy == "class_weight" else None
    best_params: dict = {}
    with tracking.run(f"lgbm_v1_{stage}", params={"stage": stage, "imbalance": best_strategy}):
        if n_trials > 0:
            log(f"Optuna tuning ({n_trials} trials)", quiet)
            best_params, best_auc = M.tune_lgbm(Xtr, ytr, Xva, yva, n_trials, class_weight=cw)
            tracking.log_metrics({"best_valid_auc": best_auc})
        if best_strategy == "undersample":  # keep it simple: undersample is only a comparison arm
            cw = None
        gbm = M.fit_lgbm(Xtr, ytr, Xva, yva, best_params, cw)
        p_va_raw, p_te_raw = gbm.predict_proba(Xva)[:, 1], gbm.predict_proba(Xte)[:, 1]
        res["gbm"] = {
            "imbalance_strategy": best_strategy,
            "params": best_params,
            "best_iteration": int(gbm.best_iteration_ or 0),
            "valid": M.evaluate(yva, p_va_raw),
            "test_raw": M.evaluate(yte, p_te_raw),
        }
        tracking.log_metrics(res["gbm"]["valid"], "valid_")
    try:
        xg = M.fit_xgb(Xtr, ytr, Xva, yva)
        res["xgboost"] = {
            "valid": M.evaluate(yva, xg.predict_proba(Xva)[:, 1]),
            "test": M.evaluate(yte, xg.predict_proba(Xte)[:, 1]),
        }
    except Exception as e:  # pragma: no cover
        res["xgboost"] = {"error": str(e)}

    # ---- 5. calibration -------------------------------------------------------------------
    log("calibration", quiet)
    method = choose_calibrator(p_va_raw, yva)
    cal_table, cals = CAL.compare_calibrators(p_va_raw, yva, p_te_raw, yte)
    cal = cals[method]
    p_va, p_te = cal.transform(p_va_raw), cal.transform(p_te_raw)
    sc_cal = pd.DataFrame(
        [
            {
                "method": "scorecard (logistic, raw)",
                "brier": M.evaluate(yte, sc_test)["brier"],
                "ece": CAL.ece(yte, sc_test),
                "mean_pred": float(sc_test.mean()),
            }
        ]
    )
    res["calibration"] = {
        "chosen": method,
        "table": pd.concat([cal_table.assign(model="LightGBM"), sc_cal.assign(model="Scorecard")]),
        "observed_default_rate_test": float(yte.mean()),
    }
    res["gbm"]["test"] = M.evaluate(yte, p_te)
    res["gbm"]["test_ci"] = M.bootstrap_ci(yte, p_te, n_boot)
    res["scorecard"]["test_ci"] = M.bootstrap_ci(yte, sc_test, n_boot)
    # the guide's story: class-weighted GBMs over-predict risk. Show that arm before/after calibration too.
    cw_m = imb_models["class_weight"]
    cw_va, cw_te = cw_m.predict_proba(Xva)[:, 1], cw_m.predict_proba(Xte)[:, 1]
    cw_tab, cw_cals = CAL.compare_calibrators(cw_va, yva, cw_te, yte)
    res["calibration"]["class_weight_arm"] = cw_tab
    pd.DataFrame(res["calibration"]["table"]).to_csv(reports_dir / f"calibration{tag}.csv", index=False)
    cw_tab.to_csv(reports_dir / f"calibration_class_weight_arm{tag}.csv", index=False)
    if make_plots:
        P.roc_compare(yte, {"Scorecard": sc_test, "LightGBM": p_te}, fig_dir / f"roc{tag}.png")
        P.reliability(
            yte,
            {
                "Scorecard": sc_test,
                "LightGBM raw": p_te_raw,
                f"LightGBM {method}": p_te,
                "LightGBM class-weight raw": cw_te,
                "LightGBM class-weight + platt": cw_cals["platt"].transform(cw_te),
            },
            fig_dir / f"reliability{tag}.png",
        )
        P.iv_plot(ivt, fig_dir / f"iv{tag}.png")

    # ---- 6. explainability ---------------------------------------------------------------
    log("SHAP / reason codes", quiet)
    vals_te, _ = X.contributions(gbm, Xte)
    imp = X.global_importance(vals_te, feats)
    imp.to_csv(reports_dir / f"shap_importance{tag}.csv", index=False)
    j = imp.merge(ivt[["feature", "iv"]], on="feature", how="inner")
    rho = spearmanr(j["mean_abs_shap"], j["iv"]).correlation if len(j) > 3 else float("nan")
    res["explain"] = {"top10": imp.head(10), "spearman_shap_vs_iv": float(rho)}
    if make_plots:
        try:
            X.save_shap_plots(vals_te, Xte, fig_dir)
        except Exception as e:  # pragma: no cover
            res["explain"]["plot_error"] = str(e)
    order = np.argsort(p_te)
    examples = []
    for label, q in (("low risk", 0.10), ("medium risk", 0.60), ("high risk", 0.99)):
        i = int(order[int(q * (len(order) - 1))])
        examples.append(
            {
                "label": label,
                "pd": float(p_te[i]),
                "score": float(S.pd_to_score(p_te[i])),
                "reason_codes": X.reason_codes(vals_te[i], feats, top_k=4),
            }
        )
    res["explain"]["examples"] = examples

    # ---- 7. business cut-off ----------------------------------------------------------------
    log("profit cut-off", quiet)
    a = B.Assumptions()
    amt_va, amt_te = (
        va["AMT_CREDIT"].fillna(va["AMT_CREDIT"].median()).to_numpy(),
        te["AMT_CREDIT"].fillna(te["AMT_CREDIT"].median()).to_numpy(),
    )
    best = B.best_cutoff(p_va, yva, amt_va, a)  # chosen on VALIDATION
    on_test = B.profit_at_cutoff(p_te, yte, amt_te, best["pd_cutoff"], a)  # test used once
    base = B.baselines(p_te, yte, amt_te, a)
    all_p = base["approve_all"]
    res["business"] = {
        "assumptions": B.assumptions_dict(a),
        "cutoff_pd": best["pd_cutoff"],
        "valid_best": best,
        "test": on_test,
        "baselines_test": base,
        "uplift_vs_approve_all_abs": on_test["profit"] - all_p,
        "uplift_vs_approve_all_pct": float((on_test["profit"] - all_p) / abs(all_p)) if all_p else None,
        "sensitivity": B.sensitivity(p_va, yva, amt_va, a),
    }
    scores_te = S.pd_to_score(p_te)
    bands = B.score_band_table(scores_te, p_te, yte)
    res["business"]["score_bands"] = bands
    bands.to_csv(reports_dir / f"score_bands{tag}.csv", index=False)
    res["business"]["sensitivity"].to_csv(reports_dir / f"cutoff_sensitivity{tag}.csv", index=False)
    top20 = bands.sort_values("band").head(2)["capture_rate_of_defaults"].sum()
    res["business"]["riskiest_20pct_capture"] = float(top20)
    if make_plots:
        P.profit_plot(
            B.profit_curve(p_va, yva, amt_va, a), best["approval_rate"], fig_dir / f"profit{tag}.png"
        )
        P.band_plot(bands, fig_dir / f"score_bands{tag}.png")

    # ---- 8. fairness ------------------------------------------------------------------------
    log("fairness audit", quiet)
    fair, fair_tables = {}, {}
    groups = {
        "gender": te["CODE_GENDER"].astype(str).fillna("unknown") if "CODE_GENDER" in te else None,
        "age_band": FAIR.age_band(te["AGE_YEARS"]) if "AGE_YEARS" in te else None,
        "family_status": te["NAME_FAMILY_STATUS"].astype(str) if "NAME_FAMILY_STATUS" in te else None,
    }
    for name, g in groups.items():
        if g is None:
            continue
        t = FAIR.group_metrics(g.to_numpy(), yte, p_te, scores_te, best["pd_cutoff"])
        fair_tables[name] = t
        fair[name] = {"table": t, "gaps": FAIR.gaps(t)}
        t.to_csv(reports_dir / f"fairness_{name}{tag}.csv", index=False)
    res["fairness"] = fair
    if make_plots and fair_tables:
        P.fairness_plot(fair_tables, fig_dir / f"fairness{tag}.png")

    # ---- 9. stability -----------------------------------------------------------------------
    log("PSI", quiet)
    top_feats = [f for f in imp["feature"].head(12) if pd.api.types.is_numeric_dtype(train[f])]
    ptab = ST.psi_table(train, test, top_feats)
    shifted = test[test["AGE_YEARS"] < 35] if "AGE_YEARS" in test else test.iloc[: len(test) // 4]
    ptab_shift = ST.psi_table(test, shifted, top_feats)
    res["stability"] = {
        "score_psi_valid_vs_test": ST.psi(p_va, p_te),
        "feature_psi_train_vs_test": ptab,
        "shift_exercise_young_vs_all": {
            "n_shifted": len(shifted),
            "score_psi": ST.psi(
                p_te, cal.transform(gbm.predict_proba(M.to_matrix(enc.transform(shifted), feats))[:, 1])
            ),
            "features": ptab_shift,
        },
        "monitoring_plan": ST.monitoring_plan(),
    }
    ptab.to_csv(reports_dir / f"psi_features{tag}.csv", index=False)
    if make_plots:
        P.psi_plot(ptab, fig_dir / f"psi{tag}.png")

    # ---- 10. artifacts ----------------------------------------------------------------------
    version = f"{__version__}-{stage}-{dt.datetime.now(dt.UTC).strftime('%Y%m%d')}"
    bundle = {
        "version": version,
        "stage": stage,
        "features": feats,
        "cat_cols": cat_cols,
        "encoder": enc,
        "model": gbm,
        "calibrator": cal,
        "cutoff_pd": best["pd_cutoff"],
        "score_params": {"base_score": C.BASE_SCORE, "base_odds": C.BASE_ODDS, "pdo": C.PDO},
        "metrics": {"test": res["gbm"]["test"], "valid": res["gbm"]["valid"]},
        "trained_at": dt.datetime.now(dt.UTC).isoformat(),
        "scorecard": sc,
    }
    joblib.dump(bundle, artifacts_dir / f"model_bundle{tag}.joblib", compress=3)
    res["bundle"] = str(artifacts_dir / f"model_bundle{tag}.joblib")
    res["runtime_seconds"] = round(time.time() - t0, 1)
    dump_json(res, reports_dir / f"metrics{tag}.json")
    if mlf:
        with tracking.run(f"final_{stage}", params={"version": version}):
            tracking.log_metrics(res["gbm"]["test"], "test_gbm_")
            tracking.log_metrics(res["scorecard"]["test"], "test_scorecard_")
            tracking.log_artifact(reports_dir / f"metrics{tag}.json")
    log(f"done in {res['runtime_seconds']}s -> {reports_dir}", quiet)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the credit-risk pipeline")
    ap.add_argument("--stage", choices=["A", "B"], default="A")
    ap.add_argument(
        "--source",
        choices=["synthetic", "real"],
        default=None,
        help="default: $CR_DATA_SOURCE or 'synthetic'",
    )
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument(
        "--n-rows", type=int, default=None, help="synthetic only: generate this many applicants in memory"
    )
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    import os

    source = a.source or os.getenv("CR_DATA_SOURCE", "synthetic")
    C.ensure_dirs()
    run(a.stage, source, a.data_dir, a.n_rows, a.n_trials, use_mlflow=not a.no_mlflow)


if __name__ == "__main__":
    main()
