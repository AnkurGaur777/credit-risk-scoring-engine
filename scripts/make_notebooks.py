"""Builds the 7 course notebooks (thin: logic lives in src/credit_risk) and executes them.
Usage: python scripts/make_notebooks.py [--no-exec]"""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NB = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
from credit_risk import config as C, data as D, features as F, scorecard as S, models as M
from credit_risk import calibration as CAL, business as B, fairness as FAIR, stability as ST
R = lambda s: json.load(open(C.REPORTS / f"metrics_{s}.json"))
def app():
    t = pd.read_parquet(C.ROOT / "data/synthetic/application_train.parquet")  # use data/raw + D.read_table for real data
    return F.engineer_application(D.clean_application(t)), t
"""
NOTEBOOKS = {
    "01_eda_and_data_audit": [
        (
            "md",
            "# 01 - EDA and data audit\nTarget rate, missingness, sentinels. *Synthetic stand-in data unless you switched to the real files.*",
        ),
        (
            "code",
            "df, raw = app()\naudit = D.audit_application(raw)\nprint({k: audit[k] for k in ['rows','target_rate','imbalance_ratio','sentinel_days_employed','xna_counts']})",
        ),
        (
            "code",
            "raw.isna().mean().sort_values(ascending=False).head(15).plot.barh(figsize=(6,4)); plt.title('Missing share'); plt.show()",
        ),
        (
            "code",
            "df.groupby(pd.qcut(df['EXT_SOURCE_2'], 10))[C.TARGET].mean().plot(marker='o'); plt.title('Default rate by EXT_SOURCE_2 decile'); plt.show()",
        ),
        (
            "md",
            "**Takeaways:** ~8% positives (use PR-AUC, not accuracy); missing external scores are informative; `DAYS_EMPLOYED` sentinel must be neutralised.",
        ),
    ],
    "02_feature_engineering": [
        (
            "md",
            "# 02 - Feature engineering\nEvery engineered feature, how it is built and why (see `docs/feature_dictionary.md`).",
        ),
        ("code", "df, _ = app()\npd.DataFrame(F.FEATURE_DICTIONARY, index=['how','why']).T.head(20)"),
        (
            "code",
            "df.groupby(pd.qcut(df['ANNUITY_INCOME_RATIO'], 10))[C.TARGET].mean().plot(marker='o'); plt.title('Default rate by payment burden decile'); plt.show()",
        ),
        (
            "md",
            "Stage B adds aggregates of bureau / previous applications / instalments / POS / cards, using only records dated before the application (`features.before_application`).",
        ),
    ],
    "03_scorecard_woe": [
        (
            "md",
            "# 03 - WoE scorecard\nMonotone binning -> WoE/IV -> logistic regression -> points. WoE = ln(good/bad), so higher = safer and coefficients are negative.",
        ),
        (
            "code",
            "df, _ = app(); tr, va, te = D.split_data(df)\nenc = F.CategoryEncoder().fit(tr, F.categorical_columns(tr[F.model_feature_columns(tr)]))\ntr, va = enc.transform(tr), enc.transform(va)\nfeats = F.model_feature_columns(tr)\nsc, ivt, rep = S.fit_scorecard(tr, tr[C.TARGET], feats)\nivt.head(12)",
        ),
        (
            "code",
            "b = sc.binners['EXT_SOURCE_MEAN'].table_; display(b[['bin','share','bad_rate','woe']])\nb.plot(x='bin', y='bad_rate', marker='o', figsize=(7,3)); plt.xticks(rotation=60); plt.show()",
        ),
        (
            "code",
            "from sklearn.metrics import roc_auc_score\nprint('valid AUC', roc_auc_score(va[C.TARGET], sc.predict_pd(va)), '| final vars', rep['n_final'])\nsc.points_table_.head(15)",
        ),
    ],
    "04_gbm_challenger_and_calibration": [
        ("md", "# 04 - GBM challenger, imbalance, calibration"),
        ("code", "m = R('A'); pd.DataFrame(m['imbalance_comparison'])"),
        (
            "code",
            "display(pd.DataFrame(m['calibration']['table'])); display(Image(str(C.FIGURES / 'reliability_A.png')))",
        ),
        (
            "md",
            "A class-weighted GBM ranks fine but over-predicts risk (mean PD far above the base rate). Platt/isotonic on the validation set fixes it without changing the ranking.",
        ),
        ("code", "display(Image(str(C.FIGURES / 'roc_A.png')))"),
    ],
    "05_shap_explainability": [
        ("md", "# 05 - SHAP and reason codes"),
        ("code", "m = R('A'); display(Image(str(C.FIGURES / 'shap_summary.png')))"),
        ("code", "pd.DataFrame(m['explain']['top10'])"),
        (
            "code",
            "for ex in m['explain']['examples']:\n    print(ex['label'], f\"PD={ex['pd']:.1%} score={ex['score']:.0f}\")\n    for r in ex['reason_codes']: print('   -', r['reason'], r['impact_log_odds'])",
        ),
        (
            "md",
            "Spearman rank agreement of mean |SHAP| vs scorecard IV is in the metrics JSON (`explain.spearman_shap_vs_iv`).",
        ),
    ],
    "06_business_cutoff_and_fairness": [
        ("md", "# 06 - Business cut-off and fairness\nEconomics are **illustrative assumptions**."),
        ("code", "m = R('A'); pd.DataFrame(m['business']['assumptions'], index=[0]).T"),
        (
            "code",
            "display(Image(str(C.FIGURES / 'profit_A.png'))); print(m['business']['test'], m['business']['baselines_test'])",
        ),
        ("code", "pd.DataFrame(m['business']['sensitivity'])"),
        (
            "code",
            "for g, f in m['fairness'].items():\n    print(g, f['gaps']); display(pd.DataFrame(f['table']).round(3))",
        ),
    ],
    "07_stability_psi_and_monitoring": [
        ("md", "# 07 - PSI and monitoring"),
        (
            "code",
            "m = R('A'); print('score PSI valid->test', m['stability']['score_psi_valid_vs_test'])\npd.DataFrame(m['stability']['feature_psi_train_vs_test'])",
        ),
        (
            "code",
            "s = m['stability']['shift_exercise_young_vs_all']; print('shifted population (under 35): score PSI', s['score_psi']); pd.DataFrame(s['features'])",
        ),
        ("code", "ST.monitoring_plan()"),
    ],
}

for name, cells in NOTEBOOKS.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell(SETUP)] + [
        nbf.v4.new_markdown_cell(src) if k == "md" else nbf.v4.new_code_cell(src) for k, src in cells
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if "--no-exec" not in sys.argv:
        NotebookClient(
            nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(NB)}}
        ).execute()
    nbf.write(nb, NB / f"{name}.ipynb")
    print("wrote", name)
