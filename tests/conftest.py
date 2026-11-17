import pandas as pd
import pytest

from credit_risk import pipeline


@pytest.fixture(scope="session")
def mini_run(tmp_path_factory):
    """Tiny end-to-end run (4,000 synthetic applicants, no tuning, no plots, no MLflow)."""
    base = tmp_path_factory.mktemp("run")
    res = pipeline.run(
        "A",
        "synthetic",
        n_rows=4000,
        n_trials=0,
        reports_dir=base / "reports",
        artifacts_dir=base / "artifacts",
        use_mlflow=False,
        n_boot=20,
        quiet=True,
        make_plots=False,
    )
    return {"res": res, "bundle": base / "artifacts" / "model_bundle_A.joblib", "base": base}


@pytest.fixture()
def toy():
    """Small hand-made application frame with the quirks of the real data."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 0],
            "CODE_GENDER": ["F", "M", "XNA", "F"],
            "DAYS_BIRTH": [-10957, -14610, -18262, -21915],  # 30, 40, 50, 60 years
            "DAYS_EMPLOYED": [-1000, 365243, -3000, -500],
            "AMT_INCOME_TOTAL": [100000, 50000, 0, 200000],
            "AMT_CREDIT": [300000, 150000, 400000, 100000],
            "AMT_ANNUITY": [15000, 12000, 20000, 5000],
            "AMT_GOODS_PRICE": [270000, 150000, 400000, 90000],
            "CNT_FAM_MEMBERS": [2, 3, 1, 4],
            "EXT_SOURCE_1": [0.5, None, 0.7, None],
            "EXT_SOURCE_2": [0.4, 0.3, 0.8, 0.6],
            "EXT_SOURCE_3": [None, 0.2, 0.6, 0.5],
            "ORGANIZATION_TYPE": ["Business", "XNA", "School", "Business"],
        }
    )
