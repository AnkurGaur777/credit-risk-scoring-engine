import numpy as np
import pandas as pd

from credit_risk import config as C
from credit_risk import data as D
from credit_risk import features as F


def test_sentinel_and_xna_and_age(toy):
    out = D.clean_application(toy)
    assert out["DAYS_EMPLOYED"].isna().tolist() == [False, True, False, False]
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1, 0, 0]
    assert out["CODE_GENDER"].isna().tolist() == [False, False, True, False]
    assert out["ORGANIZATION_TYPE"].isna().tolist() == [False, True, False, False]
    assert np.allclose(out["AGE_YEARS"], [30, 40, 50, 60], atol=0.1)
    assert np.isnan(out.loc[2, "AMT_INCOME_TOTAL"])  # income of 0 is impossible -> missing


def test_ratio_features(toy):
    f = F.engineer_application(D.clean_application(toy))
    assert np.isclose(f.loc[0, "CREDIT_INCOME_RATIO"], 3.0)
    assert np.isclose(f.loc[0, "ANNUITY_INCOME_RATIO"], 0.15)
    assert np.isclose(f.loc[0, "ANNUITY_CREDIT_RATIO"], 0.05)
    assert np.isclose(f.loc[0, "INCOME_PER_FAMILY_MEMBER"], 50000)
    assert f["EXT_SOURCE_1_MISSING"].tolist() == [0, 1, 0, 1]
    assert np.isclose(f.loc[0, "EXT_SOURCE_MEAN"], 0.45)  # mean of available scores
    assert np.isclose(f.loc[0, "EXT_SOURCE_MIN"], 0.4)


def test_gender_never_in_model_and_age_excluded_by_default(toy):
    f = F.engineer_application(D.clean_application(toy))
    cols = F.model_feature_columns(f)
    assert "CODE_GENDER" not in cols
    assert C.TARGET not in cols and C.ID_COL not in cols
    if not C.ALLOW_AGE:
        assert not set(C.AGE_DERIVED) & set(cols)


def test_split_is_stratified_disjoint_and_reproducible():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": range(5000),
            "TARGET": (rng.random(5000) < 0.08).astype(int),
            "x": rng.normal(size=5000),
        }
    )
    tr, va, te = D.split_data(df)
    assert len(tr) + len(va) + len(te) == 5000
    assert not (set(tr.SK_ID_CURR) & set(va.SK_ID_CURR)) and not (set(tr.SK_ID_CURR) & set(te.SK_ID_CURR))
    assert abs(tr.TARGET.mean() - te.TARGET.mean()) < 0.005
    assert D.split_data(df)[2].SK_ID_CURR.tolist() == te.SK_ID_CURR.tolist()


def test_category_encoder_fit_on_train_only():
    train = pd.DataFrame({"c": ["a"] * 90 + ["b"] * 9 + ["rare"] * 1})
    enc = F.CategoryEncoder(min_share=0.05).fit(train, ["c"])
    out = enc.transform(pd.DataFrame({"c": ["a", "rare", "never_seen", None]}))
    assert out["c"].astype(object).tolist()[:3] == ["a", F.RARE_LABEL, F.RARE_LABEL]
    assert pd.isna(out["c"].iloc[3])


def test_leakage_guard_drops_future_records():
    inst = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1],
            "DAYS_INSTALMENT": [-30, -10, 15],
            "DAYS_ENTRY_PAYMENT": [-31, -10, 20],
            "AMT_INSTALMENT": [100.0, 100, 100],
            "AMT_PAYMENT": [100.0, 100, 100],
        }
    )
    agg = F.aggregate_installments(inst)
    assert np.isclose(agg.loc[1, "INST_LATE_SHARE"], 0.0)  # future (positive-day) instalment ignored
    assert len(F.before_application(inst, "DAYS_INSTALMENT")) == 2
    pos = pd.DataFrame({"SK_ID_CURR": [1, 1], "MONTHS_BALANCE": [-1.0, 0.0], "SK_DPD": [0, 99]})
    assert F.aggregate_pos(pos).loc[1, "POS_MAX_DPD"] == 0  # month 0 is not strictly before the application


def test_stage_b_aggregation_shapes():
    from credit_risk import synthetic

    t = synthetic.generate(1500, seed=1)
    agg = F.build_stage_b(t)
    assert agg.index.is_unique and agg.index.name == C.ID_COL
    assert {
        "BUREAU_COUNT",
        "PREV_REFUSED_RATIO",
        "INST_LATE_SHARE",
        "POS_DPD_MONTHS",
        "CC_UTILISATION_MEAN",
    } <= set(agg.columns)
    merged = F.merge_stage_b(F.engineer_application(D.clean_application(t["application_train"])), agg)
    assert len(merged) == len(t["application_train"])
    assert (merged["PREV_COUNT"] >= 0).all()
