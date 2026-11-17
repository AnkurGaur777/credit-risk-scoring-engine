import json

import pytest
from fastapi.testclient import TestClient

from credit_risk.api import app as appmod

GOOD = {
    "AMT_INCOME_TOTAL": 135000,
    "AMT_CREDIT": 450000,
    "AMT_ANNUITY": 22500,
    "EXT_SOURCE_2": 0.52,
    "EXT_SOURCE_3": 0.4,
    "DAYS_EMPLOYED": -1800,
    "NAME_EDUCATION_TYPE": "Higher education",
}


def test_pipeline_outputs_are_sane(mini_run):
    r = mini_run["res"]
    assert 0.60 < r["scorecard"]["test"]["auc"] < 0.90
    assert 0.60 < r["gbm"]["test"]["auc"] < 0.90  # ~0.9 would signal leakage
    assert r["scorecard"]["test"]["gini"] == pytest.approx(2 * r["scorecard"]["test"]["auc"] - 1)
    assert 0.03 < r["split"]["default_rate"][2] < 0.15
    assert "CODE_GENDER" in r["excluded_by_policy"]
    assert (mini_run["base"] / "reports" / "metrics_A.json").exists()
    assert json.loads((mini_run["base"] / "reports" / "metrics_A.json").read_text())["stage"] == "A"
    assert r["business"]["test"]["profit"] >= r["business"]["baselines_test"]["approve_all"] * 0.9


@pytest.fixture()
def client(mini_run, tmp_path):
    appmod.service.bundle_path = mini_run["bundle"]
    appmod.service.audit_log = tmp_path / "audit.jsonl"
    appmod.service.load()
    return TestClient(appmod.app)


def test_health_and_score(client, tmp_path):
    h = client.get("/health").json()
    assert h["loaded"] and h["status"] == "ok"
    r = client.post("/score", json=GOOD)
    assert r.status_code == 200
    j = r.json()
    assert 0 < j["pd"] < 1 and 200 < j["score"] < 1000 and j["decision"] in ("approve", "decline")
    assert j["model_version"] == h["model_version"] and j["latency_ms"] >= 0
    log = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    assert len(log) == 1 and "AMT_INCOME_TOTAL" not in log[0]  # audit log stores no applicant data


def test_risky_applicant_scores_worse_than_safe(client):
    safe = client.post(
        "/score", json={**GOOD, "EXT_SOURCE_2": 0.9, "EXT_SOURCE_3": 0.9, "EXT_SOURCE_1": 0.9}
    ).json()
    risky = client.post(
        "/score",
        json={**GOOD, "EXT_SOURCE_2": 0.05, "EXT_SOURCE_3": 0.05, "EXT_SOURCE_1": 0.05, "AMT_ANNUITY": 80000},
    ).json()
    assert risky["pd"] > safe["pd"] and risky["score"] < safe["score"]
    assert len(risky["reason_codes"]) >= 1


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "AMT_INCOME_TOTAL": -5},
        {**GOOD, "EXT_SOURCE_2": 1.7},
        {**GOOD, "NAME_CONTRACT_TYPE": "Mortgage"},
        {**GOOD, "CODE_GENDER": "F"},
        {"AMT_CREDIT": 1000},
    ],
)
def test_validation_rejects_bad_input(client, bad):
    assert client.post("/score", json=bad).status_code == 422


def test_503_when_model_not_loaded(client):
    saved, appmod.service.bundle = appmod.service.bundle, None
    try:
        assert client.post("/score", json=GOOD).status_code == 503
        assert client.get("/health").json()["status"] == "model_not_loaded"
    finally:
        appmod.service.bundle = saved
