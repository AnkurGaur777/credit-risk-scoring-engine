"""Latency benchmark for /score: p50 / p95 over N requests.

python scripts/benchmark_api.py                       # in-process (FastAPI TestClient), needs artifacts/model_bundle_A.joblib
python scripts/benchmark_api.py --url http://localhost:8000 --n 1000   # against a running container
"""

import argparse
import json
import statistics
import time

import numpy as np

from credit_risk import config as C

SAMPLE = {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": 1,
    "AMT_INCOME_TOTAL": 135000,
    "AMT_CREDIT": 450000,
    "AMT_ANNUITY": 22500,
    "AMT_GOODS_PRICE": 430000,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "DAYS_EMPLOYED": -1800,
    "OCCUPATION_TYPE": "Laborers",
    "CNT_FAM_MEMBERS": 3,
    "EXT_SOURCE_1": 0.45,
    "EXT_SOURCE_2": 0.52,
    "EXT_SOURCE_3": 0.40,
}

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=None)
ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args()
rng = np.random.default_rng(0)

if a.url:
    import httpx

    client = httpx.Client(base_url=a.url, timeout=10)
else:
    from fastapi.testclient import TestClient

    from credit_risk.api import app as appmod

    appmod.service.load()
    appmod.service.audit_log = C.ROOT / "logs" / "benchmark_audit.jsonl"
    client = TestClient(appmod.app)

lat = []
for _i in range(a.n):
    body = {
        **SAMPLE,
        "EXT_SOURCE_2": float(rng.uniform(0.05, 0.95)),
        "AMT_CREDIT": float(rng.uniform(1e5, 1.2e6)),
    }
    t0 = time.perf_counter()
    r = client.post("/score", json=body)
    lat.append((time.perf_counter() - t0) * 1000)
    assert r.status_code == 200, r.text
out = {
    "requests": a.n,
    "mode": a.url or "in-process",
    "p50_ms": round(statistics.median(lat), 2),
    "p95_ms": round(float(np.percentile(lat, 95)), 2),
    "p99_ms": round(float(np.percentile(lat, 99)), 2),
    "mean_ms": round(statistics.mean(lat), 2),
}
print(json.dumps(out, indent=2))
(C.REPORTS / "latency.json").write_text(json.dumps(out, indent=2))
