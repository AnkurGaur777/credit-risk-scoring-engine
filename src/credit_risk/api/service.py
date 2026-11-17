"""Model loading and scoring logic (kept separate from HTTP so it is unit-testable)."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import config as C
from .. import data as D
from .. import explain as X
from .. import features as F
from ..scorecard import pd_to_score

RISK_BANDS = [(0.03, "very low"), (0.06, "low"), (0.12, "medium"), (0.20, "high")]


def risk_band(p: float) -> str:
    for ub, name in RISK_BANDS:
        if p < ub:
            return name
    return "very high"


class ScoringService:
    def __init__(self, bundle_path: str | Path | None = None, audit_log: str | Path | None = None):
        self.bundle_path = Path(
            bundle_path or os.getenv("CR_MODEL_BUNDLE", C.ARTIFACTS / "model_bundle_A.joblib")
        )
        self.audit_log = Path(audit_log or os.getenv("CR_AUDIT_LOG", C.ROOT / "logs" / "audit.jsonl"))
        self.bundle: dict | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        self.bundle = joblib.load(self.bundle_path)

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    def _frame(self, payload: dict) -> pd.DataFrame:
        df = pd.DataFrame([payload])
        df = D.clean_application(df)
        df = F.engineer_application(df)
        for c in self.bundle["features"]:  # fields the client did not send -> missing
            if c not in df:
                df[c] = np.nan
        df = self.bundle["encoder"].transform(df)
        return df[self.bundle["features"]]

    def score(self, payload: dict, top_k: int = 4) -> dict:
        t0 = time.perf_counter()
        b = self.bundle
        X_ = self._frame(payload)
        p_raw = b["model"].predict_proba(X_)[:, 1]
        p = float(np.clip(b["calibrator"].transform(p_raw)[0], 1e-4, 1 - 1e-4))
        contrib = b["model"].predict(X_, pred_contrib=True)[0][:-1]
        decision = "approve" if p <= b["cutoff_pd"] else "decline"
        rid = str(uuid.uuid4())
        out = {
            "request_id": rid,
            "model_version": b["version"],
            "pd": round(p, 5),
            "score": int(round(float(pd_to_score(p)))),
            "risk_band": risk_band(p),
            "decision": decision,
            "cutoff_pd": round(float(b["cutoff_pd"]), 5),
            "reason_codes": X.reason_codes(contrib, b["features"], top_k=top_k)
            if decision == "decline" or p >= 0.06
            else [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        self._audit(out)
        return out

    def _audit(self, out: dict) -> None:
        """Audit trail WITHOUT personal data: ids, model version, outputs only."""
        rec = {
            "ts": datetime.now(UTC).isoformat(),
            **{
                k: out[k]
                for k in ("request_id", "model_version", "pd", "score", "risk_band", "decision", "latency_ms")
            },
        }
        try:
            with self._lock:
                self.audit_log.parent.mkdir(parents=True, exist_ok=True)
                with open(self.audit_log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        except OSError:  # never fail a scoring request because logging failed
            pass
