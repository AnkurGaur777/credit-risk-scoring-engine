"""FastAPI scoring service.  uvicorn credit_risk.api.app:app --port 8000"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import Applicant, Health, ScoreResponse
from .service import ScoringService

log = logging.getLogger("credit_risk.api")
service = ScoringService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service.load()
        log.info("loaded model %s", service.bundle["version"])
    except Exception as e:  # service starts; /health reports not loaded; /score returns 503
        log.error("could not load model bundle %s: %s", service.bundle_path, e)
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    version="0.1.0",
    lifespan=lifespan,
    description="Educational demonstration of a PD model with calibrated probabilities, "
    "reason codes and a profit-based cut-off. Not for production lending decisions.",
)


@app.get("/health", response_model=Health)
def health():
    b = service.bundle
    return Health(
        status="ok" if service.loaded else "model_not_loaded",
        model_version=b["version"] if b else None,
        stage=b["stage"] if b else None,
        loaded=service.loaded,
    )


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    if not service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    payload = applicant.model_dump(exclude_none=True)
    try:
        return service.score(payload)
    except Exception as e:  # pragma: no cover
        log.exception("scoring failed")
        raise HTTPException(status_code=500, detail=f"scoring failed: {type(e).__name__}") from e


if os.getenv("CR_DEBUG_RELOAD") == "1":  # pragma: no cover

    @app.post("/admin/reload")
    def reload_model():
        service.load()
        return {"reloaded": service.bundle["version"]}
