FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CR_MODEL_BUNDLE=/app/artifacts/model_bundle_A.joblib \
    CR_AUDIT_LOG=/app/logs/audit.jsonl

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# The trained bundle is built by `make run-a` and is NOT committed to git; copy it in at build time.
COPY artifacts/model_bundle_A.joblib ./artifacts/model_bundle_A.joblib

RUN useradd -m app && mkdir -p /app/logs && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "credit_risk.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
