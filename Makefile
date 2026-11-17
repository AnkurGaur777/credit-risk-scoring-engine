.PHONY: install data run-a run-b run test lint api docker docs clean mlflow
PY ?= python

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

data:            ## synthetic stand-in data (no Kaggle account needed)
	$(PY) -m credit_risk.synthetic --n 20000

run-a:           ## Stage A: application_train only
	$(PY) -m credit_risk.pipeline --stage A

run-b:           ## Stage B: + bureau / previous apps / instalments / POS / cards
	$(PY) -m credit_risk.pipeline --stage B

run: data run-a run-b docs   ## one command: everything end to end on synthetic data

docs:            ## model card, data audit, feature dictionary, business summary PDF, architecture png
	$(PY) scripts/build_docs.py

test:
	pytest

lint:
	ruff check src tests scripts

api:
	uvicorn credit_risk.api.app:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf artifacts/*.joblib reports/*.json reports/*.csv reports/*.md reports/figures/*.png mlruns mlflow.db logs
