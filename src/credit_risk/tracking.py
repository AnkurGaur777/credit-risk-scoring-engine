"""Thin MLflow wrapper. Degrades to a no-op if MLflow is not installed so tests/API stay light."""

from __future__ import annotations

import contextlib

from . import config as C

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


def setup(experiment: str = "credit-risk") -> bool:
    if mlflow is None:
        return False
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    mlflow.set_experiment(experiment)
    return True


@contextlib.contextmanager
def run(name: str, nested: bool = False, params: dict | None = None):
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=name, nested=nested) as r:
        if params:
            mlflow.log_params(
                {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}
            )
        yield r


def log_metrics(metrics: dict, prefix: str = "") -> None:
    if mlflow is not None and mlflow.active_run():
        mlflow.log_metrics(
            {f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )


def log_artifact(path) -> None:
    if mlflow is not None and mlflow.active_run():
        mlflow.log_artifact(str(path))
