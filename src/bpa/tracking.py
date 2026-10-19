"""Thin MLflow wrapper.

* Tracking server set (MLFLOW_TRACKING_URI): params, metrics and the model are logged, and the
  model is registered (local MLflow registry, or Unity Catalog on Databricks).
* MLflow missing or unreachable: the pipeline keeps running and models are still saved with
  joblib, so the BI refresh never depends on the tracking server being up.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import joblib
import pandas as pd

log = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn  # noqa: F401
    from mlflow.models import infer_signature
    _HAS_MLFLOW = True
except Exception:  # pragma: no cover
    _HAS_MLFLOW = False

PACKAGE_DIR = Path(__file__).resolve().parent


class Run:
    def __init__(self, name: str, model_dir: Path, mlflow_run: Any | None):
        self.name = name
        self.model_dir = model_dir
        self._run = mlflow_run
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        self.version = f"{name}-{stamp}"
        if mlflow_run is not None:
            self.version = f"{name}-{mlflow_run.info.run_id[:8]}"

    def params(self, p: dict[str, Any]) -> None:
        if self._run is not None:
            mlflow.log_params({k: str(v)[:250] for k, v in p.items()})

    def metrics(self, m: dict[str, float]) -> None:
        if self._run is not None:
            mlflow.log_metrics({k: float(v) for k, v in m.items()})

    def model(self, model: Any, example: pd.DataFrame | None = None, register_as: str | None = None,
              sklearn_flavor: bool = True) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / f"{self.name}.joblib"
        joblib.dump(model, path)
        if self._run is None:
            return path
        if not sklearn_flavor:
            mlflow.log_artifact(str(path), artifact_path="model")
            return path
        # pyfunc / spark_udf return class probabilities [P(0), P(1)] instead of hard 0/1 labels
        kwargs: dict[str, Any] = dict(registered_model_name=register_as, serialization_format="cloudpickle",
                                      code_paths=[str(PACKAGE_DIR)], pyfunc_predict_fn="predict_proba")
        if example is not None:
            ex = example.head(200).copy()
            for c in ex.columns:                                  # plain types in the schema
                if isinstance(ex[c].dtype, pd.CategoricalDtype):
                    ex[c] = ex[c].astype(str)
            kwargs["signature"] = infer_signature(ex, model.predict_proba(ex))
        logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)   # cloudpickle notice is expected
        try:
            try:
                mlflow.sklearn.log_model(model, name="model", **kwargs)
            except TypeError:                                     # MLflow 2.x signature
                mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)
        except Exception as exc:
            log.warning("MLflow model logging/registration skipped: %s", exc)
        return path


@contextmanager
def start_run(experiment: str, name: str, model_dir: Path, tracking_uri: str | None) -> Iterator[Run]:
    if not _HAS_MLFLOW:
        log.info("MLflow not installed - tracking skipped")
        yield Run(name, model_dir, None)
        return
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            if tracking_uri == "databricks":
                mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(experiment)
        active = mlflow.start_run(run_name=name)
    except Exception as exc:  # tracking server down, missing DB driver, ...
        log.warning("MLflow unavailable (%s) - continuing without tracking", exc.__class__.__name__)
        yield Run(name, model_dir, None)
        return
    with active as r:
        yield Run(name, model_dir, r)
