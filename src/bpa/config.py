"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # Where bronze (raw), silver (features) and gold (BI-ready) files are written
    data_dir: Path = Path(_env("BPA_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))

    # Data volume. 50k customers x 24 months = 1.2M activity rows (default).
    # Raise to 500_000 for ~12M rows when you want to show "millions of rows" at scale.
    n_customers: int = int(_env("BPA_N_CUSTOMERS", "50000"))
    n_months: int = int(_env("BPA_N_MONTHS", "24"))
    seed: int = int(_env("BPA_SEED", "42"))

    # Scoring date = last day of the observed history
    as_of: date = date.fromisoformat(_env("BPA_AS_OF", "2026-09-30"))

    # Business rules
    churn_horizon_months: int = 3          # churn = inactive in the next 3 months
    churn_lookback_months: int = 6         # features use the last 6 months
    credit_default_dpd: int = 90           # default = 90+ days past due within 12 months
    lgd: float = 0.45                      # loss given default for unsecured personal finance
    forecast_horizon_days: int = 90

    # Optional outputs
    postgres_url: str | None = os.environ.get("BPA_POSTGRES_URL")      # e.g. postgresql+psycopg2://bpa:bpa@postgres:5432/bpa
    mlflow_uri: str | None = os.environ.get("MLFLOW_TRACKING_URI")      # e.g. http://mlflow:5000 or "databricks"
    # Databricks needs an absolute workspace path, e.g. /Users/<you>/bank-predictive-analytics
    mlflow_experiment: str = _env("BPA_MLFLOW_EXPERIMENT", "bank-predictive-analytics")
    # Registered model name prefix: "bpa-" locally, "workspace.bpa." for Unity Catalog
    registry_prefix: str = _env("BPA_REGISTRY_PREFIX", "bpa-")

    corridors: tuple[str, ...] = field(
        default=("Indonesia", "India", "Pakistan", "Philippines", "Egypt", "Bangladesh")
    )

    def __post_init__(self):
        # No tracking server configured -> local MLflow store inside the data folder
        if not self.mlflow_uri:
            object.__setattr__(self, "mlflow_uri", f"sqlite:///{(self.data_dir / 'mlflow.db').as_posix()}")

    @property
    def bronze(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold(self) -> Path:
        return self.data_dir / "gold"


def get_settings() -> Settings:
    s = Settings()
    for p in (s.bronze, s.silver, s.gold):
        p.mkdir(parents=True, exist_ok=True)
    return s
