"""Airflow DAG: synthetic bronze -> 3 models in parallel -> quality gate -> gold publish.

    generate_bronze
        |-- score_churn ------|
        |-- score_credit_risk |--> quality_gate --> publish_gold (PostgreSQL)
        |-- forecast_remittance

The quality gate runs BEFORE the PostgreSQL load, so a bad model run never reaches Power BI.

Runs daily at 05:00 Riyadh time so Power BI's scheduled refresh (e.g. 06:00) picks up fresh scores.
Each task calls one idempotent step of `bpa.pipeline`, so any task can be re-run on its own.
"""
from __future__ import annotations

from datetime import timedelta

import pendulum

try:  # Airflow 3.x
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "bi-analytics",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}


@dag(
    dag_id="bank_predictive_pipeline",
    description="Churn, credit-risk PD and remittance forecast -> PostgreSQL gold -> Power BI",
    schedule="0 5 * * *",
    start_date=pendulum.datetime(2026, 10, 1, tz="Asia/Riyadh"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["predictive", "powerbi", "banking"],
)
def bank_predictive_pipeline():
    def _run(step: str) -> None:
        from bpa.config import get_settings
        from bpa.pipeline import STEPS

        STEPS[step](get_settings())

    @task
    def generate_bronze():
        _run("generate")

    @task
    def score_churn():
        _run("churn")

    @task
    def score_credit_risk():
        _run("credit")

    @task
    def forecast_remittance():
        _run("forecast")

    @task
    def publish_gold():
        _run("publish")

    @task
    def quality_gate():
        """Stop the run before the database load (Power BI keeps yesterday's data) if the new scores look wrong."""
        import pandas as pd

        from bpa.config import get_settings

        s = get_settings()
        churn = pd.read_csv(s.gold / "fact_churn_score.csv")
        loans = pd.read_csv(s.gold / "fact_loan_risk.csv")
        metrics = pd.concat([pd.read_parquet(s.silver / f"metrics_{n}.parquet")
                             for n in ("churn", "credit_risk", "remittance_forecast")])
        auc = metrics[metrics["metric"] == "auc"].set_index("model_name")["value"]
        mape = metrics.query("model_name == 'remittance_forecast' and metric == 'mape'")["value"].iloc[0]

        checks = {
            "churn scores present": len(churn) > 0,
            "churn probability in [0,1]": churn["churn_probability"].between(0, 1).all(),
            "no duplicate customers": churn["customer_id"].is_unique,
            "PD in [0,1]": loans["pd_12m"].between(0, 1).all(),
            "churn AUC >= 0.70": auc.get("churn", 0) >= 0.70,
            "credit AUC >= 0.70": auc.get("credit_risk", 0) >= 0.70,
            "remittance MAPE <= 15%": mape <= 0.15,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"Quality gate failed: {failed}")
        print("Quality gate passed:", ", ".join(checks))

    bronze = generate_bronze()
    models = [score_churn(), score_credit_risk(), forecast_remittance()]
    bronze >> models >> quality_gate() >> publish_gold()


bank_predictive_pipeline()
