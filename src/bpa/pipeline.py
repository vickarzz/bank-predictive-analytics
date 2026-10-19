"""Pipeline entry point. Each step is idempotent and can run on its own, which is how the
Airflow DAG calls it:

    python -m bpa.pipeline generate   # bronze
    python -m bpa.pipeline churn      # model + gold fact
    python -m bpa.pipeline credit
    python -m bpa.pipeline forecast
    python -m bpa.pipeline publish    # dimensions, metrics, optional PostgreSQL load
    python -m bpa.pipeline all
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from . import churn, credit_risk, generate_data, remittance_forecast
from .config import Settings, get_settings
from .io import read_table, to_postgres, write_table

log = logging.getLogger("bpa")

CORRIDORS = pd.DataFrame({
    "corridor": ["Indonesia", "India", "Pakistan", "Philippines", "Egypt", "Bangladesh"],
    "currency": ["IDR", "INR", "PKR", "PHP", "EGP", "BDT"],
    "region": ["South-East Asia", "South Asia", "South Asia", "South-East Asia", "MENA", "South Asia"],
})


def _write_model_outputs(s: Settings, name: str, out: dict) -> None:
    for key, value in out.items():
        if key.startswith("fact_"):
            write_table(value, s.gold, key, csv=True)
    write_table(pd.DataFrame(out["metrics"]), s.silver, f"metrics_{name}")
    write_table(pd.DataFrame(out["importance"] or [], columns=["model_name", "model_version", "feature", "importance",
                                                           "importance_std", "rank"]), s.silver, f"importance_{name}")


def step_generate(s: Settings) -> None:
    generate_data.run(s)


def step_churn(s: Settings) -> None:
    _write_model_outputs(s, "churn", churn.run(s))


def step_credit(s: Settings) -> None:
    _write_model_outputs(s, "credit_risk", credit_risk.run(s))


def step_forecast(s: Settings) -> None:
    _write_model_outputs(s, "remittance_forecast", remittance_forecast.run(s))


def build_dim_date(s: Settings) -> pd.DataFrame:
    start = pd.Timestamp(s.as_of) - pd.DateOffset(months=s.n_months + 1)
    end = pd.Timestamp(s.as_of) + pd.Timedelta(days=s.forecast_horizon_days + 31)
    d = pd.DataFrame({"date": pd.date_range(start.replace(day=1), end, freq="D")})
    d["year"] = d["date"].dt.year
    d["quarter"] = "Q" + d["date"].dt.quarter.astype(str)
    d["month_num"] = d["date"].dt.month
    d["month_name"] = d["date"].dt.strftime("%b")
    d["year_month"] = d["date"].dt.strftime("%Y-%m")
    d["month_end"] = (d["date"] + pd.offsets.MonthEnd(0)).dt.strftime("%Y-%m-%d")
    d["day_of_week"] = d["date"].dt.strftime("%a")
    d["day_of_week_num"] = (d["date"].dt.dayofweek + 1) % 7 + 1      # Sun=1 ... Sat=7 (Saudi week)
    d["is_weekend"] = d["date"].dt.dayofweek.isin([4, 5])
    d["is_forecast_period"] = d["date"] > pd.Timestamp(s.as_of)
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return d


def step_publish(s: Settings) -> dict[str, pd.DataFrame]:
    customers = read_table(s.bronze, "customers")
    customers["tenure_months"] = customers["tenure_months_at_start"] + s.n_months
    customers["age_band"] = pd.cut(customers["age"], [0, 25, 35, 45, 55, 120],
                                   labels=["<=25", "26-35", "36-45", "46-55", "56+"]).astype(str)
    dim_customer = customers.drop(columns=["tenure_months_at_start"])

    activity = read_table(s.bronze, "customer_monthly_activity")
    activity["month_end"] = pd.to_datetime(activity["month_end"]).dt.strftime("%Y-%m-%d")
    fact_monthly = activity.drop(columns=["month_idx"])

    gold: dict[str, pd.DataFrame] = {
        "dim_customer": dim_customer,
        "dim_date": build_dim_date(s),
        "dim_corridor": CORRIDORS,
        "fact_customer_monthly": fact_monthly,
    }
    for name, df in gold.items():
        write_table(df, s.gold, name, csv=True)

    metrics = pd.concat([read_table(s.silver, f"metrics_{n}") for n in ("churn", "credit_risk", "remittance_forecast")],
                        ignore_index=True)
    importance = pd.concat([read_table(s.silver, f"importance_{n}") for n in ("churn", "credit_risk")], ignore_index=True)
    write_table(metrics, s.gold, "model_metrics", csv=True)
    write_table(importance, s.gold, "feature_importance", csv=True)

    for fact in ("fact_churn_score", "fact_loan_risk", "fact_remittance_daily", "fact_remittance_forecast"):
        gold[fact] = pd.read_csv(s.gold / f"{fact}.csv")
    gold["model_metrics"], gold["feature_importance"] = metrics, importance

    if s.postgres_url:
        to_postgres(gold, s.postgres_url)
    else:
        log.info("BPA_POSTGRES_URL not set - gold layer written as CSV only (%s)", s.gold)
    return gold


STEPS = {"generate": step_generate, "churn": step_churn, "credit": step_credit,
         "forecast": step_forecast, "publish": step_publish}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bank predictive analytics pipeline")
    parser.add_argument("step", choices=[*STEPS, "all"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    s = get_settings()
    order = list(STEPS) if args.step == "all" else [args.step]
    for name in order:
        t0 = time.time()
        log.info("=== step: %s ===", name)
        STEPS[name](s)
        log.info("=== %s done in %.1fs ===", name, time.time() - t0)


if __name__ == "__main__":
    main()
