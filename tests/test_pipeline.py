"""Smoke test: run the whole pipeline on a small sample and check the gold contract
that Power BI depends on (file names, key columns, value ranges)."""
import importlib

import pandas as pd
import pytest

GOLD_CONTRACT = {
    "dim_customer": ["customer_id", "segment", "region", "status"],
    "dim_date": ["date", "year_month", "is_forecast_period"],
    "dim_corridor": ["corridor", "currency"],
    "fact_customer_monthly": ["customer_id", "month_end", "txn_count", "avg_balance_sar"],
    "fact_churn_score": ["customer_id", "churn_probability", "risk_band", "primary_signal"],
    "fact_loan_risk": ["loan_id", "customer_id", "pd_12m", "risk_grade", "expected_loss_sar"],
    "fact_remittance_daily": ["date", "corridor", "amount_sar", "backtest_forecast_sar"],
    "fact_remittance_forecast": ["date", "corridor", "forecast_amount_sar", "lower_p10_sar", "upper_p90_sar"],
    "model_metrics": ["model_name", "metric", "value"],
    "feature_importance": ["model_name", "feature", "importance", "rank"],
}


@pytest.fixture(scope="module")
def gold(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    mp = pytest.MonkeyPatch()
    mp.setenv("BPA_DATA_DIR", str(data))
    mp.setenv("BPA_N_CUSTOMERS", "6000")
    mp.delenv("BPA_POSTGRES_URL", raising=False)
    mp.delenv("MLFLOW_TRACKING_URI", raising=False)
    import bpa.config
    importlib.reload(bpa.config)
    import bpa.pipeline
    importlib.reload(bpa.pipeline)
    bpa.pipeline.main(["all"])
    yield data / "gold"
    mp.undo()


@pytest.mark.parametrize("table,cols", GOLD_CONTRACT.items())
def test_gold_contract(gold, table, cols):
    df = pd.read_csv(gold / f"{table}.csv")
    assert len(df) > 0
    assert set(cols) <= set(df.columns)


def test_probabilities_in_range(gold):
    churn = pd.read_csv(gold / "fact_churn_score.csv")
    loans = pd.read_csv(gold / "fact_loan_risk.csv")
    assert churn["churn_probability"].between(0, 1).all()
    assert loans["pd_12m"].between(0, 1).all()
    assert set(loans["risk_grade"]) <= set("ABCDE")


def test_keys_are_unique_and_join(gold):
    dim = pd.read_csv(gold / "dim_customer.csv")
    churn = pd.read_csv(gold / "fact_churn_score.csv")
    assert dim["customer_id"].is_unique
    assert churn["customer_id"].isin(dim["customer_id"]).all()


def test_models_beat_random(gold):
    m = pd.read_csv(gold / "model_metrics.csv")
    auc = m[m["metric"] == "auc"].set_index("model_name")["value"]
    assert auc["churn"] > 0.65 and auc["credit_risk"] > 0.65
    mape = m[(m["model_name"] == "remittance_forecast") & (m["metric"] == "mape")]["value"].iloc[0]
    assert mape < 0.15


def test_forecast_band_brackets_point(gold):
    fc = pd.read_csv(gold / "fact_remittance_forecast.csv")
    assert (fc["lower_p10_sar"] <= fc["forecast_amount_sar"]).all()
    assert (fc["forecast_amount_sar"] <= fc["upper_p90_sar"]).all()


# ---------------------------------------------------------------------------------------
# Power BI contract: the semantic model spec must match what the pipeline produces
# ---------------------------------------------------------------------------------------
def _pbip_spec():
    import importlib.util
    import sys
    from pathlib import Path

    if "build_pbip" in sys.modules:
        return sys.modules["build_pbip"]
    path = Path(__file__).resolve().parents[1] / "powerbi" / "build_pbip.py"
    spec = importlib.util.spec_from_file_location("build_pbip", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_pbip"] = mod          # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def test_powerbi_columns_exist_in_gold(gold):
    pbip = _pbip_spec()
    for t in pbip.TABLES:
        cols = set(pd.read_csv(gold / f"{t.source}.csv", nrows=5).columns)
        missing = [c.source for c in t.columns if c.source not in cols]
        assert not missing, f"{t.name}: gold table {t.source} lacks {missing}"


def test_powerbi_dax_references_resolve():
    import re

    pbip = _pbip_spec()
    columns = {(t.name, c.name) for t in pbip.TABLES for c in t.columns}
    measures = {m.name for t in pbip.TABLES for m in t.measures}
    tables = {t.name for t in pbip.TABLES}
    for t in pbip.TABLES:
        for m in t.measures:
            for tbl, col in re.findall(r"'([^']+)'\[([^\]]+)\]", m.expr):
                assert (tbl, col) in columns, f"{m.name}: unknown column '{tbl}'[{col}]"
            bare = re.sub(r"'[^']+'\[[^\]]+\]", "", m.expr)
            for ref in re.findall(r"\[([^\]]+)\]", bare):
                assert ref in measures, f"{m.name}: unknown measure [{ref}]"
            for tbl in re.findall(r"(?:COUNTROWS|FILTER|SUMX|AVERAGEX)\('([^']+)'", m.expr):
                assert tbl in tables
    for ft, fc, tt, tc, _ in pbip.RELATIONSHIPS:
        assert (ft, fc) in columns and (tt, tc) in columns


def test_report_fields_exist_in_model():
    pbip = _pbip_spec()
    columns = {(t.name, c.name) for t in pbip.TABLES for c in t.columns}
    measures = {(t.name, m.name) for t in pbip.TABLES for m in t.measures}
    for page in pbip.build_pages():
        for v in page.visuals:
            for role in v["visual"].get("query", {}).get("queryState", {}).values():
                for p in role["projections"]:
                    kind, body = next(iter(p["field"].items()))
                    key = (body["Expression"]["SourceRef"]["Entity"], body["Property"])
                    assert key in (columns if kind == "Column" else measures), f"{page.title}: {kind} {key} not in model"
