"""PySpark and pandas churn features must agree (skipped when PySpark is not installed)."""
import importlib

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    s = (SparkSession.builder.master("local[2]").appName("bpa-parity")
         .config("spark.sql.shuffle.partitions", "4").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def test_spark_matches_pandas(spark, tmp_path, monkeypatch):
    monkeypatch.setenv("BPA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BPA_N_CUSTOMERS", "2000")
    import bpa.config
    importlib.reload(bpa.config)
    from bpa import churn, generate_data
    from bpa.spark_features import build_churn_snapshots

    s = bpa.config.get_settings()
    rng = np.random.default_rng(s.seed)
    customers, activity = generate_data.generate_customers_and_activity(s, rng)
    t = 15
    pdf = churn.build_snapshot(customers, activity, t, s, with_label=True).set_index("customer_id").sort_index()

    sdf = build_churn_snapshots(spark.createDataFrame(customers.drop(columns=["closed_month"])),
                                spark.createDataFrame(activity.drop(columns=["month_end"])), [t])
    spdf = sdf.toPandas().set_index("customer_id").sort_index()

    assert len(spdf) == len(pdf)
    assert (spdf["churn_3m"] == pdf["churn_3m"]).all()
    for col in churn.NUMERIC:
        np.testing.assert_allclose(spdf[col].astype(float), pdf[col].astype(float), atol=1e-2, err_msg=col)
