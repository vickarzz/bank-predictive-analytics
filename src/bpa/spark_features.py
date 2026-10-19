"""PySpark version of the churn feature builder (used on Databricks / any Spark cluster).

It produces exactly the same columns as `bpa.churn.build_snapshot`, but builds many snapshots
in one distributed pass, so it scales to hundreds of millions of activity rows.
`tests/test_spark_parity.py` checks the two implementations agree.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def _ratio(a, b):
    return F.round(F.when(b > 0, a / b).when(a > 0, F.lit(2.0)).otherwise(F.lit(1.0)), 4)


def build_churn_snapshots(customers: DataFrame, activity: DataFrame, snapshots: list[int],
                          lookback: int = 6, horizon: int = 3, with_label: bool = True) -> DataFrame:
    spark = SparkSession.getActiveSession()
    snaps = spark.createDataFrame([(int(t),) for t in snapshots], "snapshot_idx int")

    upper = horizon if with_label else 0
    a = (activity.join(F.broadcast(snaps),
                       (F.col("month_idx") >= F.col("snapshot_idx") - (lookback - 1))
                       & (F.col("month_idx") <= F.col("snapshot_idx") + upper))
         .withColumn("rel", F.col("month_idx") - F.col("snapshot_idx")))   # -5 .. 0 (.. +3)

    in_w = F.col("rel").between(-(lookback - 1), 0)
    recent = F.col("rel").between(-2, 0)
    prior = F.col("rel").between(-5, -3)
    now = F.col("rel") == 0

    def avg_if(cond, col):
        return F.avg(F.when(cond, F.col(col)))

    agg = a.groupBy("customer_id", "snapshot_idx").agg(
        F.round(avg_if(in_w, "txn_count"), 2).alias("txn_count_avg_6m"),
        F.max(F.when(now, F.col("txn_count"))).alias("txn_count_last"),
        avg_if(recent, "txn_count").alias("_txn_r"), avg_if(prior, "txn_count").alias("_txn_p"),
        F.round(avg_if(in_w, "txn_amount_sar"), 2).alias("txn_amount_avg_6m"),
        F.round(avg_if(in_w, "avg_balance_sar"), 2).alias("balance_avg_6m"),
        avg_if(recent, "avg_balance_sar").alias("_bal_r"), avg_if(prior, "avg_balance_sar").alias("_bal_p"),
        F.round(avg_if(in_w, "digital_logins"), 2).alias("logins_avg_6m"),
        avg_if(recent, "digital_logins").alias("_lgn_r"), avg_if(prior, "digital_logins").alias("_lgn_p"),
        F.sum(F.when(in_w, F.col("complaints"))).cast("long").alias("complaints_6m"),
        F.round(F.avg(F.when(in_w, F.col("salary_credited").cast("int"))), 3).alias("salary_credit_ratio_6m"),
        F.sum(F.when(recent & (F.col("txn_count") == 0), 1).otherwise(0)).cast("long").alias("inactive_months_last3"),
        F.max(F.when(now, F.col("is_active").cast("int"))).alias("_active_now"),
        F.max(F.when(F.col("rel") == horizon, F.col("is_active").cast("int"))).alias("_active_h"),
    )
    agg = (agg
           .withColumn("txn_trend", _ratio(F.col("_txn_r"), F.col("_txn_p")))
           .withColumn("balance_trend", _ratio(F.col("_bal_r"), F.col("_bal_p")))
           .withColumn("logins_trend", _ratio(F.col("_lgn_r"), F.col("_lgn_p"))))
    if with_label:
        agg = agg.withColumn("churn_3m", ((F.col("_active_now") == 1) & (F.col("_active_h") == 0)).cast("int"))

    c = customers.select(
        "customer_id", "segment", "region", "customer_type", "age", "product_count",
        F.col("has_salary_transfer").cast("int").alias("has_salary_transfer"),
        F.col("has_mobile_app").cast("int").alias("has_mobile_app"),
        "tenure_months_at_start",
    )
    out = (agg.filter(F.col("_active_now") == 1)
           .join(c, "customer_id")
           .withColumn("tenure_months", F.col("tenure_months_at_start") + F.col("snapshot_idx"))
           .drop("tenure_months_at_start", *[x for x in agg.columns if x.startswith("_")]))
    return out
