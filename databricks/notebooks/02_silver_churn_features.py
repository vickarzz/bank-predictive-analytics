# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — churn features with PySpark
# MAGIC Builds every training, test and scoring snapshot in **one distributed pass** with
# MAGIC `bpa.spark_features.build_churn_snapshots` (window = last 6 months, label = inactive 3 months later).
# MAGIC The same logic exists in pandas (`bpa.churn.build_snapshot`); `tests/test_spark_parity.py` proves they match.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from pyspark.sql import functions as F

from bpa.config import get_settings
from bpa.spark_features import build_churn_snapshots

s = get_settings()
m, H = s.n_months, s.churn_horizon_months
test_t = m - 1 - H
train_ts = list(range(s.churn_lookback_months + 2, test_t - H + 1))[-6:]
score_t = m - 1
print("train snapshots", train_ts, "| OOT test", test_t, "| scoring", score_t)

customers = spark.table(f"{FQ}.bronze_customers")
activity = spark.table(f"{FQ}.bronze_customer_monthly_activity")

labelled = (build_churn_snapshots(customers, activity, train_ts + [test_t], with_label=True)
            .withColumn("dataset", F.when(F.col("snapshot_idx") == test_t, "test").otherwise("train")))
scoring = (build_churn_snapshots(customers, activity, [score_t], with_label=False)
           .withColumn("churn_3m", F.lit(None).cast("int"))
           .withColumn("dataset", F.lit("score")))

features = labelled.unionByName(scoring)
save_delta(features, "silver_churn_features", comment="Churn feature snapshots (train/test/score)")

# COMMAND ----------

display(spark.table(f"{FQ}.silver_churn_features")
        .groupBy("dataset", "snapshot_idx")
        .agg(F.count("*").alias("customers"), F.round(F.avg("churn_3m"), 4).alias("churn_rate"))
        .orderBy("snapshot_idx"))
