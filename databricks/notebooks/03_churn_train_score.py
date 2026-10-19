# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Churn — train, register in Unity Catalog, score
# MAGIC * Out-of-time validation (older snapshots train, latest labelled snapshot tests)
# MAGIC * MLflow tracking + model registered as `<catalog>.<schema>.churn` with signature
# MAGIC * Distributed scoring with `mlflow.pyfunc.spark_udf` (falls back to driver scoring if unavailable)

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from bpa import churn
from bpa.config import get_settings
from bpa.metrics import classification_metrics, importance_rows, metric_rows
from bpa.tracking import start_run

s = get_settings()
feat = spark.table(f"{FQ}.silver_churn_features")
labelled = feat.filter(F.col("dataset") != "score").toPandas()
for c in churn.CATEGORICAL:
    labelled[c] = labelled[c].astype(str)
train, test = labelled[labelled.dataset == "train"], labelled[labelled.dataset == "test"]
print(f"train {len(train):,} rows | test {len(test):,} rows | churn rate {train.churn_3m.mean():.2%}")

# COMMAND ----------

with start_run(s.mlflow_experiment, "churn", s.data_dir / "models", s.mlflow_uri) as run_:
    clf = churn._model(s.seed).fit(train[churn.FEATURES], train["churn_3m"])
    oot = classification_metrics(test["churn_3m"].to_numpy(), clf.predict_proba(test[churn.FEATURES])[:, 1])
    run_.params({"model": "HistGradientBoostingClassifier", "platform": "databricks",
                 "n_train": len(train), "horizon_months": s.churn_horizon_months})
    run_.metrics({f"oot_{k}": v for k, v in oot.items()})
    imp = importance_rows(clf, test[churn.FEATURES], test["churn_3m"], "churn", run_.version, "roc_auc", s.seed)

    final = churn._model(s.seed).fit(labelled[churn.FEATURES], labelled["churn_3m"])
    run_.model(final, example=labelled[churn.FEATURES], register_as=f"{s.registry_prefix}churn")
    version = run_.version

pd.DataFrame([oot]).round(3)

# COMMAND ----------

# MAGIC %md ### Score the current month

# COMMAND ----------

import mlflow

score_sdf = feat.filter(F.col("dataset") == "score")
model_name = f"{s.registry_prefix}churn"
try:
    latest = max(int(v.version) for v in mlflow.MlflowClient().search_model_versions(f"name='{model_name}'"))
    predict = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/{latest}",
                                      result_type="array<double>", env_manager="local")
    scored = (score_sdf
              .withColumn("proba", predict(*[F.col(c) for c in churn.FEATURES]))
              .withColumn("churn_probability", F.element_at("proba", 2))      # P(churn = 1)
              .drop("proba").toPandas())
    print(f"scored with spark_udf: models:/{model_name}/{latest}")
except Exception as exc:                       # e.g. registry not reachable -> score on the driver
    print("spark_udf unavailable, scoring on driver:", exc.__class__.__name__)
    scored = score_sdf.toPandas()
    for c in churn.CATEGORICAL:
        scored[c] = scored[c].astype(str)
    scored["churn_probability"] = final.predict_proba(scored[churn.FEATURES])[:, 1]

# COMMAND ----------

scored["churn_probability"] = scored["churn_probability"].round(5)
scored["risk_decile"] = (pd.qcut(scored["churn_probability"].rank(method="first"), 10, labels=False) + 1).astype(int)
scored["risk_band"] = np.select([scored["risk_decile"].eq(10), scored["risk_decile"].ge(8)], ["High", "Medium"], "Low")
scored["balance_at_risk_sar"] = (scored["balance_avg_6m"] * scored["churn_probability"]).round(2)
scored["primary_signal"] = churn.reason_code(scored)
scored["score_date"] = pd.Timestamp(s.as_of)
scored["model_version"] = version

cols = ["customer_id", "score_date", "churn_probability", "risk_decile", "risk_band", "balance_at_risk_sar",
        "primary_signal", "txn_trend", "logins_trend", "balance_trend", "complaints_6m",
        "salary_credit_ratio_6m", "balance_avg_6m", "model_version"]
save_delta(scored[cols], "fact_churn_score", comment="Churn probability per active customer (3-month horizon)")
save_delta(pd.DataFrame(metric_rows("churn", version, s.as_of, "out_of_time", oot)), "silver_metrics_churn")
save_delta(pd.DataFrame(imp), "silver_importance_churn")

# COMMAND ----------

display(spark.table(f"{FQ}.fact_churn_score").groupBy("risk_band", "primary_signal").count().orderBy("risk_band", F.desc("count")))
