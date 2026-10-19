# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the Unity Catalog schema and the Volume that holds raw files and model artefacts.
# MAGIC Safe to re-run.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ} COMMENT 'Bank predictive analytics: churn, credit risk, remittance forecast'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {FQ}.files COMMENT 'Bronze/silver files and saved models'")
display(spark.sql(f"SHOW TABLES IN {FQ}"))

# COMMAND ----------

import mlflow, sklearn, pandas
print("mlflow", mlflow.__version__, "| scikit-learn", sklearn.__version__, "| pandas", pandas.__version__)
