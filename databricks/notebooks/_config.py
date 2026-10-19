# Databricks notebook source
# MAGIC %md
# MAGIC ## Shared configuration
# MAGIC Every notebook runs `%run ./_config` first. It sets the Unity Catalog location, points the
# MAGIC shared `bpa` package at a UC Volume, and puts the repo's `src/` folder on the Python path.
# MAGIC
# MAGIC Works on **Databricks Free Edition** (serverless) and on any workspace with Unity Catalog.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "bpa", "Schema")
dbutils.widgets.text("n_customers", "50000", "Customers to simulate")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
N_CUSTOMERS = dbutils.widgets.get("n_customers")
FQ = f"{CATALOG}.{SCHEMA}"            # fully-qualified prefix for tables, e.g. workspace.bpa
VOLUME_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/files"

# COMMAND ----------

import os
import sys

# notebooks live in <repo>/databricks/notebooks -> package lives in <repo>/src
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

USER = spark.sql("SELECT current_user()").first()[0]

# The bpa package reads its settings from environment variables at import time
os.environ.update({
    "BPA_DATA_DIR": VOLUME_DIR,
    "BPA_N_CUSTOMERS": N_CUSTOMERS,
    "MLFLOW_TRACKING_URI": "databricks",
    "BPA_MLFLOW_EXPERIMENT": f"/Users/{USER}/bank-predictive-analytics",
    "BPA_REGISTRY_PREFIX": f"{FQ}.",          # models register in Unity Catalog: workspace.bpa.churn
})
os.environ.pop("BPA_POSTGRES_URL", None)

# COMMAND ----------

def save_delta(pdf_or_sdf, table: str, comment: str | None = None):
    """Overwrite a managed Delta table from a pandas or Spark DataFrame."""
    sdf = pdf_or_sdf if hasattr(pdf_or_sdf, "rdd") else spark.createDataFrame(pdf_or_sdf)
    (sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.{table}"))
    if comment:
        spark.sql(f"COMMENT ON TABLE {FQ}.{table} IS '{comment}'")
    print(f"saved {FQ}.{table}: {spark.table(f'{FQ}.{table}').count():,} rows")

print(f"catalog/schema = {FQ} | volume = {VOLUME_DIR} | src = {SRC}")
