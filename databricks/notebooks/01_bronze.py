# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — generate synthetic source data
# MAGIC Runs the shared generator (`bpa.generate_data`) into the Volume, then registers each file as a
# MAGIC Delta table. In a real bank this notebook would instead ingest from core banking / data lake
# MAGIC (e.g. Finacle extracts, Cloudera/Hive tables) — every downstream step stays the same.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from bpa import generate_data
from bpa.config import get_settings

s = get_settings()
counts = generate_data.run(s)
counts

# COMMAND ----------

for name in ["customers", "customer_monthly_activity", "loans", "remittance_daily"]:
    save_delta(spark.read.parquet(f"{s.bronze}/{name}.parquet"), f"bronze_{name}",
               comment=f"Synthetic bronze data: {name}")

# COMMAND ----------

display(spark.sql(f"""
    SELECT date_format(month_end, 'yyyy-MM')   AS month,
           SUM(CAST(is_active AS INT))          AS active_customers,
           ROUND(SUM(txn_amount_sar) / 1e6, 1)  AS txn_amount_m_sar
    FROM {FQ}.bronze_customer_monthly_activity
    GROUP BY 1 ORDER BY 1
"""))
