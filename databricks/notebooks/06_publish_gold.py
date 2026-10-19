# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Publish gold for Power BI
# MAGIC Builds the dimensions, consolidates model metrics, applies a quality gate and optimises the
# MAGIC large fact table. Table names match the local PostgreSQL gold schema, so the same Power BI
# MAGIC semantic model works on either source.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import pandas as pd

from bpa.config import get_settings
from bpa.pipeline import CORRIDORS, build_dim_date

s = get_settings()

# ---- quality gate: stop before publishing if any model looks wrong ----
metrics = spark.sql(f"""
    SELECT * FROM {FQ}.silver_metrics_churn
    UNION ALL SELECT * FROM {FQ}.silver_metrics_credit_risk
    UNION ALL SELECT * FROM {FQ}.silver_metrics_remittance_forecast
""").toPandas()
auc = metrics[metrics.metric == "auc"].set_index("model_name")["value"]
mape = metrics.query("model_name == 'remittance_forecast' and metric == 'mape'")["value"].iloc[0]
checks = {"churn AUC >= 0.70": bool(auc.get("churn", 0) >= 0.70),
          "credit AUC >= 0.70": bool(auc.get("credit_risk", 0) >= 0.70),
          "remittance MAPE <= 15%": bool(mape <= 0.15)}
assert all(checks.values()), f"Quality gate failed: {checks}"
print("Quality gate passed:", checks)

# COMMAND ----------

customers = spark.table(f"{FQ}.bronze_customers").toPandas()
customers["tenure_months"] = customers["tenure_months_at_start"] + s.n_months
customers["age_band"] = pd.cut(customers["age"], [0, 25, 35, 45, 55, 120],
                               labels=["<=25", "26-35", "36-45", "46-55", "56+"]).astype(str)
customers["closed_month"] = pd.to_datetime(customers["closed_month"])
save_delta(customers.drop(columns=["tenure_months_at_start"]), "dim_customer")

dim_date = build_dim_date(s)
for c in ("date", "month_end"):
    dim_date[c] = pd.to_datetime(dim_date[c])
save_delta(dim_date, "dim_date")
save_delta(CORRIDORS, "dim_corridor")

save_delta(spark.table(f"{FQ}.bronze_customer_monthly_activity").drop("month_idx"), "fact_customer_monthly")

metrics["run_date"] = pd.to_datetime(metrics["run_date"])
save_delta(metrics, "model_metrics")
save_delta(spark.sql(f"SELECT * FROM {FQ}.silver_importance_churn UNION ALL "
                     f"SELECT * FROM {FQ}.silver_importance_credit_risk"), "feature_importance")

# COMMAND ----------

# Faster DirectQuery / refresh on the biggest table
spark.sql(f"OPTIMIZE {FQ}.fact_customer_monthly ZORDER BY (customer_id)")
display(spark.sql(f"SHOW TABLES IN {FQ}"))
