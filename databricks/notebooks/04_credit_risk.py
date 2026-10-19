# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Credit risk — 12-month PD, grade and expected loss
# MAGIC Champion/challenger (gradient boosting vs logistic scorecard) chosen on out-of-time AUC.
# MAGIC Customer type, age and region are excluded from the model by design (fair-lending hygiene).
# MAGIC Uses the same `bpa.credit_risk` code as the local Airflow pipeline.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import pandas as pd

from bpa import credit_risk
from bpa.config import get_settings

s = get_settings()
out = credit_risk.run(s)

fact = out["fact_loan_risk"].copy()
for c in ("score_date", "origination_date"):
    fact[c] = pd.to_datetime(fact[c])

save_delta(fact, "fact_loan_risk", comment="PD, risk grade and expected loss per live personal-finance contract")
save_delta(pd.DataFrame(out["metrics"]), "silver_metrics_credit_risk")
save_delta(pd.DataFrame(out["importance"]), "silver_importance_credit_risk")

# COMMAND ----------

display(spark.sql(f"""
    SELECT risk_grade, COUNT(*) AS loans,
           ROUND(AVG(pd_12m), 4) AS avg_pd,
           ROUND(SUM(ead_sar) / 1e6, 1) AS ead_m_sar,
           ROUND(SUM(expected_loss_sar) / 1e6, 2) AS el_m_sar
    FROM {FQ}.fact_loan_risk GROUP BY risk_grade ORDER BY risk_grade
"""))
