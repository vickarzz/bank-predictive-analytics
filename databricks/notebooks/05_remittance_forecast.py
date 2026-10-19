# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Remittance forecast — 90 days per corridor with P10–P90 band
# MAGIC Log-linear trend + gradient-boosted calendar effects (payday ~27th, Fri/Sat, Ramadan, days to Eid).
# MAGIC Backtest on the last 90 days is stored alongside actuals so Power BI can show forecast vs actual.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import pandas as pd

from bpa import remittance_forecast
from bpa.config import get_settings

s = get_settings()
out = remittance_forecast.run(s)

daily, fc = out["fact_remittance_daily"].copy(), out["fact_remittance_forecast"].copy()
daily["date"], fc["date"] = pd.to_datetime(daily["date"]), pd.to_datetime(fc["date"])

save_delta(daily, "fact_remittance_daily", comment="Daily remittance actuals per corridor with 90-day backtest")
save_delta(fc, "fact_remittance_forecast", comment="90-day remittance forecast per corridor (mean, P10, P90)")
save_delta(pd.DataFrame(out["metrics"]), "silver_metrics_remittance_forecast")

# COMMAND ----------

display(spark.table(f"{FQ}.silver_metrics_remittance_forecast"))
