"""Use case 3 - Daily remittance volume forecast per corridor (90-day horizon).

Design
* log(amount) = linear trend per corridor + calendar effects learned by gradient boosting
  (payday window, weekend, Ramadan, days-to-Eid). Detrending lets the tree model
  extrapolate growth, which trees cannot do on their own.
* P10 / P90 prediction bands from quantile gradient boosting on the residual.
* Backtest: hold out the last 90 days, report MAPE / WAPE per corridor, then refit and forecast.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import Settings, get_settings
from .generate_data import remittance_calendar
from .io import read_table
from .metrics import metric_rows
from .tracking import start_run

log = logging.getLogger(__name__)

CAL_FEATURES = ["dow", "dom", "month", "payday_window", "days_to_eid", "pre_eid_window", "ramadan", "weekend", "corridor"]


class TrendCalendarForecaster:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.trend: dict[str, np.ndarray] = {}
        self.origin: pd.Timestamp | None = None

    def _t(self, dates: pd.Series) -> np.ndarray:
        return ((pd.to_datetime(dates) - self.origin).dt.days / 365.0).to_numpy()

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        cal = remittance_calendar(pd.DatetimeIndex(pd.to_datetime(df["date"])))
        cal["corridor"] = pd.Categorical(df["corridor"].to_numpy(), categories=self.corridors)
        return cal[CAL_FEATURES]

    def _base(self, df: pd.DataFrame) -> np.ndarray:
        t = self._t(df["date"])
        coef = np.vstack([self.trend[c] for c in df["corridor"]])
        return coef[:, 0] * t + coef[:, 1]

    def fit(self, df: pd.DataFrame) -> "TrendCalendarForecaster":
        df = df.reset_index(drop=True)
        self.origin = pd.to_datetime(df["date"]).min()
        self.corridors = sorted(df["corridor"].unique())
        y = np.log(df["amount_sar"].to_numpy())
        t = self._t(df["date"])
        for c in self.corridors:
            m = (df["corridor"] == c).to_numpy()
            self.trend[c] = np.polyfit(t[m], y[m], 1)
        resid = y - self._base(df)
        X = self._X(df)
        kw = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                  categorical_features="from_dtype", random_state=self.seed)
        self.mean_ = HistGradientBoostingRegressor(**kw).fit(X, resid)
        self.p10_ = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **kw).fit(X, resid)
        self.p90_ = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **kw).fit(X, resid)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        X, base = self._X(df), self._base(df)
        out = df[["date", "corridor"]].copy()
        out["forecast_amount_sar"] = np.exp(base + self.mean_.predict(X)).round(2)
        lo, hi = np.exp(base + self.p10_.predict(X)), np.exp(base + self.p90_.predict(X))
        out["lower_p10_sar"] = np.minimum(lo, out["forecast_amount_sar"]).round(2)
        out["upper_p90_sar"] = np.maximum(hi, out["forecast_amount_sar"]).round(2)
        return out


def run(s: Settings | None = None) -> dict:
    s = s or get_settings()
    hist = read_table(s.bronze, "remittance_daily")
    hist["date"] = pd.to_datetime(hist["date"])
    H = s.forecast_horizon_days
    cutoff = hist["date"].max() - pd.Timedelta(days=H)
    train, hold = hist[hist["date"] <= cutoff], hist[hist["date"] > cutoff]

    with start_run(s.mlflow_experiment, "remittance_forecast", s.data_dir / "models", s.mlflow_uri) as run_:
        bt_model = TrendCalendarForecaster(s.seed).fit(train)
        bt = bt_model.predict(hold).merge(hold[["date", "corridor", "amount_sar"]], on=["date", "corridor"])
        err = (bt["forecast_amount_sar"] - bt["amount_sar"]).abs()
        overall = {
            "mape": float((err / bt["amount_sar"]).mean()),
            "wape": float(err.sum() / bt["amount_sar"].sum()),
            "p10_p90_coverage": float(((bt["amount_sar"] >= bt["lower_p10_sar"]) & (bt["amount_sar"] <= bt["upper_p90_sar"])).mean()),
            "n_test_days": float(H),
        }
        per_corridor = (bt.assign(ape=err / bt["amount_sar"]).groupby("corridor")["ape"].mean())
        log.info("Remittance backtest: MAPE %.2f%% | WAPE %.2f%% | P10-P90 coverage %.0f%%",
                 100 * overall["mape"], 100 * overall["wape"], 100 * overall["p10_p90_coverage"])
        run_.params({"model": "log-linear trend + HistGradientBoosting residual (mean, P10, P90)",
                     "horizon_days": H, "backtest_cutoff": cutoff.date().isoformat()})
        run_.metrics({f"backtest_{k}": v for k, v in overall.items()}
                     | {f"backtest_mape_{c.lower()}": v for c, v in per_corridor.items()})

        final = TrendCalendarForecaster(s.seed).fit(hist)
        run_.model(final, sklearn_flavor=False)
        version = run_.version

    future_dates = pd.date_range(hist["date"].max() + pd.Timedelta(days=1), periods=H, freq="D")
    grid = pd.DataFrame([(d, c) for c in s.corridors for d in future_dates], columns=["date", "corridor"])
    fc = final.predict(grid)
    fc["date"] = fc["date"].dt.strftime("%Y-%m-%d")
    fc["model_version"] = version

    actual = hist.merge(bt[["date", "corridor", "forecast_amount_sar"]].rename(columns={"forecast_amount_sar": "backtest_forecast_sar"}),
                        on=["date", "corridor"], how="left")
    actual["date"] = actual["date"].dt.strftime("%Y-%m-%d")
    actual["is_backtest_period"] = actual["backtest_forecast_sar"].notna()

    rows = metric_rows("remittance_forecast", version, s.as_of, "backtest_90d", overall)
    rows += metric_rows("remittance_forecast", version, s.as_of, "backtest_90d",
                        {f"mape_{c.lower()}": v for c, v in per_corridor.items()})
    return {"fact_remittance_daily": actual, "fact_remittance_forecast": fc, "metrics": rows, "importance": []}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run()["fact_remittance_forecast"].head())
