"""Use case 1 - Customer churn (3-month horizon).

Design
* Snapshot-based: features use the 6 months up to snapshot month t; label = customer is
  active at t but inactive at t+3.
* Out-of-time validation: train on older snapshots, test on the most recent labelled one.
* Final model is refit on all labelled snapshots and scores the current month (as_of).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .config import Settings, get_settings
from .io import read_table, write_table
from .metrics import classification_metrics, importance_rows, metric_rows
from .tracking import start_run
from .transformers import CategoryCaster

log = logging.getLogger(__name__)

CATEGORICAL = ["segment", "region", "customer_type"]
NUMERIC = [
    "tenure_months", "age", "product_count", "has_salary_transfer", "has_mobile_app",
    "txn_count_avg_6m", "txn_count_last", "txn_trend", "txn_amount_avg_6m",
    "balance_avg_6m", "balance_trend", "logins_avg_6m", "logins_trend",
    "complaints_6m", "salary_credit_ratio_6m", "inactive_months_last3",
]
FEATURES = CATEGORICAL + NUMERIC


def _panel(activity: pd.DataFrame, col: str, n: int, m: int) -> np.ndarray:
    return activity[col].to_numpy().reshape(n, m)


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.round(np.where(b > 0, a / np.maximum(b, 1e-9), np.where(a > 0, 2.0, 1.0)), 4)


def build_snapshot(customers: pd.DataFrame, activity: pd.DataFrame, t: int, s: Settings,
                   with_label: bool) -> pd.DataFrame:
    n, m = len(customers), s.n_months
    act = activity.sort_values(["customer_id", "month_idx"])
    L = s.churn_lookback_months
    w = slice(t - L + 1, t + 1)
    recent, prior = slice(t - 2, t + 1), slice(t - 5, t - 2)

    txn = _panel(act, "txn_count", n, m)
    amt = _panel(act, "txn_amount_sar", n, m)
    bal = _panel(act, "avg_balance_sar", n, m)
    lgn = _panel(act, "digital_logins", n, m)
    cmp_ = _panel(act, "complaints", n, m)
    sal = _panel(act, "salary_credited", n, m)
    alive = _panel(act, "is_active", n, m)

    df = customers[["customer_id", "segment", "region", "customer_type", "age", "product_count",
                    "has_salary_transfer", "has_mobile_app"]].copy()
    df["snapshot_idx"] = t
    df["tenure_months"] = customers["tenure_months_at_start"].to_numpy() + t
    df["txn_count_avg_6m"] = txn[:, w].mean(1).round(2)
    df["txn_count_last"] = txn[:, t]
    df["txn_trend"] = _ratio(txn[:, recent].mean(1), txn[:, prior].mean(1))
    df["txn_amount_avg_6m"] = amt[:, w].mean(1).round(2)
    df["balance_avg_6m"] = bal[:, w].mean(1).round(2)
    df["balance_trend"] = _ratio(bal[:, recent].mean(1), bal[:, prior].mean(1))
    df["logins_avg_6m"] = lgn[:, w].mean(1).round(2)
    df["logins_trend"] = _ratio(lgn[:, recent].mean(1), lgn[:, prior].mean(1))
    df["complaints_6m"] = cmp_[:, w].sum(1)
    df["salary_credit_ratio_6m"] = sal[:, w].mean(1).round(3)
    df["inactive_months_last3"] = (txn[:, recent] == 0).sum(1)
    df["is_active_now"] = alive[:, t]

    if with_label:
        df["churn_3m"] = (alive[:, t] & ~alive[:, t + s.churn_horizon_months]).astype(int)

    df = df[df["is_active_now"]].drop(columns="is_active_now")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    for c in ("has_salary_transfer", "has_mobile_app"):
        df[c] = df[c].astype(int)
    return df.reset_index(drop=True)


def reason_code(df: pd.DataFrame) -> pd.Series:
    """Readable early-warning signal for front-line teams (rule-based, complements the model)."""
    rules = [
        (df["salary_credit_ratio_6m"].lt(0.7) & df["has_salary_transfer"].eq(1), "Salary transfer stopped"),
        (df["complaints_6m"].ge(2), "Repeated complaints"),
        (df["logins_trend"].lt(0.6), "Digital engagement dropping"),
        (df["balance_trend"].lt(0.6), "Balance declining"),
        (df["txn_trend"].lt(0.6), "Transactions declining"),
        (df["product_count"].le(1), "Single-product customer"),
    ]
    out = pd.Series("No strong signal", index=df.index)
    for mask, label in reversed(rules):
        out = out.mask(mask, label)
    return out


def _model(seed: int) -> Pipeline:
    return Pipeline([
        ("cast", CategoryCaster(CATEGORICAL)),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=80,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            categorical_features="from_dtype", random_state=seed,
        )),
    ])


def run(s: Settings | None = None) -> dict:
    s = s or get_settings()
    customers = read_table(s.bronze, "customers")
    activity = read_table(s.bronze, "customer_monthly_activity")

    m, H, L = s.n_months, s.churn_horizon_months, s.churn_lookback_months
    last_labelled = m - 1 - H
    test_t = last_labelled
    train_ts = [t for t in range(L + 2, test_t - H + 1)][-6:]     # 6 older snapshots, no label leakage
    log.info("Churn: train snapshots %s, OOT test snapshot %s, scoring snapshot %s", train_ts, test_t, m - 1)

    train = pd.concat([build_snapshot(customers, activity, t, s, True) for t in train_ts], ignore_index=True)
    test = build_snapshot(customers, activity, test_t, s, True)
    write_table(train, s.silver, "churn_train_features")

    with start_run(s.mlflow_experiment, "churn", s.data_dir / "models", s.mlflow_uri) as run_:
        clf = _model(s.seed).fit(train[FEATURES], train["churn_3m"])
        p_test = clf.predict_proba(test[FEATURES])[:, 1]
        oot = classification_metrics(test["churn_3m"].to_numpy(), p_test)
        log.info("Churn OOT: AUC %.3f | PR-AUC %.3f | lift@10%% %.2f", oot["auc"], oot["pr_auc"], oot["top_decile_lift"])
        run_.params({"model": "HistGradientBoostingClassifier", "train_snapshots": train_ts, "test_snapshot": test_t,
                     "horizon_months": H, "n_train": len(train),
                     **{k.split("__")[1]: v for k, v in clf.get_params().items()
                        if k in ("clf__learning_rate", "clf__max_leaf_nodes", "clf__min_samples_leaf")}})
        run_.metrics({f"oot_{k}": v for k, v in oot.items()})
        imp = importance_rows(clf, test[FEATURES], test["churn_3m"], "churn", run_.version, "roc_auc", s.seed)

        # refit on everything labelled, then score the current month
        full = pd.concat([train, test], ignore_index=True)
        final = _model(s.seed).fit(full[FEATURES], full["churn_3m"])
        run_.model(final, example=full[FEATURES], register_as=f"{s.registry_prefix}churn" if s.mlflow_uri else None)
        version = run_.version

    current = build_snapshot(customers, activity, m - 1, s, False)
    current["churn_probability"] = final.predict_proba(current[FEATURES])[:, 1].round(5)
    current["risk_decile"] = (pd.qcut(current["churn_probability"].rank(method="first"), 10, labels=False) + 1).astype(int)
    current["risk_band"] = np.select([current["risk_decile"].eq(10), current["risk_decile"].ge(8)], ["High", "Medium"], "Low")
    current["balance_at_risk_sar"] = (current["balance_avg_6m"] * current["churn_probability"]).round(2)
    current["primary_signal"] = reason_code(current)
    current["score_date"] = s.as_of.isoformat()
    current["model_version"] = version

    scores = current[["customer_id", "score_date", "churn_probability", "risk_decile", "risk_band",
                      "balance_at_risk_sar", "primary_signal", "txn_trend", "logins_trend", "balance_trend",
                      "complaints_6m", "salary_credit_ratio_6m", "balance_avg_6m", "model_version"]]
    return {
        "fact_churn_score": scores,
        "metrics": metric_rows("churn", version, s.as_of, "out_of_time", oot),
        "importance": imp,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = run()
    print(out["fact_churn_score"].head())
