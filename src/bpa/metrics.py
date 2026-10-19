"""Model evaluation helpers that produce rows for the gold `model_metrics` table."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def top_decile_lift(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    cut = np.quantile(y_score, 0.9)
    top = y_true[y_score >= cut]
    return float(top.mean() / max(y_true.mean(), 1e-9))


def classification_metrics(y_true, y_score) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "pr_auc": average_precision_score(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "top_decile_lift": top_decile_lift(y_true, y_score),
        "base_rate": float(np.mean(y_true)),
        "n_test": float(len(y_true)),
    }


def metric_rows(model_name: str, version: str, run_date: date, dataset: str, metrics: dict[str, float]) -> list[dict]:
    return [
        {"model_name": model_name, "model_version": version, "run_date": run_date.isoformat(),
         "dataset": dataset, "metric": k, "value": round(float(v), 6)}
        for k, v in metrics.items()
    ]


def importance_rows(model, X: pd.DataFrame, y, model_name: str, version: str, scoring: str, seed: int = 42,
                    max_rows: int = 20_000) -> list[dict]:
    if len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    r = permutation_importance(model, X, y, scoring=scoring, n_repeats=3, random_state=seed, n_jobs=1)
    out = [
        {"model_name": model_name, "model_version": version, "feature": f,
         "importance": round(float(m), 6), "importance_std": round(float(s), 6)}
        for f, m, s in zip(X.columns, r.importances_mean, r.importances_std)
    ]
    out.sort(key=lambda d: d["importance"], reverse=True)
    for rank, row in enumerate(out, 1):
        row["rank"] = rank
    return out
