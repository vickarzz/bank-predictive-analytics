"""Use case 2 - Credit risk: 12-month probability of default (PD) for personal finance.

Design
* Training population: loans with a complete 12-month performance window.
* Out-of-time split: the most recent 25% of originations are the test set.
* Champion = gradient boosting; challenger = logistic regression (scorecard-style baseline).
* Protected / proxy attributes (customer type, age, region) are deliberately excluded.
* Output: PD, risk grade (A-E), and expected loss EL = PD x LGD x EAD for every live loan.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Settings, get_settings
from .io import read_table
from .metrics import classification_metrics, importance_rows, metric_rows
from .tracking import start_run
from .transformers import CategoryCaster

log = logging.getLogger(__name__)

CATEGORICAL = ["product", "employer_sector"]
NUMERIC = ["dbr", "bureau_score", "prior_delinquencies", "years_employed", "monthly_salary_sar",
           "principal_sar", "tenor_months", "profit_rate", "has_salary_transfer"]
FEATURES = CATEGORICAL + NUMERIC

GRADES = [(0.01, "A"), (0.03, "B"), (0.07, "C"), (0.15, "D"), (1.01, "E")]


def grade(pd_: np.ndarray) -> np.ndarray:
    out = np.full(len(pd_), "E", dtype=object)
    for cut, g in reversed(GRADES):
        out[pd_ < cut] = g
    return out


def dbr_band(dbr: pd.Series) -> pd.Series:
    return pd.cut(dbr, [0, 0.25, 0.33, 0.45, 1.0], labels=["<=25%", "25-33%", "33-45%", ">45%"], include_lowest=True).astype(str)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["has_salary_transfer"] = X["has_salary_transfer"].astype(int)
    return X


def _gbm(seed: int) -> Pipeline:
    return Pipeline([
        ("cast", CategoryCaster(CATEGORICAL)),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=2.0, early_stopping=True, categorical_features="from_dtype", random_state=seed,
        )),
    ])


def _logit() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ])),
        ("lr", LogisticRegression(max_iter=2000, C=0.5)),
    ])


def run(s: Settings | None = None) -> dict:
    s = s or get_settings()
    loans = read_table(s.bronze, "loans")
    loans["origination_date"] = pd.to_datetime(loans["origination_date"])

    dev = loans[loans["performance_window_complete"]].sort_values("origination_date")
    cut = int(len(dev) * 0.75)
    train, test = dev.iloc[:cut], dev.iloc[cut:]
    log.info("Credit: %s train / %s OOT test loans, default rate %.2f%%", len(train), len(test), 100 * dev["default_12m"].mean())

    with start_run(s.mlflow_experiment, "credit_risk", s.data_dir / "models", s.mlflow_uri) as run_:
        Xtr, Xte, y_te = _prep(train), _prep(test), test["default_12m"].to_numpy()
        candidates = {"gradient_boosting": _gbm, "logistic_scorecard": lambda seed: _logit()}
        fitted, results = {}, {}
        for name, make in candidates.items():
            fitted[name] = make(s.seed).fit(Xtr, train["default_12m"])
            results[name] = classification_metrics(y_te, fitted[name].predict_proba(Xte)[:, 1])
            log.info("Credit OOT %-20s AUC %.3f | KS %.3f", name, results[name]["auc"], results[name]["ks"])
        # Champion = best out-of-time AUC. Ties (<0.005) go to the more explainable scorecard.
        best = max(results, key=lambda k: results[k]["auc"])
        if abs(results["gradient_boosting"]["auc"] - results["logistic_scorecard"]["auc"]) < 0.005:
            best = "logistic_scorecard"
        challenger = next(k for k in results if k != best)
        log.info("Credit champion: %s", best)

        run_.params({"champion": best, "challenger": challenger, "n_train": len(train), "n_test": len(test),
                     "lgd": s.lgd, "excluded_attributes": "customer_type, age, region"})
        run_.metrics({**{f"oot_{k}": v for k, v in results[best].items()},
                      **{f"challenger_oot_{k}": v for k, v in results[challenger].items()}})
        imp = importance_rows(fitted[best], Xte, y_te, "credit_risk", run_.version, "roc_auc", s.seed)

        X_dev = _prep(dev)
        final = candidates[best](s.seed).fit(X_dev, dev["default_12m"])
        run_.model(final, example=X_dev, register_as=f"{s.registry_prefix}credit_risk" if s.mlflow_uri else None)
        version = run_.version
        m_champ, m_chall = results[best], results[challenger]

    live = loans[loans["outstanding_sar"] > 0].copy()
    live["pd_12m"] = final.predict_proba(_prep(live))[:, 1].round(5)
    live["risk_grade"] = grade(live["pd_12m"].to_numpy())
    live["lgd"] = s.lgd
    live["ead_sar"] = live["outstanding_sar"]
    live["expected_loss_sar"] = (live["pd_12m"] * s.lgd * live["ead_sar"]).round(2)
    live["dbr_band"] = dbr_band(live["dbr"])
    live["origination_date"] = live["origination_date"].dt.strftime("%Y-%m-%d")
    live["score_date"] = s.as_of.isoformat()
    live["model_version"] = version
    live["actual_default_12m"] = live["default_12m"].replace(-1, np.nan)

    cols = ["loan_id", "customer_id", "score_date", "product", "origination_date", "principal_sar", "outstanding_sar",
            "tenor_months", "profit_rate", "monthly_salary_sar", "dbr", "dbr_band", "employer_sector", "bureau_score",
            "prior_delinquencies", "pd_12m", "risk_grade", "lgd", "ead_sar", "expected_loss_sar",
            "actual_default_12m", "model_type", "model_version"]
    live["model_type"] = best
    rows = (metric_rows("credit_risk", version, s.as_of, "out_of_time", m_champ)
            + metric_rows(f"credit_risk_challenger_{challenger}", version, s.as_of, "out_of_time", m_chall))
    return {"fact_loan_risk": live[cols], "metrics": rows, "importance": imp}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run()["fact_loan_risk"].head())
