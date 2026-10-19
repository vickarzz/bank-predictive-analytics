# Model card

All three models are trained on **synthetic** data from `bpa.generate_data`. The numbers below show the method works end to end. They do not predict performance on real bank data.

## 1. Customer churn (`bpa.churn`)

| | |
|---|---|
| Target | Customer active at month *t* and inactive at *t + 3* |
| Population | Customers active at the snapshot month |
| Features | Last 6 months: transaction count/amount, balance, digital logins, complaints, salary-credit ratio, recent-vs-prior trends, inactive months; static: segment, region, tenure, product count, salary transfer, mobile app |
| Algorithm | `HistGradientBoostingClassifier` inside a pipeline that casts categoricals, so it accepts plain strings |
| Validation | Train on 6 older snapshots, test on the most recent labelled snapshot (out-of-time) |
| Result | AUC 0.82, PR-AUC 0.28 at a 2.1% base rate, top-decile lift 5.3× |
| Top drivers | Salary-credit ratio, latest transaction count, product count, salary transfer |
| Outputs | `churn_probability`, `risk_decile`, `risk_band` (High = top decile, Medium = deciles 8–9), `balance_at_risk_sar`, `primary_signal` (rule-based reason code) |
| Intended use | Prioritising retention outreach; not for pricing or credit decisions |

## 2. Credit risk — 12-month PD (`bpa.credit_risk`)

| | |
|---|---|
| Target | 90+ days past due within 12 months of origination |
| Population | Personal finance contracts with a complete 12-month window (training); all live contracts (scoring) |
| Features | DBR, bureau score, prior delinquencies, years employed, salary, principal, tenor, profit rate, employer sector, product, salary transfer |
| Excluded on purpose | Customer type, age, region |
| Algorithms | Logistic scorecard (champion) vs gradient boosting (challenger); champion = higher out-of-time AUC, with ties going to the scorecard |
| Validation | Newest 25% of originations held out (out-of-time) |
| Result | Scorecard AUC 0.80 / KS 0.47; GBM AUC 0.76 / KS 0.41 |
| Outputs | `pd_12m`, `risk_grade` (A < 1%, B < 3%, C < 7%, D < 15%, E ≥ 15%), `expected_loss_sar = PD × LGD (45%) × EAD` |
| Intended use | Portfolio monitoring and provisioning discussion; a production PD model would need regulatory validation (SAMA / IFRS 9) |

## 3. Remittance forecast (`bpa.remittance_forecast`)

| | |
|---|---|
| Target | Daily outbound remittance amount (SAR) per corridor |
| Method | log(amount) = linear trend per corridor + gradient-boosted residual on calendar features (payday window ~27th, Fri/Sat, Ramadan, days to Eid); quantile models give P10/P90 |
| Validation | Last 90 days held out |
| Result | MAPE 6.1%, WAPE 6.0%; P10–P90 coverage 73% (target 80%) |
| Intended use | Liquidity and FX planning per corridor |

## Monitoring built into the pipeline

- The quality gate blocks publishing if churn AUC < 0.70, credit AUC < 0.70, remittance MAPE > 15%, or probabilities fall outside [0, 1].
- Every run writes `model_metrics` and `feature_importance` to the gold layer. The *Model Health* page in Power BI reads them.
- With a tracking server configured, every run is logged to MLflow and the models are registered (local registry or Unity Catalog).
