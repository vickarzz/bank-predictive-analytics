"""Synthetic Saudi retail-banking data (bronze layer).

Nothing here comes from a real bank. The generator plants realistic, *learnable*
patterns so the models have something genuine to find:

* customers   - one row per customer (segment, region, tenure, salary transfer ...)
* activity    - one row per customer per month (transactions, balance, logins, complaints)
* loans       - personal finance contracts with bureau score, DBR and 12-month default outcome
* remittance  - daily outbound remittance volume per corridor (payday, weekend, Ramadan/Eid effects)
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .io import write_table

log = logging.getLogger(__name__)

REGIONS = np.array(["Riyadh", "Makkah", "Eastern", "Madinah", "Qassim", "Asir"])
REGION_P = np.array([0.34, 0.24, 0.18, 0.09, 0.08, 0.07])
SEGMENTS = np.array(["Mass", "Mass Affluent", "Affluent", "Private"])
SEGMENT_P = np.array([0.62, 0.25, 0.10, 0.03])
SECTORS = np.array(["Government", "Semi-Government", "Private - Large", "Private - SME"])
SECTOR_P = np.array([0.38, 0.14, 0.30, 0.18])

# Eid dates (Gregorian) used to shape remittance seasonality
EID_DATES = pd.to_datetime([
    "2024-04-10", "2024-06-16", "2025-03-30", "2025-06-06",
    "2026-03-20", "2026-05-27", "2027-03-10", "2027-05-16",
])
RAMADAN_STARTS = pd.to_datetime(["2024-03-11", "2025-03-01", "2026-02-18", "2027-02-08"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def month_ends(s: Settings) -> pd.DatetimeIndex:
    end = pd.Timestamp(s.as_of) + pd.offsets.MonthEnd(0)
    return pd.date_range(end=end, periods=s.n_months, freq="ME")


# --------------------------------------------------------------------------------------
# Customers + monthly activity
# --------------------------------------------------------------------------------------
def generate_customers_and_activity(s: Settings, rng: np.random.Generator):
    n, m = s.n_customers, s.n_months
    ids = np.arange(1, n + 1)

    segment = rng.choice(SEGMENTS, n, p=SEGMENT_P)
    seg_rank = pd.Series(segment).map({"Mass": 0, "Mass Affluent": 1, "Affluent": 2, "Private": 3}).to_numpy()

    region = rng.choice(REGIONS, n, p=REGION_P)
    age = np.clip(rng.normal(37, 10, n), 21, 70).round().astype(int)
    tenure0 = rng.gamma(2.0, 30, n).round().astype(int) + 1          # tenure (months) at start of window
    salary_transfer = rng.random(n) < (0.55 + 0.08 * seg_rank)
    products = np.clip(rng.poisson(1.6 + 0.6 * seg_rank + 0.5 * salary_transfer), 1, 7)
    digital = rng.random(n) < 0.78                                     # has the mobile app
    complaint_prone = rng.random(n) < 0.12
    customer_type = np.where(rng.random(n) < 0.68, "Saudi", "Expat")

    base_balance = np.exp(rng.normal(9.0 + 0.9 * seg_rank, 0.8))       # SAR
    base_txn = rng.gamma(4, 4, n) * (1 + 0.3 * seg_rank)
    base_logins = np.where(digital, rng.gamma(3, 5, n), rng.gamma(1, 0.8, n))
    avg_ticket = np.exp(rng.normal(5.2 + 0.4 * seg_rank, 0.5))

    # ---- churn hazard (monthly). Drivers the model should rediscover. ----
    logit = (
        -3.4
        - 1.1 * salary_transfer
        - 0.35 * (products - 1)
        + 1.2 * complaint_prone
        - 0.45 * digital
        - 0.012 * np.minimum(tenure0, 120)
        + 0.25 * (age < 26)
        + 0.30 * (customer_type == "Expat")
        + rng.normal(0, 0.35, n)
    )
    hazard = _sigmoid(logit)
    # month the customer leaves (index into the window); >= m means still active
    churn_month = rng.geometric(hazard) + 1                             # first month the customer is gone

    # ---- monthly panel (n x m) ----
    mi = np.arange(m)[None, :]
    to_churn = churn_month[:, None] - mi                                # months until churn
    active = to_churn > 0
    # Most leavers disengage gradually over 2-6 months; ~35% leave abruptly with no warning.
    fade_len = rng.integers(2, 7, n)[:, None]
    gradual = (rng.random(n) < 0.65)[:, None]
    fade_floor = rng.uniform(0.2, 0.7, n)[:, None]
    fade = np.where(gradual, np.clip(to_churn / fade_len, fade_floor, 1.0), 1.0)
    # Loyal customers also have temporary dips (travel, summer, job change) -> realistic false positives
    dip_start = rng.integers(0, m, n)[:, None]
    has_dip = (rng.random(n) < 0.18)[:, None]
    dip = np.where(has_dip & (mi >= dip_start) & (mi < dip_start + rng.integers(1, 4, n)[:, None]),
                   rng.uniform(0.25, 0.7, n)[:, None], 1.0)
    fade = np.where(active, fade * dip, 0.0)
    season = 1 + 0.08 * np.sin(2 * np.pi * (mi + 3) / 12)

    lam_txn = base_txn[:, None] * fade * season
    txn_count = rng.poisson(lam_txn)
    txn_amount = txn_count * avg_ticket[:, None] * rng.lognormal(0, 0.25, (n, m))
    avg_balance = np.where(
        active, base_balance[:, None] * (0.3 + 0.7 * fade) * rng.lognormal(0, 0.18, (n, m)),
        rng.uniform(0, 50, (n, m)),
    )
    logins = rng.poisson(base_logins[:, None] * fade)
    near = (to_churn > 0) & (to_churn <= 4) & gradual
    complaints = rng.poisson(0.03 + 0.10 * complaint_prone[:, None] + 0.25 * near * complaint_prone[:, None])
    salary_credited = salary_transfer[:, None] & active & (rng.random((n, m)) < np.where(near, 0.55, 0.97))

    months = month_ends(s)
    activity = pd.DataFrame({
        "customer_id": np.repeat(ids, m).astype(np.int32),
        "month_end": np.tile(months.values, n),
        "month_idx": np.tile(np.arange(m), n).astype(np.int16),
        "is_active": active.ravel(),
        "txn_count": txn_count.ravel().astype(np.int32),
        "txn_amount_sar": txn_amount.ravel().round(2),
        "avg_balance_sar": avg_balance.ravel().round(2),
        "digital_logins": logins.ravel().astype(np.int32),
        "complaints": complaints.ravel().astype(np.int16),
        "salary_credited": salary_credited.ravel(),
    })

    churned_in_window = churn_month < m
    customers = pd.DataFrame({
        "customer_id": ids.astype(np.int32),
        "segment": segment,
        "region": region,
        "customer_type": customer_type,
        "age": age,
        "tenure_months_at_start": tenure0,
        "has_salary_transfer": salary_transfer,
        "product_count": products,
        "has_mobile_app": digital,
        "status": np.where(churned_in_window, "Closed", "Active"),
        "closed_month": np.where(churned_in_window, pd.Series(months).reindex(np.minimum(churn_month, m - 1)).dt.strftime("%Y-%m-%d").to_numpy(), None),
    })
    return customers, activity


# --------------------------------------------------------------------------------------
# Loans (personal finance) with 12-month default outcome
# --------------------------------------------------------------------------------------
def generate_loans(s: Settings, customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_loans = int(len(customers) * 0.32)
    cust = customers.sample(n_loans, random_state=s.seed, replace=False).reset_index(drop=True)
    seg_rank = cust["segment"].map({"Mass": 0, "Mass Affluent": 1, "Affluent": 2, "Private": 3}).to_numpy()

    sector = rng.choice(SECTORS, n_loans, p=SECTOR_P)
    salary = np.exp(rng.normal(9.1 + 0.45 * seg_rank, 0.35)).round(-1)               # monthly SAR
    years_employed = np.clip(rng.gamma(2.2, 3.0, n_loans), 0.3, 35).round(1)
    bureau_score = np.clip(rng.normal(690 + 25 * seg_rank, 70, n_loans), 300, 900).round().astype(int)
    prior_delinq = rng.poisson(np.where(bureau_score < 600, 1.1, 0.15))
    tenor = rng.choice([12, 24, 36, 48, 60], n_loans, p=[0.08, 0.17, 0.25, 0.25, 0.25])
    profit_rate = np.clip(0.035 + 0.00012 * (750 - bureau_score) + rng.normal(0, 0.006, n_loans), 0.025, 0.12)
    principal = np.clip(salary * rng.uniform(4, 18, n_loans), 10_000, 1_500_000).round(-2)
    installment = (principal * (1 + profit_rate * tenor / 12) / tenor).round(2)
    other_obligations = salary * rng.beta(2, 8, n_loans)
    dbr = np.clip((installment + other_obligations) / salary, 0.05, 0.75)              # SAMA cap is 33%-45% typical

    start = pd.Timestamp(s.as_of) - pd.DateOffset(months=s.n_months)
    span_days = (pd.Timestamp(s.as_of) - start).days
    origination = start + pd.to_timedelta(rng.integers(0, span_days, n_loans), unit="D")

    product = rng.choice(["Murabaha Personal Finance", "Tawarruq Cash Finance", "Auto Ijara"], n_loans, p=[0.6, 0.3, 0.1])

    logit = (
        -3.3
        + 5.0 * (dbr - 0.33)
        - 0.012 * (bureau_score - 690)
        + 0.55 * prior_delinq
        - 0.06 * np.minimum(years_employed, 15)
        + pd.Series(sector).map({"Government": -0.7, "Semi-Government": -0.4, "Private - Large": 0.0, "Private - SME": 0.6}).to_numpy()
        - 0.5 * cust["has_salary_transfer"].to_numpy()
        + rng.normal(0, 0.3, n_loans)
    )
    pd12 = _sigmoid(logit)
    defaulted = rng.random(n_loans) < pd12
    months_on_book = ((pd.Timestamp(s.as_of) - origination).days / 30.44).astype(int)
    performance_window_complete = months_on_book >= 12

    loans = pd.DataFrame({
        "loan_id": np.arange(100001, 100001 + n_loans, dtype=np.int64),
        "customer_id": cust["customer_id"].to_numpy(),
        "product": product,
        "origination_date": origination.strftime("%Y-%m-%d"),
        "principal_sar": principal,
        "tenor_months": tenor,
        "profit_rate": profit_rate.round(4),
        "monthly_installment_sar": installment,
        "monthly_salary_sar": salary,
        "dbr": dbr.round(4),
        "employer_sector": sector,
        "years_employed": years_employed,
        "bureau_score": bureau_score,
        "prior_delinquencies": prior_delinq,
        "has_salary_transfer": cust["has_salary_transfer"].to_numpy(),
        "months_on_book": months_on_book,
        "performance_window_complete": performance_window_complete,
        # outcome only observable once the 12-month window has passed
        "default_12m": np.where(performance_window_complete, defaulted.astype(int), -1),
    })
    # outstanding balance (simple straight-line amortisation) used for EAD
    paid_ratio = np.clip(months_on_book / tenor, 0, 1)
    loans["outstanding_sar"] = (principal * (1 - paid_ratio)).round(2)
    return loans


# --------------------------------------------------------------------------------------
# Daily remittance per corridor
# --------------------------------------------------------------------------------------
def remittance_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DataFrame({"date": dates})
    d["dow"] = d["date"].dt.dayofweek               # Mon=0 ... Sun=6
    d["dom"] = d["date"].dt.day
    d["month"] = d["date"].dt.month
    # Saudi payroll lands around the 27th; volume stays high for ~5 days
    d["payday_window"] = ((d["dom"] >= 27) | (d["dom"] <= 2)).astype(int)
    # days until the next Eid (capped) and Ramadan flag
    eid = EID_DATES.values
    nxt = np.searchsorted(eid, d["date"].values)
    nxt = np.clip(nxt, 0, len(eid) - 1)
    d["days_to_eid"] = np.clip((eid[nxt] - d["date"].values).astype("timedelta64[D]").astype(int), -1, 60)
    d.loc[d["days_to_eid"] < 0, "days_to_eid"] = 60
    d["pre_eid_window"] = (d["days_to_eid"] <= 10).astype(int)
    ram = np.zeros(len(d), dtype=int)
    for r in RAMADAN_STARTS:
        ram |= ((d["date"] >= r) & (d["date"] < r + pd.Timedelta(days=30))).to_numpy()
    d["ramadan"] = ram
    d["weekend"] = d["dow"].isin([4, 5]).astype(int)  # Fri-Sat
    return d


def generate_remittance(s: Settings, rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp(s.as_of) - pd.DateOffset(months=s.n_months) + pd.Timedelta(days=1)
    dates = pd.date_range(start, pd.Timestamp(s.as_of), freq="D")
    cal = remittance_calendar(dates)
    base = {"Indonesia": 2.4e6, "India": 6.5e6, "Pakistan": 4.8e6, "Philippines": 2.1e6, "Egypt": 3.9e6, "Bangladesh": 2.7e6}
    growth = {"Indonesia": 0.10, "India": 0.06, "Pakistan": 0.05, "Philippines": 0.04, "Egypt": -0.02, "Bangladesh": 0.08}
    t = np.arange(len(dates)) / 365.0
    frames = []
    for c in s.corridors:
        level = base[c] * (1 + growth[c]) ** t
        mult = (
            1
            + 0.38 * cal["payday_window"]
            + 0.55 * cal["pre_eid_window"] * (1 - cal["days_to_eid"] / 12)
            + 0.12 * cal["ramadan"]
            + np.where(cal["dow"] == 5, 0.18, 0) - np.where(cal["dow"] == 4, 0.22, 0)   # Sat peak, Fri dip
        )
        amount = level * mult * rng.lognormal(0, 0.07, len(dates))
        avg_ticket = {"Indonesia": 1350, "India": 1650, "Pakistan": 1500, "Philippines": 1250, "Egypt": 2100, "Bangladesh": 1100}[c]
        frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "corridor": c,
            "amount_sar": amount.round(2),
            "txn_count": (amount / avg_ticket * rng.lognormal(0, 0.03, len(dates))).round().astype(int),
        }))
    return pd.concat(frames, ignore_index=True)


def run(s: Settings | None = None) -> dict[str, int]:
    s = s or get_settings()
    rng = np.random.default_rng(s.seed)
    log.info("Generating %s customers x %s months", f"{s.n_customers:,}", s.n_months)
    customers, activity = generate_customers_and_activity(s, rng)
    loans = generate_loans(s, customers, rng)
    remittance = generate_remittance(s, rng)

    write_table(customers, s.bronze, "customers")
    write_table(activity, s.bronze, "customer_monthly_activity")
    write_table(loans, s.bronze, "loans")
    write_table(remittance, s.bronze, "remittance_daily")
    counts = {"customers": len(customers), "activity": len(activity), "loans": len(loans), "remittance": len(remittance)}
    log.info("Bronze rows: %s", counts)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
