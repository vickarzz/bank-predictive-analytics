"""Live demo of the Bank Predictive Analytics pipeline outputs.

Reads the gold layer (CSV). By default it uses the sample shipped in ./demo_data
(20k synthetic customers); point BPA_GOLD_DIR at data/gold to use your own run.

    streamlit run streamlit_app/streamlit_app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

GOLD = Path(os.environ.get("BPA_GOLD_DIR", Path(__file__).parent / "demo_data"))

# ---- palette (validated categorical order + reserved status colours) ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"High": "#d03b3b", "Medium": "#fab219", "Low": "#0ca30c"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

st.set_page_config(page_title="Bank Predictive Analytics", page_icon="📊", layout="wide")


@st.cache_data
def load() -> dict[str, pd.DataFrame]:
    t = {p.stem: pd.read_csv(p) for p in GOLD.glob("*.csv")}
    for name, cols in {"fact_remittance_daily": ["date"], "fact_remittance_forecast": ["date"],
                       "fact_loan_risk": ["origination_date"]}.items():
        for c in cols:
            t[name][c] = pd.to_datetime(t[name][c])
    return t


def style(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=INK2), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2), traceorder="normal"),
        hoverlabel=dict(bgcolor="white", font_color=INK), bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def bars(df: pd.DataFrame, x: str, y: str, fmt: str, horizontal: bool = False, colors=None) -> go.Figure:
    color = colors if colors is not None else SERIES[0]
    trace = (go.Bar(y=df[x], x=df[y], orientation="h", marker_color=color, hovertemplate=f"%{{y}}<br>%{{x:{fmt}}}<extra></extra>")
             if horizontal else
             go.Bar(x=df[x], y=df[y], marker_color=color, hovertemplate=f"%{{x}}<br>%{{y:{fmt}}}<extra></extra>"))
    fig = go.Figure(trace)
    fig.update_traces(marker_line_width=0, marker_cornerradius=4)
    if horizontal:
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return style(fig)


def sar(v: float) -> str:
    for unit, div in (("bn", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"SAR {v / div:,.1f}{unit}"
    return f"SAR {v:,.0f}"


def pick(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """Keep rows whose column value is in the selected list; an empty selection keeps everything."""
    mask = pd.Series(True, index=df.index)
    for col, values in filters.items():
        if values:
            mask &= df[col].isin(values)
    return df[mask]


def metric(df: pd.DataFrame, model: str, name: str) -> float:
    s = df[(df.model_name == model) & (df.metric == name)]["value"]
    return float(s.iloc[0]) if len(s) else float("nan")


t = load()
cust, churn, loans = t["dim_customer"], t["fact_churn_score"], t["fact_loan_risk"]
daily, fc, mm, fi = t["fact_remittance_daily"], t["fact_remittance_forecast"], t["model_metrics"], t["feature_importance"]
churn = churn.merge(cust[["customer_id", "segment", "region"]], on="customer_id", how="left")

st.title("Bank Predictive Analytics")
st.caption("Churn · credit risk · remittance forecasting for a synthetic Saudi retail bank. "
           "Models are trained and validated out-of-time by the pipeline; this app reads its gold output. "
           "[Source code](https://github.com/vickarzz/bank-predictive-analytics)")

tab_over, tab_churn, tab_credit, tab_rem, tab_model = st.tabs(
    ["Overview", "Churn risk", "Credit risk", "Remittance forecast", "Model health"])

# ------------------------------------------------------------------ Overview
with tab_over:
    c = st.columns(6)
    c[0].metric("High-risk customers", f"{(churn.risk_band == 'High').sum():,}")
    c[1].metric("Balance at risk", sar(churn.balance_at_risk_sar.sum()))
    c[2].metric("Credit exposure (EAD)", sar(loans.ead_sar.sum()))
    c[3].metric("Expected loss", sar(loans.expected_loss_sar.sum()))
    c[4].metric("Remittance next 90d", sar(fc.forecast_amount_sar.sum()))
    c[5].metric("Churn model AUC", f"{metric(mm, 'churn', 'auc'):.2f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Balance at risk by segment")
        d = churn.groupby("segment", as_index=False).balance_at_risk_sar.sum().sort_values("balance_at_risk_sar", ascending=False)
        st.plotly_chart(bars(d, "segment", "balance_at_risk_sar", ",.0f", horizontal=True), width="stretch")
    with right:
        st.subheader("Expected loss by risk grade (SAR)")
        d = loans.groupby("risk_grade", as_index=False).expected_loss_sar.sum()
        st.plotly_chart(bars(d, "risk_grade", "expected_loss_sar", ",.0f"), width="stretch")
    st.subheader("Total daily remittance — last 180 days and next 90 days (SAR)")
    a = daily.groupby("date", as_index=False).amount_sar.sum()
    a = a[a.date >= a.date.max() - pd.Timedelta(days=180)]
    f = fc.groupby("date", as_index=False).forecast_amount_sar.sum()
    fig = go.Figure([
        go.Scatter(x=a.date, y=a.amount_sar, name="Actual", line=dict(color=SERIES[0], width=2)),
        go.Scatter(x=f.date, y=f.forecast_amount_sar, name="Forecast", line=dict(color=SERIES[1], width=2)),
    ])
    fig.update_traces(hovertemplate="%{y:,.0f}")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 320, legend=True), width="stretch")

# ------------------------------------------------------------------ Churn
with tab_churn:
    f1, f2, _ = st.columns([1, 1, 2])
    seg = f1.multiselect("Segment", sorted(churn.segment.unique()))
    reg = f2.multiselect("Region", sorted(churn.region.unique()))
    d = pick(churn, segment=seg, region=reg)
    c = st.columns(5)
    c[0].metric("Customers scored", f"{len(d):,}")
    c[1].metric("High-risk customers", f"{(d.risk_band == 'High').sum():,}")
    c[2].metric("Expected churners (3M)", f"{d.churn_probability.sum():,.0f}")
    c[3].metric("Balance at risk", sar(d.balance_at_risk_sar.sum()))
    c[4].metric("Avg churn probability", f"{d.churn_probability.mean():.2%}")
    left, right = st.columns(2)
    with left:
        st.subheader("High-risk customers by primary signal")
        s = d[d.risk_band == "High"].groupby("primary_signal", as_index=False).size().sort_values("size", ascending=False)
        st.plotly_chart(bars(s, "primary_signal", "size", ",.0f", horizontal=True), width="stretch")
    with right:
        st.subheader("Average churn probability by risk decile")
        s = d.groupby("risk_decile", as_index=False).agg(p=("churn_probability", "mean"), band=("risk_band", "first"))
        fig = bars(s, "risk_decile", "p", ".2%", colors=[STATUS[b] for b in s.band])
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(dtick=1, title_text="Decile (10 = highest risk)", title_font=dict(color=MUTED))
        st.plotly_chart(fig, width="stretch")
        st.caption("Colour = risk band: red High (decile 10), amber Medium (8–9), green Low.")
    st.subheader("Retention worklist — highest churn probability first")
    w = d.sort_values("churn_probability", ascending=False).head(200)
    st.dataframe(
        w[["customer_id", "segment", "region", "risk_band", "primary_signal", "churn_probability", "balance_at_risk_sar"]],
        hide_index=True, width="stretch",
        column_config={"customer_id": st.column_config.NumberColumn("Customer ID", format="%d"),
                       "segment": "Segment", "region": "Region", "risk_band": "Risk band", "primary_signal": "Primary signal",
                       "churn_probability": st.column_config.ProgressColumn("Churn probability", format="percent", min_value=0, max_value=1),
                       "balance_at_risk_sar": st.column_config.NumberColumn("Balance at risk (SAR)", format="%,.0f")})

# ------------------------------------------------------------------ Credit
with tab_credit:
    f1, f2, _ = st.columns([1, 1, 2])
    prod = f1.multiselect("Product", sorted(loans["product"].unique()))
    sect = f2.multiselect("Employer sector", sorted(loans.employer_sector.unique()))
    d = pick(loans, product=prod, employer_sector=sect)
    c = st.columns(5)
    c[0].metric("Live loans", f"{len(d):,}")
    c[1].metric("EAD", sar(d.ead_sar.sum()))
    c[2].metric("Expected loss", sar(d.expected_loss_sar.sum()))
    c[3].metric("EL rate", f"{d.expected_loss_sar.sum() / max(d.ead_sar.sum(), 1):.2%}")
    c[4].metric("Loans above 45% DBR", f"{(d.dbr > 0.45).sum():,}")
    left, right = st.columns(2)
    with left:
        st.subheader("EAD by risk grade (SAR)")
        st.plotly_chart(bars(d.groupby("risk_grade", as_index=False).ead_sar.sum(), "risk_grade", "ead_sar", ",.0f"),
                        width="stretch")
    with right:
        st.subheader("Average 12-month PD by DBR band")
        order = ["<=25%", "25-33%", "33-45%", ">45%"]
        s = d.groupby("dbr_band", as_index=False).pd_12m.mean().set_index("dbr_band").reindex(order).reset_index()
        fig = bars(s, "dbr_band", "pd_12m", ".2%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Watchlist — largest expected loss first")
    st.dataframe(
        d.sort_values("expected_loss_sar", ascending=False).head(200)[
            ["loan_id", "product", "risk_grade", "dbr_band", "employer_sector", "bureau_score", "pd_12m", "ead_sar", "expected_loss_sar"]],
        hide_index=True, width="stretch",
        column_config={"loan_id": st.column_config.NumberColumn("Loan ID", format="%d"), "product": "Product",
                       "risk_grade": "Grade", "dbr_band": "DBR band", "employer_sector": "Employer sector",
                       "bureau_score": "Bureau score",
                       "pd_12m": st.column_config.NumberColumn("PD 12M", format="percent"),
                       "ead_sar": st.column_config.NumberColumn("EAD (SAR)", format="%,.0f"),
                       "expected_loss_sar": st.column_config.NumberColumn("Expected loss (SAR)", format="%,.0f")})

# ------------------------------------------------------------------ Remittance
with tab_rem:
    corridors = list(fc.corridor.drop_duplicates())
    pick = st.selectbox("Corridor", ["All corridors"] + corridors)
    da = daily if pick == "All corridors" else daily[daily.corridor == pick]
    df_ = fc if pick == "All corridors" else fc[fc.corridor == pick]
    a = da.groupby("date", as_index=False)[["amount_sar", "backtest_forecast_sar"]].sum(min_count=1)
    f = df_.groupby("date", as_index=False)[["forecast_amount_sar", "lower_p10_sar", "upper_p90_sar"]].sum()
    bt = da[da.is_backtest_period]
    mape = ((bt.backtest_forecast_sar - bt.amount_sar).abs() / bt.amount_sar).mean()
    c = st.columns(4)
    c[0].metric("Forecast next 90 days", sar(f.forecast_amount_sar.sum()))
    c[1].metric("Low case (P10)", sar(f.lower_p10_sar.sum()))
    c[2].metric("High case (P90)", sar(f.upper_p90_sar.sum()))
    c[3].metric("Backtest MAPE (90 days)", f"{mape:.1%}")
    st.subheader("Daily remittance: actual, backtest and forecast with P10–P90 band (SAR)")
    recent = a[a.date >= a.date.max() - pd.Timedelta(days=180)]
    fig = go.Figure([
        go.Scatter(x=f.date, y=f.upper_p90_sar, line=dict(width=0), hoverinfo="skip", showlegend=False),
        go.Scatter(x=f.date, y=f.lower_p10_sar, line=dict(width=0), fill="tonexty",
                   fillcolor="rgba(235,104,52,0.15)", name="P10–P90 band", hovertemplate="P10 %{y:,.0f}<extra></extra>"),
        go.Scatter(x=recent.date, y=recent.amount_sar, name="Actual", line=dict(color=SERIES[0], width=2),
                   hovertemplate="Actual %{y:,.0f}<extra></extra>"),
        go.Scatter(x=recent.date, y=recent.backtest_forecast_sar, name="Backtest", line=dict(color=INK2, width=2, dash="dot"),
                   hovertemplate="Backtest %{y:,.0f}<extra></extra>"),
        go.Scatter(x=f.date, y=f.forecast_amount_sar, name="Forecast", line=dict(color=SERIES[1], width=2),
                   hovertemplate="Forecast %{y:,.0f}<extra></extra>"),
    ])
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 380, legend=True), width="stretch")
    st.caption("Peaks: Saudi payday window (~27th of the month), Saturdays, and the days before Eid. Last 180 days of history shown.")
    st.subheader("90-day forecast by corridor (SAR)")
    s = fc.groupby("corridor", as_index=False).forecast_amount_sar.sum().sort_values("forecast_amount_sar", ascending=False)
    colors = [SERIES[corridors.index(x) % len(SERIES)] for x in s.corridor]   # colour follows the corridor
    st.plotly_chart(bars(s, "corridor", "forecast_amount_sar", ",.0f", horizontal=True, colors=colors),
                    width="stretch")

# ------------------------------------------------------------------ Model health
with tab_model:
    c = st.columns(4)
    c[0].metric("Churn AUC (out-of-time)", f"{metric(mm, 'churn', 'auc'):.3f}")
    c[1].metric("Credit AUC (out-of-time)", f"{metric(mm, 'credit_risk', 'auc'):.3f}")
    c[2].metric("Credit Gini", f"{metric(mm, 'credit_risk', 'gini'):.3f}")
    c[3].metric("Remittance MAPE (backtest)", f"{metric(mm, 'remittance_forecast', 'mape'):.2%}")
    left, right = st.columns(2)
    with left:
        model = st.radio("Feature importance for", ["churn", "credit_risk"], horizontal=True,
                         format_func=lambda m: {"churn": "Churn", "credit_risk": "Credit risk"}[m])
        s = fi[fi.model_name == model].sort_values("importance", ascending=False).head(10)
        st.plotly_chart(bars(s, "feature", "importance", ".4f", horizontal=True), width="stretch")
        st.caption("Permutation importance = drop in AUC when the feature is shuffled.")
    with right:
        st.subheader("All metrics")
        st.dataframe(mm[["model_name", "dataset", "metric", "value", "model_version"]], hide_index=True,
                     width="stretch", height=420,
                     column_config={"model_name": "Model", "dataset": "Dataset", "metric": "Metric",
                                    "value": st.column_config.NumberColumn("Value", format="%.4f"),
                                    "model_version": "Version"})

st.divider()
st.caption("All data is synthetic. Built with Python, scikit-learn, MLflow, Airflow / Databricks and Power BI · "
           "Ahmad Zulfikar Rafsanjani")
