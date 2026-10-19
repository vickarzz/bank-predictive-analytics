"""Generate the Power BI project (PBIP: TMDL semantic model + PBIR report) from code.

    python powerbi/build_pbip.py

Why generate instead of hand-editing? The model, measures and report layout live in one
reviewable Python spec, every rebuild is deterministic (stable IDs), and the output is plain
text that diffs cleanly in Git. Open powerbi/BankPredictive.pbip in Power BI Desktop afterwards.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "BankPredictive"
SM_DIR = ROOT / f"{NAME}.SemanticModel"
RP_DIR = ROOT / f"{NAME}.Report"
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"


def guid(*parts: str) -> str:
    """Deterministic UUID so rebuilding does not churn every lineageTag in Git."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "bpa/" + "/".join(parts)))


def hexid(*parts: str) -> str:
    return hashlib.sha1("/".join(parts).encode()).hexdigest()[:20]


def q(name: str) -> str:
    """Quote a TMDL/DAX object name when needed."""
    return f"'{name}'" if any(ch in name for ch in " -()%/&.'") or name[0].isdigit() else name


# ======================================================================================
# Semantic model spec
# ======================================================================================
@dataclass
class Col:
    source: str
    name: str
    dtype: str                      # int64 | double | string | dateTime | boolean
    fmt: str | None = None
    summarize: str = "none"
    hidden: bool = False
    key: bool = False
    sort_by: str | None = None


@dataclass
class Measure:
    name: str
    expr: str
    fmt: str
    folder: str
    description: str = ""


@dataclass
class Table:
    name: str
    source: str                     # gold table / CSV file name
    columns: list[Col]
    measures: list[Measure] = field(default_factory=list)
    description: str = ""
    is_date: bool = False


SAR = "#,0"
PCT = "0.0%"
PCT2 = "0.00%"

M_TYPES = {"int64": "Int64.Type", "double": "type number", "string": "type text",
           "dateTime": "type date", "boolean": "type logical"}

TABLES: list[Table] = [
    Table("Customer", "dim_customer", [
        Col("customer_id", "Customer ID", "int64", "0", key=True),
        Col("segment", "Segment", "string"),
        Col("region", "Region", "string"),
        Col("customer_type", "Customer Type", "string"),
        Col("age", "Age", "int64", "0"),
        Col("age_band", "Age Band", "string"),
        Col("has_salary_transfer", "Has Salary Transfer", "boolean"),
        Col("product_count", "Product Count", "int64", "0"),
        Col("has_mobile_app", "Has Mobile App", "boolean"),
        Col("status", "Status", "string"),
        Col("closed_month", "Closed Month", "dateTime", "yyyy-mm-dd"),
        Col("tenure_months", "Tenure Months", "int64", "0"),
    ], description="One row per customer (synthetic)."),

    Table("Date", "dim_date", [
        Col("date", "Date", "dateTime", "yyyy-mm-dd", key=True),
        Col("year", "Year", "int64", "0"),
        Col("quarter", "Quarter", "string"),
        Col("month_num", "Month Num", "int64", "0", hidden=True),
        Col("month_name", "Month", "string", sort_by="Month Num"),
        Col("year_month", "Year Month", "string"),
        Col("month_end", "Month End", "dateTime", "yyyy-mm-dd"),
        Col("day_of_week", "Day of Week", "string", sort_by="Day of Week Num"),
        Col("day_of_week_num", "Day of Week Num", "int64", "0", hidden=True),
        Col("is_weekend", "Is Weekend", "boolean"),
        Col("is_forecast_period", "Is Forecast Period", "boolean"),
    ], description="Calendar (Saudi week: Sun=1). Marked as date table.", is_date=True),

    Table("Corridor", "dim_corridor", [
        Col("corridor", "Corridor", "string", key=True),
        Col("currency", "Currency", "string"),
        Col("region", "Corridor Region", "string"),
    ]),

    Table("Customer Monthly", "fact_customer_monthly", [
        Col("customer_id", "Customer ID", "int64", "0", hidden=True),
        Col("month_end", "Month End", "dateTime", "yyyy-mm-dd", hidden=True),
        Col("is_active", "Is Active", "boolean"),
        Col("txn_count", "Txn Count", "int64", "#,0", "sum"),
        Col("txn_amount_sar", "Txn Amount SAR", "double", SAR, "sum"),
        Col("avg_balance_sar", "Avg Balance SAR", "double", SAR, "average"),
        Col("digital_logins", "Digital Logins", "int64", "#,0", "sum"),
        Col("complaints", "Complaints", "int64", "#,0", "sum"),
        Col("salary_credited", "Salary Credited", "boolean"),
    ], measures=[
        Measure("Active Customers", "CALCULATE(DISTINCTCOUNT('Customer Monthly'[Customer ID]), 'Customer Monthly'[Is Active] = TRUE())", "#,0", "Activity"),
        Measure("Transaction Amount (SAR)", "SUM('Customer Monthly'[Txn Amount SAR])", SAR, "Activity"),
        Measure("Average Balance (SAR)", "AVERAGE('Customer Monthly'[Avg Balance SAR])", SAR, "Activity"),
        Measure("Digital Logins", "SUM('Customer Monthly'[Digital Logins])", "#,0", "Activity"),
    ], description="Customer x month activity panel (~1.2M rows at default scale)."),

    Table("Churn Score", "fact_churn_score", [
        Col("customer_id", "Customer ID", "int64", "0"),
        Col("score_date", "Score Date", "dateTime", "yyyy-mm-dd"),
        Col("churn_probability", "Churn Probability", "double", PCT2),
        Col("risk_decile", "Risk Decile", "int64", "0"),
        Col("risk_band", "Risk Band", "string"),
        Col("balance_at_risk_sar", "Balance at Risk SAR", "double", SAR, "sum"),
        Col("primary_signal", "Primary Signal", "string"),
        Col("txn_trend", "Txn Trend", "double", "0.00"),
        Col("logins_trend", "Logins Trend", "double", "0.00"),
        Col("balance_trend", "Balance Trend", "double", "0.00"),
        Col("complaints_6m", "Complaints 6M", "int64", "0"),
        Col("salary_credit_ratio_6m", "Salary Credit Ratio 6M", "double", PCT),
        Col("balance_avg_6m", "Balance Avg 6M SAR", "double", SAR),
        Col("model_version", "Model Version", "string"),
    ], measures=[
        Measure("Customers Scored", "COUNTROWS('Churn Score')", "#,0", "Churn"),
        Measure("High Risk Customers", "CALCULATE([Customers Scored], 'Churn Score'[Risk Band] = \"High\")", "#,0", "Churn"),
        Measure("High Risk Share", "DIVIDE([High Risk Customers], [Customers Scored])", PCT, "Churn"),
        Measure("Avg Churn Probability", "AVERAGE('Churn Score'[Churn Probability])", PCT2, "Churn"),
        Measure("Expected Churners (3M)", "SUM('Churn Score'[Churn Probability])", "#,0", "Churn",
                "Sum of probabilities = expected number of customers leaving in the next 3 months."),
        Measure("Balance at Risk (SAR)", "SUM('Churn Score'[Balance at Risk SAR])", SAR, "Churn",
                "6-month average balance x churn probability."),
    ], description="Churn probability per active customer, 3-month horizon."),

    Table("Loan Risk", "fact_loan_risk", [
        Col("loan_id", "Loan ID", "int64", "0"),
        Col("customer_id", "Customer ID", "int64", "0", hidden=True),
        Col("score_date", "Score Date", "dateTime", "yyyy-mm-dd"),
        Col("product", "Product", "string"),
        Col("origination_date", "Origination Date", "dateTime", "yyyy-mm-dd"),
        Col("principal_sar", "Principal SAR", "double", SAR, "sum"),
        Col("outstanding_sar", "Outstanding SAR", "double", SAR, "sum"),
        Col("tenor_months", "Tenor Months", "int64", "0"),
        Col("profit_rate", "Profit Rate", "double", PCT2),
        Col("monthly_salary_sar", "Monthly Salary SAR", "double", SAR),
        Col("dbr", "DBR", "double", PCT),
        Col("dbr_band", "DBR Band", "string"),
        Col("employer_sector", "Employer Sector", "string"),
        Col("bureau_score", "Bureau Score", "int64", "0"),
        Col("prior_delinquencies", "Prior Delinquencies", "int64", "0"),
        Col("pd_12m", "PD 12M", "double", PCT2),
        Col("risk_grade", "Risk Grade", "string"),
        Col("lgd", "LGD", "double", PCT),
        Col("ead_sar", "EAD SAR", "double", SAR, "sum"),
        Col("expected_loss_sar", "Expected Loss SAR", "double", SAR, "sum"),
        Col("actual_default_12m", "Actual Default 12M", "double", "0"),
        Col("model_type", "Model Type", "string"),
        Col("model_version", "Model Version", "string"),
    ], measures=[
        Measure("Live Loans", "COUNTROWS('Loan Risk')", "#,0", "Credit"),
        Measure("EAD (SAR)", "SUM('Loan Risk'[EAD SAR])", SAR, "Credit", "Exposure at default = outstanding balance."),
        Measure("Expected Loss (SAR)", "SUM('Loan Risk'[Expected Loss SAR])", SAR, "Credit", "PD x LGD x EAD."),
        Measure("EL Rate", "DIVIDE([Expected Loss (SAR)], [EAD (SAR)])", PCT2, "Credit"),
        Measure("Avg PD", "AVERAGE('Loan Risk'[PD 12M])", PCT2, "Credit"),
        Measure("EAD-Weighted PD", "DIVIDE(SUMX('Loan Risk', 'Loan Risk'[PD 12M] * 'Loan Risk'[EAD SAR]), [EAD (SAR)])", PCT2, "Credit"),
        Measure("Loans Above 45% DBR", "CALCULATE([Live Loans], 'Loan Risk'[DBR] > 0.45)", "#,0", "Credit"),
        Measure("Loans Originated", "CALCULATE(COUNTROWS('Loan Risk'), USERELATIONSHIP('Loan Risk'[Origination Date], 'Date'[Date]))", "#,0", "Credit",
                "Uses the inactive Origination Date relationship."),
    ], description="12-month PD, risk grade and expected loss per live personal-finance contract."),

    Table("Remittance Actual", "fact_remittance_daily", [
        Col("date", "Date", "dateTime", "yyyy-mm-dd", hidden=True),
        Col("corridor", "Corridor", "string", hidden=True),
        Col("amount_sar", "Amount SAR", "double", SAR, "sum"),
        Col("txn_count", "Txn Count", "int64", "#,0", "sum"),
        Col("backtest_forecast_sar", "Backtest Forecast SAR", "double", SAR, "sum"),
        Col("is_backtest_period", "Is Backtest Period", "boolean"),
    ], measures=[
        Measure("Remittance Actual (SAR)", "SUM('Remittance Actual'[Amount SAR])", SAR, "Remittance"),
        Measure("Backtest Forecast (SAR)", "SUM('Remittance Actual'[Backtest Forecast SAR])", SAR, "Remittance"),
        Measure("Backtest MAPE",
                "AVERAGEX(FILTER('Remittance Actual', NOT ISBLANK('Remittance Actual'[Backtest Forecast SAR])), "
                "DIVIDE(ABS('Remittance Actual'[Backtest Forecast SAR] - 'Remittance Actual'[Amount SAR]), 'Remittance Actual'[Amount SAR]))",
                PCT2, "Remittance", "Mean absolute percentage error on the 90-day hold-out."),
    ], description="Daily outbound remittance actuals per corridor, with 90-day backtest forecast."),

    Table("Remittance Forecast", "fact_remittance_forecast", [
        Col("date", "Date", "dateTime", "yyyy-mm-dd", hidden=True),
        Col("corridor", "Corridor", "string", hidden=True),
        Col("forecast_amount_sar", "Forecast Amount SAR", "double", SAR, "sum"),
        Col("lower_p10_sar", "Lower P10 SAR", "double", SAR, "sum"),
        Col("upper_p90_sar", "Upper P90 SAR", "double", SAR, "sum"),
        Col("model_version", "Model Version", "string"),
    ], measures=[
        Measure("Remittance Forecast (SAR)", "SUM('Remittance Forecast'[Forecast Amount SAR])", SAR, "Remittance"),
        Measure("Forecast P10 (SAR)", "SUM('Remittance Forecast'[Lower P10 SAR])", SAR, "Remittance"),
        Measure("Forecast P90 (SAR)", "SUM('Remittance Forecast'[Upper P90 SAR])", SAR, "Remittance"),
    ], description="90-day forecast per corridor with P10-P90 band."),

    Table("Model Metrics", "model_metrics", [
        Col("model_name", "Model Name", "string"),
        Col("model_version", "Model Version", "string"),
        Col("run_date", "Run Date", "dateTime", "yyyy-mm-dd"),
        Col("dataset", "Dataset", "string"),
        Col("metric", "Metric", "string"),
        Col("value", "Value", "double", "0.000", "sum"),
    ], measures=[
        Measure("Metric Value", "SUM('Model Metrics'[Value])", "0.000", "Model health"),
        Measure("Churn AUC", "CALCULATE([Metric Value], 'Model Metrics'[Model Name] = \"churn\", 'Model Metrics'[Metric] = \"auc\")", "0.000", "Model health"),
        Measure("Credit AUC", "CALCULATE([Metric Value], 'Model Metrics'[Model Name] = \"credit_risk\", 'Model Metrics'[Metric] = \"auc\")", "0.000", "Model health"),
        Measure("Credit Gini", "CALCULATE([Metric Value], 'Model Metrics'[Model Name] = \"credit_risk\", 'Model Metrics'[Metric] = \"gini\")", "0.000", "Model health"),
        Measure("Remittance MAPE", "CALCULATE([Metric Value], 'Model Metrics'[Model Name] = \"remittance_forecast\", 'Model Metrics'[Metric] = \"mape\")", PCT2, "Model health"),
    ], description="Out-of-time / backtest metrics written by every pipeline run."),

    Table("Feature Importance", "feature_importance", [
        Col("model_name", "Model Name", "string"),
        Col("model_version", "Model Version", "string"),
        Col("feature", "Feature", "string"),
        Col("importance", "Importance", "double", "0.0000", "sum"),
        Col("importance_std", "Importance Std", "double", "0.0000"),
        Col("rank", "Rank", "int64", "0"),
    ], measures=[
        Measure("Permutation Importance", "SUM('Feature Importance'[Importance])", "0.0000", "Model health",
                "Drop in AUC when the feature is shuffled (out-of-time sample)."),
    ]),
]

RELATIONSHIPS = [  # (from table, from col, to table, to col, active)
    ("Churn Score", "Customer ID", "Customer", "Customer ID", True),
    ("Loan Risk", "Customer ID", "Customer", "Customer ID", True),
    ("Customer Monthly", "Customer ID", "Customer", "Customer ID", True),
    ("Customer Monthly", "Month End", "Date", "Date", True),
    ("Remittance Actual", "Date", "Date", "Date", True),
    ("Remittance Forecast", "Date", "Date", "Date", True),
    ("Remittance Actual", "Corridor", "Corridor", "Corridor", True),
    ("Remittance Forecast", "Corridor", "Corridor", "Corridor", True),
    ("Loan Risk", "Origination Date", "Date", "Date", False),
]

PARAMETERS = [  # (name, default, description, allowed values)
    ("SourceType", "CSV", "Where to read the gold tables from", ["CSV", "PostgreSQL", "Databricks"]),
    ("CsvFolder", "C:\\bank-predictive-analytics\\data\\gold\\", "Folder with gold CSV files (end with a backslash)", None),
    ("PgServer", "localhost:5432", "PostgreSQL host:port (Docker stack)", None),
    ("PgDatabase", "bpa", "PostgreSQL database (schema gold)", None),
    ("DbxServerHostname", "dbc-00000000-0000.cloud.databricks.com", "Databricks SQL warehouse server hostname", None),
    ("DbxHttpPath", "/sql/1.0/warehouses/0000000000000000", "Databricks SQL warehouse HTTP path", None),
    ("DbxCatalog", "workspace", "Unity Catalog catalog", None),
    ("DbxSchema", "bpa", "Unity Catalog schema", None),
]


def m_source(t: Table) -> str:
    cols = ", ".join(f'"{c.source}"' for c in t.columns)
    types = ", ".join(f'{{"{c.source}", {M_TYPES[c.dtype]}}}' for c in t.columns)
    renames = ", ".join(f'{{"{c.source}", "{c.name}"}}' for c in t.columns if c.source != c.name)
    return f"""let
    Source =
        if SourceType = "PostgreSQL" then
            PostgreSQL.Database(PgServer, PgDatabase){{[Schema = "gold", Item = "{t.source}"]}}[Data]
        else if SourceType = "Databricks" then
            Databricks.Catalogs(DbxServerHostname, DbxHttpPath, [Catalog = null, Database = null, EnableAutomaticProxyDiscovery = null])
                {{[Name = DbxCatalog, Kind = "Database"]}}[Data]
                {{[Name = DbxSchema, Kind = "Schema"]}}[Data]
                {{[Name = "{t.source}", Kind = "Table"]}}[Data]
        else
            Table.TransformColumns(
                Table.PromoteHeaders(
                    Csv.Document(File.Contents(CsvFolder & "{t.source}.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
                    [PromoteAllScalars = true]),
                {{}}, each if _ = "" then null else _),
    Selected = Table.SelectColumns(Source, {{{cols}}}),
    Typed = Table.TransformColumnTypes(Selected, {{{types}}}, "en-US"),
    Renamed = Table.RenameColumns(Typed, {{{renames}}})
in
    Renamed"""


def indent(text: str, n: int) -> str:
    return "\n".join(("\t" * n + line) if line.strip() else "" for line in text.splitlines())


def table_tmdl(t: Table) -> str:
    out = []
    if t.description:
        out.append(f"/// {t.description}")
    out.append(f"table {q(t.name)}")
    out.append(f"\tlineageTag: {guid('table', t.name)}")
    if t.is_date:
        out.append("\tdataCategory: Time")
    out.append("")
    for m in t.measures:
        if m.description:
            out.append(f"\t/// {m.description}")
        out.append(f"\tmeasure {q(m.name)} = {m.expr}")
        out.append(f"\t\tformatString: {m.fmt}")
        out.append(f"\t\tdisplayFolder: {m.folder}")
        out.append(f"\t\tlineageTag: {guid('measure', t.name, m.name)}")
        out.append("")
    for c in t.columns:
        out.append(f"\tcolumn {q(c.name)}")
        out.append(f"\t\tdataType: {c.dtype}")
        if c.fmt:
            out.append(f"\t\tformatString: {c.fmt}")
        if c.key:
            out.append("\t\tisKey")
        if c.hidden:
            out.append("\t\tisHidden")
        out.append(f"\t\tlineageTag: {guid('column', t.name, c.name)}")
        out.append(f"\t\tsummarizeBy: {c.summarize}")
        out.append(f"\t\tsourceColumn: {c.name}")
        if c.sort_by:
            out.append(f"\t\tsortByColumn: {q(c.sort_by)}")
        out.append("")
        out.append("\t\tannotation SummarizationSetBy = Automatic")
        out.append("")
    out.append(f"\tpartition {q(t.name)} = m")
    out.append("\t\tmode: import")
    out.append("\t\tsource =")
    out.append(indent(m_source(t), 4))
    out.append("")
    out.append("\tannotation PBI_ResultType = Table")
    out.append("")
    return "\n".join(out)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, obj) -> None:
    write(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def build_semantic_model() -> None:
    d = SM_DIR / "definition"
    write_json(SM_DIR / "definition.pbism", {
        "$schema": f"{SCHEMA}/item/semanticModel/definitionProperties/1.0.0/schema.json",
        "version": "4.2", "settings": {}})
    write_json(SM_DIR / ".platform", {
        "$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "SemanticModel", "displayName": NAME},
        "config": {"version": "2.0", "logicalId": guid("platform", "semanticModel")}})
    write(d / "database.tmdl", "database\n\tcompatibilityLevel: 1567\n\n")

    query_order = [p[0] for p in PARAMETERS] + [t.name for t in TABLES]
    model = ["model Model", "\tculture: en-US", "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
             "\tsourceQueryCulture: en-US", "\tdataAccessOptions", "\t\tlegacyRedirects", "\t\treturnErrorValuesAsNull", "",
             "annotation __PBI_TimeIntelligenceEnabled = 0", "",
             f"annotation PBI_QueryOrder = {json.dumps(query_order)}", ""]
    model += [f"ref table {q(t.name)}" for t in TABLES] + [""]
    write(d / "model.tmdl", "\n".join(model))

    exprs = []
    for name, default, desc, allowed in PARAMETERS:
        meta = ['IsParameterQuery = true']
        if allowed:
            meta.append("List = {" + ", ".join(f'"{a}"' for a in allowed) + "}")
            meta.append(f'DefaultValue = "{default}"')
        meta += ['Type = "Text"', "IsParameterQueryRequired = true"]
        exprs += [f"/// {desc}",
                  f'expression {name} = "{default}" meta [{", ".join(meta)}]',
                  f"\tlineageTag: {guid('expression', name)}", "",
                  "\tannotation PBI_ResultType = Text", ""]
    write(d / "expressions.tmdl", "\n".join(exprs))

    rels = []
    for ft, fc, tt, tc, active in RELATIONSHIPS:
        rels.append(f"relationship {guid('rel', ft, fc, tt, tc)}")
        if not active:
            rels.append("\tisActive: false")
        rels.append(f"\tfromColumn: {q(ft)}.{q(fc)}")
        rels.append(f"\ttoColumn: {q(tt)}.{q(tc)}")
        rels.append("")
    write(d / "relationships.tmdl", "\n".join(rels))

    for t in TABLES:
        write(d / "tables" / f"{t.name}.tmdl", table_tmdl(t))


# ======================================================================================
# Report spec (PBIR)
# ======================================================================================
PAGE_W, PAGE_H = 1280, 720
THEME_FILE = "BankPredictiveTheme.json"
THEME = {
    "name": "Bank Predictive",
    "dataColors": ["#0F766E", "#2563EB", "#D97706", "#DC2626", "#7C3AED", "#64748B", "#0891B2", "#65A30D"],
    "foreground": "#1F2937", "foregroundNeutralSecondary": "#4B5563", "foregroundNeutralTertiary": "#9CA3AF",
    "background": "#FFFFFF", "backgroundLight": "#F3F4F6", "backgroundNeutral": "#E5E7EB",
    "tableAccent": "#0F766E", "good": "#0F766E", "neutral": "#D97706", "bad": "#DC2626",
    "maximum": "#0F766E", "center": "#FDE68A", "minimum": "#FEF2F2",
    "textClasses": {
        "title": {"fontFace": "Segoe UI Semibold", "fontSize": 12, "color": "#1F2937"},
        "callout": {"fontFace": "Segoe UI Semibold", "fontSize": 26, "color": "#0F766E"},
        "label": {"fontFace": "Segoe UI", "fontSize": 10, "color": "#4B5563"},
    },
    "visualStyles": {
        "page": {"*": {"background": [{"color": {"solid": {"color": "#F3F4F6"}}, "transparency": 0}]}},
        "*": {"*": {
            "background": [{"show": True, "color": {"solid": {"color": "#FFFFFF"}}, "transparency": 0}],
            "border": [{"show": True, "color": {"solid": {"color": "#E5E7EB"}}, "radius": 8}],
            "title": [{"show": True, "fontSize": 11, "fontColor": {"solid": {"color": "#1F2937"}}}],
        }},
    },
}


def lit(v: str) -> dict:
    return {"expr": {"Literal": {"Value": v}}}


def field(kind: str, entity: str, prop: str) -> dict:
    return {kind: {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def proj(kind: str, entity: str, prop: str, label: str | None = None, active: bool = False) -> dict:
    p = {"field": field(kind, entity, prop), "queryRef": f"{entity}.{prop}", "nativeQueryRef": label or prop}
    if label:
        p["displayName"] = label
    if active:
        p["active"] = True
    return p


def C(entity, prop, label=None):      # column projection
    return ("Column", entity, prop, label)


def M(entity, prop, label=None):      # measure projection
    return ("Measure", entity, prop, label)


class Page:
    def __init__(self, key: str, title: str, subtitle: str):
        self.name = hexid("page", key)
        self.title = title
        self.visuals: list[dict] = []
        self.add_text(title, subtitle)

    def _container(self, kind: str, x, y, w, h, title: str | None, query: dict | None, objects: dict | None = None,
                   sort: dict | None = None) -> dict:
        idx = len(self.visuals)
        visual: dict = {"visualType": kind}
        if query:
            visual["query"] = {"queryState": query}
            if sort:
                visual["query"]["sortDefinition"] = sort
        if objects:
            visual["objects"] = objects
        vco = {"title": [{"properties": {"show": lit("true" if title else "false")}}]}
        if title:
            vco["title"][0]["properties"]["text"] = lit(f"'{title}'")
        visual["visualContainerObjects"] = vco
        visual["drillFilterOtherVisuals"] = True
        v = {
            "$schema": f"{SCHEMA}/item/report/definition/visualContainer/2.4.0/schema.json",
            "name": hexid("visual", self.name, str(idx)),
            "position": {"x": x, "y": y, "z": (idx + 1) * 1000, "height": h, "width": w, "tabOrder": idx * 1000},
            "visual": visual,
        }
        self.visuals.append(v)
        return v

    @staticmethod
    def _role(items) -> dict:
        return {"projections": [proj(k, e, p, lbl, active=(k == "Column")) for k, e, p, lbl in items]}

    def add_text(self, title: str, subtitle: str):
        objects = {"general": [{"properties": {"paragraphs": [
            {"textRuns": [{"value": title, "textStyle": {"fontSize": "20pt", "fontWeight": "bold", "color": "#0F766E"}}]},
            {"textRuns": [{"value": subtitle, "textStyle": {"fontSize": "10pt", "color": "#4B5563"}}]},
        ]}}]}
        v = self._container("textbox", 16, 8, 900, 64, None, None, objects)
        v["visual"]["visualContainerObjects"].update({
            "background": [{"properties": {"show": lit("false")}}],
            "border": [{"properties": {"show": lit("false")}}]})

    def card(self, x, y, w, h, m, title):
        return self._container("card", x, y, w, h, title, {"Values": self._role([m])},
                               {"labels": [{"properties": {"fontSize": lit("22D")}}],
                                "categoryLabels": [{"properties": {"show": lit("false")}}]})

    def chart(self, kind, x, y, w, h, title, category, values, series=None, sort_desc=None):
        query = {"Category": self._role(category), "Y": self._role(values)}
        if series:
            query["Series"] = self._role(series)
        sort = None
        if sort_desc:
            k, e, p, _ = sort_desc
            sort = {"sort": [{"field": field(k, e, p), "direction": "Descending"}], "isDefaultSort": True}
        return self._container(kind, x, y, w, h, title, query, sort=sort)

    def table(self, x, y, w, h, title, values, sort_desc=None):
        sort = None
        if sort_desc:
            k, e, p, _ = sort_desc
            sort = {"sort": [{"field": field(k, e, p), "direction": "Descending"}], "isDefaultSort": True}
        return self._container("tableEx", x, y, w, h, title, {"Values": self._role(values)}, sort=sort)

    def slicer(self, x, y, w, h, col, title):
        return self._container("slicer", x, y, w, h, title, {"Values": self._role([col])},
                               {"data": [{"properties": {"mode": lit("'Dropdown'")}}]})


def build_pages() -> list[Page]:
    pages = []
    G, top = 16, 84                       # gutter, first row y
    cw = (PAGE_W - 7 * G) // 6            # card width for 6 cards

    # ---------------- 1. Executive overview ----------------
    p = Page("overview", "Executive Overview", "Predictive signals across churn, credit risk and remittance — synthetic data, as of the latest pipeline run")
    cards = [M("Churn Score", "High Risk Customers"), M("Churn Score", "Balance at Risk (SAR)"),
             M("Loan Risk", "EAD (SAR)"), M("Loan Risk", "Expected Loss (SAR)"),
             M("Remittance Forecast", "Remittance Forecast (SAR)"), M("Model Metrics", "Churn AUC")]
    titles = ["High-risk customers", "Balance at risk (SAR)", "Credit exposure EAD (SAR)",
              "Expected loss (SAR)", "Remittance next 90 days (SAR)", "Churn model AUC"]
    for i, (m, t) in enumerate(zip(cards, titles)):
        p.card(G + i * (cw + G), top, cw, 110, m, t)
    row2 = top + 110 + G
    half = (PAGE_W - 3 * G) // 2
    p.chart("clusteredBarChart", G, row2, half, 250, "Balance at risk by segment",
            [C("Customer", "Segment")], [M("Churn Score", "Balance at Risk (SAR)")],
            sort_desc=M("Churn Score", "Balance at Risk (SAR)"))
    p.chart("clusteredColumnChart", 2 * G + half, row2, half, 250, "Expected loss by risk grade",
            [C("Loan Risk", "Risk Grade")], [M("Loan Risk", "Expected Loss (SAR)")])
    row3 = row2 + 250 + G
    p.chart("lineChart", G, row3, PAGE_W - 2 * G, PAGE_H - row3 - G, "Daily remittance: actual vs forecast (SAR)",
            [C("Date", "Date")], [M("Remittance Actual", "Remittance Actual (SAR)", "Actual"),
                                  M("Remittance Forecast", "Remittance Forecast (SAR)", "Forecast")])
    pages.append(p)

    # ---------------- 2. Churn ----------------
    p = Page("churn", "Churn Risk", "3-month churn probability per active customer, with the main early-warning signal")
    sw = 200
    p.slicer(PAGE_W - 2 * (sw + G), 16, sw, 56, C("Customer", "Segment"), "Segment")
    p.slicer(PAGE_W - (sw + G), 16, sw, 56, C("Customer", "Region"), "Region")
    cards = [(M("Churn Score", "Customers Scored"), "Customers scored"),
             (M("Churn Score", "High Risk Customers"), "High-risk customers"),
             (M("Churn Score", "High Risk Share"), "High-risk share"),
             (M("Churn Score", "Expected Churners (3M)"), "Expected churners (3M)"),
             (M("Churn Score", "Balance at Risk (SAR)"), "Balance at risk (SAR)"),
             (M("Churn Score", "Avg Churn Probability"), "Avg churn probability")]
    for i, (m, t) in enumerate(cards):
        p.card(G + i * (cw + G), top, cw, 100, m, t)
    row2 = top + 100 + G
    third = (PAGE_W - 4 * G) // 3
    p.chart("clusteredBarChart", G, row2, third, 230, "High-risk customers by primary signal",
            [C("Churn Score", "Primary Signal")], [M("Churn Score", "High Risk Customers")],
            sort_desc=M("Churn Score", "High Risk Customers"))
    p.chart("clusteredColumnChart", 2 * G + third, row2, third, 230, "Average churn probability by risk decile",
            [C("Churn Score", "Risk Decile")], [M("Churn Score", "Avg Churn Probability")])
    p.chart("clusteredBarChart", 3 * G + 2 * third, row2, third, 230, "Expected churners by region",
            [C("Customer", "Region")], [M("Churn Score", "Expected Churners (3M)")],
            sort_desc=M("Churn Score", "Expected Churners (3M)"))
    row3 = row2 + 230 + G
    p.table(G, row3, PAGE_W - 2 * G, PAGE_H - row3 - G, "Retention worklist — highest churn probability first", [
        C("Churn Score", "Customer ID"), C("Customer", "Segment"), C("Customer", "Region"),
        C("Churn Score", "Risk Band"), C("Churn Score", "Primary Signal"),
        M("Churn Score", "Avg Churn Probability", "Churn probability"),
        M("Churn Score", "Balance at Risk (SAR)", "Balance at risk (SAR)"),
    ], sort_desc=M("Churn Score", "Avg Churn Probability"))
    pages.append(p)

    # ---------------- 3. Credit risk ----------------
    p = Page("credit", "Credit Risk", "12-month probability of default, risk grade and expected loss (PD x LGD x EAD) for live personal finance")
    p.slicer(PAGE_W - 2 * (sw + G), 16, sw, 56, C("Loan Risk", "Product"), "Product")
    p.slicer(PAGE_W - (sw + G), 16, sw, 56, C("Loan Risk", "Employer Sector"), "Employer sector")
    cards = [(M("Loan Risk", "Live Loans"), "Live loans"), (M("Loan Risk", "EAD (SAR)"), "EAD (SAR)"),
             (M("Loan Risk", "Expected Loss (SAR)"), "Expected loss (SAR)"), (M("Loan Risk", "EL Rate"), "EL rate"),
             (M("Loan Risk", "EAD-Weighted PD"), "EAD-weighted PD"), (M("Loan Risk", "Loans Above 45% DBR"), "Loans above 45% DBR")]
    for i, (m, t) in enumerate(cards):
        p.card(G + i * (cw + G), top, cw, 100, m, t)
    p.chart("clusteredColumnChart", G, row2, third, 230, "EAD by risk grade (SAR)",
            [C("Loan Risk", "Risk Grade")], [M("Loan Risk", "EAD (SAR)")])
    p.chart("clusteredColumnChart", 2 * G + third, row2, third, 230, "Average PD by DBR band",
            [C("Loan Risk", "DBR Band")], [M("Loan Risk", "Avg PD")])
    p.chart("clusteredBarChart", 3 * G + 2 * third, row2, third, 230, "Expected loss by employer sector (SAR)",
            [C("Loan Risk", "Employer Sector")], [M("Loan Risk", "Expected Loss (SAR)")],
            sort_desc=M("Loan Risk", "Expected Loss (SAR)"))
    p.table(G, row3, PAGE_W - 2 * G, PAGE_H - row3 - G, "Watchlist — largest expected loss first", [
        C("Loan Risk", "Loan ID"), C("Loan Risk", "Product"), C("Loan Risk", "Risk Grade"),
        C("Loan Risk", "DBR Band"), C("Loan Risk", "Employer Sector"),
        M("Loan Risk", "Avg PD", "PD 12M"), M("Loan Risk", "EAD (SAR)"), M("Loan Risk", "Expected Loss (SAR)"),
    ], sort_desc=M("Loan Risk", "Expected Loss (SAR)"))
    pages.append(p)

    # ---------------- 4. Remittance ----------------
    p = Page("remittance", "Remittance Forecast", "90-day daily forecast per corridor with P10-P90 band; dotted history shows the 90-day backtest")
    p.slicer(PAGE_W - (sw + G), 16, sw, 56, C("Corridor", "Corridor"), "Corridor")
    cards = [(M("Remittance Forecast", "Remittance Forecast (SAR)"), "Forecast next 90 days (SAR)"),
             (M("Remittance Forecast", "Forecast P10 (SAR)"), "Low case P10 (SAR)"),
             (M("Remittance Forecast", "Forecast P90 (SAR)"), "High case P90 (SAR)"),
             (M("Remittance Actual", "Backtest MAPE"), "Backtest MAPE")]
    cw4 = (PAGE_W - 5 * G) // 4
    for i, (m, t) in enumerate(cards):
        p.card(G + i * (cw4 + G), top, cw4, 100, m, t)
    p.chart("lineChart", G, row2, PAGE_W - 2 * G, 280, "Actual, backtest and forecast with P10-P90 band (SAR)",
            [C("Date", "Date")], [
                M("Remittance Actual", "Remittance Actual (SAR)", "Actual"),
                M("Remittance Actual", "Backtest Forecast (SAR)", "Backtest"),
                M("Remittance Forecast", "Remittance Forecast (SAR)", "Forecast"),
                M("Remittance Forecast", "Forecast P10 (SAR)", "P10"),
                M("Remittance Forecast", "Forecast P90 (SAR)", "P90")])
    row3b = row2 + 280 + G
    p.chart("clusteredBarChart", G, row3b, half, PAGE_H - row3b - G, "Forecast by corridor (SAR)",
            [C("Corridor", "Corridor")], [M("Remittance Forecast", "Remittance Forecast (SAR)")],
            sort_desc=M("Remittance Forecast", "Remittance Forecast (SAR)"))
    p.chart("clusteredColumnChart", 2 * G + half, row3b, half, PAGE_H - row3b - G, "Actual remittance by day of week (SAR)",
            [C("Date", "Day of Week")], [M("Remittance Actual", "Remittance Actual (SAR)")])
    pages.append(p)

    # ---------------- 5. Model health ----------------
    p = Page("models", "Model Health", "Out-of-time and backtest metrics from the latest run, plus what drives each model")
    p.slicer(PAGE_W - (sw + G), 16, sw, 56, C("Feature Importance", "Model Name"), "Model")
    cards = [(M("Model Metrics", "Churn AUC"), "Churn AUC (out-of-time)"), (M("Model Metrics", "Credit AUC"), "Credit AUC (out-of-time)"),
             (M("Model Metrics", "Credit Gini"), "Credit Gini"), (M("Model Metrics", "Remittance MAPE"), "Remittance MAPE (backtest)")]
    for i, (m, t) in enumerate(cards):
        p.card(G + i * (cw4 + G), top, cw4, 100, m, t)
    p.chart("clusteredBarChart", G, row2, half, PAGE_H - row2 - G, "Permutation importance by feature",
            [C("Feature Importance", "Feature")], [M("Feature Importance", "Permutation Importance")],
            sort_desc=M("Feature Importance", "Permutation Importance"))
    p.table(2 * G + half, row2, half, PAGE_H - row2 - G, "All metrics", [
        C("Model Metrics", "Model Name"), C("Model Metrics", "Dataset"), C("Model Metrics", "Metric"),
        M("Model Metrics", "Metric Value", "Value"), C("Model Metrics", "Model Version")])
    pages.append(p)
    return pages


def build_report() -> None:
    d = RP_DIR / "definition"
    write_json(RP_DIR / "definition.pbir", {
        "$schema": f"{SCHEMA}/item/report/definitionProperties/2.0.0/schema.json",
        "version": "4.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}})
    write_json(RP_DIR / ".platform", {
        "$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "Report", "displayName": NAME},
        "config": {"version": "2.0", "logicalId": guid("platform", "report")}})
    write_json(RP_DIR / "StaticResources" / "RegisteredResources" / THEME_FILE, THEME)
    write_json(d / "version.json", {"$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json",
                                    "version": "2.0.0"})
    write_json(d / "report.json", {
        "$schema": f"{SCHEMA}/item/report/definition/report/3.0.0/schema.json",
        "themeCollection": {"customTheme": {
            "name": THEME_FILE,
            "reportVersionAtImport": {"visual": "2.4.0", "report": "3.0.0", "page": "2.0.0"},
            "type": "RegisteredResources"}},
        "resourcePackages": [{"name": "RegisteredResources", "type": "RegisteredResources",
                              "items": [{"name": THEME_FILE, "path": THEME_FILE, "type": "CustomTheme"}]}],
        "settings": {"useStylableVisualContainerHeader": True, "exportDataMode": "AllowSummarized",
                     "defaultDrillFilterOtherVisuals": True, "allowChangeFilterTypes": True,
                     "useEnhancedTooltips": True, "useDefaultAggregateDisplayName": True},
    })
    pages = build_pages()
    write_json(d / "pages" / "pages.json", {
        "$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
        "pageOrder": [p.name for p in pages], "activePageName": pages[0].name})
    for p in pages:
        write_json(d / "pages" / p.name / "page.json", {
            "$schema": f"{SCHEMA}/item/report/definition/page/2.0.0/schema.json",
            "name": p.name, "displayName": p.title, "displayOption": "FitToPage",
            "height": PAGE_H, "width": PAGE_W})
        for v in p.visuals:
            write_json(d / "pages" / p.name / "visuals" / v["name"] / "visual.json", v)


def main() -> None:
    for folder in (SM_DIR, RP_DIR):
        if folder.exists():
            shutil.rmtree(folder)
    build_semantic_model()
    build_report()
    write_json(ROOT / f"{NAME}.pbip", {
        "$schema": f"{SCHEMA}/pbip/pbipProperties/1.0.0/schema.json",
        "version": "1.0", "artifacts": [{"report": {"path": f"{NAME}.Report"}}],
        "settings": {"enableAutoRecovery": True}})
    n_vis = sum(1 for _ in RP_DIR.rglob("visual.json"))
    n_meas = sum(len(t.measures) for t in TABLES)
    print(f"PBIP written to {ROOT}: {len(TABLES)} tables, {n_meas} measures, {len(RELATIONSHIPS)} relationships, {n_vis} visuals")


if __name__ == "__main__":
    main()
