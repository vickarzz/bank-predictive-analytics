"""Read/write helpers. Bronze/silver are Parquet; gold is written as CSV (for Power BI)
and optionally loaded into PostgreSQL."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DATE_COLUMNS = {"date", "month_end", "score_date", "origination_date", "run_date", "closed_month"}


def write_table(df: pd.DataFrame, folder: Path, name: str, csv: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if csv:
        path = folder / f"{name}.csv"
        df.to_csv(path, index=False)
    else:
        path = folder / f"{name}.parquet"
        # millisecond timestamps: Spark cannot read pandas' default nanosecond Parquet timestamps
        df.to_parquet(path, index=False, coerce_timestamps="ms", allow_truncated_timestamps=True)
    log.info("wrote %-38s %10s rows -> %s", name, f"{len(df):,}", path)
    return path


def read_table(folder: Path, name: str) -> pd.DataFrame:
    pq = folder / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(folder / f"{name}.csv")


def to_postgres(tables: dict[str, pd.DataFrame], url: str, schema: str = "gold") -> None:
    """Replace gold tables in PostgreSQL using COPY (≈1M rows in a few seconds).

    Each table is loaded into a staging table and swapped in one transaction, so Power BI
    never reads a half-loaded table. Requires sqlalchemy + psycopg2."""
    import io as _io

    from sqlalchemy import Date, create_engine, text

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
    for name, df in tables.items():
        df = df.copy()
        date_cols = sorted(DATE_COLUMNS.intersection(df.columns))   # real DATE columns for Power BI
        for c in date_cols:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
        stage = f"{name}__stage"
        df.head(0).to_sql(stage, engine, schema=schema, if_exists="replace", index=False,
                          dtype={c: Date() for c in date_cols})
        buf = _io.StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            cols = ", ".join(f'"{c}"' for c in df.columns)
            cur.copy_expert(f'COPY {schema}."{stage}" ({cols}) FROM STDIN WITH (FORMAT csv)', buf)
            cur.execute(f'DROP TABLE IF EXISTS {schema}."{name}"')
            cur.execute(f'ALTER TABLE {schema}."{stage}" RENAME TO "{name}"')
            raw.commit()
        finally:
            raw.close()
        log.info("loaded %s.%s (%s rows)", schema, name, f"{len(df):,}")
