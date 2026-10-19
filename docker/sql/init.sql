-- Runs once when the Postgres volume is first created.

-- Metadata databases for the platform services
CREATE DATABASE airflow;
CREATE DATABASE mlflow;

-- Gold layer lives in database "bpa", schema "gold" (tables are (re)created by the pipeline)
\connect bpa
CREATE SCHEMA IF NOT EXISTS gold;

-- Least-privilege account for Power BI: read-only on the gold schema
CREATE ROLE powerbi_reader LOGIN PASSWORD 'powerbi_reader';
GRANT CONNECT ON DATABASE bpa TO powerbi_reader;
GRANT USAGE ON SCHEMA gold TO powerbi_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA gold TO powerbi_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE bpa IN SCHEMA gold GRANT SELECT ON TABLES TO powerbi_reader;
