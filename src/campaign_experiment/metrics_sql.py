"""The SQL metric layer: metric definitions live in .sql files run on DuckDB.

Keeping definitions in SQL means the numbers in the report are the numbers a
warehouse query would produce, and a reviewer can read a metric's definition
without reading Python.
"""

from __future__ import annotations

from importlib import resources

import duckdb
import pandas as pd

# Columns a segment query may group by. The query text is built from this
# list, so anything else is rejected before it reaches SQL.
SEGMENT_COLUMNS = {"newbie", "channel", "purchase_line", "zip_code", "history_segment", "recency"}


def _sql(name: str) -> str:
    return resources.files("campaign_experiment.sql").joinpath(f"{name}.sql").read_text("utf-8")


def connect(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with the experiment loaded as table `customers`."""
    con = duckdb.connect(database=":memory:")
    # The categorical `arm` column is a pandas convenience; SQL uses `segment`.
    con.register("customers_df", df.drop(columns=["arm"], errors="ignore"))
    con.execute("CREATE TABLE customers AS SELECT * FROM customers_df")
    con.unregister("customers_df")
    return con


def arm_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(_sql("arm_metrics")).df().set_index("arm")


def segment_metrics(con: duckdb.DuckDBPyConnection, segments: list[str]) -> pd.DataFrame:
    unknown = set(segments) - SEGMENT_COLUMNS
    if unknown or not segments:
        raise ValueError(f"segments must be a non-empty subset of {sorted(SEGMENT_COLUMNS)}")
    # UNPIVOT needs one value type, so every segment column is cast to text.
    cast = ", ".join(f"CAST({c} AS VARCHAR) AS {c}" for c in segments)
    query = _sql("segment_metrics").format(cast_columns=cast, segment_columns=", ".join(segments))
    return con.execute(query).df()
