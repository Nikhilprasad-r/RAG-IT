"""Read-only SQL tools.

The database comes from the SQL_ANALYST_DATABASE_URL environment variable, in
SQLAlchemy form (postgresql+psycopg://..., sqlite:///analytics.db, ...). Needs
`ragkit-agents[sql]` plus the driver for your database.

Queries are checked to be a single SELECT, run in a read-only transaction where
the database supports it (Postgres, SQLite) and always rolled back. Still give
the agent a database user that can only read.
"""

import os
import re
from functools import cache

from sqlalchemy import Engine, create_engine, inspect, text

MAX_ROWS = 200
MAX_CELL = 200
STATEMENT_TIMEOUT_MS = 15_000

_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRINGS = re.compile(r"'(?:[^']|'')*'")
_WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|attach|detach|"
    r"pragma|copy|call|exec|execute|vacuum|reindex|lock|set)\b",
    re.IGNORECASE,
)


@cache
def _engine() -> Engine:
    url = os.environ.get("SQL_ANALYST_DATABASE_URL")
    if not url:
        raise RuntimeError("Set SQL_ANALYST_DATABASE_URL to the database to analyse.")
    return create_engine(url)


def list_tables() -> str:
    """List the tables and views in the database."""
    inspector = inspect(_engine())
    names = inspector.get_table_names() + inspector.get_view_names()
    return "\n".join(sorted(names)) or "The database has no tables."


def describe_table(table: str) -> str:
    """Show a table's columns with their types, plus three example rows."""
    inspector = inspect(_engine())
    if table not in inspector.get_table_names() + inspector.get_view_names():
        return f"No table named {table!r}. Call list_tables to see the names."
    columns = "\n".join(
        f"- {c['name']}: {c['type']}{'' if c.get('nullable', True) else ' NOT NULL'}"
        for c in inspector.get_columns(table)
    )
    quoted = _engine().dialect.identifier_preparer.quote(table)
    return f"Columns of {table}:\n{columns}\n\nExample rows:\n" + run_query(
        f"SELECT * FROM {quoted} LIMIT 3"
    )


def run_query(sql: str) -> str:
    """Run one read-only SELECT query and return up to 200 rows as a Markdown table."""
    statement = _COMMENTS.sub(" ", sql).strip().rstrip(";").strip()
    keywords = _STRINGS.sub("''", statement)  # 'deleted' or ';' inside a literal is fine
    if ";" in keywords:
        return "Only one statement is allowed."
    if not re.match(r"(?is)^(select|with)\b", keywords) or _WRITE_WORDS.search(keywords):
        return "Only read-only SELECT queries are allowed."

    engine = _engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            if engine.dialect.name == "postgresql":
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}"))
            elif engine.dialect.name == "sqlite":
                connection.exec_driver_sql("PRAGMA query_only = ON")
            result = connection.execute(text(statement))
            columns = list(result.keys())
            rows = result.fetchmany(MAX_ROWS + 1)
        except Exception as exc:
            return f"The query failed: {exc}"
        finally:
            transaction.rollback()
            if engine.dialect.name == "sqlite":
                connection.exec_driver_sql("PRAGMA query_only = OFF")

    if not rows:
        return "The query returned no rows."
    shown = rows[:MAX_ROWS]
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in shown]
    if len(rows) > MAX_ROWS:
        lines.append(f"\n(Showing the first {MAX_ROWS} rows; add filters or aggregates.)")
    return "\n".join(lines)


def _cell(value: object) -> str:
    text_value = "NULL" if value is None else str(value).replace("|", "\\|").replace("\n", " ")
    return text_value if len(text_value) <= MAX_CELL else text_value[: MAX_CELL - 1] + "…"
