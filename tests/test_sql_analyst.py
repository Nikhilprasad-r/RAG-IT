import importlib.util
import sqlite3
from pathlib import Path
from types import ModuleType

import pytest

from ragkit.agents.templates import TEMPLATES_DIR


@pytest.fixture
def sql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    db = tmp_path / "shop.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, plan TEXT);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL,
                                 status TEXT);
            INSERT INTO customers VALUES (1, 'Ada', 'pro'), (2, 'Grace', 'team');
            INSERT INTO orders VALUES (1, 1, 20.0, 'paid'), (2, 1, 15.5, 'deleted'),
                                      (3, 2, 99.0, 'paid');
            """
        )
        conn.executemany(
            "INSERT INTO orders (customer_id, total, status) VALUES (1, 1.0, 'paid')",
            [()] * 250,
        )
    monkeypatch.setenv("SQL_ANALYST_DATABASE_URL", f"sqlite:///{db}")
    spec = importlib.util.spec_from_file_location(
        "sql_tools_under_test", TEMPLATES_DIR / "sql-analyst" / "tools.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_tools(sql: ModuleType) -> None:
    assert sql.list_tables() == "customers\norders"
    description = sql.describe_table("customers")
    assert "- name: VARCHAR NOT NULL" in description or "- name: TEXT NOT NULL" in description
    assert "| 1 | Ada | pro |" in description
    assert "No table named 'nope'" in sql.describe_table("nope")


def test_query_returns_a_markdown_table(sql: ModuleType) -> None:
    result = sql.run_query(
        "SELECT c.name, SUM(o.total) AS spent FROM orders o JOIN customers c "
        "ON c.id = o.customer_id WHERE o.status = 'paid' GROUP BY c.name ORDER BY spent DESC"
    )

    assert result.splitlines()[0] == "| name | spent |"
    assert "| Ada | 270.0 |" in result
    assert "| Grace | 99.0 |" in result


def test_string_literals_with_keywords_are_allowed(sql: ModuleType) -> None:
    result = sql.run_query("SELECT COUNT(*) AS n FROM orders WHERE status = 'deleted; drop'")
    assert "| 0 |" in result
    assert "| 1 |" in sql.run_query("SELECT COUNT(*) FROM orders WHERE status = 'deleted'")


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM orders",
        "UPDATE orders SET total = 0",
        "SELECT 1; DROP TABLE orders",
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
        "PRAGMA writable_schema = ON",
        "ATTACH DATABASE 'other.db' AS other",
    ],
)
def test_writes_are_refused(sql: ModuleType, statement: str) -> None:
    result = sql.run_query(statement)

    assert result.startswith("Only")
    assert "| 3 |" in sql.run_query(
        "SELECT COUNT(*) FROM customers c, orders o WHERE o.id <= 3 AND c.id = 1"
    )


def test_rows_are_capped_and_errors_are_reported(sql: ModuleType) -> None:
    capped = sql.run_query("SELECT id FROM orders")
    assert "Showing the first 200 rows" in capped
    rows = [line for line in capped.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 201  # header + 200 rows

    assert sql.run_query("SELECT nope FROM orders").startswith("The query failed:")


def test_database_stays_writable_for_others_after_a_query(sql: ModuleType) -> None:
    sql.run_query("SELECT 1")
    with sql._engine().connect() as connection:
        connection.exec_driver_sql("INSERT INTO customers VALUES (3, 'Linus', 'free')")
        connection.commit()
    assert "| 3 |" in sql.run_query("SELECT COUNT(*) FROM customers")
