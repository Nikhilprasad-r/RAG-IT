# sql-analyst

Answers questions about a SQL database. It lists tables, inspects their
columns and sample rows, writes a SELECT, and explains the result with the SQL
it used.

Setup:

```bash
uv add "ragkit-agents[sql]" psycopg          # plus the driver for your database
export SQL_ANALYST_DATABASE_URL=postgresql+psycopg://readonly:pass@host/db
ragkit chat sql-analyst "Which five customers spent the most this year?"
```

Safety: `run_query` accepts a single SELECT (or WITH) statement, rejects write
keywords, runs in a read-only transaction on Postgres and SQLite, enforces a
15-second statement timeout on Postgres, returns at most 200 rows, and always
rolls back. Treat these checks as a backstop and connect with a database user
that can only read.

| File | What to change |
|---|---|
| `tools.py` | Row limits, timeouts, or a different connection setting |
| `prompts/system.md` | Business definitions ("active customer means...") and naming conventions |
