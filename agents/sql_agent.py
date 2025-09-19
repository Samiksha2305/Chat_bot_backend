# sql_agent.py
import os
import json
import time
import sqlite3
import threading
import contextlib
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from openai import RateLimitError, APIStatusError  # type: ignore
except Exception:
    RateLimitError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore


class SQLAgent:
    """
    SQL helper agent that can:
      - create tables (DDL) in an in-memory SQLite database
      - run arbitrary SQL (DML/SELECT), including MULTIPLE statements per call
    via OpenAI tool/function-calling.

    Web-safe:
      * SQLite opened with check_same_thread=False
      * All DB ops are guarded by an RLock
      * Fresh DB per run() so tables don't persist across clicks

    Output behavior:
      * If the final statement is SELECT, returns a Markdown table (deterministic)
      * Otherwise returns a short success summary
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = (
            "You are a helpful SQL assistant. "
            "If the schema is missing, create it first using DDL (CREATE TABLE ...). "
            "Prefer batching: you may send CREATE/INSERT/SELECT in a single sql_run separated by semicolons. "
            "When a SELECT is executed, the server will render the final result as a Markdown table."
        ),
        pragmas: Optional[List[str]] = None,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        tool_loop_sleep_sec: float = 0.4,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.tool_loop_sleep_sec = tool_loop_sleep_sec

        # connection + lock are set in _reset_db()
        self.conn: Optional[sqlite3.Connection] = None
        self._db_lock: Optional[threading.RLock] = None
        self._reset_db()

        # Tool specs the model can call
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "sql_create_schema",
                    "description": "Execute one or more DDL statements to define tables, indexes, etc. Use semicolons to separate statements.",
                    "parameters": {
                        "type": "object",
                        "properties": {"schema_sql": {"type": "string"}},
                        "required": ["schema_sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sql_run",
                    "description": "Run SQL. MULTIPLE statements allowed (separated by semicolons). If the final statement is SELECT, return its table; else return a rowcount summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ]

        # Optional startup pragmas
        if pragmas:
            with self._db_lock:
                cur = self.conn.cursor()
                for p in pragmas:
                    cur.execute(p)
                self.conn.commit()

    # ---------- DB lifecycle ----------
    def _reset_db(self):
        """Fresh in-memory DB; safe for cross-thread web usage."""
        with contextlib.suppress(Exception):
            if getattr(self, "conn", None):
                self.conn.close()
        self._db_lock = threading.RLock()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    # ---------- Small helpers ----------
    def _is_select(self, sql: str) -> bool:
        return sql.lstrip().lower().startswith("select")

    def _format_table_md(self, cols: List[str], rows: List[List[Any]]) -> str:
        if not rows:
            # still show header
            header = "| " + " | ".join(cols) + " |"
            sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
            return "\n".join([header, sep])
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines  = []
        for r in rows:
            lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
        return "\n".join([header, sep] + lines)

    # ---------- Tool implementations ----------
    def sql_create_schema(self, schema_sql: str) -> Dict[str, Any]:
        try:
            with self._db_lock:
                self.conn.executescript(schema_sql)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}

    def sql_run(self, query: str) -> Dict[str, Any]:
        """
        Execute one or more SQL statements in order.
        - If the final statement is SELECT, return {columns, rows}.
        - Otherwise return {rowcount} summarizing non-SELECT effects.
        """
        try:
            stmts = [s.strip() for s in query.split(";") if s.strip()]
            if not stmts:
                return {"status": "error", "message": "Empty SQL."}

            with self._db_lock:
                cur = self.conn.cursor()
                total_rowcount = 0
                last_select_cols: Optional[List[str]] = None
                last_select_rows: Optional[List[List[Any]]] = None

                for stmt in stmts:
                    if self._is_select(stmt):
                        cur.execute(stmt)
                        last_select_cols = [c[0] for c in cur.description]
                        last_select_rows = [list(r) for r in cur.fetchall()]
                    else:
                        cur.execute(stmt)
                        total_rowcount += max(cur.rowcount, 0)

                self.conn.commit()

            if last_select_rows is not None:
                return {"status": "ok", "columns": last_select_cols, "rows": last_select_rows, "rowcount": len(last_select_rows)}
            else:
                return {"status": "ok", "rowcount": total_rowcount}
        except Exception as e:
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}

    # ---------- OpenAI call with retry/backoff ----------
    def _chat_with_retry(self, *, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None):
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto" if tools else "none",
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                )
            except RateLimitError as e:
                last_err = e
            except APIStatusError as e:
                if getattr(e, "status_code", None) == 429:
                    last_err = e
                else:
                    raise
            except Exception as e:
                last_err = e
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_sec * (2 ** attempt))
                continue
            raise last_err

    # ---------- Main interaction loop ----------
    def run(self, user_message: str) -> str:
        """
        The model can call tools; we capture any SELECT result and return it as a table.
        """
        # Fresh DB on every web click so you don't keep old tables
        self._reset_db()

        user_message = user_message.strip()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        last_table: Optional[Tuple[List[str], List[List[Any]]]] = None
        last_tool_fp: Optional[Tuple[str, str]] = None

        for _ in range(8):
            resp = self._chat_with_retry(messages=messages, tools=self.tools)
            msg = resp.choices[0].message

            if msg.tool_calls:
                for call in msg.tool_calls:
                    fn = call.function.name
                    args = json.loads(call.function.arguments or "{}")

                    fp = (fn, json.dumps(args, sort_keys=True))
                    if fp == last_tool_fp:
                        result = {"status": "error", "message": "Duplicate identical tool call skipped."}
                    else:
                        if fn == "sql_create_schema":
                            result = self.sql_create_schema(**args)
                        elif fn == "sql_run":
                            result = self.sql_run(**args)
                            if result.get("status") == "ok" and "columns" in result:
                                last_table = (result["columns"], result["rows"])
                        else:
                            result = {"status": "error", "message": f"Unknown tool {fn}"}
                        last_tool_fp = fp

                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": fn,
                        "content": json.dumps(result),
                    })

                time.sleep(self.tool_loop_sleep_sec)
                continue

            # Model returned final text. Prefer a deterministic table if we have one.
            if last_table:
                cols, rows = last_table
                return self._format_table_md(cols, rows)
            return (msg.content or "").strip()

        # fallback if no final assistant content but we did run a SELECT
        if last_table:
            cols, rows = last_table
            return self._format_table_md(cols, rows)
        return "Finished without a final message. Try adding explicit SQL, e.g., CREATE/INSERT/SELECT separated by semicolons."

    def close(self):
        with contextlib.suppress(Exception):
            if self.conn:
                self.conn.close()


# ---------- Tiny CLI for quick testing ----------
if __name__ == "__main__":
    agent = SQLAgent()
    print("SQLAgent ready. Example prompts:\n"
          "  - Create a table sales(region TEXT, amount INT); insert rows; show total by region.\n"
          "  - Make a customers + orders schema and compute revenue by customer.\n")
    try:
        prompt = input("Your request: ").strip()
        if not prompt:
            prompt = (
                "CREATE TABLE sales(region TEXT, amount INT); "
                "INSERT INTO sales VALUES ('West',10),('East',20),('West',5),('North',30); "
                "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY region;"
            )
        print("\n--- Answer ---")
        print(agent.run(prompt))
    finally:
        agent.close()
