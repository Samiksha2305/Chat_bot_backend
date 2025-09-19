# analyst_agent.py
import os, io, json, sys, time, re
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for servers/CI
import matplotlib.pyplot as plt
from openai import OpenAI


class AnalystAgent:
    """
    Data analyst agent with tools to:
      - load_csv(csv_text):   load CSV from raw text
      - load_csv_file(path):  load CSV from a local file path
      - plot_hist(column):    save a histogram PNG for a numeric column (legacy)
      - plot_chart(...):      generic chart tool (histogram, bar, line, scatter, box, pie)

    If your prompt explicitly asks for a chart kind (scatter/histogram/line/box/pie/bar),
    we deterministically force that chart right after loading the CSV (using parsed args
    when available, otherwise sensible defaults). Any later model plot calls are ignored.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = (
            "You are a concise data analyst. You can load CSVs (from raw text or a file path), "
            "summarize shape/columns/dtypes/nulls/basic stats, and create ONE chart if asked. "
            "Use plot_chart for histogram, bar, pie, line, scatter, or box. "
            "If plotting fails, explain why. Be brief and actionable."
        )
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

        os.makedirs("data", exist_ok=True)

        self._df: Optional[pd.DataFrame] = None
        self._last_loaded_path: Optional[str] = None

        # last generated plot (so server can show it)
        self.last_plot_path: Optional[str] = None

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "load_csv",
                    "description": "Load CSV provided as raw text; return shape/columns/dtypes/nulls/describe.",
                    "parameters": {
                        "type": "object",
                        "properties": {"csv_text": {"type": "string"}},
                        "required": ["csv_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "load_csv_file",
                    "description": "Load CSV from a local file path; return shape/columns/dtypes/nulls/describe.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            },
            # Legacy histogram tool (kept for back-compat)
            {
                "type": "function",
                "function": {
                    "name": "plot_hist",
                    "description": "Plot histogram for a numeric column of the loaded DataFrame and save to a PNG.",
                    "parameters": {
                        "type": "object",
                        "properties": {"column": {"type": "string"}},
                        "required": ["column"],
                    },
                },
            },
            # Generic chart tool
            {
                "type": "function",
                "function": {
                    "name": "plot_chart",
                    "description": (
                        "Create a chart from the loaded DataFrame. "
                        "Supported kinds: histogram, bar, line, scatter, box, pie. "
                        "For bar/pie with categories, aggregate with 'count', 'sum', or 'mean' of value_column."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["histogram", "bar", "line", "scatter", "box", "pie"]},
                            "x": {"type": "string", "description": "X column (categorical or numeric)."},
                            "y": {"type": "string", "description": "Y column (numeric for line/scatter/box)."},
                            "value_column": {"type": "string", "description": "For bar/pie when agg is 'sum' or 'mean'."},
                            "agg": {"type": "string", "enum": ["count", "sum", "mean"], "description": "Aggregation for bar/pie."},
                            "bins": {"type": "integer", "description": "Bins for histogram."},
                            "normalize": {"type": "boolean", "description": "Normalize bar/pie to proportions."},
                            "title": {"type": "string", "description": "Optional chart title."}
                        },
                        "required": ["kind"]
                    },
                },
            },
        ]

    # ------------------ intent parsing & detection ------------------
    def _parse_chart_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Returns dict(kind, x, y, bins, agg, value_column, normalize, title) or None.
        Looks for explicit kind words, quoted column names, bins/agg/normalize/title.
        """
        t = text.lower()

        # chart kind (via synonyms & word boundaries)
        synonyms = {
            "histogram": [r"\bhist(?:ogram)?\b", r"\bdist(?:ribution)?\b"],
            "scatter": [r"\bscatter\b", r"\bxy\s*plot\b"],
            "line": [r"\bline\b", r"\btime\s*series\b", r"\btrend\b"],
            "box": [r"\bbox(?:-?\s*plot)?\b"],
            "pie": [r"\bpie\b", r"\bdonut\b"],
            "bar": [r"\bbar\b", r"\bcolumn\b"]
        }
        kind = None
        for k in ["scatter", "histogram", "line", "box", "pie", "bar"]:  # prefer specific kinds first
            if any(re.search(p, t) for p in synonyms[k]):
                kind = k
                break

        # columns: look for quoted names (keep original case)
        col_pat = r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]"
        cols = re.findall(col_pat, text)

        # bins
        bins = None
        m = re.search(r"\bbins?\s*[:=]*\s*(\d+)", t)
        if m:
            bins = int(m.group(1))

        # agg
        agg = None
        for a in ["count", "sum", "mean"]:
            if re.search(rf"\b{a}\b", t):
                agg = a
                break

        # normalize
        normalize = bool(re.search(r"\b(normalize|proportion|percentage|percent)\b", t))

        # title
        title = None
        m = re.search(r"(?:title|titled)\s*[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]", text, flags=re.I)
        if m:
            title = m.group(1)

        # assign x/y/value heuristically by kind
        x = y = value_col = None
        if kind == "scatter":
            if len(cols) >= 2:
                x, y = cols[0], cols[1]
        elif kind in ("histogram", "line", "box"):
            if len(cols) >= 1:
                if kind == "histogram":
                    x = cols[0]
                elif kind == "line":
                    if len(cols) >= 2:
                        x, y = cols[0], cols[1]
                    else:
                        y = cols[0]
                elif kind == "box":
                    if len(cols) >= 2:
                        x, y = cols[0], cols[1]  # group by x, box of y
                    else:
                        y = cols[0]
        elif kind in ("bar", "pie"):
            if len(cols) >= 1:
                x = cols[0]
            if len(cols) >= 2:
                value_col = cols[1]

        if not kind:
            return None
        return dict(kind=kind, x=x, y=y, bins=bins, agg=agg, value_column=value_col, normalize=normalize, title=title)

    def _detect_requested_kind(self, text: str) -> Optional[str]:
        """Backup: simple detector with synonyms."""
        t = text.lower()
        synonyms = {
            "histogram": [r"\bhist(?:ogram)?\b", r"\bdist(?:ribution)?\b"],
            "scatter": [r"\bscatter\b", r"\bxy\s*plot\b"],
            "line": [r"\bline\b", r"\btime\s*series\b", r"\btrend\b"],
            "box": [r"\bbox(?:-?\s*plot)?\b"],
            "pie": [r"\bpie\b", r"\bdonut\b"],
            "bar": [r"\bbar\b", r"\bcolumn\b"]
        }
        for k in ["scatter", "histogram", "line", "box", "pie", "bar"]:
            if any(re.search(p, t) for p in synonyms[k]):
                return k
        return None

    # ------------------ dataframe helpers ------------------
    def _summarize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            desc = df.describe(include="all").to_dict()
        except Exception:
            desc = {}
        nulls = df.isnull().sum().to_dict()
        dtypes = {c: str(t) for c, t in df.dtypes.to_dict().items()}
        return {
            "status": "ok",
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(df.columns),
            "dtypes": dtypes,
            "nulls": nulls,
            "describe": desc,
            "source_path": self._last_loaded_path or "",
        }

    def _new_plot_path(self, stem: str = "analyst_chart") -> str:
        ts = int(time.time() * 1000)
        return f"data/{stem}_{ts}.png"

    def _auto_chart_args(self, kind: str) -> Dict[str, Any]:
        """
        Sensible defaults when user didn't specify columns:
        - histogram: first numeric -> x
        - scatter: first two numerics -> x,y
        - line: first numeric -> y (x=index)
        - box: first numeric -> y; if a categorical exists, use it as x group
        - bar/pie: first categorical -> x (count)
        """
        if self._df is None:
            return {}
        df = self._df
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        if kind == "histogram":
            return {"kind": "histogram", "x": num_cols[0], "bins": 20} if num_cols else {}

        if kind == "scatter":
            return {"kind": "scatter", "x": num_cols[0], "y": num_cols[1]} if len(num_cols) >= 2 else {}

        if kind == "line":
            return {"kind": "line", "y": num_cols[0], "title": f"Line of {num_cols[0]} (index on x)"} if num_cols else {}

        if kind == "box":
            if not num_cols:
                return {}
            return {"kind": "box", "x": cat_cols[0], "y": num_cols[0]} if cat_cols else {"kind": "box", "y": num_cols[0]}

        if kind in ("bar", "pie"):
            return {"kind": kind, "x": cat_cols[0], "agg": "count", "title": f"{kind.capitalize()} of {cat_cols[0]} (count)"} if cat_cols else {}

        return {}

    # ------------------ tool implementations ------------------
    def load_csv(self, csv_text: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as e:
            return {"status": "error", "message": f"Failed to read CSV: {type(e).__name__}: {e}"}
        self._df = df
        self._last_loaded_path = None
        return self._summarize_df(df)

    def load_csv_file(self, file_path: str) -> Dict[str, Any]:
        try:
            if not os.path.isfile(file_path):
                return {"status": "error", "message": f"File not found: {file_path}"}
            df = pd.read_csv(file_path)
        except Exception as e:
            return {"status": "error", "message": f"Failed to read file: {type(e).__name__}: {e}"}
        self._df = df
        self._last_loaded_path = os.path.abspath(file_path)
        return self._summarize_df(df)

    def plot_hist(self, column: str) -> Dict[str, Any]:
        if self._df is None:
            return {"status": "error", "message": "No data loaded yet. Call load_csv/load_csv_file first."}
        if column not in self._df.columns:
            return {"status": "error", "message": f"Column '{column}' not found."}

        series = pd.to_numeric(self._df[column], errors="coerce").dropna()
        if series.empty:
            return {"status": "error", "message": f"Column '{column}' is non-numeric or empty after coercion."}

        try:
            out = self._new_plot_path("analyst_histogram")
            plt.figure()
            series.hist(bins=12)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            self.last_plot_path = out
            return {"status": "ok", "image_path": out}
        except Exception as e:
            plt.close()
            return {"status": "error", "message": f"Plot failed: {type(e).__name__}: {e}"}

    def plot_chart(
        self,
        kind: str,
        x: Optional[str] = None,
        y: Optional[str] = None,
        value_column: Optional[str] = None,
        agg: Optional[str] = None,
        bins: int = 12,
        normalize: bool = False,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._df is None:
            return {"status": "error", "message": "No data loaded yet. Use load_csv or load_csv_file first."}

        df = self._df.copy()
        out_path = self._new_plot_path("analyst_chart")

        try:
            plt.figure()

            if kind == "histogram":
                if not x or x not in df.columns:
                    return {"status": "error", "message": "For histogram, provide a numeric 'x' column."}
                s = pd.to_numeric(df[x], errors="coerce").dropna()
                if s.empty:
                    return {"status": "error", "message": f"Column '{x}' is non-numeric/empty after coercion."}
                s.hist(bins=bins)
                plt.xlabel(x); plt.ylabel("Frequency")

            elif kind == "scatter":
                if not x or not y or x not in df.columns or y not in df.columns:
                    return {"status": "error", "message": "For scatter, provide numeric 'x' and 'y' columns."}
                xs = pd.to_numeric(df[x], errors="coerce")
                ys = pd.to_numeric(df[y], errors="coerce")
                m = xs.notna() & ys.notna()
                if not m.any():
                    return {"status": "error", "message": "No numeric data after coercion for scatter."}
                plt.scatter(xs[m], ys[m])
                plt.xlabel(x); plt.ylabel(y)

            elif kind == "line":
                if y and y in df.columns:
                    ys = pd.to_numeric(df[y], errors="coerce")
                    if x and x in df.columns:
                        if pd.api.types.is_numeric_dtype(df[x]):
                            xs = pd.to_numeric(df[x], errors="coerce")
                        else:
                            xs = df[x].astype(str)
                        plt.plot(xs, ys); plt.xlabel(x)
                    else:
                        plt.plot(ys.reset_index(drop=True)); plt.xlabel("index")
                    plt.ylabel(y)
                else:
                    return {"status": "error", "message": "For line, provide numeric 'y' (and optional 'x')."}

            elif kind == "box":
                if not y or y not in df.columns:
                    return {"status": "error", "message": "For box, provide numeric 'y' (and optional 'x' for groups)."}
                if x and x in df.columns:
                    df.boxplot(column=y, by=x, grid=False, rot=45)
                    plt.suptitle(""); plt.xlabel(x); plt.ylabel(y)
                else:
                    vals = pd.to_numeric(df[y], errors="coerce").dropna()
                    if vals.empty:
                        return {"status": "error", "message": f"No numeric data in '{y}' for box plot."}
                    plt.boxplot(vals, vert=True); plt.ylabel(y)

            elif kind in ("bar", "pie"):
                if not x or x not in df.columns:
                    return {"status": "error", "message": "For bar/pie, provide categorical 'x' column."}
                if agg in (None, "count"):
                    series = df[x].value_counts(dropna=False)
                else:
                    if not value_column or value_column not in df.columns:
                        return {"status": "error", "message": "For bar/pie with 'sum'/'mean', provide 'value_column'."}
                    if agg == "sum":
                        series = df.groupby(x)[value_column].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
                    elif agg == "mean":
                        series = df.groupby(x)[value_column].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
                    else:
                        return {"status": "error", "message": "agg must be one of: count, sum, mean."}

                if normalize:
                    total = series.sum()
                    if total:
                        series = series / total

                if kind == "bar":
                    series.plot(kind="bar"); plt.xlabel(x); plt.ylabel("value")
                else:
                    series.plot(kind="pie", autopct="%1.1f%%"); plt.ylabel("")

            else:
                return {"status": "error", "message": f"Unsupported kind '{kind}'."}

            plt.title(title or kind.capitalize())
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            self.last_plot_path = out_path
            return {"status": "ok", "image_path": out_path}

        except Exception as e:
            plt.close()
            return {"status": "error", "message": f"Plot failed: {type(e).__name__}: {e}"}

    # ------------------ agent loop (with deterministic override) ------------------
    def run(self, user_message: str) -> str:
        """
        The model can call load_csv / load_csv_file, then plot_hist or plot_chart.
        If the user explicitly asks for a specific chart kind (scatter/histogram/pie/bar/line/box),
        we force that chart once after the CSV loads (auto-selecting reasonable columns if missing).
        Later stray plot tool-calls are ignored.
        """
        self.last_plot_path = None
        parsed = self._parse_chart_intent(user_message)
        requested_kind = parsed["kind"] if parsed else self._detect_requested_kind(user_message)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        forced_done = False  # once True, ignore any further plot tool calls

        for _ in range(8):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                for call in msg.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments or "{}")

                    # If we already forced the correct chart, ignore later plot calls from the model
                    if forced_done and name in ("plot_hist", "plot_chart"):
                        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": name,
                            "content": json.dumps({"status": "ok", "message": "Plot already generated; ignoring extra plot call."}),
                        })
                        continue

                    if name == "load_csv":
                        result = self.load_csv(**args)
                    elif name == "load_csv_file":
                        result = self.load_csv_file(**args)
                    elif name == "plot_hist":
                        result = self.plot_hist(**args)
                    elif name == "plot_chart":
                        result = self.plot_chart(**args)
                    else:
                        result = {"status": "error", "message": f"Unknown tool {name}"}

                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result),
                    })

                    # Deterministic override: immediately after LOAD, force the requested chart once
                    if name in ("load_csv", "load_csv_file") and requested_kind and not forced_done:
                        if parsed and parsed.get("kind"):
                            k = parsed["kind"]
                            if k == "histogram":
                                forced_args = {"column": parsed.get("x") or parsed.get("value_column") or parsed.get("y") or ""}
                                forced = self.plot_hist(**forced_args)
                                forced_name = "plot_hist"
                            else:
                                # filter None so tool args stay clean
                                raw_args = {
                                    "kind": k,
                                    "x": parsed.get("x"),
                                    "y": parsed.get("y"),
                                    "value_column": parsed.get("value_column"),
                                    "agg": parsed.get("agg"),
                                    "bins": parsed.get("bins") or 12,
                                    "normalize": bool(parsed.get("normalize")),
                                    "title": parsed.get("title"),
                                }
                                forced_args = {kk: vv for kk, vv in raw_args.items() if vv is not None}
                                forced = self.plot_chart(**forced_args)
                                forced_name = "plot_chart"
                        else:
                            auto_args = self._auto_chart_args(requested_kind)
                            if not auto_args:
                                # Can't auto-plot (e.g., missing numeric/categorical). Skip forcing.
                                forced = {"status": "error", "message": "Could not auto-select columns for requested chart."}
                                forced_name = "plot_chart"
                                forced_args = {"kind": requested_kind}
                            else:
                                if requested_kind == "histogram":
                                    forced_args = {"column": auto_args.get("x", "")}
                                    forced = self.plot_hist(**forced_args)
                                    forced_name = "plot_hist"
                                else:
                                    forced_args = auto_args
                                    forced = self.plot_chart(**forced_args)
                                    forced_name = "plot_chart"

                        fake_id = f"forced-{requested_kind}"
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": fake_id,
                                "type": "function",
                                "function": {"name": forced_name, "arguments": json.dumps(forced_args)}
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": fake_id,
                            "name": forced_name,
                            "content": json.dumps(forced)
                        })
                        forced_done = True
                continue

            return (msg.content or "").strip()

        return "I hit the tool-call limit. Try a more specific request."


# --------------- CLI ---------------
if __name__ == "__main__":
    agent = AnalystAgent()

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        prompt = (
            f"Load this CSV from file path: {csv_path}\n"
            "Summarize shape, columns, dtypes, nulls, and key stats. "
            "Then make a scatter plot if appropriate."
        )
        print(agent.run(prompt))
        sys.exit(0)

    print("AnalystAgent ready. Try:\n  python analyst_agent.py ./data/myfile.csv\n")
    sample_csv = "city,price\nSeattle,10\nSeattle,12\nAustin,7\nAustin,8\n"
    prompt = (
        "Here is a CSV:\n```csv\n" + sample_csv + "\n```\n"
        "Load it, summarize key stats, and plot a histogram of 'price'."
    )
    print(agent.run(prompt))
