import os
import sys
import re
import textwrap
from typing import Optional, Tuple, List, Union

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

# ---------- Helpers ----------

_RANGE_RE = re.compile(r"^\s*(\d+)\s*[:\-]\s*(\d+)\s*$")

def _parse_usecols_expr(expr: str) -> List[Union[int, str]]:
    """
    Parse a comma-separated expression of column selectors into a list usable by read_csv(usecols=...).
    Supports names, positions, and ranges (inclusive), e.g.:
      - "Account,Project"
      - "0,3,5"
      - "0-5,9,Name"
      - "2:7"
    """
    tokens = [t.strip() for t in (expr or "").split(",") if t.strip()]
    out: List[Union[int, str]] = []
    for tok in tokens:
        m = _RANGE_RE.match(tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            out.extend(range(lo, hi + 1))
            continue
        if tok.isdigit():
            out.append(int(tok))
            continue
        out.append(tok)  # treat as name
    return out

def csv_to_text(csv_path: str, usecols: Optional[str]) -> Tuple[str, int]:
    """
    Read CSV and return CSV text (header + rows) plus number of data rows.
    - Accepts COLUMNS env with names/positions/ranges.
    - Gracefully ignores out-of-range indices or missing names.
    - Truncates very long cells to ~500 chars.
    """
    cols = _parse_usecols_expr(usecols) if usecols else None

    try:
        df = pd.read_csv(csv_path, usecols=cols, dtype=str, encoding_errors="ignore")
    except ValueError:
        # Repair: validate against header, keep only valid selectors; else read all.
        header_only = pd.read_csv(csv_path, nrows=0, dtype=str, encoding_errors="ignore")
        names = list(header_only.columns)
        max_idx = len(names) - 1
        fixed: List[Union[int, str]] = []
        if isinstance(cols, list):
            for c in cols:
                if isinstance(c, int):
                    if 0 <= c <= max_idx:
                        fixed.append(c)
                elif c in names:
                    fixed.append(c)
        if fixed:
            try:
                df = pd.read_csv(csv_path, usecols=fixed, dtype=str, encoding_errors="ignore")
            except Exception:
                df = pd.read_csv(csv_path, dtype=str, encoding_errors="ignore")
                keep = [names[i] for i in fixed if isinstance(i, int)] + [c for c in fixed if isinstance(c, str)]
                keep = [c for c in keep if c in df.columns]
                if keep:
                    df = df[keep]
        else:
            df = pd.read_csv(csv_path, dtype=str, encoding_errors="ignore")

    n_rows = len(df)
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].map(lambda x: x if len(x) <= 500 else (x[:450] + "...[TRUNC]..."))
    return df.to_csv(index=False), n_rows

def _float_or_none(s: Optional[str]) -> Optional[float]:
    if s is None or s.strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _int_or_default(s: Optional[str], default: int) -> int:
    try:
        return int(s) if s else default
    except ValueError:
        return default

def _effort_or_none(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    v = s.strip().lower()
    return v if v in {"minimal", "low", "medium", "high"} else None

# ---------- Main (Call #1 only) ----------

def main():
    load_dotenv()

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    if not endpoint or not key:
        print("ERROR: Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in env/.env", file=sys.stderr)
        sys.exit(2)

    csv_path = os.environ.get("CSV_PATH")
    if not csv_path:
        print("ERROR: Please set CSV_PATH in your .env", file=sys.stderr)
        sys.exit(2)

    prompt_summary_user = (os.environ.get("PROMPT_SUMMARY") or "").strip()
    if not prompt_summary_user:
        prompt_summary_user = textwrap.dedent("""
        You are a precise data analyst.

        Task:
        1) Read the CSV below (first line is the header). Each row is a suspended business project.
        2) Briefly summarize the main reasons they did not proceed (≤ 200 words).
        3) Combine largely similar reasons into a concise, canonical list (each ≤ 4 words).

        Return ONLY the summary text. No code blocks. No tables. No CSV.
        """).strip()

    max_output_tokens = _int_or_default(os.environ.get("MAX_OUTPUT_TOKENS"), 2400)
    temperature = _float_or_none(os.environ.get("TEMPERATURE"))
    reasoning_effort = _effort_or_none(os.environ.get("REASONING_EFFORT"))
    columns = os.environ.get("COLUMNS", "").strip() or None

    csv_block, _ = csv_to_text(csv_path, columns)

    prompt_summary = (prompt_summary_user + "\n\n" + textwrap.dedent(f"""
    CSV data:
    ```csv
    {csv_block}
    ```
    """).strip()).strip()

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)

    kwargs_sum = {
        "model": deployment,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_summary}]}],
        "max_output_tokens": max_output_tokens,
        "store": True,
    }
    if temperature is not None:
        kwargs_sum["temperature"] = temperature
    if reasoning_effort:
        kwargs_sum["reasoning"] = {"effort": reasoning_effort}

    resp_sum = client.responses.create(**kwargs_sum)
    try:
        summary_text = (resp_sum.output_text or "").strip()
    except Exception:
        summary_text = (resp_sum.model_dump().get("output", {}).get("text", "") or "").strip()

    if not summary_text:
        print("ERROR: Empty summary from model.", file=sys.stderr)
        sys.exit(3)

    for line in summary_text.splitlines():
        print(line.rstrip())

if __name__ == "__main__":
    main()
