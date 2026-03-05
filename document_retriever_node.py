"""Document Retriever Node — SQL agent for fast document listing.
Operates like Step 2E in semantic_node.py: LLM with bound SQL tool ->
tool-calling loop -> inline-formatted markdown response.
Routes here when intent = document_listing. Skips RAG and formatter
entirely — returns a pre-formatted markdown list directly.
Uses *_det columns only — for both display (SELECT) and filtering (WHERE).
Supports both PostgreSQL (local) and Databricks (development) backends.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from config import METADATA_TABLE_NAME, DB_BACKEND
from models import PipelineState, DocumentListingOutput
from persistence import pool as pg_pool
from utils import safe_utf8, sanitize_any, create_llm, execute_tool_calls


# ---------------------------------------------------------------------------
# Database query layer (supports PostgreSQL and Databricks)
# ---------------------------------------------------------------------------

_databricks_connection = None


def _get_databricks_connection():
    """Return a reusable Databricks SQL connection (created on first call)."""
    global _databricks_connection
    if _databricks_connection is None:
        from databricks import sql as databricks_sql
        _databricks_connection = databricks_sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME", ""),
            http_path=os.getenv("DATABRICKS_HTTP_PATH", ""),
            access_token=os.getenv("DATABRICKS_ACCESS_TOKEN", ""),
        )
    return _databricks_connection


def _execute_query(query: str, max_rows: int = 200) -> Tuple[List[str], List[Dict[str, Any]], int]:
    """Execute a read-only SQL query against the active backend.

    Returns (columns, rows, total_count).
    Routes to PostgreSQL or Databricks based on DB_BACKEND config.
    """
    if DB_BACKEND == "databricks":
        conn = _get_databricks_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            raw_rows = cursor.fetchmany(max_rows)
            rows = [dict(zip(columns, row)) for row in raw_rows]
            total = cursor.rowcount if cursor.rowcount and cursor.rowcount >= 0 else len(rows)
            return columns, rows, total
        finally:
            cursor.close()
    else:
        with pg_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                raw_rows = cur.fetchmany(max_rows)
                if raw_rows and isinstance(raw_rows[0], dict):
                    rows = list(raw_rows)
                else:
                    rows = [dict(zip(columns, row)) for row in raw_rows]
                total = cur.rowcount if cur.rowcount >= 0 else len(rows)
                return columns, rows, total


# ---------------------------------------------------------------------------
# Dynamic column value discovery (cached at module load)
# ---------------------------------------------------------------------------

_TABLE_NAME = METADATA_TABLE_NAME

_FILTERABLE_COLUMNS = {
    "file_category_det": "file_category",
    "file_sub_category_det": "file_sub_category",
    "product_category_det": "product_category",
    "product_sub_category_det": "product_sub_category",
    "brand_det": "brand",
    "sub_brand_variant_det": "sub_brand_variant",
    "country_det": "country",
    "region_det": "region",
    "file_time_period_det": "time_period",
}


def _fetch_distinct_column_values() -> Dict[str, List[str]]:
    """Query the metadata table for distinct values of every filterable
    column.  Returns a dict mapping display names to sorted value lists.
    Called once at module load and cached in ``_COLUMN_VALUES``.
    """
    result: Dict[str, List[str]] = {}
    for col, display in _FILTERABLE_COLUMNS.items():
        try:
            query = (
                f"SELECT DISTINCT {col} FROM {_TABLE_NAME} "
                f"WHERE {col} IS NOT NULL AND {col} != '' "
                f"ORDER BY {col}"
            )
            columns, rows, total = _execute_query(query, max_rows=500)
            result[display] = [row[col] for row in rows if row.get(col)]
        except Exception as e:
            # Column may not exist in this environment — skip gracefully
            print(f"  Note: column {col} not available ({e})")
    return result


# Cached at import time — lightweight queries, no repeated DB hits per request
_COLUMN_VALUES: Dict[str, List[str]] = _fetch_distinct_column_values()

# Build the set of columns that actually exist in this environment
_AVAILABLE_COLUMNS = {col for col, display in _FILTERABLE_COLUMNS.items()
                      if display in _COLUMN_VALUES and _COLUMN_VALUES[display]}


def _format_column_values_for_prompt() -> str:
    """Format the cached distinct values into a text block suitable for
    injection into the LLM system prompt."""
    if not _COLUMN_VALUES:
        return "(Could not load column values from database.)"
    sections = []
    for display_name, values in _COLUMN_VALUES.items():
        if values:
            quoted = ", ".join(f"'{v}'" for v in values)
            sections.append(f"  {display_name}: {quoted}")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_filename(raw: str) -> str:
    """Clean up raw document filenames for display.
    Strips leading numeric IDs/UUIDs, underscores, and file extensions
    to produce a human-readable title.
    """
    name = raw.strip()
    # Remove file extension
    name = re.sub(r'\.(pdf|pptx?|docx?|xlsx?|csv)$', '', name, flags=re.IGNORECASE)
    # Remove leading numeric ID prefix (e.g., '171452770_')
    name = re.sub(r'^\d{6,}_', '', name)
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name or raw


def _clean_value(val: Any) -> str:
    """Clean a metadata value for display: trim whitespace, handle
    None/empty, and apply title case where appropriate."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "n/a", "na", ""):
        return ""
    return s


def _format_results(listed_docs: List[Dict[str, Any]]) -> str:
    """Format document rows into clean, readable markdown output."""
    if not listed_docs:
        return "No documents found matching your query."

    count = len(listed_docs)
    lines = [f"Here are **{count}** document(s) matching your query:\n"]

    # All possible display columns (alias -> human-readable label)
    col_checks = [
        ("file_category", "Category"),
        ("file_sub_category", "Sub-Category"),
        ("product_category", "Product"),
        ("product_sub_category", "Product Sub-Category"),
        ("brand", "Brand"),
        ("sub_brand_variant", "Sub-Brand / Variant"),
        ("time_period", "Period"),
        ("country", "Country"),
        ("region", "Region"),
    ]

    # Determine which columns actually have data in the result set
    active_cols = [
        (key, label) for key, label in col_checks
        if any(_clean_value(row.get(key)) for row in listed_docs)
    ]

    has_rich_metadata = len(active_cols) > 0

    if has_rich_metadata and count >= 3:
        # ── Table format for 3+ results with metadata ──
        header = "| # | Document Title | " + " | ".join(lbl for _, lbl in active_cols) + " |"
        separator = "|:---:|---|" + "|".join("---" for _ in active_cols) + "|"
        lines.append(header)
        lines.append(separator)

        for i, row in enumerate(listed_docs, 1):
            title = _clean_filename(row.get("document_title", "Untitled"))
            col_vals = []
            for key, _ in active_cols:
                v = _clean_value(row.get(key))
                col_vals.append(v if v else "\u2014")
            lines.append(f"| {i} | {title} | {' | '.join(col_vals)} |")
    else:
        # ── Numbered list for fewer results or sparse metadata ──
        for i, row in enumerate(listed_docs, 1):
            title = _clean_filename(row.get("document_title", "Untitled"))
            meta_parts = []
            for key, label in col_checks:
                val = _clean_value(row.get(key))
                if val:
                    meta_parts.append(f"**{label}:** {val}")
            line = f"{i}. **{title}**"
            if meta_parts:
                line += "  \n   " + " | ".join(meta_parts)
            lines.append(line)

    if count >= 200:
        lines.append(f"\n*Showing first 200 results. Please refine your filters to narrow the list.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL execution tool — the LLM builds the query, this tool runs it
# ---------------------------------------------------------------------------

# Build the tool docstring dynamically based on the configured table name
_TOOL_DOCSTRING = f"""Execute a read-only SQL query against the {_TABLE_NAME} table and return results as JSON.

RULES:
- Only SELECT statements are allowed.
- The query MUST target the {_TABLE_NAME} table.
- Maximum 200 rows returned.

NAMING CONVENTION:
  Columns suffixed with "_det" hold AI-determined metadata (classified after upload).
  "document_title" is the original filename set at creation time — it is NOT
  AI-determined, so it has no _det suffix.
  The time-period column carries a "file_" prefix: file_time_period_det.

COMPLETE COLUMN LIST (all VARCHAR):
  document_title                — original document filename (no _det suffix)
  file_category_det             — e.g. Link Testing, Brand Health Track
  file_sub_category_det         — file sub-category
  product_category_det          — e.g. Household Insecticide, Personal Wash
  product_sub_category_det      — product sub-category
  brand_det                     — e.g. Good Knight, Cinthol
  sub_brand_variant_det         — sub-brand or variant
  country_det                   — e.g. India, Indonesia
  region_det                    — region
  file_time_period_det          — time period

COLUMN USAGE:
  Use _det columns for both SELECT and WHERE.  Alias them without the suffix:
    file_category_det AS file_category, brand_det AS brand, ...
  Use ILIKE for case-insensitive filtering:
    brand_det ILIKE '%Godrej%'
  Refer to AVAILABLE COLUMN VALUES in the system prompt for valid values.

EXAMPLE QUERIES:
-- List all U&A reports
SELECT DISTINCT
    document_title,
    file_category_det AS file_category,
    brand_det AS brand,
    file_time_period_det AS time_period,
    country_det AS country
FROM {_TABLE_NAME}
WHERE file_category_det ILIKE '%Usage & Attitude%'
ORDER BY document_title
LIMIT 200;

-- Count reports by category
SELECT
    file_category_det AS file_category,
    COUNT(DISTINCT document_title) AS doc_count
FROM {_TABLE_NAME}
GROUP BY file_category
ORDER BY doc_count DESC;

-- List brand equity reports for a specific brand
SELECT DISTINCT
    document_title,
    brand_det AS brand,
    sub_brand_variant_det AS sub_brand_variant,
    file_time_period_det AS time_period,
    country_det AS country,
    region_det AS region
FROM {_TABLE_NAME}
WHERE file_category_det ILIKE '%Brand equity%'
  AND brand_det ILIKE '%Godrej%'
ORDER BY document_title
LIMIT 200;
"""


@tool
def execute_metadata_sql(query: str) -> str:
    """Execute a read-only SQL query against the metadata table and return results as JSON."""
    # ── Safety checks ──
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed.", "query": query})

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]
    for kw in forbidden:
        # Match as whole word to avoid false positives (e.g., "SELECTED")
        if f" {kw} " in f" {normalized} " or normalized.startswith(f"{kw} "):
            return json.dumps({"error": f"Forbidden keyword: {kw}", "query": query})

    if _TABLE_NAME.lower() not in query.lower():
        return json.dumps({"error": f"Query must target the {_TABLE_NAME} table.", "query": query})

    # ── Execute via the unified query interface ──
    try:
        columns, rows, total = _execute_query(query, max_rows=200)

        return json.dumps({
            "columns": columns,
            "rows": rows,
            "returned_count": len(rows),
            "total_count": total,
        }, default=str)

    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


# Inject the dynamic docstring so the LLM sees the correct table name and columns
execute_metadata_sql.__doc__ = _TOOL_DOCSTRING


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def document_retriever_node(state: PipelineState) -> Dict[str, Any]:
    """
    Document retriever node — SQL agent for document listing queries.
    Pattern: Same as semantic_node Step 2D (tool-calling loop).
    1. LLM receives user query + chat history + SQL tool
    2. LLM generates SQL via tool call
    3. Tool executes query, returns results
    4. LLM formats results into markdown response
    5. Returns pre-formatted output (skips RAG + formatter)
    """

    print("\n" + "=" * 70)
    print("DOCUMENT RETRIEVER NODE (SQL Agent)")
    print(f"  Backend: {DB_BACKEND} | Table: {_TABLE_NAME}")
    print("=" * 70)

    user_query = state.user_query
    enriched_query = state.enriched_query or user_query
    messages = state.messages or []

    print(f"  Query: {user_query}")
    print(f"  Enriched: {enriched_query}")

    # ── Build prompt ──
    tools = [execute_metadata_sql]
    tools_map = {"execute_metadata_sql": execute_metadata_sql}
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    # Build the dynamic values block for the system prompt
    column_values_block = _format_column_values_for_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a document listing agent for an enterprise document management system.
Your job is to query the {_TABLE_NAME} table to find and list documents matching the user's request.
INSTRUCTIONS:
1. Analyse the user's query and chat history to understand what documents they want.
2. Build a SQL query using the execute_metadata_sql tool.
3. Use _det columns for both SELECT and WHERE. Alias them without the suffix for display.
4. Use ILIKE for case-insensitive matching on categories, brands, etc.
5. Always SELECT DISTINCT on document_title to avoid duplicates.
6. Include all relevant metadata columns in SELECT for richer results.

NAMING CONVENTION (understand this, don't memorise column names):
  "_det" columns = AI-determined metadata, classified after upload.
  "document_title" = original filename from creation time, NOT AI-determined, so no _det.
  The time-period column is prefixed with "file_": file_time_period_det.
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
AVAILABLE COLUMN VALUES (loaded from database \u2014 use these for accurate filtering)
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
""" + column_values_block + f"""
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
RESULT VERIFICATION (CRITICAL \u2014 run after EVERY query that returns rows)
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
After getting results, CHECK whether they actually match ALL the user's
requested filters. Do NOT blindly accept results.
VERIFY:
1. If user asked for MULTIPLE categories/brands (e.g., "bars and insecticides"),
   check that results contain BOTH. If results only cover one, the other term
   likely did not match any real metadata value.
2. If a filter term is VAGUE or INFORMAL (e.g., "bars", "soaps", "sprays"),
   check the returned product_category / brand values. If none of them
   obviously correspond to the vague term, that term is AMBIGUOUS.
3. For any ambiguous or unmatched term, run a discovery query:
     SELECT DISTINCT product_category_det AS product_category
     FROM {_TABLE_NAME}
     WHERE file_category_det ILIKE '%Link testing%'
     ORDER BY product_category;
4. Present the unmatched term + discovered options to the user:
     "I found results for **insecticides** but couldn't match **'bars'** to a
     known product category. Available categories include:
     1. Personal Wash
     2. Hair Care
     3. Home Care
     Which one did you mean by 'bars'?"
NEVER silently ignore a filter term that produced no matching results.
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
ZERO-RESULTS FALLBACK (CRITICAL)
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
If your first query returns 0 rows, DO NOT give up. The user's terms may not
match the exact metadata values. Follow this escalation:
STEP A \u2014 Broaden the failing filter.
  Remove the most restrictive filter (usually product_category or brand) and re-query.
  Example: if ILIKE '%Soap%' returned 0, try without the product_category filter.
STEP B \u2014 Discover available values.
  Run a discovery query to show the user what values actually exist:
    SELECT DISTINCT product_category_det AS product_category
    FROM {_TABLE_NAME}
    WHERE file_category_det ILIKE '%Link testing%'
      AND country_det ILIKE '%India%'
    ORDER BY product_category;
  Or for brands:
    SELECT DISTINCT brand_det AS brand
    FROM {_TABLE_NAME}
    WHERE file_category_det ILIKE '%Link testing%'
    ORDER BY brand;
STEP C \u2014 Present options to the user.
  Format as a clarifying question:
    "I couldn't find link testing reports matching 'Soap'. Here are the available
    product categories for link testing reports:
    1. Personal Wash
    2. Hair Care
    3. Home Care
    Which category would you like to see?"
IMPORTANT:
- You have up to 4 tool calls. Use them: initial query -> broaden -> discover -> (optional retry).
- NEVER return "no results found" without first trying Steps A and B.
- If discovery also returns 0, THEN say no documents exist for that report type.
- When presenting options, keep the format conversational and helpful.
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
FORMATTING (only when results are found \u2014 Python handles the actual formatting)
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
When results ARE found, you can stop \u2014 Python code will format the rows.
When results are NOT found and you're asking a clarifying question, write
the clarifying message as your final text response.
"""),
        MessagesPlaceholder("messages"),
        ("human", "Find documents for: {enriched_query}"),
    ])

    agent_messages = list(prompt.format_messages(
        messages=messages,
        enriched_query=enriched_query,
    ))

    # ── Tool-calling loop (Step 2D pattern) ──
    all_new_messages = []
    max_iterations = 5
    response = None
    last_tool_rows: List[Dict[str, Any]] = []  # Collect raw rows for Python formatting

    for iteration in range(max_iterations):
        print(f"\n--- SQL Agent Iteration {iteration + 1} ---")

        try:
            response = llm_with_tools.invoke(agent_messages)
        except Exception as e:
            print(f"  LLM error: {str(e)[:200]}")
            raise

        if response.content:
            response.content = safe_utf8(response.content)

        agent_messages.append(response)
        all_new_messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                tc_name = tc.name if hasattr(tc, "name") else tc.get("name", "?")
                tc_args = tc.args if hasattr(tc, "args") else tc.get("args", {})
                print(f"\n{'_' * 60}")
                print(f"  SQL Tool Call [{iteration}]: {tc_name}")
                print(f"{'_' * 60}")
                sql_query = tc_args.get("query", "") if isinstance(tc_args, dict) else ""
                print(f"  SQL: {sql_query}")
                print(f"{'_' * 60}")

            tool_messages = execute_tool_calls(response.tool_calls, tools_map)

            # Log results and collect rows
            for tm in tool_messages:
                try:
                    result = json.loads(tm.content)
                    if "error" in result:
                        print(f"  SQL Error: {result['error']}")
                    else:
                        print(f"  Returned {result.get('returned_count', '?')} rows "
                              f"(total: {result.get('total_count', '?')})")
                        fetched = result.get("rows", [])
                        if fetched:          # never overwrite with an empty result
                            last_tool_rows = fetched
                except Exception:
                    pass

                if tm.content:
                    tm.content = safe_utf8(tm.content)

            agent_messages.extend(tool_messages)
            all_new_messages.extend(tool_messages)
        else:
            # No more tool calls — LLM finished
            print("  SQL agent finished")
            break

    # ── Determine final response ──
    llm_final_text = (response.content.strip() if response and response.content else "")

    if not last_tool_rows:
        # No SQL results — LLM produced either a clarification question or a no-results message
        print("  Using LLM's clarification/explanation text")
        final_response = llm_final_text or "No documents found matching your query."
        listed_docs = []
    else:
        # Normal case — format document rows in Python
        listed_docs = last_tool_rows
        final_response = _format_results(listed_docs)

    print(f"\n  Document listing response ({len(final_response)} chars)")
    print(f"   Preview: {final_response[:300]}...")

    # ── Build output for state ──
    listing_output = DocumentListingOutput(
        formatted_response=final_response,
        documents=listed_docs,
        total_count=len(listed_docs),
        query_used=enriched_query,
    )

    # Add the formatted response as an AI message for chat history
    ai_message = AIMessage(
        content=safe_utf8(final_response),
        metadata={"type": "document_listing", "node": "document_retriever", "doc_count": len(listed_docs)},
    )
    all_new_messages.append(ai_message)

    # If no rows were returned and the LLM asked a question, flag the state so
    # semantic_node routes the user's next reply straight back here (issue 5 fix).
    is_asking_clarification = (not last_tool_rows) and ("?" in final_response)

    return {
        "messages": sanitize_any(all_new_messages),
        "document_listing_output": sanitize_any(listing_output.dict()),
        "clarification_message": safe_utf8(final_response),  # For app.py response chain
        "semantic_chitchat": False,
        "awaiting_clarification": is_asking_clarification,
        "rag_output": None,
    }
