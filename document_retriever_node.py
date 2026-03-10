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
_INVALID_SESSION_ERRORS = ("INVALID_STATE", "Invalid SessionHandle")


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


def _reset_databricks_connection():
    """Close and clear the cached Databricks connection after session expiry."""
    global _databricks_connection
    if _databricks_connection is not None:
        try:
            _databricks_connection.close()
        except Exception:
            pass
    _databricks_connection = None


def _execute_query(query: str, max_rows: int = 200) -> Tuple[List[str], List[Dict[str, Any]], int]:
    """Execute a read-only SQL query against the active backend.

    Returns (columns, rows, total_count).
    Routes to PostgreSQL or Databricks based on DB_BACKEND config.
    """
    if DB_BACKEND == "databricks":
        def _run_query(conn):
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

        conn = _get_databricks_connection()
        try:
            return _run_query(conn)
        except Exception as exc:
            message = str(exc)
            if any(err in message for err in _INVALID_SESSION_ERRORS):
                print("  Databricks session invalid; refreshing connection and retrying once")
                _reset_databricks_connection()
                conn = _get_databricks_connection()
                return _run_query(conn)
            raise
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
    injection into the LLM system prompt.
    Uses actual SQL column names (e.g. brand_det) so the LLM can reference
    them directly when building WHERE clauses."""
    if not _COLUMN_VALUES:
        return "(Could not load column values from database.)"
    # Reverse map: display_name -> actual SQL column name
    display_to_col = {display: col for col, display in _FILTERABLE_COLUMNS.items()}
    sections = []
    for display_name, values in _COLUMN_VALUES.items():
        if values:
            col_name = display_to_col.get(display_name, display_name)
            quoted = ", ".join(f"'{v}'" for v in values[:50])
            suffix = f" (+{len(values) - 50} more)" if len(values) > 50 else ""
            sections.append(f"  {col_name}: {quoted}{suffix}")
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


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize row keys so _det-suffixed columns are also accessible by their alias.
    e.g. file_category_det -> also stored as file_category, brand_det -> brand.
    This handles LLM queries that omit AS aliases in the SELECT clause.
    """
    normalized = []
    for row in rows:
        new_row = dict(row)
        for k, v in row.items():
            if k.endswith("_det"):
                alias = k[:-4]  # drop _det suffix
                if alias not in new_row:
                    new_row[alias] = v
        normalized.append(new_row)
    return normalized


def _format_results(listed_docs: List[Dict[str, Any]]) -> str:
    """Format document rows into clean, readable markdown output."""
    listed_docs = _normalize_rows(listed_docs)
    if not listed_docs:
        return (
            "No documents were found matching your request.\n\n"
            "**Suggestions:**\n"
            "- Try broadening your search (e.g. remove a specific filter)\n"
            "- Check the spelling of brand or category names\n"
            "- Ask me what categories, brands, or countries are available"
        )

    count = len(listed_docs)

    # All possible display columns — SQL alias -> human-readable label.
    # Handles both the old alias style (brand, time_period) and the newer
    # full-suffix-dropped style (file_category, file_time_period) so queries
    # from either convention are rendered correctly.
    col_checks = [
        ("file_category",        "Category"),
        ("file_sub_category",    "Sub-Category"),
        ("product_category",     "Product Category"),
        ("product_sub_category", "Product Sub-Category"),
        ("brand",                "Brand"),
        ("sub_brand_variant",    "Sub-Brand / Variant"),
        ("file_time_period",     "Time Period"),
        ("time_period",          "Time Period"),   # legacy alias
        ("country",              "Country"),
        ("region",               "Region"),
    ]

    # Deduplicate: skip a column key if its label was already shown (legacy aliases)
    seen_labels: set = set()
    deduped_col_checks = []
    for key, label in col_checks:
        if label not in seen_labels:
            deduped_col_checks.append((key, label))
            seen_labels.add(label)

    # Determine which columns actually have data in this result set
    active_cols = [
        (key, label) for key, label in deduped_col_checks
        if any(_clean_value(row.get(key)) for row in listed_docs)
    ]

    doc_word = "document" if count == 1 else "documents"
    lines = [f"Here are **{count} {doc_word}** matching your request:\n"]

    if active_cols and count >= 3:
        # --- Table format for 3+ results with metadata ---
        header    = "| # | Document | " + " | ".join(lbl for _, lbl in active_cols) + " |"
        separator = "|:---:|:---|" + "".join(":---|" for _ in active_cols)
        lines.append(header)
        lines.append(separator)

        for i, row in enumerate(listed_docs, 1):
            title = _clean_filename(row.get("document_title", "Untitled"))
            col_vals = [
                _clean_value(row.get(key)) or "\u2014" for key, _ in active_cols
            ]
            lines.append(f"| {i} | {title} | {' | '.join(col_vals)} |")

        # Summary footer
        lines.append("")
        if count >= 200:
            lines.append(
                "> **Note:** Showing the first 200 results. "
                "Refine your filters (brand, category, country, time period) to narrow the list."
            )
        elif count > 5:
            lines.append(
                f"> Tip: You can ask me to filter this list further by brand, category, "
                "country, or time period."
            )

    else:
        # --- Card-style list for 1-2 results or sparse metadata ---
        for i, row in enumerate(listed_docs, 1):
            title = _clean_filename(row.get("document_title", "Untitled"))
            lines.append(f"\n**{i}. {title}**")
            for key, label in active_cols:
                val = _clean_value(row.get(key))
                if val:
                    lines.append(f"   - **{label}:** {val}")

        if count >= 200:
            lines.append(
                "\n> **Note:** Showing the first 200 results. "
                "Refine your filters to narrow the list."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL execution tool — the LLM builds the query, this tool runs it
# ---------------------------------------------------------------------------

def _build_tool_docstring() -> str:
    """Build the SQL tool docstring dynamically.
    Only includes columns confirmed to exist in _AVAILABLE_COLUMNS so the LLM
    never references a column that isn't in the real table.
    """
    col_descriptions = {
        "file_category_det":        "file category",
        "file_sub_category_det":    "file sub-category",
        "product_category_det":     "product category",
        "product_sub_category_det": "product sub-category",
        "brand_det":                "brand",
        "sub_brand_variant_det":    "sub-brand or variant",
        "country_det":              "country",
        "region_det":               "region",
        "file_time_period_det":     "time period",
    }

    # Column list — only those confirmed to exist in this environment
    col_lines = [
        "  document_title                    — original document filename (no _det suffix)"
    ]
    for col, _display in _FILTERABLE_COLUMNS.items():
        if col in _AVAILABLE_COLUMNS:
            desc = col_descriptions.get(col, _display)
            col_lines.append(f"  {col:<34} — {desc}")
    col_list = "\n".join(col_lines)

    # Alias examples from the first few available columns
    alias_pairs = []
    for col in list(_AVAILABLE_COLUMNS)[:4]:
        alias = col.replace("_det", "")
        alias_pairs.append(f"{col} AS {alias}")
    alias_example = ", ".join(alias_pairs)
    if len(_AVAILABLE_COLUMNS) > 4:
        alias_example += ", ..."

    # Example SELECT using available columns
    ex_cols = ["DISTINCT document_title"]
    for col in ["file_category_det", "brand_det", "file_time_period_det", "country_det"]:
        if col in _AVAILABLE_COLUMNS:
            alias = col.replace("_det", "")
            ex_cols.append(f"{col} AS {alias}")
    ex_select = ",\n    ".join(ex_cols)

    ex_where_col = "file_category_det" if "file_category_det" in _AVAILABLE_COLUMNS else (
        next(iter(sorted(_AVAILABLE_COLUMNS))) if _AVAILABLE_COLUMNS else "file_category_det"
    )

    return f"""Execute a read-only SQL query against the {_TABLE_NAME} table and return results as JSON.

RULES:
- Only SELECT statements are allowed.
- The query MUST target the {_TABLE_NAME} table.
- Maximum 200 rows returned.

NAMING CONVENTION:
  Columns suffixed with "_det" hold AI-determined metadata (classified after upload).
  "document_title" is the original filename — it is NOT AI-determined and has no _det suffix.
  The time-period column has a "file_" prefix: file_time_period_det.

AVAILABLE COLUMNS — ONLY use these exact column names. Do NOT invent or guess others.
{col_list}

COLUMN USAGE:
  Alias _det columns by dropping the "_det" suffix in your SELECT clause:
    {alias_example}
  Use ILIKE for case-insensitive WHERE filtering:
    brand_det ILIKE '%Godrej%'
  Use exact values from AVAILABLE COLUMN VALUES in the system prompt for precise filtering.

EXAMPLE QUERY:
SELECT
    {ex_select}
FROM {_TABLE_NAME}
WHERE {ex_where_col} ILIKE '%Usage & Attitude%'
ORDER BY document_title
LIMIT 200;
"""


# Build the dynamic docstring once at module load (after _AVAILABLE_COLUMNS is known)
_TOOL_DOCSTRING = _build_tool_docstring()


@tool
def execute_metadata_sql(query: str) -> str:
    """Execute a read-only SQL query against the metadata table and return results as JSON."""
    # -- Safety checks --
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

    # -- Execute via the unified query interface --
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

    # -- Build prompt --
    tools = [execute_metadata_sql]
    tools_map = {"execute_metadata_sql": execute_metadata_sql}
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    # Build the dynamic values block for the system prompt
    column_values_block = _format_column_values_for_prompt()

    # Build a comma-separated list of columns that actually exist in this environment
    _available_col_list = "document_title, " + ", ".join(
        col for col in _FILTERABLE_COLUMNS if col in _AVAILABLE_COLUMNS
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a document listing agent for an enterprise document management system.
Your job is to query the {_TABLE_NAME} table to find and list documents matching the user's request.

INSTRUCTIONS:
1. Analyse the user's query and chat history to understand what documents they want.
2. Build a SQL query using the execute_metadata_sql tool.
3. Use ILIKE for case-insensitive matching on categories, brands, etc.
4. Always SELECT DISTINCT on document_title to avoid duplicates.
5. Include all relevant metadata columns in SELECT for richer results.
6. ONLY use the columns listed under AVAILABLE COLUMNS below. Never invent column names.

AVAILABLE COLUMNS (the only columns that exist in this table):
  {_available_col_list}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE COLUMN VALUES (loaded live from database — use for accurate filtering)
Each entry shows the exact SQL column name followed by its valid values.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""" + column_values_block + f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULT VERIFICATION & ZERO-RESULTS HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After every query, verify results match ALL the user's requested filters.
- If a filter term returned no matches, run a discovery query using one of the
  AVAILABLE COLUMNS above, e.g.:
    SELECT DISTINCT brand_det FROM {_TABLE_NAME} WHERE brand_det ILIKE '%soap%'
- Present discovered options to the user. NEVER silently ignore an unmatched filter.

If a query returns 0 rows:
1. Broaden: remove the most restrictive filter and retry.
2. Discover: query DISTINCT values for the failing column (use only AVAILABLE COLUMNS).
3. Ask: present available values as a clarifying question.
NEVER return "no results found" without trying steps 1-2 first. You have up to 4 tool calls.

When results are found, stop. For clarifications, write your question as the final response.
"""),
        MessagesPlaceholder("messages"),
        ("human", "Find documents for: {enriched_query}"),
    ])

    agent_messages = list(prompt.format_messages(
        messages=messages,
        enriched_query=enriched_query,
    ))

    # -- Tool-calling loop (Step 2D pattern) --
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

    # -- Determine final response --
    llm_final_text = (response.content.strip() if response and response.content else "")

    # Detect aggregate results (COUNT, GROUP BY, etc.) — rows exist but
    # contain no document_title, so they aren't listable documents.
    is_aggregate = (
        last_tool_rows
        and "document_title" not in last_tool_rows[0]
    )

    if not last_tool_rows or is_aggregate:
        # No document rows — use LLM's text (covers counts, clarifications, etc.)
        print("  Using LLM's text response (no document rows or aggregate result)")
        final_response = llm_final_text or "No documents found matching your query."
        listed_docs = []
    else:
        # Normal case — format document rows in Python
        listed_docs = last_tool_rows
        final_response = _format_results(listed_docs)

    print(f"\n  Document listing response ({len(final_response)} chars)")
    print(f"   Preview: {final_response[:300]}...")

    # -- Build output for state --
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
