"""Document Retriever Node — SQL agent for fast document listing.
Operates like Step 2E in semantic_node.py: LLM with bound SQL tool →
tool-calling loop → inline-formatted markdown response.
Routes here when intent = document_listing. Skips RAG and formatter
entirely — returns a pre-formatted markdown list directly.
Uses COALESCE(*_det, *_ai) so deterministic metadata is preferred
and AI-generated values are the fallback.
"""

import json
import re
from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from models import PipelineState, DocumentListingOutput
from persistence import pool as pg_pool
from utils import safe_utf8, sanitize_any, create_llm, execute_tool_calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_filename(raw: str) -> str:
    """Clean up raw document filenames for display.
    Strips leading numeric IDs/UUIDs, underscores, and file extensions
    to produce a human-readable title.
    Examples:
        '171452770_Hospital Link Express - QRE.pdf' → 'Hospital Link Express - QRE'
        'Cinthol Lime Fragrance Test (New Vendor)-Report-210421.pdf' → 'Cinthol Lime Fragrance Test (New Vendor)-Report-210421'
        'Citron_20-101250-01_Recall 1_V2.docx' → 'Citron 20-101250-01 Recall 1 V2'
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


# ---------------------------------------------------------------------------
# SQL execution tool — the LLM builds the query, this tool runs it
# ---------------------------------------------------------------------------

@tool
def execute_metadata_sql(query: str) -> str:
    """Execute a read-only SQL query against the metadata_gcpl table and return results as JSON.
    RULES:
    - Only SELECT statements are allowed.
    - The query MUST target the metadata_gcpl table.
    - Maximum 200 rows returned.
    AVAILABLE COLUMNS (all VARCHAR):
        document_title          — Document name / title
        file_category_det       — Deterministic file category
        file_category_ai        — AI-generated file category
        product_category_det    — Deterministic product category
        product_category_ai     — AI-generated product category
        brand_det               — Deterministic brand name
        brand_ai                — AI-generated brand name
        file_time_period_det    — Deterministic time period
        file_time_period_ai     — AI-generated time period
        country_det             — Deterministic country
        country_ai              — AI-generated country
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    COLUMN USAGE RULES (CRITICAL)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The _det and _ai columns may contain DIFFERENTLY WORDED values for the
    same concept (e.g., _det='Link Test', _ai='Link testing'). To avoid
    missed matches:
    FOR SELECT (display): Use COALESCE to prefer _det over _ai.
        COALESCE(file_category_det, file_category_ai) AS file_category
        COALESCE(product_category_det, product_category_ai) AS product_category
        COALESCE(brand_det, brand_ai) AS brand
        COALESCE(file_time_period_det, file_time_period_ai) AS time_period
        COALESCE(country_det, country_ai) AS country
    FOR WHERE (filtering): Use OR across BOTH columns to catch all matches.
        (file_category_det ILIKE '%X%' OR file_category_ai ILIKE '%X%')
        (product_category_det ILIKE '%X%' OR product_category_ai ILIKE '%X%')
        (brand_det ILIKE '%X%' OR brand_ai ILIKE '%X%')
        (file_time_period_det ILIKE '%X%' OR file_time_period_ai ILIKE '%X%')
        (country_det ILIKE '%X%' OR country_ai ILIKE '%X%')
    WRONG — misses rows where only _ai matches:
        WHERE COALESCE(file_category_det, file_category_ai) ILIKE '%Link testing%'
    CORRECT — catches matches in either column:
        WHERE (file_category_det ILIKE '%Link testing%' OR file_category_ai ILIKE '%Link testing%')
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    EXACT file_category values (case-insensitive matching recommended via ILIKE):
        'Analysis', 'Annual presentation', 'Brand equity', 'Brand Health track',
        'Concept testing', 'Home panel', 'Link testing', 'Media Optimization',
        'Miscellaneous', 'Needscope', 'Post Launch Evaluation',
        'Product acceptance testing', 'Product Performance Evaluation',
        'Retail audit', 'Usage/Attitude (U&A)'
    EXAMPLE QUERIES:
    -- List all U&A reports
    SELECT DISTINCT
        document_title,
        COALESCE(file_category_det, file_category_ai) AS file_category,
        COALESCE(brand_det, brand_ai) AS brand,
        COALESCE(file_time_period_det, file_time_period_ai) AS time_period,
        COALESCE(country_det, country_ai) AS country
    FROM metadata_gcpl
    WHERE (file_category_det ILIKE '%Usage/Attitude%' OR file_category_ai ILIKE '%Usage/Attitude%')
    ORDER BY document_title
    LIMIT 200;
    -- Count reports by category
    SELECT
        COALESCE(file_category_det, file_category_ai) AS file_category,
        COUNT(DISTINCT document_title) AS doc_count
    FROM metadata_gcpl
    GROUP BY file_category
    ORDER BY doc_count DESC;
    -- List brand equity reports for a specific brand
    SELECT DISTINCT
        document_title,
        COALESCE(brand_det, brand_ai) AS brand,
        COALESCE(file_time_period_det, file_time_period_ai) AS time_period,
        COALESCE(country_det, country_ai) AS country
    FROM metadata_gcpl
    WHERE (file_category_det ILIKE '%Brand equity%' OR file_category_ai ILIKE '%Brand equity%')
      AND (brand_det ILIKE '%Godrej%' OR brand_ai ILIKE '%Godrej%')
    ORDER BY document_title
    LIMIT 200;
    """
    # ── Safety checks ──
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed.", "query": query})

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]
    for kw in forbidden:
        # Match as whole word to avoid false positives (e.g., "SELECTED")
        if f" {kw} " in f" {normalized} " or normalized.startswith(f"{kw} "):
            return json.dumps({"error": f"Forbidden keyword: {kw}", "query": query})

    if "metadata_gcpl" not in query.lower():
        return json.dumps({"error": "Query must target the metadata_gcpl table.", "query": query})

    # ── Execute ──
    try:
        with pg_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchmany(200)

                # The pool uses dict_row, so rows are already dicts.
                # If they're tuples (no row_factory), convert manually.
                if rows and isinstance(rows[0], dict):
                    results = list(rows)
                else:
                    results = [dict(zip(columns, row)) for row in rows]

                total = cur.rowcount if cur.rowcount >= 0 else len(results)

                return json.dumps({
                    "columns": columns,
                    "rows": results,
                    "returned_count": len(results),
                    "total_count": total,
                }, default=str)

    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


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
    print("📋 DOCUMENT RETRIEVER NODE (SQL Agent)")
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document listing agent for an enterprise document management system.
Your job is to query the metadata_gcpl table to find and list documents matching the user's request.
INSTRUCTIONS:
1. Analyse the user's query and chat history to understand what documents they want.
2. Build a SQL query using the execute_metadata_sql tool.
3. ALWAYS use COALESCE(column_det, column_ai) for metadata columns to prefer deterministic values.
4. Use ILIKE for case-insensitive matching on categories, brands, etc.
5. Always SELECT DISTINCT on document_title to avoid duplicates.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULT VERIFICATION (CRITICAL — run after EVERY query that returns rows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
     SELECT DISTINCT COALESCE(product_category_det, product_category_ai) AS product_category
     FROM metadata_gcpl
     WHERE (file_category_det ILIKE '%Link testing%' OR file_category_ai ILIKE '%Link testing%')
     ORDER BY product_category;
4. Present the unmatched term + discovered options to the user:
     "I found results for **insecticides** but couldn't match **'bars'** to a
     known product category. Available categories include:
     1. Personal Wash
     2. Hair Care
     3. Home Care
     Which one did you mean by 'bars'?"
NEVER silently ignore a filter term that produced no matching results.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZERO-RESULTS FALLBACK (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If your first query returns 0 rows, DO NOT give up. The user's terms may not
match the exact metadata values. Follow this escalation:
STEP A — Broaden the failing filter.
  Remove the most restrictive filter (usually product_category or brand) and re-query.
  Example: if ILIKE '%Soap%' returned 0, try without the product_category filter.
STEP B — Discover available values.
  Run a discovery query to show the user what values actually exist:
    SELECT DISTINCT COALESCE(product_category_det, product_category_ai) AS product_category
    FROM metadata_gcpl
    WHERE COALESCE(file_category_det, file_category_ai) ILIKE '%Link testing%'
      AND COALESCE(country_det, country_ai) ILIKE '%India%'
    ORDER BY product_category;
  Or for brands:
    SELECT DISTINCT COALESCE(brand_det, brand_ai) AS brand
    FROM metadata_gcpl
    WHERE COALESCE(file_category_det, file_category_ai) ILIKE '%Link testing%'
    ORDER BY brand;
STEP C — Present options to the user.
  Format as a clarifying question:
    "I couldn't find link testing reports matching 'Soap'. Here are the available product categories for link testing reports:
    1. Personal Wash
    2. Hair Care
    3. Home Care
    Which category would you like to see?"
IMPORTANT:
- You have up to 4 tool calls. Use them: initial query → broaden → discover → (optional retry).
- NEVER return "no results found" without first trying Steps A and B.
- If discovery also returns 0, THEN say no documents exist for that report type.
- When presenting options, keep the format conversational and helpful.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING (only when results are found — Python handles the actual formatting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When results ARE found, you can stop — Python code will format the rows.
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
            print(f"⚠️ LLM error: {str(e)[:200]}")
            raise

        if response.content:
            response.content = safe_utf8(response.content)

        agent_messages.append(response)
        all_new_messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                tc_name = tc.name if hasattr(tc, "name") else tc.get("name", "?")
                tc_args = tc.args if hasattr(tc, "args") else tc.get("args", {})
                print(f"\n{'─' * 60}")
                print(f"🔧 SQL Tool Call [{iteration}]: {tc_name}")
                print(f"{'─' * 60}")
                sql_query = tc_args.get("query", "") if isinstance(tc_args, dict) else ""
                print(f"  SQL: {sql_query}")
                print(f"{'─' * 60}")

            tool_messages = execute_tool_calls(response.tool_calls, tools_map)

            # Log results and collect rows
            for tm in tool_messages:
                try:
                    result = json.loads(tm.content)
                    if "error" in result:
                        print(f"  ❌ SQL Error: {result['error']}")
                    else:
                        print(f"  ✅ Returned {result.get('returned_count', '?')} rows "
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
            print("✅ SQL agent finished")
            break

    # ── Determine final response ──
    # The only reliable signal is whether SQL returned rows:
    #   • rows returned  → format them in Python (deterministic, consistent)
    #   • no rows        → LLM wrote a clarification question or "no results" explanation; use that text
    # Avoid keyword heuristics ("which", "available", "?") — they match normal result text too.
    llm_final_text = (response.content.strip() if response and response.content else "")

    if not last_tool_rows:
        # No SQL results — LLM produced either a clarification question or a no-results message
        print("💬 Using LLM's clarification/explanation text")
        final_response = llm_final_text or "No documents found matching your query."
        listed_docs = []
    elif last_tool_rows:
        # Normal case — format document rows in Python
        listed_docs = last_tool_rows
        lines = [f"Found **{len(listed_docs)}** document(s) matching your query.\n"]

        # Check if results have enough metadata columns for a table
        has_rich_metadata = any(
            row.get("brand") or row.get("time_period") or row.get("product_category")
            for row in listed_docs
        )

        if has_rich_metadata and len(listed_docs) >= 3:
            # ── Table format for 3+ results with metadata ──
            # Determine which columns have data
            col_checks = [
                ("file_category", "Category"),
                ("product_category", "Product"),
                ("brand", "Brand"),
                ("time_period", "Period"),
                ("country", "Country"),
            ]
            active_cols = [
                (key, label) for key, label in col_checks
                if any(row.get(key) for row in listed_docs)
            ]

            # Build header
            header = "| # | Document Title | " + " | ".join(lbl for _, lbl in active_cols) + " |"
            separator = "|---|---|" + "|".join("---" for _ in active_cols) + "|"
            lines.append(header)
            lines.append(separator)

            for i, row in enumerate(listed_docs, 1):
                title = _clean_filename(row.get("document_title", "Untitled"))
                cols = " | ".join(row.get(key, "—") or "—" for key, _ in active_cols)
                lines.append(f"| {i} | **{title}** | {cols} |")
        else:
            # ── Numbered list for fewer results or sparse metadata ──
            for i, row in enumerate(listed_docs, 1):
                title = _clean_filename(row.get("document_title", "Untitled"))
                meta_parts = []
                for key, label in [
                    ("file_category", "Category"),
                    ("product_category", "Product"),
                    ("brand", "Brand"),
                    ("time_period", "Period"),
                    ("country", "Country"),
                ]:
                    val = row.get(key)
                    if val:
                        meta_parts.append(f"{label}: {val}")
                meta_str = " | ".join(meta_parts)
                line = f"{i}. **{title}**"
                if meta_str:
                    line += f" — {meta_str}"
                lines.append(line)

        final_response = "\n".join(lines)
    else:
        listed_docs = []
        final_response = llm_final_text or "No documents found matching your query."

    print(f"\n📋 Document listing response ({len(final_response)} chars)")
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
