# rag_node.py
"""RAG Node - Full prompt with retrieval, synthesis, and citations"""

from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from tools import azure_ai_search
from models import RAGOutput, RetrievedDoc, PipelineState
import json
import hashlib
import pathlib
from memory_store import store
from langgraph.types import RunnableConfig
from utils import safe_utf8, sanitize_any, create_llm, filter_sensitive_content, execute_tool_calls

# Load summarization schemas once at module level
_SCHEMAS_PATH = pathlib.Path(__file__).parent / "schemas.json"
with open(_SCHEMAS_PATH) as _f:
    _SUMMARIZATION_SCHEMAS: dict = json.load(_f)



# Keywords that signal the user wants a chart or graph — routes to formatter_node.
_CHART_KEYWORDS = frozenset({
    'chart', 'graph', 'plot', 'visualize', 'visualise', 'visualization', 'visualisation',
    'bar chart', 'pie chart', 'bar graph', 'line graph', 'line chart', 'trend chart',
    'draw', 'diagram', 'mermaid',
})

def _wants_chart(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _CHART_KEYWORDS)

@tool
def get_summarization_schema(file_category_ai: str) -> str:
    """
    Summarization schema helper (tool).

    Use when summarizing a document:
    - Input: file_category_ai (e.g., "Link Test", "U&A", "Concept Test", "Dipstick")
    - Returns: JSON with:
      • slots: all fields to extract
      • section_hints: section names to target during retrieval

    Workflow:
    1) Identify file_category_ai.
       - If unknown: call azure_ai_search with selectFields="document_title,file_category_ai,content_path" and a query matching the document name/brand.
    2) Call this tool to get slots + section_hints.
    3) Retrieve chunks:
       - filter: file_category_ai eq '<value>' and text_document_id ne ''
       - selectFields: "content_text,document_title,content_path,locationMetadata"
       - paginate: top_k=100, skip as needed (max 4 calls)
    4) Use the slots and section_hints to guide retrieval and organise your answer.
    5) Write the final answer as a business report in Markdown — prose paragraphs, section
       headings, bullet lists, and tables. Do NOT output raw JSON.

    Examples:
    - get_summarization_schema("U&A")
    - get_summarization_schema("Concept Test")
    """
    schema = _SUMMARIZATION_SCHEMAS.get(file_category_ai)
    if not schema:
        # Try case-insensitive match
        for key, val in _SUMMARIZATION_SCHEMAS.items():
            if key.lower() == file_category_ai.lower().strip():
                schema = val
                break

    if schema:
        return json.dumps(schema, indent=2)

    return json.dumps({
        "error": f"No schema found for '{file_category_ai}'",
        "available_categories": list(_SUMMARIZATION_SCHEMAS.keys()),
    })




def _extract_retrieved_docs_from_messages(agent_messages: list) -> list:
    """
    Parse unique retrieved documents directly from azure_ai_search ToolMessages.
    Deterministic — no second LLM call needed. Deduplicates by document_title,
    keeping the highest score and collecting all seen page numbers.
    """
    from models import RetrievedDoc
    from langchain_core.messages import AIMessage

    # Build a map: tool_call_id → tool_name so we only process azure_ai_search results
    tool_call_names: dict = {}
    for msg in agent_messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tc in (msg.tool_calls or []):
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if tc_id and tc_name:
                    tool_call_names[tc_id] = tc_name

    seen: dict = {}  # filename → {content_path, score, pages: set}
    for msg in agent_messages:
        if not isinstance(msg, ToolMessage):
            continue
        # Only process azure_ai_search results
        tc_id = getattr(msg, "tool_call_id", None)
        if tool_call_names.get(tc_id) != "azure_ai_search":
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        for doc in data.get("documents", []):
            title = doc.get("document_title", "")
            if not title:
                continue
            path = doc.get("content_path", "")
            score = float(doc.get("score") or 0.0)
            page = doc.get("page_number")
            if title not in seen:
                seen[title] = {"content_path": path, "score": score, "pages": set()}
            else:
                if score > seen[title]["score"]:
                    seen[title]["score"] = score
                    seen[title]["content_path"] = path
            if page is not None:
                seen[title]["pages"].add(int(page))

    result = []
    for filename, info in seen.items():
        pages_set = info["pages"]
        if pages_set:
            lo, hi = min(pages_set), max(pages_set)
            pages_str = str(lo) if lo == hi else f"{lo}-{hi}"
        else:
            pages_str = None
        result.append(RetrievedDoc(
            filename=filename,
            content_path=info["content_path"],
            score=round(info["score"], 4),
            pages=pages_str,
            description=filename,  # filename is already descriptive
        ))
    return result


def _count_search_calls(agent_messages: list) -> int:
    """Count how many azure_ai_search tool calls were made."""
    from langchain_core.messages import AIMessage
    count = 0
    for msg in agent_messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tc in (msg.tool_calls or []):
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name == "azure_ai_search":
                    count += 1
    return count


def rag_node(state: PipelineState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Full RAG node with multi-phase document retrieval, page-level extraction, synthesis,
    citations, and quality standards. Uses full enterprise-grade prompt.
    """
    print("\n" + "="*70)
    print("📚 RAG NODE")
    print("RAG MESSAGES:")
    last_msgs = state.messages[-5:]
    for i, msg in enumerate(last_msgs):
        print(f"{i+1}. {msg.type}: {msg.content[:200]}") 

    user_query = state.user_query
    enriched_query = state.enriched_query or ""
    messages = state.messages

    # Early return: semantic_specific already answered directly from history.
    # enriched_query="" is the convention used to signal "no retrieval needed".
    if not enriched_query and state.clarification_message:
        print("⏭️  RAG NODE: skipping — semantic node already answered directly")
        return {
            "messages": sanitize_any(state.messages),
            "rag_output": sanitize_any(RAGOutput(
                retrieved_docs=[],
                final_answer=state.clarification_message,
                total_searches=0,
            ).dict()),
        }

    thread_id = config.get("configurable", {}).get("thread_id", "default")

    # Reuse memories already loaded by semantic_node — avoids a second DB round-trip.
    # Each entry is a plain dict: {filename, content_path, description, pages, score}.
    user_memories: list = state.user_memories or []

    # Format memories for the RAG prompt
    if user_memories:
        memories_text = "Previously retrieved documents:\n"
        for i, mem in enumerate(user_memories, start=1):
            filename = mem.get("filename", "")
            content_path = mem.get("content_path", "")
            description_preview = mem.get("description", "")[:300]
            pages = mem.get("pages", "")
            memories_text += f"- Doc {i}: {filename} ({content_path}) [Pages: {pages}] {description_preview}\n"
    else:
        memories_text = "No previously retrieved documents."

    # ------------------------------------------------------------------
    # Summarization pre-load: if semantic_node typed this as a
    # summarization task, call get_summarization_schema in Python
    # *before* the agent loop so the schema is always in context.
    # This is deterministic — no keyword scanning, no LLM decision.
    # ------------------------------------------------------------------
    task_type: str = state.task_type or "other"
    document_category: str = state.document_category or ""
    schema_injection: str = ""

    if task_type == "summarization":
        if document_category:
            schema_result = get_summarization_schema.invoke({"file_category_ai": document_category})
            parsed = json.loads(schema_result)
            if "error" not in parsed:
                schema_injection = (
                    f"SUMMARIZATION SCHEMA (pre-loaded for \"{document_category}\"):\n"
                    f"{schema_result}\n\n"
                    f"Use these slots and section_hints to structure your retrieval and answer.\n"
                    f"Retrieve the document content using this exact approach:\n"
                    f"  filter: document_title eq '<exact filename from user query>' and text_document_id ne ''\n"
                    f"  top_k: 100\n"
                    f"  select_fields: content_text,document_title,content_path,locationMetadata\n"
                )
                print(f"✅ Pre-loaded schema for '{document_category}'")
            else:
                # Category not in schemas — tell the agent which categories exist
                available = list(_SUMMARIZATION_SCHEMAS.keys())
                schema_injection = (
                    f"This is a summarization task. Call get_summarization_schema first "
                    f"with the best-matching category. Available: {available}."
                )
                print(f"⚠️  No schema for '{document_category}', injecting category hint")
        else:
            # Category unknown — let the agent discover and call the tool
            available = list(_SUMMARIZATION_SCHEMAS.keys())
            schema_injection = (
                f"This is a summarization task. Your FIRST tool call must be "
                f"get_summarization_schema(file_category_ai). "
                f"Available categories: {available}. "
                f"Infer the best match from the document name or context."
            )
            print("⚠️  Summarization task but no document_category — injecting tool-call instruction")

    llm = create_llm()
    tools = [azure_ai_search, get_summarization_schema]
    tools_map = {
        "azure_ai_search": azure_ai_search,
        "get_summarization_schema": get_summarization_schema,
    }
    llm_with_tools = llm.bind_tools(tools)

    # Focused RAG prompt - tool description handles "how to use the tool"
    prompt_text = """You are a RAG retrieval and analysis agent. Use the azure_ai_search tool to find documents, then synthesize professional answers with citations.

⚠️  YOUR CURRENT TASK — answer ONLY this question:
"{enriched_query}"

The conversation history below is BACKGROUND CONTEXT only.
You are NOT answering any earlier question from history — answer ONLY the question above.
Do NOT copy or re-use any prior AI response from the conversation history as your answer.

---
PREVIOUSLY RETRIEVED DOCUMENTS:
---
{memories_text}

---
PHASE 0: REPEAT-QUESTION SHORTCUT (before any tool call only)
---
SKIP search ONLY when the user repeats the exact same question from the immediately
preceding exchange and you already answered it.

NEVER SKIP when the current question differs in any way from the last question —
even if the topic is similar. Always search for fresh content.

⚠️  SYNTHESIS RULE: Once you have made any tool call and received results,
your final answer MUST be composed from those tool results only.
Never fall back to a prior AI response in the conversation history.
Conversation history is context — it is never the answer to the current query.

IF SKIPPING (exact repeat only):
  → Start: "As I just mentioned..." then restate the answer
  → retrieved_docs: [] | total_searches: 0
---

STEP 1: Assess Query Scope
- Determine if the query requires single or multiple documents
- Identify the primary domain/topic (e.g., market share analysis, consumer insights, financial metrics)
- Establish relevance criteria for document selection

STEP 2: Execute Strategic Search
Call tool: azure ai search
Search Strategy Guidelines:
- Use domain-specific keywords from the query
- Look for high-scoring documents
- For broad exploratory search: use general terms
- For targeted retrieval: use focused terms after identifying relevant sources
- ALWAYS do at least 2 searches for broad/multi-dimensional queries. First search = broad terms; follow-up searches = varied terms, different filters, or segment-level keywords to gather comprehensive coverage.
- If conclusive results are not found, formulate queries as KEYWORD EXPANSIONS, not natural-language phrases
- NEVER stop at one search if the first results are partial or older-period data. Try alternate
  phrasings: e.g. if "Lux brand growth" returns only India 2019 data, follow up with
  `"Lux" growth share sales market performance`, "Lux market share segment".
- Reason: Azure AI Search ranks on keyword overlap. Multi-keyword queries match docs that
  use "sales", "penetration", "volume", "gains", "decline", etc. — not just those containing
  the exact phrase. More keywords = broader recall.

CRITICAL: selectFields USAGE
- When LISTING documents (names, links, counts): use selectFields: "document_title,content_path"
- When READING/ANALYZING content: ALWAYS include "content_text"in selectFields (e.g., "document_title,content_path,locationMetadata/pageNumber,content_text")

STEP 3: For Recommendation/Judgment Questions
Use targeted query terms that capture both sides of the answer. Include positive AND negative terms in a single search.
Example: query: "recommend not recommend conclusion risk concern overall"
This ensures relevance ranking surfaces chunks from both supporting AND contradicting sections.
DO NOT use selectFields without "content_text" for these - you need the full text to analyze.
If results only show one perspective, do one follow-up search with opposing terms (e.g., "not recommend risk caution decline").
Always check if results contain contradicting viewpoints before giving a final answer.

STEP 4: Domain-Filtered Document Selection
CRITICAL RULES:
- Retrieve context ONLY from documents matching the query domain
- Do NOT mix content across unrelated documents
- Do NOT answer from wrong documents just because wording appears similar
- Prioritize depth over breadth: one highly relevant document > multiple loosely related ones

RETRIEVAL STRATEGY — Pick the right approach for each query type:

1. CONTENT SEARCH ("What does X say about Y", "Find insights on Z"):
   → Use keyword search with top_k=10-50. No facets needed.
   → Include content_text in select_fields if you need to read the text.

2. PAGE-SPECIFIC ("What's on page 6 of Report.pdf"):
   → Use filter with locationMetadata/pageNumber.

3. SUMMARIZATION ("Summarize document X", "Summarize these documents", "Summarize observations in document X", "Summarize section in X"):
   → Identify file category(file_category_ai) from user query or memory (use azure_ai_search with selectFields to find it when unknown).
   → Call get_summarization_schema(file_category_ai) to get slots + section_hints; use them to target retrieval.
   → Then call azure_ai_search with "query="*", index_type="main_data", top_k=100, filter="document_title eq 'Report.pdf'", select_fields="content_text,document_title,content_path,locationMetadata" to retrieve relevant chunks.
   → Compose a business report style answer using the slots and section_hints:
        - For each section, write in a business report style. Avoid single-line slot responses.
        - Include quantitative tables where applicable (e.g., metrics vs norms).
   → Fill every slot; set null for missing fields and do not fabricate information.
   → Add extra crucial details in additional_notes if needed.
   → For multi-document requests:
        - Apply full schema pipeline separately per document. Use get_summarization_schema to get the right schema for each document type.
        - DO NOT merge.

SYNTHESIS RULES:
- Answer the CURRENT enriched_query. Never answer an older question from the conversation history.
- If tool calls were made this turn, base your answer ENTIRELY on those tool results — do not use any prior AI response as your answer.
- Executive Summary (2-3 sentences), then Detailed Analysis with inline citations, then Key Takeaways (3-5 bullets).
- Use business report formatting: clear section headings, bullet lists, and tables for numeric comparisons.
- Avoid terse one-liners; provide explanatory sentences grounded in retrieved evidence.
- ALWAYS cite with page numbers: 📄 [filename](content_path) (Page N)
- For listing queries: return ALL documents from search, not a filtered subset.
- Evidence-based claims only — do not fabricate information.

CRITICAL:
- Include page numbers in every citation from locationMetadata/pageNumber.
- Use content_path from search results for links — NEVER reconstruct URLs.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            MessagesPlaceholder("messages"),
            ("human", "Original query: {user_query}\nEnriched query: {enriched_query}"),
        ]
    )

    initial_messages = prompt.format_messages(
        user_query=user_query, enriched_query=enriched_query, messages=messages, memories_text=memories_text
    )

    agent_messages = list(initial_messages)

    # Inject schema context as the last message before the loop so it is
    # the most-recent context the model sees — impossible to overlook.
    from langchain_core.messages import SystemMessage as _SM
    if schema_injection:
        agent_messages.append(_SM(content=schema_injection))

    # Always pin the current task as the very last message so the LLM
    # cannot drift to answering an earlier question from the history window.
    agent_messages.append(_SM(
        content=(
            f"REMINDER — you are answering ONLY this question:\n"
            f"\"{enriched_query}\"\n"
            f"Do NOT answer any prior question from the conversation history."
        )
    ))

    all_new_messages = []
    max_iterations = 10
    response = None

    for iteration in range(max_iterations):
        try:
            # 🔹 Filter sensitive content from messages before sending
            filtered_messages = []
            for msg in agent_messages:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    msg.content = filter_sensitive_content(msg.content)
                filtered_messages.append(msg)
            
            response = llm_with_tools.invoke(filtered_messages)

        except ValueError as e:
            if "content filter" in str(e).lower():
                print(f"\n⚠️ CONTENT FILTER TRIGGERED (Iteration {iteration})")
                print(f"   Error: {str(e)[:200]}")
                print(f"   Attempting to generate simpler response...")
                
                # Fallback: Try with simplified prompt
                from langchain_core.messages import HumanMessage, AIMessage
                fallback_messages = [
                    ("system", "Generate a professional data analysis response with citations."),
                    ("human", f"User query: {user_query}")
                ]
                fallback_prompt = ChatPromptTemplate.from_messages(fallback_messages)
                fallback_llm = create_llm()
                
                try:
                    response = fallback_llm.invoke(
                        fallback_prompt.format_messages(user_query=user_query)
                    )
                    print(f"   ✓ Fallback response generated successfully")
                    break
                except Exception as fallback_error:
                    print(f"   ✗ Fallback also failed: {str(fallback_error)[:100]}")
                    raise
            else:
                raise

        # 🔹 Sanitize AI message content before saving
        if response.content:
            response.content = safe_utf8(response.content)

        # Append AI message
        agent_messages.append(response)
        all_new_messages.append(response)

        if response.tool_calls:
            # Log tool call args (clean single-block format)
            for tc in response.tool_calls:
                tc_name = tc.name if hasattr(tc, "name") else tc.get("name", "?")
                tc_args = tc.args if hasattr(tc, "args") else tc.get("args", {})
                print(f"\n{'─'*60}")
                print(f"🔧 Tool Call [{iteration}]: {tc_name}")
                print(f"{'─'*60}")
                for k, v in (tc_args.items() if isinstance(tc_args, dict) else {}):
                    print(f"  {k}: {v}")
                print(f"{'─'*60}")

            tool_messages = execute_tool_calls(response.tool_calls, tools_map)

            # Log search results
            parsed_results = {}
            for tool_msg in tool_messages:
                try:
                    if tool_msg.content not in parsed_results:
                        tool_result = json.loads(tool_msg.content)
                        parsed_results[tool_msg.content] = tool_result
                    else:
                        tool_result = parsed_results[tool_msg.content]

                    if isinstance(tool_result, dict):
                        total = tool_result.get("totalCount", "?")
                        docs = tool_result.get("documents", [])
                        facets_data = tool_result.get("facets", {})
                        unique_count = tool_result.get("uniqueDocumentCount", tool_result.get("uniqueDocumentsInBatch", "?"))
                        has_more = tool_result.get("hasMoreResults", False)

                        print(f"\n{'─'*60}")
                        print(f"📊 Search Results [{iteration}]")
                        print(f"{'─'*60}")
                        print(f"  Total chunks: {total} | Returned: {len(docs) if isinstance(docs, list) else '?'} | Unique docs: {unique_count} | More: {has_more}")

                        # Facets
                        if facets_data:
                            for fn, fv in facets_data.items():
                                if isinstance(fv, list):
                                    print(f"  Facet [{fn}]: {len(fv)} values")

                        # Top 5 docs
                        if isinstance(docs, list) and docs:
                            print(f"  {'─'*56}")
                            for i, d in enumerate(docs[:5]):
                                title = d.get("document_title", d.get("source", "?"))
                                page = d.get("page_number", "?")
                                score = d.get("score", "?")
                                content_preview = (d.get("content_text", "") or "")
                                print(f"  [{i+1}] {title} (p.{page}) score={score}")
                                if content_preview:
                                    print(f"      {content_preview}...")
                            if len(docs) > 5:
                                print(f"  ... +{len(docs) - 5} more")

                        if has_more:
                            print(f"  ⚠ More results available: nextSkip={tool_result.get('nextSkip')}, remaining={tool_result.get('remainingChunks')}")
                        print(f"{'─'*60}")
                except (json.JSONDecodeError, Exception) as e:
                    print(f"⚠️ Result parsing error: {str(e)}")

            # 🔹 Sanitize all tool messages
            for tm in tool_messages:
                if tm.content:
                    tm.content = safe_utf8(tm.content)

            agent_messages.extend(tool_messages)
            all_new_messages.extend(tool_messages)
        else:
            break

    raw_output = response.content if response else ""

    # DEBUG: Print raw output before parsing
    print("\n" + "-"*70)
    print("🐛 DEBUG: RAG NODE - Raw LLM Output")
    print("-"*70)
    print(f"Raw Output Length: {len(raw_output)} chars")
    print(f"Raw Output (first 500 chars):\n{raw_output[:500]}")
    print(f"Raw Output (last 500 chars):\n{raw_output[-500:]}")
    print(f"Total docs count in raw output: {raw_output.count('filename')}")
    print("-"*70)

    # Build RAGOutput directly — no second LLM call needed.
    # retrieved_docs are parsed from actual tool results (deterministic, exact).
    # raw_output IS the final answer; no extraction step can improve on it.
    retrieved_docs = _extract_retrieved_docs_from_messages(agent_messages)
    total_searches = _count_search_calls(agent_messages)
    output = RAGOutput(
        retrieved_docs=retrieved_docs,
        final_answer=raw_output,
        total_searches=total_searches,
    )
    
    # DEBUG: Print structured output before returning
    # print("\n" + "-"*70)
    # print("🐛 DEBUG: RAG NODE - Structured Output")
    # print("-"*70)
    # print(f"Output Type: {type(output)}")
    # print(f"Output Dict:\n{json.dumps(output.dict(), indent=2, default=str)}")
    # print("-"*70)
    if output.retrieved_docs:
        # Batch-write all docs in parallel using a thread pool.
        # Each store.put() is an independent DB round-trip; parallelising removes
        # the serial N × latency bottleneck.
        import concurrent.futures

        _ns = ("rag_memory", state.user_id, thread_id)

        def _put_doc(d):
            store.put(
                namespace=_ns,
                key=hashlib.sha256(
                    f"{d.content_path}|{d.pages}".encode()
                ).hexdigest()[:24],
                value={
                    "type": "retrieved_doc",
                    "filename": safe_utf8(d.filename),
                    "content_path": safe_utf8(d.content_path),
                    "description": safe_utf8(d.description),
                    "pages": d.pages,
                    "score": d.score,
                },
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(output.retrieved_docs), 8)
        ) as pool:
            list(pool.map(_put_doc, output.retrieved_docs))

        print(f"📚 Stored {len(output.retrieved_docs)} RAG docs to PostgresStore (parallel)")


    needs_formatter = _wants_chart(user_query) or _wants_chart(enriched_query or "")
    print(f"📊 needs_formatter: {needs_formatter}")

    return {
        "messages": sanitize_any(all_new_messages),
        "rag_output": sanitize_any(output.dict()),
        "needs_formatter": needs_formatter,
    }
