# rag_node.py
"""RAG Node - Full prompt with retrieval, synthesis, and citations"""

from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import azure_ai_search
from models import RAGOutput, PipelineState
import asyncio
import config
import json
from memory_store import store


# ----- Add this helper at the top of rag_node.py -----
def safe_utf8(text: str) -> str:
    if not text:
        return ""
    # Replace invalid UTF-8 characters with '?'
    # Also remove null bytes which PostgreSQL cannot handle in JSON
    cleaned = text.encode("utf-8", errors="replace").decode("utf-8")
    return cleaned.replace("\x00", "")


def filter_sensitive_content(text: str) -> str:
    """Remove or replace content that might trigger Azure content filters"""
    if not text:
        return ""
    
    # Replace potentially problematic patterns
    filtered = text
    
    # Common filter triggers - replace with safe alternatives
    filter_patterns = {
        r'(?i)content.*filter': 'content validation',
        r'(?i)harmful': 'inappropriate',
        r'(?i)violence|violent': 'aggressive',
        r'(?i)hate|hateful': 'prejudiced',
        r'(?i)abuse|abusive': 'harmful behavior',
    }
    
    import re
    for pattern, replacement in filter_patterns.items():
        try:
            filtered = re.sub(pattern, replacement, filtered)
        except:
            pass
    
    return filtered


# ------------------------------------------------------
def sanitize_any(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        return safe_utf8(obj)
    if isinstance(obj, list):
        return [sanitize_any(i) for i in obj]
    if isinstance(obj, dict):
        return {k: sanitize_any(v) for k, v in obj.items()}
    return obj


def create_llm():
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=1,
        timeout=120.0,
        max_retries=3,
    )


async def _invoke_single_tool_async(tool_call, tools_map: dict) -> ToolMessage:
    """Execute a single tool call asynchronously and return a ToolMessage."""
    if hasattr(tool_call, "name"):
        tool_name = tool_call.name
        tool_args = tool_call.args
        tool_id = tool_call.id
    else:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            return ToolMessage(
                content=json.dumps({"error": f"Invalid JSON in tool args: {tool_args}"}),
                tool_call_id=tool_id,
            )

    if tool_name in tools_map:
        try:
            # Run sync tool.invoke in a thread to avoid blocking the event loop
            result = await asyncio.to_thread(tools_map[tool_name].invoke, tool_args)
            return ToolMessage(content=result, tool_call_id=tool_id)
        except Exception as e:
            return ToolMessage(
                content=json.dumps({"error": str(e)}), tool_call_id=tool_id
            )
    else:
        return ToolMessage(
            content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
            tool_call_id=tool_id,
        )


def execute_tool_calls(tool_calls: list, tools_map: dict) -> list:
    """Execute tool calls in parallel using asyncio.gather."""
    if len(tool_calls) <= 1:
        # Single call — run synchronously (no async overhead)
        return [_invoke_single_tool_sync(tc, tools_map) for tc in tool_calls]

    print(f"  ⚡ Executing {len(tool_calls)} tool calls in parallel (asyncio.gather)")

    async def _gather_all():
        return await asyncio.gather(
            *[_invoke_single_tool_async(tc, tools_map) for tc in tool_calls]
        )

    # If an event loop is already running, use it; otherwise create one
    if _is_event_loop_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(asyncio.run, _gather_all()).result()
        return list(results)
    else:
        return list(asyncio.run(_gather_all()))


def _is_event_loop_running() -> bool:
    """Check if an asyncio event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
        return loop.is_running()
    except RuntimeError:
        return False


def _invoke_single_tool_sync(tool_call, tools_map: dict) -> ToolMessage:
    """Synchronous fallback for single tool call execution."""
    if hasattr(tool_call, "name"):
        tool_name = tool_call.name
        tool_args = tool_call.args
        tool_id = tool_call.id
    else:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            return ToolMessage(
                content=json.dumps({"error": f"Invalid JSON in tool args: {tool_args}"}),
                tool_call_id=tool_id,
            )

    if tool_name in tools_map:
        try:
            result = tools_map[tool_name].invoke(tool_args)
            return ToolMessage(content=result, tool_call_id=tool_id)
        except Exception as e:
            return ToolMessage(
                content=json.dumps({"error": str(e)}), tool_call_id=tool_id
            )
    else:
        return ToolMessage(
            content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
            tool_call_id=tool_id,
        )

def rag_node(state: PipelineState) -> Dict[str, Any]:
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

    user_memories = store.search(
        ("rag_memory", state.user_id),
        query=None,   # no semantic filtering, just fetch recent
        limit=5
    )

# 🔹 Step 2: Format them for prompt
    if user_memories:
        docs_text = "Previously retrieved documents for reference:\n"
        for i, doc in enumerate(user_memories, start=1):
            filename = doc.value.get("filename", "")
            content_path = doc.value.get("content_path", "")
            description_preview = doc.value.get("description", "")[:300]
            pages = doc.value.get("pages", "")
            docs_text += f"- Doc {i}: {filename} ({content_path}) [Pages: {pages}] {description_preview}\n"
    else:
        docs_text = "No prior retrieved documents."

    llm = create_llm()
    tools = [azure_ai_search]
    tools_map = {"azure_ai_search": azure_ai_search}
    llm_with_tools = llm.bind_tools(tools)

    # Focused RAG prompt - tool description handles "how to use the tool"
    prompt_text = """You are a RAG retrieval and analysis agent. Use the azure_ai_search tool to find documents, then synthesize professional answers with citations.

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

1. LISTING/COUNTING ("List all X", "How many X"):
   → Use facets in ONE call. Never loop per document.
   → Include ALL documents from facets in your response — do NOT filter or subset them.
   → When listing documents with content_path, ALWAYS add "and text_document_id ne ''" to filter. Without this filter, you may get image paths instead of PDF paths. DO NOT return image paths when user is asking for documents.
    Examples:
    - Wrong: "file_category_ai eq 'Brand equity'" → Returns image paths ❌
    - Right: "file_category_ai eq 'Brand equity' and text_document_id ne ''" → Returns PDF paths ✅
   → STOP IMMEDIATELY after this one call. Do NOT paginate (no skip calls). Do NOT search for individual documents.
   → Use the "document_list" array directly to build your answer — it already has all unique document names and their URLs.
   → Include ALL documents from the document_list — do NOT filter or subset them.

2. CONTENT SEARCH ("What does X say about Y", "Find insights on Z"):
   → Use keyword search with top_k=10-50. No facets needed.
   → Include content_text in select_fields if you need to read the text.

3. PAGE-SPECIFIC ("What's on page 6 of Report.pdf"):
   → Use filter with locationMetadata/pageNumber.

4. SUMMARIZATION ("Summarize document X", "Give me a summary of X"):
   → First call: top_k=100, select_fields="content_text,document_title,content_path,locationMetadata". Check totalCount.
   → If totalCount <= 300: paginate to read all chunks (top_k=100, skip=100, skip=200).
   → If totalCount > 300: sample beginning (already have first 100), middle (skip=totalCount/2, top_k=100), end (skip=totalCount-100, top_k=100). Max 4 calls total.

SYNTHESIS RULES:
- Executive Summary (2-3 sentences), then Detailed Analysis with inline citations, then Key Takeaways (3-5 bullets).
- ALWAYS cite with page numbers: 📄 [filename](content_path) (Page N)
- For listing queries: return ALL documents from search, not a filtered subset.
- Evidence-based claims only — do not fabricate information.

OUTPUT — Return valid JSON:
{{
  "retrieved_docs": [
    {{"filename": "string", "content_path": "string", "score": 0.0, "pages": "string", "description": "string"}}
  ],
  "final_answer": "string (markdown with citations)",
  "search_strategy": "string (brief description of approach taken)",
  "reasoning": "string (why this strategy was chosen)",
  "total_searches": 0
}}

CRITICAL:
- For listing queries, retrieved_docs MUST contain ALL documents found (e.g., if facets return 36 documents, include all 36).
- Include page numbers in every citation from locationMetadata/pageNumber.
- Use content_path from search results for links — never reconstruct URLs.
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            MessagesPlaceholder("messages"),
            ("human", "Original query: {user_query}\nEnriched query: {enriched_query}"),
        ]
    )

    initial_messages = prompt.format_messages(
        user_query=user_query, enriched_query=enriched_query, messages=messages,memories_text=docs_text
    )

    print("\n" + "🔍"*35)
    print("DEBUG: RAG PHASE 0 - Checking if history/memories are present")
    print("🔍"*35)

    # Check system message for memories
    if initial_messages and len(initial_messages) > 0:
        first_msg = initial_messages[0]
        if hasattr(first_msg, 'content'):
            content = str(first_msg.content)
            if "PREVIOUSLY RETRIEVED DOCUMENTS" in content:
                print("✅ Memories section found in system prompt")
                # Extract memories section
                start = content.find("PREVIOUSLY RETRIEVED DOCUMENTS")
                end = content.find("PHASE 0", start)
                if start != -1 and end != -1:
                    memories_section = content[start:end]
                    print(f"📚 Memories preview:\n{memories_section[:500]}")
            else:
                print("❌ Memories section NOT found in system prompt")

    # Check chat history messages
    history_count = sum(1 for msg in initial_messages 
                    if msg.__class__.__name__ in ['HumanMessage', 'AIMessage'])
    print(f"💬 Chat history messages: {history_count}")

    print("="*70 + "\n")

    agent_messages = list(initial_messages)
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

    llm_structured = create_llm().with_structured_output(
        RAGOutput, method="function_calling"
    )
    output: RAGOutput = llm_structured.invoke(raw_output)
    
    # DEBUG: Print structured output before returning
    print("\n" + "-"*70)
    print("🐛 DEBUG: RAG NODE - Structured Output")
    print("-"*70)
    print(f"Output Type: {type(output)}")
    print(f"Output Dict:\n{json.dumps(output.dict(), indent=2, default=str)}")
    print("-"*70)
    if output.retrieved_docs:
        for i, d in enumerate(output.retrieved_docs):
            store.put(
                namespace=("rag_memory", state.user_id),  # tuple namespace
                key=f"doc_{i}_{hash(d.description) % 100000}",  # unique key per doc
                value={
                    "type": "retrieved_doc",
                    "filename": safe_utf8(d.filename),
                    "content_path": safe_utf8(d.content_path),
                    "description": safe_utf8(d.description),
                    "pages": d.pages,
                    "score": d.score,
                }
            )
        print(f"📚 Stored {len(output.retrieved_docs)} RAG docs to PostgresStore")


    return {
        "messages": sanitize_any(all_new_messages),
        "rag_output": sanitize_any(output.dict()),
    }
