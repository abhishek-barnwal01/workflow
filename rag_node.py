# rag_node.py
"""RAG Node - Full prompt with retrieval, synthesis, and citations"""

from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import ToolMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import azure_ai_search
from models import RAGOutput, PipelineState
import config
import json
from memory_store import store


# ----- Add this helper at the top of rag_node.py -----
def safe_utf8(text: str) -> str:
    if not text:
        return ""
    # Replace invalid UTF-8 characters with '?'
    return text.encode("utf-8", errors="replace").decode("utf-8")


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
    )


def execute_tool_calls(tool_calls: list, tools_map: dict) -> list:
    tool_messages = []
    for tool_call in tool_calls:
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
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {"error": f"Invalid JSON in tool args: {tool_args}"}
                        ),
                        tool_call_id=tool_id,
                    )
                )
                continue

        if tool_name in tools_map:
            try:
                result = tools_map[tool_name].invoke(tool_args)
                tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}), tool_call_id=tool_id
                    )
                )
        else:
            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    tool_call_id=tool_id,
                )
            )
    return tool_messages


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
    query=None,   # no semantic filtering, just fetch all
    limit=20      # adjust as needed
)

# 🔹 Step 2: Format them for prompt
    if user_memories:
        docs_text = "Previously retrieved documents for reference:\n"
        for i, doc in enumerate(user_memories, start=1):
            content_preview = doc.value.get("content", "")[:300]  # first 300 chars
            docs_text += f"- Doc {i}: {content_preview}\n"
    else:
        docs_text = "No prior retrieved documents."

    llm = create_llm()
    tools = [azure_ai_search]
    tools_map = {"azure_ai_search": azure_ai_search}
    llm_with_tools = llm.bind_tools(tools)

    # Full enterprise RAG prompt (exact copy from user)
    prompt_text = """You are an enterprise-grade RAG retrieval and analysis agent. You provide CMI-level research with rigorous document handling and citation practices.

---
PHASE 1: INTELLIGENT DOCUMENT DISCOVERY
---

STEP 1: Assess Query Scope
- Determine if the query requires single or multiple documents
- Identify the primary domain/topic (e.g., market share analysis, consumer insights, financial metrics)
- Establish relevance criteria for document selection

STEP 2: Execute Strategic Search
Call tool: azure_ai_search(query="...", index_type="main_data", top_k=?)

YOU decide the optimal top_k based on:
- Query complexity (simple question = fewer results needed)
- Expected document count (niche topic = broader search)
- Initial exploration vs. targeted retrieval
- Tool metadata feedback from previous searches

Search Strategy Guidelines:
- For broad exploratory search: Use higher top_k to see document landscape
- For targeted retrieval: Use focused top_k after identifying relevant sources
- Use domain-specific keywords from enriched query
- Look for high-scoring documents (>0.7 typically indicates strong relevance)
- Adjust top_k dynamically based on what you find

STEP 3: Domain-Filtered Document Selection
CRITICAL RULES:
✓ Retrieve context ONLY from documents matching the query domain
✓ Do NOT mix content across unrelated documents
✓ Do NOT answer from wrong documents just because wording appears similar
✓ Prioritize depth over breadth: one highly relevant document > multiple loosely related ones

STEP 4-6: Page-level content extraction
- Identify relevant pages in each document
- Retrieve all content from relevant pages
- Iterate if answer incomplete (adjust top_k or retrieve additional pages)
- Maintain domain relevance
- Use tool metadata suggestions to guide search

STEP 7: Synthesize Professional Answer
- Executive Summary (2-3 sentences)
- Detailed Analysis (2-4 paragraphs) with inline citations
- Key Takeaways (3-5 bullets)

DOCUMENT REFERENCE FORMATTING & PAGE NUMBERS
CRITICAL: Extract and include page numbers for EVERY source document

From tool results, you'll see:
- "document_title": the filename
- "content_path": the full path/URL
- "page_number": the page number (CRITICAL - include this!)

YOU MUST:
✓ Extract the page_number from EVERY tool result
✓ Format pages as "p.12" (single page) or "pp.5-7" (range)
✓ If multiple pages from same doc, combine as "pp.5,8,12"
✓ Clean filename by removing UUID prefixes (e.g., "abc123_report.pdf" → "report.pdf")
✓ Store in retrieved_docs with ALL fields populated

Output JSON schema:
{{
  "retrieved_docs": [
    {{
      "filename": "clean filename without UUID",
      "content_path": "full path from tool",
      "score": float,
      "pages": "p.12 or pp.5-7 or pp.5,8,12",
      "description": "what info from this doc was used"
    }}
  ],
  "final_answer": "string",
  "search_strategy": "string",
  "reasoning": "string",
  "total_searches": int
}}

QUALITY STANDARDS
- CMI-grade professional tone
- Evidence-based claims only
- Precise citations
- Logical, structured analysis
- Actionable insights
- Domain-appropriate terminology
- Clickable PDF links in correct format
- Clear separation of summary vs. detailed analysis
- Transparent about search strategy
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

    agent_messages = list(initial_messages)
    all_new_messages = []
    max_iterations = 5
    response = None

    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(agent_messages)

        # 🔹 Sanitize AI message content before saving
        if response.content:
            response.content = safe_utf8(response.content)

        # Append AI message
        agent_messages.append(response)
        all_new_messages.append(response)

        if response.tool_calls:
            tool_messages = execute_tool_calls(response.tool_calls, tools_map)

            # 🔹 Sanitize all tool messages
            for tm in tool_messages:
                if tm.content:
                    tm.content = safe_utf8(tm.content)

            agent_messages.extend(tool_messages)
            all_new_messages.extend(tool_messages)
        else:
            break

    raw_output = response.content if response else ""

    llm_structured = create_llm().with_structured_output(
        RAGOutput, method="function_calling"
    )
    output: RAGOutput = llm_structured.invoke(raw_output)
    if output.retrieved_docs:
        for i, d in enumerate(output.retrieved_docs):
            store.put(
                namespace=("rag_memory", state.user_id),  # tuple namespace
                key=f"doc_{i}_{hash(d.content) % 100000}",  # unique key per doc
                value={
                    "type": "retrieved_doc",
                    "content": safe_utf8(d.content),
                    "source": getattr(d, "source", ""),
                    "score": getattr(d, "score", 0.0),
                }
            )
        print(f"📚 Stored {len(output.retrieved_docs)} RAG docs to PostgresStore")


    return {
        "messages": sanitize_any(all_new_messages),
        "rag_output": sanitize_any(output.dict()),
    }
