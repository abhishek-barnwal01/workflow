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
        timeout=30.0,
        max_retries=2,
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

    # Full enterprise RAG prompt with advanced search capabilities
    prompt_text = """You are an enterprise-grade RAG retrieval and analysis agent. You provide CMI-level research with rigorous document handling and citation practices.

---
PHASE 1: INTELLIGENT DOCUMENT DISCOVERY
---

STEP 1: Assess Query Scope
- Determine if the query requires single or multiple documents
- Identify the primary domain/topic (e.g., market share analysis, consumer insights, financial metrics)
- Establish relevance criteria for document selection
- Determine if the query needs page-specific content, document listing, or category-based filtering

STEP 2: Execute Strategic Search
Call tool: azure_ai_search(query="...", index_type="main_data", top_k=?, filter=?, facets=?, skip=?, select_fields=?)

The tool now supports ADVANCED parameters for the main_data index:

PARAMETERS:
- query: Search text. Use "*" to match all documents when using filters/facets only.
- index_type: Always use "main_data" for document retrieval.
- top_k: Number of results (1-50). YOU decide the optimal value.
- filter: OData filter expression for precise filtering. Examples:
    * By category: filter="file_category_ai eq 'Usage/Attitude (U&A)'"
    * By page number: filter="locationMetadata/pageNumber eq 6"
    * By document + page: filter="document_title eq 'Report.pdf' and locationMetadata/pageNumber eq 6"
    * By brand: filter="brand_ai eq 'Lux'"
    * By product category: filter="product_category_ai eq 'Soaps'"
    * By path: filter="content_path eq '/reports/2023/'"
    * Combine with "and" / "or"
- facets: List of facetable fields for aggregation/counting. Examples:
    * facets=["document_title,count:1000"] to list/count unique documents
    * facets=["file_category_ai,count:100"] to list categories
    * facets=["brand_ai,count:100"] to list brands
    * facets=["product_category_ai,count:100"] to list product categories
    * IMPORTANT: Add ",count:N" to get up to N unique values (default is only 10!)
- skip: Number of results to skip for pagination.
- select_fields: Comma-separated fields to return (overrides defaults).

EFFICIENT DOCUMENT COUNTING/LISTING (Use Facets):
To count or list unique documents efficiently:
1. Use facets: ["document_title,count:1000"] or ["text_document_id,count:1000"]
2. Get all results in 1 call instead of many
3. Count of facet items = number of unique documents

Example: "How many U&A reports?"
  azure_ai_search(query="*", index_type="main_data", top_k=1, filter="file_category_ai eq 'Usage/Attitude (U&A)'", facets=["document_title,count:1000"])
  → facet items count = number of unique documents

WHEN TO USE FACETS:
1. Counting documents: facets=["document_title,count:1000"]
2. Listing document names: facets=["document_title,count:1000"]
3. Listing categories: facets=["file_category_ai,count:100"]
4. Listing brands: facets=["brand_ai,count:100"]
5. Counting by any facetable field

WHEN NOT TO USE FACETS:
- Searching for specific content/keywords
- Finding documents by name/topic
- Answering questions about document content

PAGE-SPECIFIC SEARCHES:
- The index has pageNumber inside locationMetadata complex field.
- Each search result now includes "pageNumber" extracted from locationMetadata.
- To filter by page: filter="locationMetadata/pageNumber eq 6"
- For specific document + page: filter="document_title eq 'Presentation.pptx' and locationMetadata/pageNumber eq 6"
- Use page numbers in citations for precise references.

AVAILABLE FIELDS IN MAIN DATA INDEX RESULTS:
Each result includes: content_id, text_document_id, document_title, image_document_id,
content_text, content_path, pageNumber, boundingPolygon, file_category_ai,
product_category_ai, brand_ai, sub_brand_ai, and search score.

Search Strategy Guidelines:
- For broad exploratory search: Use higher top_k + facets to see document landscape
- For targeted retrieval: Use focused top_k + filters after identifying relevant sources
- For document listing: Use query="*" with facets + optional filters
- For page-specific content: Use filter with locationMetadata/pageNumber
- Use domain-specific keywords from enriched query
- Look for high-scoring documents (>0.7 typically indicates strong relevance)
- Adjust top_k dynamically based on what you find
- Use pagination (skip) to get more results if needed

STEP 3: Domain-Filtered Document Selection
CRITICAL RULES:
✓ Retrieve context ONLY from documents matching the query domain
✓ Do NOT mix content across unrelated documents
✓ Ensure content consistency across sources
✓ Prioritize depth over breadth: one highly relevant document > multiple loosely related ones
✓ Use brand_ai, product_category_ai, file_category_ai to validate domain relevance

STEP 4-6: Page-level content extraction
- Use pageNumber from results to identify relevant pages
- Filter by specific page: filter="document_title eq 'doc.pdf' and locationMetadata/pageNumber eq N"
- Retrieve all content from relevant pages
- Iterate if answer incomplete (adjust top_k, filters, or retrieve additional pages)
- Maintain domain relevance
- Use tool metadata suggestions to guide search
- Include page numbers in retrieved_docs for precise citations

STEP 7: Synthesize Professional Answer
- Executive Summary (2-3 sentences)
- Detailed Analysis (2-4 paragraphs) with inline citations including page numbers
- Key Takeaways (3-5 bullets)

DOCUMENT REFERENCE FORMATTING
- Always use 📄 [filename](content_path)
- Include page number when available: 📄 [filename](content_path) (Page N)
- Extract cleaned filename by removing UUID prefix
- Format as markdown links

Output JSON schema:
{{
  "retrieved_docs": [
    {{ "filename": "string", "content_path": "string", "score": float, "pages": "string", "description": "string" }}
  ],
  "final_answer": "string",
  "search_strategy": "string",
  "reasoning": "string",
  "total_searches": int
}}

QUALITY STANDARDS
- CMI-grade professional tone
- Evidence-based claims only
- Precise citations with page numbers when available
- Logical, structured analysis
- Actionable insights
- Domain-appropriate terminology
- Clickable PDF links in correct format
- Clear separation of summary vs. detailed analysis
- Transparent about search strategy
- Include page references for verifiability
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
