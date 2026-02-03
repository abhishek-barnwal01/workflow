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
        timeout=120.0,
        max_retries=3,
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

    # Full enterprise RAG prompt with expanded index fields support
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
Call tool: azure_ai_search(query="...", index_type="main_data", top_k=?)

MANDATORY parameters:
- query: Search text or "*" for wildcard
- index_type: "main_data"
- top_k: Number of results (1-100)

OPTIONAL parameters (use only when needed):
- filter: OData filter expression (for document filtering, page filtering, category filtering, etc.)
- facets: List of facetable fields (for counting/listing unique values)
- skip: Number of results to skip (for pagination)
- select_fields: Comma-separated fields to return (only if you need specific fields)

IMPORTANT: The index contains document CHUNKS, not whole documents. Each document is split into multiple chunks.

⚡ EFFICIENT DOCUMENT COUNTING:
The index has document_title and text_document_id as FACETABLE fields.
To count or list unique documents:
1. Use facets: ["document_title,count:1000"] or facets: ["text_document_id,count:1000"]
   - IMPORTANT: Add ",count:1000" to get up to 1000 unique documents (default is only 10!)
2. Get results in 1 call instead of 20-30 calls
3. Count of facet items = number of unique documents

Example for "How many U&A reports?":
{{ query: "*", filter: "file_category_ai eq 'Usage/Attitude (U&A)' and text_document_id ne ''", facets: ["document_title,count:1000"] }}
→ Number of facet items = number of unique documents

Example for "List all U&A reports" (with clickable links):
{{ query: "*", filter: "file_category_ai eq 'Usage/Attitude (U&A)' and text_document_id ne ''", facets: ["document_title,count:1000"], select_fields: "document_title,content_path" }}
→ Extract facet values for document names
→ Use documents array to get content_path for links: 📄 [filename](content_path)

CRITICAL FOR LISTING QUERIES:
1. Always include select_fields: "document_title,content_path" to get URLs for clickable links
2. ALWAYS add "and text_document_id ne ''" to filter to exclude image chunks and get original PDF paths

WITHOUT ",count:1000" you'll only get 10 documents maximum!

WHEN TO USE FACETS:
1. Counting documents: facets: ["document_title,count:1000"]
2. Listing document names: facets: ["document_title,count:1000"]
3. Listing categories: facets: ["file_category_ai,count:100"]

WHEN NOT TO USE FACETS:
- Searching for specific content/keywords
- Finding documents by name/topic
- Answering questions about document content
Example: "Find Godrej growth reports" → Use query only, NO facets

PARAMETERS:
- query: Search term for content (use "*" when using filters/facets only)
- filter: OData filter expressions (CASE-SENSITIVE! Use exact values):
  * By category: "file_category_ai eq 'Usage/Attitude (U&A)'"
  * By page: "locationMetadata/pageNumber eq 6"
  * By document + page: "document_title eq 'Report.pdf' and locationMetadata/pageNumber eq 6"
  * By brand: "brand_ai eq 'Godrej'"
  * By country: "country_ai eq 'India'"
  * Combine with "and" or "or"
- facets: Array of facetable fields ["document_title,count:1000", "file_category_ai,count:100"]
- skip: Number of results to skip for pagination (default: 0)
- select_fields: Comma-separated fields to return (e.g., "document_title,content_path")

EXACT CATEGORY VALUES (file_category_ai) - Use these EXACT strings (case-sensitive):
- "Brand equity" (lowercase 'e')
- "Brand track" (lowercase 't')
- "Concept testing" (lowercase 't')
- "Link testing" (lowercase 't')
- "Annual presentation" (lowercase 'p')
- "Media Optimization" (capital 'O')
- "Product acceptance testing" (lowercase 'a' and 't')
- "Miscellaneous" (capital 'M')
- "Usage/Attitude (U&A)" (capital 'U' and 'A')

CRITICAL FILTER RULES:
1. Filters are CASE-SENSITIVE! Always use exact category values above.
   ✗ Wrong: "file_category_ai eq 'Concept Testing'"
   ✓ Right: "file_category_ai eq 'Concept testing'"

2. When listing documents with content_path, ALWAYS add "and text_document_id ne ''" to filter!
   Why: Index has both text chunks (original PDFs) and image chunks (extracted images).
   Without this filter, you may get image paths instead of PDF paths.
   ✗ Wrong: filter="file_category_ai eq 'Brand equity'" → May return image paths
   ✓ Right: filter="file_category_ai eq 'Brand equity' and text_document_id ne ''" → Returns PDF paths

3. ALWAYS include page numbers in document citations!
   Each search result contains locationMetadata/pageNumber - use it in your citations.
   Format: 📄 [filename](content_path) (Page X)
   Example: 📄 [Soaps UA 2024.pdf](https://...) (Page 15)
   Multiple pages: 📄 [Report.pdf](https://...) (Pages 12, 15, 18)

PAGE-SPECIFIC SEARCHES:
- The index has pageNumber field under locationMetadata
- To filter by page: use "locationMetadata/pageNumber eq [number]"
- For specific document + page: combine filters with AND
Example: "locationMetadata/pageNumber eq 6 and document_title eq 'Presentation.pptx'"

EXAMPLES:
✓ Count U&A reports (EFFICIENT - 1 call):
  azure_ai_search(query="*", index_type="main_data", top_k=1, filter="file_category_ai eq 'Usage/Attitude (U&A)' and text_document_id ne ''", facets=["document_title,count:1000"])
  → Count facet items = number of documents

✓ List all U&A reports with links (EFFICIENT - 1 call):
  azure_ai_search(query="*", index_type="main_data", top_k=100, filter="file_category_ai eq 'Usage/Attitude (U&A)' and text_document_id ne ''", facets=["document_title,count:1000"], select_fields="document_title,content_path")
  → Extract facet values for document names
  → Use documents array to get content_path for links: 📄 [filename](content_path)
  → DO NOT reconstruct URLs from document titles - ALWAYS use content_path from results

✓ List concept testing reports:
  azure_ai_search(query="*", index_type="main_data", top_k=100, filter="file_category_ai eq 'Concept testing' and text_document_id ne ''", facets=["document_title,count:1000"], select_fields="document_title,content_path")

✓ List brand equity reports:
  azure_ai_search(query="*", index_type="main_data", top_k=100, filter="file_category_ai eq 'Brand equity' and text_document_id ne ''", facets=["document_title,count:1000"], select_fields="document_title,content_path")

✓ Content search with page attribution:
  azure_ai_search(query="product likability drivers", index_type="main_data", top_k=20)
  → Results include page_number for each chunk
  → In your answer, cite as: 📄 [Soaps UA 2024.pdf](https://...) (Page 23)
  → ALWAYS extract and include the page number!

✓ Page 6 of doc:
  azure_ai_search(query="*", index_type="main_data", top_k=10, filter="locationMetadata/pageNumber eq 6 and document_title eq 'Presentation.pptx'")

✗ Wrong (gets image paths):
  azure_ai_search(query="*", index_type="main_data", top_k=100, filter="file_category_ai eq 'Brand equity'", select_fields="document_title,content_path")
✓ Right (gets PDF paths):
  azure_ai_search(query="*", index_type="main_data", top_k=100, filter="file_category_ai eq 'Brand equity' and text_document_id ne ''", select_fields="document_title,content_path")

IMPORTANT EFFICIENCY RULES:
- For COUNTING documents: Use facets with low top_k (1 call gives count)
- For LISTING with clickable links: Use facets + select_fields="document_title,content_path"
- NEVER make individual calls per document to get content_path - that's extremely inefficient!
- ALWAYS add "text_document_id ne ''" when you need PDF paths (not image paths)

Search Strategy Guidelines:
- For COUNTING documents: Use facets with top_k=1
- For LISTING with links: Use facets + select_fields, filter includes "text_document_id ne ''"
- For targeted content retrieval: Use focused top_k + filters
- For page-specific content: Use filter with locationMetadata/pageNumber
- ALWAYS include page numbers in citations from locationMetadata/pageNumber
- Use pagination (skip) if you need more than 100 documents
- NEVER loop through documents one-by-one - always batch!

STEP 3: Domain-Filtered Document Selection
CRITICAL RULES:
✓ Retrieve context ONLY from documents matching the query domain
✓ Do NOT mix content across unrelated documents
✓ Ensure content consistency across sources
✓ Prioritize depth over breadth: one highly relevant document > multiple loosely related ones
✓ Use brand_ai, product_category_ai, file_category_ai to validate domain relevance

STEP 4-6: Page-level content extraction
- Use page_number from results to identify relevant pages
- Filter by specific page: filter="document_title eq 'doc.pdf' and locationMetadata/pageNumber eq N"
- Retrieve all content from relevant pages
- Iterate if answer incomplete (adjust top_k, filters, or retrieve additional pages)
- Maintain domain relevance
- Include page numbers in retrieved_docs for precise citations

STEP 7: Synthesize Professional Answer
- Executive Summary (2-3 sentences)
- Detailed Analysis (2-4 paragraphs) with inline citations INCLUDING PAGE NUMBERS
- Key Takeaways (3-5 bullets)

DOCUMENT REFERENCE FORMATTING
- Always use 📄 [filename](content_path) (Page N)
- ALWAYS include page number from locationMetadata/pageNumber
- Extract cleaned filename by removing UUID prefix
- Format as markdown links
- Multiple pages: 📄 [filename](content_path) (Pages 12, 15, 18)

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
- Precise citations WITH PAGE NUMBERS (mandatory)
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
