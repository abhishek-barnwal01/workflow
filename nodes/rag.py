"""RAG Node - Document retrieval and answer generation"""
from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import azure_ai_search
from models import RAGOutput, RetrievedDoc, SearchRecord, PipelineState
import config
import json


def create_llm():
    """Create LLM instance"""
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,  # Updated param name
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0.7
    )


def execute_tool_calls(tool_calls: list, tools_map: dict) -> list:
    """Execute tool calls and return ToolMessages"""
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        if tool_name in tools_map:
            try:
                result = tools_map[tool_name].invoke(tool_args)
                tool_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_id
                ))
            except Exception as e:
                tool_messages.append(ToolMessage(
                    content=json.dumps({"error": str(e)}),
                    tool_call_id=tool_id
                ))
        else:
            tool_messages.append(ToolMessage(
                content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                tool_call_id=tool_id
            ))
    
    return tool_messages


def format_search_history(search_history: List[Dict]) -> str:
    """Format previous searches for the prompt"""
    if not search_history:
        return "No previous searches in this request."
    
    text = "Previous searches in this request:\n"
    for i, search in enumerate(search_history, 1):
        text += f"{i}. Query: '{search.get('query')}' | Index: {search.get('index_type')} | "
        text += f"Results: {search.get('result_count')} | Avg Score: {search.get('avg_score', 0):.2f}\n"
    return text


def format_iteration_feedback(feedback: List[str]) -> str:
    """Format evaluator feedback for the prompt"""
    if not feedback:
        return "No feedback from previous iterations."
    
    text = "Feedback from evaluator (address these gaps):\n"
    for i, fb in enumerate(feedback, 1):
        text += f"{i}. {fb}\n"
    return text


def rag_node(state: PipelineState) -> Dict[str, Any]:
    """
    RAG retrieval and generation node.
    
    LangChain 1.x pattern: Uses llm.bind_tools() + manual tool execution loop
    
    Responsibilities:
    1. Read scratchpad to avoid duplicate searches
    2. Use evaluator feedback to guide search strategy
    3. Search main_data index for documents
    4. Generate answer from retrieved documents
    5. Update scratchpad with new searches
    
    Inputs from state:
    - enriched_query: Semantically enriched query
    - user_query: Original query
    - domain_context: Context from semantic node
    - search_history: Previous searches (avoid duplicates)
    - retrieved_doc_ids: Already retrieved docs
    - iteration_feedback: Evaluator suggestions
    - iteration_count: Current iteration number
    
    Outputs to state:
    - retrieved_docs: List of retrieved documents
    - rag_answer: Generated answer
    - search_strategy: Explanation of search approach
    - search_history: Updated with new searches
    - retrieved_doc_ids: Updated with new doc IDs
    - iteration_count: Incremented
    """
    
    print("\n" + "="*70)
    print("📚 RAG NODE")
    print("="*70)
    
    enriched_query = state.get("enriched_query", state["user_query"])
    original_query = state["user_query"]
    domain_context = state.get("domain_context", {})
    search_history = state.get("search_history", [])
    retrieved_doc_ids = state.get("retrieved_doc_ids", [])
    iteration_feedback = state.get("iteration_feedback", [])
    iteration_count = state.get("iteration_count", 0)
    
    print(f"Iteration: {iteration_count + 1}")
    print(f"Original Query: {original_query}")
    print(f"Enriched Query: {enriched_query}")
    print(f"Previous Searches: {len(search_history)}")
    print(f"Already Retrieved Docs: {len(retrieved_doc_ids)}")
    
    if iteration_feedback:
        print(f"📋 Evaluator Feedback:")
        for fb in iteration_feedback:
            print(f"   - {fb}")
    
    # Format context for prompt
    search_history_text = format_search_history(search_history)
    feedback_text = format_iteration_feedback(iteration_feedback)
    already_retrieved_text = f"Already retrieved doc IDs (skip these): {retrieved_doc_ids}" if retrieved_doc_ids else ""
    
    # Create LLM with tools bound
    llm = create_llm()
    tools = [azure_ai_search]
    tools_map = {"azure_ai_search": azure_ai_search}
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = f"""You are an enterprise-grade RAG retrieval and analysis agent.

═══════════════════════════════════════════════════════════════════════════
SCRATCHPAD - AVOID DUPLICATE WORK
═══════════════════════════════════════════════════════════════════════════

{search_history_text}

{already_retrieved_text}

{feedback_text}

IMPORTANT: 
- Do NOT repeat the exact same searches
- If feedback says "missing X", search specifically for X
- Skip documents already in retrieved_doc_ids

═══════════════════════════════════════════════════════════════════════════
PHASE 1: INTELLIGENT DOCUMENT DISCOVERY
═══════════════════════════════════════════════════════════════════════════

Use azure_ai_search tool with index_type="main_data"

YOU decide the optimal top_k (10-100) based on:
- Query complexity
- Previous search results
- Evaluator feedback

═══════════════════════════════════════════════════════════════════════════
PHASE 2: SYNTHESIS & CITATION
═══════════════════════════════════════════════════════════════════════════

Structure your answer as:

**Executive Summary** (2-3 sentences)

**Detailed Analysis** (2-4 paragraphs with citations)

**Key Takeaways** (3-5 bullet points)

CITATION FORMAT: 📄 [filename](content_path)

═══════════════════════════════════════════════════════════════════════════
YOUR RESPONSE MUST INCLUDE
═══════════════════════════════════════════════════════════════════════════

1. RETRIEVED DOCUMENTS LIST with scores
2. FINAL ANSWER with citations
3. SEARCH STRATEGY explaining your approach
4. REASONING for document selection
5. TOTAL SEARCHES count

═══════════════════════════════════════════════════════════════════════════
QUALITY STANDARDS
═══════════════════════════════════════════════════════════════════════════

✓ Evidence-based claims only
✓ Precise citations with page numbers
✓ Professional tone
✓ Actionable insights

✗ Do NOT repeat previous searches exactly
✗ Do NOT retrieve already-seen documents
✗ Do NOT ignore evaluator feedback
"""

    user_prompt = f"""Original Query: {original_query}

Enriched Query: {enriched_query}

Domain Context: {json.dumps(domain_context)}

Conduct comprehensive research. Address any evaluator feedback.
YOU decide all search parameters autonomously."""

    # Build messages for tool-calling loop
    agent_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Tool-calling loop (max 5 iterations)
    max_iterations = 5
    total_searches = 0
    response = None
    
    for iteration in range(max_iterations):
        print(f"\n--- Tool Iteration {iteration + 1} ---")
        
        response = llm_with_tools.invoke(agent_messages)
        
        # Check if there are tool calls
        if response.tool_calls:
            total_searches += len(response.tool_calls)
            print(f"🔧 Tool calls: {len(response.tool_calls)} (total: {total_searches})")
            
            # Add AI message with tool calls
            agent_messages.append(response)
            
            # Execute tools
            tool_messages = execute_tool_calls(response.tool_calls, tools_map)
            agent_messages.extend(tool_messages)
        else:
            # No more tool calls, we have the final response
            print("✅ Final response received")
            break
    
    raw_output = response.content if response else ""
    
    # Parse with structured output
    llm_structured = create_llm().with_structured_output(RAGOutput)
    output: RAGOutput = llm_structured.invoke(raw_output)
    
    # Override total_searches with actual count
    output.total_searches = total_searches
    
    print(f"\n✅ Retrieved {len(output.retrieved_docs)} documents")
    print(f"Total searches: {output.total_searches}")
    
    # Update scratchpad
    new_doc_ids = [doc.content_id for doc in output.retrieved_docs if doc.content_id]
    updated_doc_ids = list(set(retrieved_doc_ids + new_doc_ids))
    
    # Create search record for this iteration
    new_search = SearchRecord(
        query=enriched_query,
        index_type="main_data",
        top_k=50,  # Default, actual value determined by agent
        result_count=len(output.retrieved_docs),
        doc_ids=new_doc_ids,
        avg_score=sum(d.score for d in output.retrieved_docs) / len(output.retrieved_docs) if output.retrieved_docs else 0
    )
    updated_search_history = search_history + [new_search.model_dump()]
    
    print(f"\n📊 RAG Answer Preview:")
    print("="*70)
    print(output.final_answer[:300] + "..." if len(output.final_answer) > 300 else output.final_answer)
    print("="*70)
    
    return {
        "retrieved_docs": [doc.model_dump() for doc in output.retrieved_docs],
        "rag_answer": output.final_answer,
        "search_strategy": output.search_strategy,
        "search_history": updated_search_history,
        "retrieved_doc_ids": updated_doc_ids,
        "iteration_count": iteration_count + 1,
    }