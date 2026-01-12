"""Semantic Node - Query enrichment and ambiguity detection"""
from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import azure_ai_search
from models import SemanticOutput, PipelineState
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


def semantic_node(state: PipelineState) -> Dict[str, Any]:
    """
    Semantic enrichment node.
    
    LangChain 1.x pattern: Uses llm.bind_tools() + manual tool execution loop
    
    Responsibilities:
    1. Load user memories from store
    2. Check for explicit memory commands
    3. Search semantic index for entity enrichment
    4. Detect ambiguity requiring user clarification
    5. Enrich query with domain context
    
    Inputs from state:
    - user_query: Original user question
    - user_id: User identifier
    - messages: Chat history
    
    Outputs to state:
    - user_memories: Loaded memories
    - enriched_query: Enhanced query
    - domain_context: Extracted context
    - ambiguity_detected: Ambiguity info if any
    """
    
    print("\n" + "="*70)
    print("🧠 SEMANTIC NODE")
    print("="*70)
    
    user_query = state["user_query"]
    user_id = state["user_id"]
    messages = state.get("messages", [])
    
    print(f"Query: {user_query}")
    print(f"User ID: {user_id}")
    
    # Step 1: Load user memories (if store module exists)
    user_memories = []
    try:
        from memory.store import get_user_memories, handle_explicit_memory_command
        handle_explicit_memory_command(user_id, user_query)
        user_memories = get_user_memories(user_id, limit=10)
        if user_memories:
            print(f"📚 Loaded {len(user_memories)} user memories")
            for mem in user_memories[:3]:
                print(f"   - [{mem.get('type')}] {mem.get('content', '')[:50]}...")
    except ImportError:
        print("   ⚠️ Memory store not available")
    
    # Step 2: Format chat history
    history_text = ""
    for msg in messages[-10:]:  # Last 10 messages
        role = getattr(msg, 'type', 'unknown')
        content = getattr(msg, 'content', str(msg))
        history_text += f"{role}: {content}\n"
    
    if history_text:
        print(f"Chat History: {len(messages)} messages")
    
    # Step 3: Format user memories for prompt
    memories_text = ""
    if user_memories:
        memories_text = "Known facts about this user:\n"
        for mem in user_memories:
            memories_text += f"- [{mem.get('type')}] {mem.get('content')}\n"
    
    # Step 4: Create LLM with tools bound
    llm = create_llm()
    tools = [azure_ai_search]
    tools_map = {"azure_ai_search": azure_ai_search}
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = f"""You are a semantic enrichment agent.

Use the azure_ai_search tool to search the SEMANTIC index for entity values and business context.
You autonomously decide:
- What to search for
- How many results (top_k: 10-100)
- If you need multiple searches

{memories_text}

---
STEP 1: Search for relevant entities
---

Call: azure_ai_search(query="relevant search terms", index_type="semantic", top_k=?)

---
STEP 2: CHECK CHAT HISTORY FIRST (CRITICAL)
---
BEFORE marking anything as ambiguous:
1. READ the chat history carefully
2. If previous conversation provides context → USE IT, NOT ambiguous
3. If user says "ALL" or "all of them" → Include all options, NOT ambiguous
4. Only mark ambiguous if: multiple values exist AND no context in history

---
STEP 3: Extract options from search results
---
READ the content returned by the tool and extract specific values.

If multiple values exist AND no context resolves them:
- ambiguity_detected.ambiguous = true
- ambiguity_detected.options MUST be populated

Each option MUST be structured as:
{{ 
  "label": "Display Name",
  "value": "lowercase_underscore_value"
}}

---
OUTPUT REQUIREMENTS
---

You MUST return ONLY valid JSON in this exact structure:

{{
  "enriched_query": "string",
  "domain_context": {{}},
  "ambiguity_detected": {{
    "ambiguous": false,
    "entity": "string or null",
    "options": [],
    "reason": "string or null"
  }},
  "reasoning": "string"
}}

---
CRITICAL RULES
---
- If ambiguous = true → options MUST NOT be empty
- If options is empty → ambiguous MUST be false
- If ambiguous = false → options MUST be an empty array []
- If ambiguity is resolved by history → ambiguous = false
- If search fails → ambiguous = false and explain in reasoning
"""

    user_prompt = f"""Query: {user_query}

Chat History:
{history_text or "No history"}

Task: Enrich this query and determine if clarification is needed.

IMPORTANT: 
1. Check chat history FIRST before marking ambiguous
2. Use user memories to provide context
3. Populate ambiguity_detected.options ONLY inside JSON if required"""

    # Build messages for tool-calling loop
    agent_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Tool-calling loop (max 5 iterations)
    max_iterations = 5
    response = None
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        response = llm_with_tools.invoke(agent_messages)
        
        # Check if there are tool calls
        if response.tool_calls:
            print(f"🔧 Tool calls: {len(response.tool_calls)}")
            
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
    
    print("\n📄 SEMANTIC AGENT RAW OUTPUT:")
    print("="*70)
    print(raw_output[:500] + "..." if len(raw_output) > 500 else raw_output)
    print("="*70)
    
    # Step 5: Parse with structured output
    llm_structured = create_llm().with_structured_output(SemanticOutput)
    output: SemanticOutput = llm_structured.invoke(raw_output)
    
    print(f"\n✅ STRUCTURED OUTPUT:")
    print(f"   Enriched: {output.enriched_query}")
    print(f"   Ambiguous: {output.ambiguity_detected.ambiguous}")
    
    if output.ambiguity_detected.ambiguous:
        print(f"   Entity: {output.ambiguity_detected.entity}")
        print(f"   Options: {len(output.ambiguity_detected.options)}")
        for opt in output.ambiguity_detected.options[:5]:
            print(f"      - {opt.label}")
    
    # Step 6: Return state updates
    return {
        "user_memories": user_memories,
        "enriched_query": output.enriched_query,
        "domain_context": output.domain_context,
        "ambiguity_detected": output.ambiguity_detected.model_dump() if output.ambiguity_detected else None,
    }