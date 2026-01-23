"""Semantic Node - Query enrichment and ambiguity detection"""

from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import azure_ai_search
from models import AmbiguityInfo, IntentClassification, SemanticOutput, PipelineState
import config
import json
from memory_store import store  # <-- your PostgresStore




# ----- Add this helper -----
def safe_utf8(text: str) -> str:
    if not text:
        return ""
    # Replace invalid UTF-8 characters with '?'
    return text.encode("utf-8", errors="replace").decode("utf-8")


# ----------------------------


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
    """Create LLM instance"""
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,  # Updated param name
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=1,
    )


def execute_tool_calls(tool_calls: list, tools_map: dict) -> list:
    """
    Execute tool calls and return ToolMessages.

    Handles both dict-like and ToolCall object formats.
    Properly handles stringified tool arguments.
    """
    tool_messages = []
    for tool_call in tool_calls:
        # Handle both dict and ToolCall object formats
        if hasattr(tool_call, "name"):
            # ToolCall object (LangChain format)
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id
        else:
            # Dict format (fallback)
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

        # Handle stringified args (some providers return JSON strings)
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


def semantic_node(state: PipelineState) -> Dict[str, Any]:
    """
    Two-step semantic enrichment node.

    Step 1: Intent classification (no tools)
    Step 2: Conditional processing based on intent
        - chitchat: Generate friendly response, end flow
        - direct: Pass enriched query to RAG (no tool)
        - semantic: Full tool-calling loop with ambiguity detection

    CLARIFICATION MODE: If awaiting_clarification is True, skip Step 1
    and go directly to Step 2C (semantic enrichment) using clarification response.
    """

    print("\n" + "=" * 70)
    print("🧠 SEMANTIC NODE")
    print("=" * 70)

    user_query = state.user_query
    user_id = state.user_id
    messages = state.messages or []
    awaiting_clarification = state.awaiting_clarification
    previous_ambiguity = state.previous_ambiguity

    print(f"Query: {user_query}")
    print(f"User ID: {user_id}")
    print(f"Clarification Mode: {awaiting_clarification}")
    if awaiting_clarification and previous_ambiguity:
        print(f"Previous Entity: {previous_ambiguity.entity}")
        print(f"Previous Options: {[opt.label for opt in previous_ambiguity.options[:5]]}")
    
    # Step 1: Load user memories from PostgresStore
    try:
        user_memories = store.search(
            ("rag_memory", user_id),
            query=None,
            limit=10
        )
        print(f"📚 Loaded {len(user_memories)} user memories")
    except Exception as e:
        print(f"⚠️ Could not load memories: {e}")
        user_memories = []

    # Get chat history
    chat_history = messages
    if chat_history:
        print(f"📜 Chat History: {len(messages)} messages available")

    # Format user memories for prompt
    memories_text = ""
    if user_memories:
        memories_text = "Known facts about this user:\n"
        for mem in user_memories:
            mem_dict = mem.value
            mem_type = mem_dict.get("type", "unknown")
            mem_content = mem_dict.get("content", "")
            memories_text += f"- [{mem_type}] {mem_content}\n"
    else:
        memories_text = "No prior user memories stored."

    # ========================================================================
    # CLARIFICATION RESPONSE MODE: Skip intent classification
    # ========================================================================

    if awaiting_clarification and previous_ambiguity:
        print("\n" + "-" * 70)
        print("🔄 CLARIFICATION RESPONSE DETECTED")
        print("-" * 70)
        print(f"✅ User responded to clarification with: '{user_query}'")
        print(f"✅ Previous ambiguity: {previous_ambiguity.entity}")
        print(f"✅ Options were: {[opt.label for opt in previous_ambiguity.options]}")
        print("➡️  SKIPPING INTENT CLASSIFICATION")
        print("➡️  ROUTING DIRECTLY TO SEMANTIC ENRICHMENT (STEP 2C)")
        print("-" * 70)

        # Skip to Step 2C (semantic enrichment) with clarification context
        # Set a flag to indicate we're in clarification mode
        intent = IntentClassification(
            intent_type="semantic",
            reasoning="User responding to clarification question - routing directly to semantic enrichment",
            confidence=1.0
        )
    else:
        # ========================================================================
        # STEP 1: INTENT CLASSIFICATION (Agentic, with chat history access)
        # ========================================================================

        print("\n" + "-" * 70)
        print("STEP 1: Intent Classification")
        print("-" * 70)

        llm = create_llm()

        # Create prompt with MessagesPlaceholder for automatic chat history injection
        intent_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for an enterprise RAG system.

        Classify the user's query into ONE of these categories:

        1. **chitchat**: Greetings, thanks, farewells, casual conversation
        - Examples: "hi", "hello", "thanks", "thank you", "bye", "goodbye", "how are you"
        - Action: Respond with friendly message, don't search documents

        2. **direct**: Simple general knowledge questions that don't require company documents
        - Examples: "what is GDP", "define market share", "explain EBITDA", "what is ROI"
        - Characteristics: Definitional, general concepts, no possessive pronouns (our/my)
        - Action: Enrich query with context, pass to RAG without semantic tool

        3. **semantic**: Domain-specific questions requiring company document search
        - Examples: "what is OUR market share", "show Q3 sales", "compare regions", "all products"
        - Characteristics: References company data, uses possessive pronouns, mentions entities
        - Action: Use semantic search tool to find entities and detect ambiguity

        IMPORTANT:
        - Check chat history BELOW to understand context
        - Use conversation flow to inform classification
        - A follow-up question may reference previous context

        Analyze the query and return your classification with reasoning."""),
            MessagesPlaceholder("messages"),  # Chat history auto-injected here
            ("human", "Query: {user_query}\n\nClassify this query's intent.")
        ])

        # Format messages with chat history
        intent_messages = intent_prompt_template.format_messages(
            messages=chat_history,
            user_query=user_query
        )

        # Invoke with structured output
        llm_structured = llm.with_structured_output(IntentClassification, method="function_calling")

        try:
            intent: IntentClassification = llm_structured.invoke(intent_messages)
        except Exception as e:
            # Handle content filter or other API errors
            error_msg = str(e)
            print(f"⚠️ Intent Classification Error: {error_msg[:200]}")
            if "content_filter" in error_msg.lower() or "jailbreak" in error_msg.lower():
                print("   Content filter triggered - defaulting to 'direct' intent")
                # Fallback: treat as direct question
                intent = IntentClassification(
                    intent_type="direct",
                    reasoning="Content filter triggered during intent classification, defaulting to direct",
                    confidence=0.5
                )
            else:
                # Re-raise other errors
                raise

        print(f"✅ Intent: {intent.intent_type}")
        print(f"   Confidence: {intent.confidence:.2f}")
        print(f"   Reasoning: {intent.reasoning}")
    
    # Track new messages for state
    all_new_messages = []
    
    # ========================================================================
    # STEP 2A: CHITCHAT - Generate friendly response and END
    # ========================================================================

    if intent.intent_type == "chitchat":
        print("\n" + "-" * 70)
        print("STEP 2A: Chitchat Response Generation")
        print("-" * 70)
        
        # Agentic chitchat with chat history
        chitchat_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful enterprise RAG assistant.

    Generate a brief, friendly response to the user's greeting or casual message.
    - Be warm and professional
    - Keep response under 2 sentences
    - Offer to help with document-related questions
    - Use chat history to provide context if relevant

    {memories_text}"""),
            MessagesPlaceholder("messages"),  # Chat history auto-injected
            ("human", "{user_query}")
        ])
        
        chitchat_messages = chitchat_prompt_template.format_messages(
            memories_text=memories_text,
            messages=chat_history,
            user_query=user_query
        )

        try:
            chitchat_response = llm.invoke(chitchat_messages)
            friendly_message = safe_utf8(chitchat_response.content)
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Chitchat Error: {error_msg[:200]}")
            if "content_filter" in error_msg.lower() or "jailbreak" in error_msg.lower():
                print("   Content filter triggered - using fallback greeting")
                friendly_message = "Hello! I'm here to help you find information in your documents. What would you like to know?"
            else:
                raise

        print(f"💬 Chitchat Response: {friendly_message}")
        
        # Store the chitchat exchange
        chitchat_ai_message = AIMessage(
            content=friendly_message,
            metadata={"type": "chitchat_response", "node": "semantic"}
        )
        all_new_messages.append(chitchat_ai_message)
        
        # Return with semantic_chitchat flag set → triggers END via router
        return {
            "messages": sanitize_any(all_new_messages),
            "user_memories": sanitize_any(user_memories),
            "clarification_message": friendly_message,  # For backwards compat
            "semantic_chitchat": True,  # Flag to END immediately
            "awaiting_clarification": False,  # Clear clarification flag
            "previous_ambiguity": None,  # Clear previous ambiguity
            "enriched_query": safe_utf8(user_query),
            "domain_context": None,
            "ambiguity_detected": sanitize_any(
                AmbiguityInfo(ambiguous=False).model_dump()
            ),
        }
    
    # ========================================================================
    # STEP 2B: DIRECT - Simple enrichment without tool
    # ========================================================================

    elif intent.intent_type == "direct":
        print("\n" + "-" * 70)
        print("STEP 2B: Direct Question - Light Enrichment (No Tool)")
        print("-" * 70)
        
        # Agentic enrichment with chat history
        enrichment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query enrichment agent.

    The user asked a general knowledge question that doesn't require searching company documents.

    Your job:
    1. Rephrase the query to be clear and specific
    2. Add any helpful context from chat history if available
    3. Do NOT search for domain entities

    Return JSON with:
    {{
    "enriched_query": "clear, specific version of the query",
    "domain_context": null,
    "ambiguity_detected": {{
        "ambiguous": false,
        "entity": null,
        "options": [],
        "reason": null
    }},
    "reasoning": "brief explanation of enrichment"
    }}"""),
            MessagesPlaceholder("messages"),  # Chat history auto-injected
            ("human", "Query: {user_query}\n\nEnrich this query without searching documents.")
        ])
        
        enrichment_messages = enrichment_prompt.format_messages(
            messages=chat_history,
            user_query=user_query
        )

        llm_structured = llm.with_structured_output(SemanticOutput, method="function_calling")

        try:
            output: SemanticOutput = llm_structured.invoke(enrichment_messages)
        except Exception as e:
            # Handle content filter or other API errors
            error_msg = str(e)
            print(f"⚠️ LLM Error: {error_msg[:200]}")
            if "content_filter" in error_msg.lower() or "jailbreak" in error_msg.lower():
                print("   Content filter triggered - using fallback enrichment")
                # Fallback: just use the original query
                output = SemanticOutput(
                    enriched_query=user_query,
                    domain_context=None,
                    ambiguity_detected=AmbiguityInfo(ambiguous=False),
                    reasoning="Content filter triggered, using original query without enrichment"
                )
            else:
                # Re-raise other errors
                raise

        print(f"✅ Enriched Query: {output.enriched_query}")
        print(f"   Reasoning: {output.reasoning}")
        
        # Store reasoning
        if output.reasoning:
            reasoning_message = AIMessage(
                content=safe_utf8(output.reasoning),
                metadata={"type": "internal_reasoning", "node": "semantic"}
            )
            all_new_messages.append(reasoning_message)
        
        return {
            "messages": sanitize_any(all_new_messages),
            "user_memories": sanitize_any(user_memories),
            "clarification_message": None,  # No clarification for direct questions
            "semantic_chitchat": False,  # Clear the flag - this is not chitchat
            "awaiting_clarification": False,  # Clear clarification flag
            "previous_ambiguity": None,  # Clear previous ambiguity
            "enriched_query": safe_utf8(output.enriched_query),
            "domain_context": None,
            "ambiguity_detected": sanitize_any(
                output.ambiguity_detected.model_dump()
            ),
        }
    
    # ========================================================================
    # STEP 2C: SEMANTIC - Full tool-calling loop with ambiguity detection
    # ========================================================================

    else:  # intent.intent_type == "semantic"
        print("\n" + "-" * 70)
        print("STEP 2C: Semantic Enrichment - Full Tool Loop")
        print("-" * 70)

        # Bind tools for semantic search
        tools = [azure_ai_search]
        tools_map = {"azure_ai_search": azure_ai_search}
        llm = create_llm()
        llm_with_tools = llm.bind_tools(tools)

        # Build clarification context string
        clarification_context = ""
        if awaiting_clarification and previous_ambiguity:
            # Join options as comma-separated string (avoid {} in template)
            options_str = ", ".join([opt.label for opt in previous_ambiguity.options])
            clarification_context = f"""
---
CLARIFICATION CONTEXT (CRITICAL)
---
The user was previously asked to clarify an ambiguity:
- Entity: {previous_ambiguity.entity}
- Options provided: {options_str}

The user's current response is their clarification: "{user_query}"

YOUR TASK:
1. Parse the user's response (could be a number like "1", option name like "Lux", or "ALL")
2. Map it to the correct option from the previous clarification
3. Enrich the ORIGINAL query (from chat history) with the selected option
4. Set ambiguous = false (ambiguity is now resolved)
5. Do NOT ask for clarification again

Example:
- Original query: "what is market share of soap"
- User's clarification response: "Lux"
- Enriched query: "what is market share of Lux soap"
---
"""

        # Create prompt template with tool usage instructions
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a semantic enrichment agent.
{clarification_context}

Use the azure_ai_search tool to search the SEMANTIC index for entity values and business context.
You autonomously decide:
- What to search for
- How many results (top_k: 10-100)
- If you need multiple searches

----
CONVERSATION HISTORY (Chat history - automatically injected below)
----

You have access to the full conversation history below. Use it to:
- Resolve ambiguities from previous context
- Understand follow-up questions
- Avoid asking for clarification if context is already clear
- Reference previous responses and tool calls

---
STEP 1: Search for relevant entities
---

Call: azure_ai_search(query="relevant search terms", index_type="semantic", top_k=?)

---
STEP 2: CHECK CHAT HISTORY FIRST (CRITICAL)
---
BEFORE marking anything as ambiguous:
1. READ the chat history carefully (available below current message)
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
"""),
            MessagesPlaceholder("messages"),
            ("human", """Query: {user_query}

Task: Enrich this query and determine if clarification is needed.

IMPORTANT:
1. Check chat history ABOVE before marking ambiguous
2. Use user memories to provide context
3. Populate ambiguity_detected.options ONLY inside JSON if required"""),
        ])
        
        # Format initial messages with history
        initial_messages = prompt.format_messages(
            memories_text=memories_text,
            messages=chat_history,
            user_query=user_query,
        )
        
        # Start tool-calling loop
        agent_messages = list(initial_messages)
        max_iterations = 5
        response = None
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            response = llm_with_tools.invoke(agent_messages)

            # Sanitize AI message content
            if response.content:
                response.content = safe_utf8(response.content)

            # Check if there are tool calls
            if response.tool_calls:
                print(f"🔧 Tool calls: {len(response.tool_calls)}")

                # Add AI message with tool calls
                agent_messages.append(response)
                all_new_messages.append(response)

                # Execute tools
                tool_messages = execute_tool_calls(response.tool_calls, tools_map)

                # Sanitize all tool messages
                for tm in tool_messages:
                    if tm.content:
                        tm.content = safe_utf8(tm.content)

                agent_messages.extend(tool_messages)
                all_new_messages.extend(tool_messages)
            else:
                # No more tool calls, we have the final response
                all_new_messages.append(response)
                print("✅ Final response received")
                break
        
        raw_output = response.content if response else ""

        print("\n📄 SEMANTIC AGENT RAW OUTPUT:")
        print("=" * 70)
        print(raw_output[:500] + "..." if len(raw_output) > 500 else raw_output)
        print("=" * 70)

        # Parse with structured output
        llm_structured = create_llm().with_structured_output(
            SemanticOutput, method="function_calling"
        )
        output: SemanticOutput = llm_structured.invoke(raw_output)

        # Store reasoning
        if output.reasoning:
            reasoning_message = AIMessage(
                content=safe_utf8(output.reasoning),
                metadata={"type": "internal_reasoning", "node": "semantic"}
            )
            all_new_messages.append(reasoning_message)

            print("\n📌 Internal Reasoning Message to store:")
            print(reasoning_message.content[:500] + "..." if len(reasoning_message.content) > 500 else reasoning_message.content)

        print(f"\n✅ STRUCTURED OUTPUT:")
        print(f"   Enriched: {output.enriched_query}")
        print(f"   Ambiguous: {output.ambiguity_detected.ambiguous}")

        if output.ambiguity_detected.ambiguous:
            print(f"   Entity: {output.ambiguity_detected.entity}")
            print(f"   Options: {len(output.ambiguity_detected.options)}")
            for opt in output.ambiguity_detected.options[:5]:
                print(f"      - {opt.label}")
        
        # Return state updates
        return {
            "messages": sanitize_any(all_new_messages),
            "user_memories": sanitize_any(user_memories),
            "clarification_message": None,  # Clarification will be set by clarification_node if needed
            "semantic_chitchat": False,  # Clear the flag - this is not chitchat
            "awaiting_clarification": False,  # Clear clarification flag after processing
            "previous_ambiguity": None,  # Clear previous ambiguity
            "enriched_query": safe_utf8(output.enriched_query),
            "domain_context": sanitize_any(output.domain_context),
            "ambiguity_detected": sanitize_any(
                output.ambiguity_detected.model_dump()
                if output.ambiguity_detected
                else None
            ),
        }
