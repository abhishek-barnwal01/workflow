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
        print("➡️  ROUTING DIRECTLY TO SEMANTIC ENRICHMENT (STEP 2D)")
        print("-" * 70)

        # Skip to Step 2D (semantic_broad enrichment) with clarification context
        # Set a flag to indicate we're in clarification mode
        intent = IntentClassification(
            intent_type="semantic_broad",
            reasoning="User responding to clarification question - routing directly to semantic enrichment with tool",
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
            ("system", """You are an advanced intent classifier for an enterprise RAG system.

        Classify the user's query into ONE of these categories:

        1. **chitchat**: Greetings, thanks, farewells, casual conversation
        - Examples: "hi", "hello", "thanks", "thank you", "bye", "goodbye", "how are you"
        - Action: Respond with friendly message, don't search documents

        2. **direct**: Simple general knowledge questions that don't require company documents
        - Examples: "what is GDP", "define market share", "explain EBITDA", "what is ROI"
        - Characteristics: Definitional, general concepts, no company-specific data needed
        - Action: Enrich query with context, pass to RAG without semantic tool

        3. **semantic_specific**: Specific, targeted questions about known entities or facts
        - Examples: "what is Lux market share in Q3", "show Godrej No.1 sales value growth", "Nielsen IQ data for soap category", "List all U&A reports", "How many Dipstick reports do we have", "list brand equity reports", "show all concept test documents"
        - Characteristics:
          * Question mentions SPECIFIC entities (brand names, products, metrics, time periods)
          * Question asks to LIST or COUNT a SPECIFIC KNOWN report type/category (U&A, Dipstick, Concept Test, Brand Equity, etc.)
          * User knows EXACTLY what they're looking for
          * Question is NARROW and FOCUSED on particular data points
          * NOT exploratory - the entity type is already specified
        - Action: Modify query for RAG search - DO NOT use semantic AI search tool
        - IMPORTANT: "List all X reports" where X is a specific report type (U&A, Dipstick, etc.) is ALWAYS semantic_specific

        4. **semantic_broad**: High-level, exploratory questions requiring entity discovery
        - Examples: "what products do we have", "show all regions", "compare all brands", "what are our top segments", "what types of reports exist"
        - Characteristics:
          * Question is OPEN-ENDED or EXPLORATORY about UNKNOWN entities
          * User wants to DISCOVER what entities/options/categories EXIST
          * Question is BROAD and generic (not specifying a known category)
          * May have AMBIGUITY that needs resolution (e.g., "soap" could mean multiple brands)
        - Action: Use semantic AI search tool to discover entities and detect ambiguities
        - NOTE: "List all X" where X is GENERIC (products, brands, regions) = semantic_broad
        - NOTE: "List all X" where X is SPECIFIC KNOWN TYPE (U&A, Dipstick) = semantic_specific (NOT this category)

        CRITICAL DECISION LOGIC:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Ask yourself these questions in order:

        Q0: "Is the user asking to LIST/COUNT a SPECIFIC KNOWN report type?"
            Known report types: U&A, Usage & Attitude, Dipstick, Brand Equity, Concept Test, Brand Health, Market Research, Nielsen, etc.
            YES (e.g., "list U&A reports", "how many Dipstick reports") → semantic_specific (STOP HERE)
            NO → Continue to Q1

        Q1: "Does the user mention SPECIFIC entities/brands/products/reports by name?"
            YES → semantic_specific
            NO → Continue to Q2

        Q2: "Is the question EXPLORATORY or asking to DISCOVER/LIST GENERIC options?"
            YES (e.g., "what products exist", "show all brands") → Check Q3
            NO → semantic_specific

        Q3: "Check the chat history BELOW - does it provide context about the entities in this question?"
            YES (context available in history) → This is semantic_specific (history provides specifics)
            NO (truly exploratory with no prior context) → semantic_broad

        Q4: "Could there be AMBIGUITY that needs resolution before answering?"
            YES (and no context in history to resolve it) → semantic_broad
            NO → semantic_specific
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        EXAMPLES TO LEARN FROM:
        ✓ "What is Lux market share?" → semantic_specific (specific brand mentioned)
        ✓ "Show GN1 sales in MAT Dec'22" → semantic_specific (specific brand + time period)
        ✓ "Nielsen IQ RMS data for soap" → semantic_specific (specific source + category)
        ✓ "List all U&A reports" → semantic_specific (U&A is a SPECIFIC KNOWN report type)
        ✓ "How many Dipstick reports?" → semantic_specific (Dipstick is a SPECIFIC KNOWN report type)
        ✓ "Show brand equity reports" → semantic_specific (Brand Equity is a SPECIFIC KNOWN report type)
        ✓ "List concept test documents" → semantic_specific (Concept Test is a SPECIFIC KNOWN report type)

        ✗ "What is our market share for soap?" → semantic_broad (ambiguous, no history context)
        ✗ "Show all products" → semantic_broad (exploratory, "products" is GENERIC)
        ✗ "Compare regions" → semantic_broad (open-ended, no context in history)
        ✗ "What brands do we have?" → semantic_broad (discovery question, no prior context)
        ✗ "What types of reports exist?" → semantic_broad (asking to DISCOVER categories)

        KEY DISTINCTION FOR "LIST ALL X":
        - If X is a SPECIFIC KNOWN TYPE (U&A, Dipstick, Brand Equity, Concept Test) → semantic_specific
        - If X is GENERIC/UNKNOWN (products, brands, reports, documents) → semantic_broad

        BUT WITH HISTORY CONTEXT:
        ✓ "Compare regions" (when history shows specific regions already mentioned) → semantic_specific
        ✓ "Show all of them" (when history clarifies what "them" refers to) → semantic_specific

        IMPORTANT:
        - "List all U&A reports", "list brand equity reports" = ALWAYS semantic_specific (known report types)
        - ALWAYS check chat history BELOW before marking as semantic_broad
        - Use conversation context to determine if entities are already known/established
        - A follow-up question may reference previous context (making it specific)
        - Only mark semantic_broad if question is truly exploratory WITH NO historical context

        Analyze the query and return your classification with detailed reasoning."""),
            MessagesPlaceholder("messages"),  # Chat history auto-injected here
            ("human", "Query: {user_query}\n\nClassify this query's intent using the decision logic above. IMPORTANT: Before classifying as semantic_broad, check the chat history to see if context makes it semantic_specific instead.")
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
    # STEP 2C: SEMANTIC_SPECIFIC - Query modification without AI search tool
    # ========================================================================

    elif intent.intent_type == "semantic_specific":
        print("\n" + "-" * 70)
        print("STEP 2C: Semantic Specific - Query Modification (No AI Search Tool)")
        print("-" * 70)
        print("ℹ️  User knows exactly what they want - specific entities mentioned")
        print("ℹ️  Modifying query for RAG node - RAG will handle the search")

        # Agentic enrichment with chat history but NO TOOLS
        enrichment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query enrichment agent for specific, targeted questions.

    The user asked a SPECIFIC question with clear entities mentioned.
    
    Your job:
    1. Use chat history to infer context (e.g., if user previously asked "list all U&A reports" and now says "concept testing", infer they mean "list all concept testing reports")
    2. Rephrase the query briefly and clearly for RAG search
    3. Preserve all entity names exactly as mentioned
    4. Keep enrichment MINIMAL - do not enumerate document types, synonyms, or variants
    5. Set ambiguous = false (this is a specific query)
    
    Return JSON with:
    {{
    "enriched_query": "brief, clear version of the query with context applied",
    "domain_context": {{"query_type": "specific"}},
    "ambiguity_detected": {{
        "ambiguous": false,
        "entity": null,
        "options": [],
        "reason": null
    }},
    "reasoning": "one-line explanation of enrichment"
    }}

    CRITICAL:
    - Keep enriched_query SHORT and DIRECT
    - Do NOT add verbose descriptions, document type enumerations, or synonyms
    - Example: User says "concept testing" after "list all U&A reports" → enriched_query = "list all concept testing reports"
    - Example: NOT "Retrieve and list all documents that specifically reference 'Concept testing'..."
    - ambiguous MUST be false
    - options MUST be empty array []"""),
            MessagesPlaceholder("messages"),  # Chat history auto-injected
            ("human", "Query: {user_query}\n\nEnrich this specific query briefly, using chat history for context.")
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
                    domain_context={"query_type": "specific"},
                    ambiguity_detected=AmbiguityInfo(ambiguous=False),
                    reasoning="Content filter triggered, using original query without enrichment"
                )
            else:
                # Re-raise other errors
                raise

        print(f"✅ Enriched Query: {output.enriched_query}")
        print(f"   Reasoning: {output.reasoning}")
        print("➡️  Passing to RAG node for document search")

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
            "clarification_message": None,  # No clarification for specific questions
            "semantic_chitchat": False,  # Clear the flag - this is not chitchat
            "awaiting_clarification": False,  # Clear clarification flag
            "previous_ambiguity": None,  # Clear previous ambiguity
            "enriched_query": safe_utf8(output.enriched_query),
            "domain_context": sanitize_any(output.domain_context),
            "ambiguity_detected": sanitize_any(
                output.ambiguity_detected.model_dump()
            ),
        }

    # ========================================================================
    # STEP 2D: SEMANTIC_BROAD - Full tool-calling loop with ambiguity detection
    # ========================================================================

    else:  # intent.intent_type == "semantic_broad"
        print("\n" + "-" * 70)
        print("STEP 2D: Semantic Broad - Full AI Search Tool Loop")
        print("-" * 70)
        print("ℹ️  High-level/exploratory question - using AI search to discover entities")
        print("ℹ️  Will detect ambiguities and ask for clarification if needed")

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
            ("system", """You are a semantic enrichment agent for HIGH-LEVEL, EXPLORATORY queries.
{clarification_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR MISSION: Use the azure_ai_search tool to DISCOVER entities and detect ambiguities
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are handling a BROAD/EXPLORATORY query that needs entity discovery.
The user wants to:
- Discover what options are available (e.g., "what products do we have")
- Get an overview (e.g., "compare all regions")
- Resolve ambiguity (e.g., "market share of soap" - which soap brand?)

USE THE TOOL to search the SEMANTIC index for entity values and business context.

You autonomously decide:
- What to search for (entities, categories, metrics)
- How many results to retrieve (top_k: 10-100)
- If you need multiple searches to fully understand the domain

----
CONVERSATION HISTORY (Chat history - automatically injected below)
----

You have access to the full conversation history below. Use it to:
- Resolve ambiguities from previous context
- Understand follow-up questions
- Avoid asking for clarification if context is already clear
- Reference previous responses and tool calls

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: SEARCH FOR ENTITIES
Call the tool to discover available entities:
    azure_ai_search(query="relevant search terms", index_type="semantic", top_k=?)


STEP 2: CHECK CHAT HISTORY FIRST (CRITICAL)
BEFORE marking anything as ambiguous:
1. READ the chat history carefully (available below current message)
2. If previous conversation provides context → USE IT, NOT ambiguous
3. If user says "ALL" or "all of them" → Include all options, NOT ambiguous
4. Only mark ambiguous if: multiple values exist AND no context in history

STEP 3: EXTRACT OPTIONS FROM SEARCH RESULTS
READ the content returned by the tool and extract specific entity values.

If multiple values exist AND no context resolves them:
✓ ambiguity_detected.ambiguous = true
✓ ambiguity_detected.entity = "the entity name (e.g., 'brand', 'product')"
✓ ambiguity_detected.options MUST be populated with all discovered options
✓ ambiguity_detected.reason = "explain why clarification is needed"

Each option MUST be structured as:
{{
  "label": "Display Name (e.g., 'Lux')",
  "value": "lowercase_underscore_value (e.g., 'lux')"
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You MUST return ONLY valid JSON in this exact structure:

{{
  "enriched_query": "enriched version of the query with discovered entities",
  "domain_context": {{"discovered_entities": ["list of discovered entities"]}},
  "ambiguity_detected": {{
    "ambiguous": false or true,
    "entity": "string or null",
    "options": [],
    "reason": "string or null"
  }},
  "reasoning": "explain your search strategy and findings"
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ If ambiguous = true → options MUST NOT be empty
✓ If options is empty → ambiguous MUST be false
✓ If ambiguous = false → options MUST be an empty array []
✓ If ambiguity is resolved by history → ambiguous = false
✓ If search fails → ambiguous = false and explain in reasoning
✓ ALWAYS use the tool at least once - this is a discovery/exploratory query
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""),
            MessagesPlaceholder("messages"),
            ("human", """Query: {user_query}

Task: Use the azure_ai_search tool to discover entities and enrich this BROAD query.

IMPORTANT:
1. This is a HIGH-LEVEL query - use the tool to discover entities
2. Check chat history ABOVE before marking ambiguous
3. Use user memories to provide context
4. Populate ambiguity_detected.options if multiple entities found"""),
        ])
        
        # Format initial messages with history
        initial_messages = prompt.format_messages(
            clarification_context=clarification_context,
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
