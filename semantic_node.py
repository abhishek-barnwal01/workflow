"""Semantic Node - Query enrichment and ambiguity detection"""

from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import azure_ai_search
from models import AmbiguityInfo, IntentClassification, SemanticOutput, PipelineState
import json
from memory_store import store
from langgraph.types import RunnableConfig
from utils import safe_utf8, sanitize_any, create_llm, execute_tool_calls


def semantic_node(state: PipelineState, config: RunnableConfig = None) -> Dict[str, Any]:
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
    
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    # Step 1: Load user memories from PostgresStore (once — rag_node reuses via state)
    try:
        _raw_items = store.search(
            ("rag_memory", user_id, thread_id),
            query=None,
            limit=10,
        )
        # Convert store Item objects to plain dicts so they serialise cleanly in PipelineState
        user_memories = [item.value for item in _raw_items] if _raw_items else []
        print(f"📚 Loaded {len(user_memories)} user memories")
    except Exception as e:
        print(f"⚠️ Could not load memories: {e}")
        user_memories = []

    # Get chat history
    chat_history = messages
    if chat_history:
        print(f"📜 Chat History: {len(messages)} messages available")

    # Format previously retrieved documents for prompt
    # (written by rag_node into PostgresStore after each retrieval)
    if user_memories:
        memories_text = "Previously retrieved documents:\n"
        for mem_dict in user_memories:
            filename = mem_dict.get("filename", "")
            content_path = mem_dict.get("content_path", "")
            description = mem_dict.get("description", "")[:300]
            pages = mem_dict.get("pages", "")
            memories_text += f"- {filename} ({content_path}) [Pages: {pages}] {description}\n"
    else:
        memories_text = "No previously retrieved documents."

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
            ("system", """You are a query enrichment agent for general knowledge questions.

Produce a self-contained, standalone version of the user's query.

RULES:
1. Resolve references — substitute pronouns ("it", "that", "the same") using prior context.
2. Carry forward topic — if the user narrows or pivots on a prior topic, preserve the base topic.
3. Keep enriched_query concise; add only what is needed to remove ambiguity.

Return JSON: enriched_query, domain_context (null), ambiguity_detected (ambiguous: false), reasoning."""),
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
            if "content_filter" in error_msg.lower() or "jailbreak" in error_msg.lower() or "filtered" in error_msg.lower() or "400" in error_msg:
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
            ("system", """You are a query enrichment agent for a document retrieval system.

USER MEMORIES (previously retrieved documents):
{memories_text}

─── ENRICHMENT RULES ────
Produce a concise, standalone enriched_query for RAG search:

1. Resolve elliptical references
   Replace shorthand with the full entity inferred from the most recent relevant turn.
   e.g. "summarize jan 2021" after a GN1 link-test discussion
     → "summarize No1 TVC Refresh Creative Brief Jan 2021 GN1 link test India"

2. Carry forward unrestated context
   Inherit brand, product, report type, category, geography, and time period from
   prior turns when the user omits them. Override only what the user explicitly changes.
   e.g. user asked about U&A reports, then says "concept testing"
     → "list all concept testing reports India"

3. Apply defaults (only when absent from both the query and history)
   - No time period → prepend "latest"
   - No geography  → append "India"

─── DIRECT ANSWER SHORTCUT ────
Set enriched_query = "" only when the COMPLETE answer already exists verbatim in a
prior AI response — not when prior turns merely list document titles or links.
Place the full answer in reasoning, starting with "Based on our previous discussion…"

─── SUMMARIZATION EXCEPTION ────
When the user wants to read, summarize, or get insights from a specific document:
- Always set task_type = "summarization" and provide a non-empty enriched_query.
- Infer document_category from the document name or context.
- Never use the direct-answer shortcut; document content is not stored in history.

─── OUTPUT ────
{{
  "enriched_query": "standalone RAG query, or '' if answering directly",
  "domain_context": {{"query_type": "specific"}},
  "ambiguity_detected": {{"ambiguous": false, "entity": null, "options": [], "reason": null}},
  "reasoning": "direct answer text, or one-line enrichment rationale",
  "task_type": "summarization | listing | content_search | other",
  "document_category": "Link Test | U&A | Concept Test | Dipstick | null"
}}"""),
            MessagesPlaceholder("messages"),  # Chat history auto-injected
            ("human", "Query: {user_query}\n\nCheck history first. If answer exists, set enriched_query='' and put full answer in reasoning otherwise enrich this specific query briefly, using chat history for context." )
        ])

        enrichment_messages = enrichment_prompt.format_messages(
            memories_text=memories_text,
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
        
        # ✅ Python guard: summarization tasks must always go to RAG — no history shortcut.
        # This is a typed signal check (output.task_type), not keyword scanning.
        if output.task_type == "summarization" and (not output.enriched_query or output.enriched_query.strip() == ""):
            print("⚠️  GUARD: summarization task had empty enriched_query — forcing RAG routing")
            output.enriched_query = user_query

        # ✅ PHASE 0: Check if answered directly from history
        if not output.enriched_query or output.enriched_query.strip() == "":
            print("✅ ANSWERED DIRECTLY FROM HISTORY - SKIPPING RAG NODE")
            print(f"   Direct answer: {output.reasoning[:200]}...")

            answer_message = AIMessage(
                content=safe_utf8(output.reasoning),
                metadata={"type": "direct_answer", "node": "semantic", "source": "history"}
            )
            all_new_messages.append(answer_message)

            return {
                "messages": sanitize_any(all_new_messages),
                "user_memories": sanitize_any(user_memories),
                "clarification_message": safe_utf8(output.reasoning),  # ✅ Signals END
                "semantic_chitchat": False,
                "awaiting_clarification": False,
                "previous_ambiguity": None,
                "enriched_query": "",
                "domain_context": sanitize_any(output.domain_context),
                "ambiguity_detected": sanitize_any(output.ambiguity_detected.model_dump()),
                "task_type": output.task_type,
                "document_category": output.document_category,
            }

        print(f"✅ Enriched Query: {output.enriched_query}")
        print(f"   Reasoning: {output.reasoning}")
        print(f"   Task Type: {output.task_type} | Document Category: {output.document_category}")
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
            "task_type": output.task_type,
            "document_category": output.document_category,
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
Previously retrieved documents (from memory):
{memories_text}

STEP 1 — CHECK HISTORY
Skip the tool ONLY if the previous AI message already answered this exact question or ambugity is resolved by history context.
For all other cases — including new chats and follow-up questions — proceed to STEP 2.

STEP 2 — SEARCH (mandatory for any new or exploratory question)
Call azure_ai_search on the semantic index to DISCOVER entities and detect ambiguities.
The user wants to:
- Discover what options are available (e.g., "what products do we have")
- Get an overview (e.g., "compare all regions")
- Resolve ambiguity (e.g., "market share of soap" - which soap brand?)
- If no geography in query or history → include "India" in search text
- If no time period in query or history → include "latest" in search text

STEP 3 — DECIDE
- One clear match, or user said "all" → enriched_query = short keyword query for RAG, ambiguous = false
- Multiple matches, nothing in history resolves them → ambiguous = true, populate all options
- Search failed or no results → best-effort enriched_query from query alone, ambiguous = false

OUTPUT:
- enriched_query: short keyword query, not a sentence or description
- ambiguity_detected.options: populated only when ambiguous = true; empty array [] otherwise
"""),
            MessagesPlaceholder("messages"),
            ("human", "Query: {user_query}"),
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
        
        # Extract structured output from the full agent_messages context (includes tool results),
        # NOT from raw_output alone — that would lose all search result context.
        llm_structured = create_llm().with_structured_output(
            SemanticOutput, method="function_calling"
        )
        output: SemanticOutput = llm_structured.invoke(agent_messages)

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
            # Cap at 10 options — clarification_node further limits to 5 for display.
            # Without a cap the LLM can return hundreds, bloating state and breaking the UI.
            if len(output.ambiguity_detected.options) > 10:
                output.ambiguity_detected.options = output.ambiguity_detected.options[:10]
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
