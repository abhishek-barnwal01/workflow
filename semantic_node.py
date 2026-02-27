"""Semantic Node - Query enrichment and ambiguity detection"""

from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import azure_ai_search
from models import AmbiguityInfo, IntentClassification, SemanticOutput, UnifiedSemanticOutput, PipelineState
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
        # UNIFIED STEP: Intent Classification + Enrichment (single LLM call)
        # ========================================================================
        # For chitchat → routes to response generation
        # For direct / semantic_specific → enrichment is already done (saves 1 LLM call)
        # For semantic_broad → routes to tool-calling loop

        print("\n" + "-" * 70)
        print("UNIFIED STEP: Intent Classification + Enrichment")
        print("-" * 70)

        llm = create_llm()

        unified_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an advanced intent classifier AND query enrichment agent for an enterprise RAG system.
In ONE pass, classify the intent AND produce an enriched query.

USER MEMORIES (previously retrieved documents):
{memories_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — INTENT CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classify into ONE category:

1. **chitchat**: Greetings, thanks, farewells, casual conversation
   - Examples: "hi", "hello", "thanks", "bye", "how are you"
   - Set enriched_query = "" (enrichment not needed)

2. **direct**: Simple general knowledge questions not requiring company documents
   - Examples: "what is GDP", "define market share", "explain EBITDA"

3. **document_listing**: Requests to LIST, COUNT, or SHOW available documents/reports
   - Examples: "list all U&A reports", "show brand equity reports", "how many link testing reports do we have", "what reports are available for Cinthol", "show all reports for India 2023"
   - Characteristics: user wants a LIST of document titles/metadata — NOT content analysis
   - Trigger words: "list", "show", "how many", "count", "what reports", "which documents", "available documents"
   - IMPORTANT: If user asks to LIST or COUNT documents by category, brand, country, or time period → ALWAYS document_listing
   - Set enriched_query = concise search description for SQL (e.g., "U&A reports India", "brand equity Godrej 2023")

4. **semantic_specific**: Specific, targeted questions about CONTENT within documents
   - Examples: "what is Lux market share in Q3", "summarize the GN1 link test", "what does the U&A study say about purchase drivers"
   - Characteristics: mentions SPECIFIC entities AND wants to READ/ANALYSE content
   - IMPORTANT: "Summarize X report" = semantic_specific (needs content), "List all X reports" = document_listing (needs metadata only)

5. **semantic_broad**: High-level, exploratory questions requiring entity discovery
   - Examples: "what products do we have", "show all regions", "compare all brands"
   - Characteristics: OPEN-ENDED, EXPLORATORY about UNKNOWN entities, BROAD and generic
   - Set enriched_query = "" (tool-calling loop will handle)

DECISION LOGIC (in order):
Q0: Asking to LIST, COUNT, or SHOW documents/reports? → document_listing
Q1: Listing/counting a SPECIFIC KNOWN report type? → document_listing
Q2: Asking to READ, SUMMARIZE, or ANALYSE content? → semantic_specific
Q3: Mentions SPECIFIC entities by name for content questions? → semantic_specific
Q4: EXPLORATORY / GENERIC discovery? → Check Q5
Q5: Chat history provides context? → semantic_specific, else → semantic_broad
Q6: Unresolved AMBIGUITY with no history context? → semantic_broad

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — ENRICHMENT (for direct / semantic_specific only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Produce a concise, standalone enriched_query for RAG search:

1. Resolve elliptical references
   Replace shorthand with full entity from most recent relevant turn.
   e.g. "summarize jan 2021" after GN1 discussion → "summarize No1 TVC Refresh Creative Brief Jan 2021 GN1 link test India"

2. Carry forward unrestated context
   Inherit brand, product, report type, category, geography, time period from prior turns.
   Override only what user explicitly changes.

3. Apply defaults (only when absent from both query and history)
   - No time period → prepend "latest"
   - No geography → append "India"

─── DIRECT ANSWER SHORTCUT ────────────────────────────────────────────────
Set enriched_query = "" only when the COMPLETE answer already exists verbatim in a
prior AI response — not when prior turns merely list document titles or links.
Place the full answer in reasoning, starting with "Based on our previous discussion…"

─── SUMMARIZATION EXCEPTION ───────────────────────────────────────────────
When user wants to read, summarize, or get insights from a specific document:
- Always set task_type = "summarization" and provide a non-empty enriched_query.
- Infer document_category from document name or context.
- Never use the direct-answer shortcut; document content is not stored in history.

─── OUTPUT ────────────────────────────────────────────────────────────────
Return all fields: intent_type, confidence, enriched_query, domain_context,
ambiguity_detected, reasoning, task_type, document_category."""),
            MessagesPlaceholder("messages"),
            ("human", "Query: {user_query}\n\nClassify intent AND enrich in one step. Check chat history before marking semantic_broad.")
        ])

        unified_messages = unified_prompt_template.format_messages(
            memories_text=memories_text,
            messages=chat_history,
            user_query=user_query
        )

        llm_unified = llm.with_structured_output(UnifiedSemanticOutput, method="function_calling")

        try:
            unified: UnifiedSemanticOutput = llm_unified.invoke(unified_messages)
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Unified Classification Error: {error_msg[:200]}")
            if "content_filter" in error_msg.lower() or "jailbreak" in error_msg.lower():
                print("   Content filter triggered - defaulting to 'direct' intent")
                unified = UnifiedSemanticOutput(
                    intent_type="direct",
                    confidence=0.5,
                    enriched_query=user_query,
                    ambiguity_detected=AmbiguityInfo(ambiguous=False),
                    reasoning="Content filter triggered, using original query",
                )
            else:
                raise

        # Map to the existing IntentClassification for downstream compat
        intent = IntentClassification(
            intent_type=unified.intent_type,
            reasoning=unified.reasoning or "",
            confidence=unified.confidence,
        )

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
            chitchat_llm = create_llm()
            chitchat_response = chitchat_llm.invoke(chitchat_messages)
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
        print("STEP 2B: Direct Question - Enrichment already done in unified call")
        print("-" * 70)

        # Enrichment was already computed in the unified call — no extra LLM round-trip
        output = SemanticOutput(
            enriched_query=unified.enriched_query or user_query,
            domain_context=unified.domain_context,
            ambiguity_detected=unified.ambiguity_detected,
            reasoning=unified.reasoning,
            task_type=unified.task_type,
            document_category=unified.document_category,
        )

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
            "clarification_message": None,
            "semantic_chitchat": False,
            "awaiting_clarification": False,
            "previous_ambiguity": None,
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
        print("STEP 2C: Semantic Specific - Enrichment already done in unified call")
        print("-" * 70)
        print("ℹ️  User knows exactly what they want - specific entities mentioned")
        print("ℹ️  Enrichment computed in unified call — no extra LLM round-trip")

        # Enrichment was already computed in the unified call
        output = SemanticOutput(
            enriched_query=unified.enriched_query,
            domain_context=unified.domain_context or {"query_type": "specific"},
            ambiguity_detected=unified.ambiguity_detected,
            reasoning=unified.reasoning,
            task_type=unified.task_type,
            document_category=unified.document_category,
        )

        # ✅ Python guard: summarization tasks must always go to RAG — no history shortcut.
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

    elif intent.intent_type == "semantic_broad":
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
                # No more tool calls — skip this text response and go straight
                # to structured extraction (saves 1 LLM call).
                print("✅ No more tool calls — extracting structured output directly")
                break

        # Single structured extraction call replaces the old pattern of
        # (text response iteration) + (separate extraction call) = 2 calls → 1 call.
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

    # ========================================================================
    # STEP 2E: DOCUMENT_LISTING - Route to SQL-based document retriever
    # ========================================================================

    elif intent.intent_type == "document_listing":
        print("\n" + "-" * 70)
        print("STEP 2E: Document Listing - Routing to SQL document retriever")
        print("-" * 70)
        print("ℹ️  User wants to list/count documents — fast SQL path, no RAG needed")

        # Enrichment was already computed in the unified call
        listing_query = unified.enriched_query or user_query

        print(f"✅ Listing Query: {listing_query}")
        print(f"   Reasoning: {unified.reasoning}")
        print("➡️  Routing to document_retriever node")

        if unified.reasoning:
            reasoning_message = AIMessage(
                content=safe_utf8(unified.reasoning),
                metadata={"type": "internal_reasoning", "node": "semantic"}
            )
            all_new_messages.append(reasoning_message)

        return {
            "messages": sanitize_any(all_new_messages),
            "user_memories": sanitize_any(user_memories),
            "clarification_message": None,
            "semantic_chitchat": False,
            "awaiting_clarification": False,
            "previous_ambiguity": None,
            "enriched_query": safe_utf8(listing_query),
            "domain_context": sanitize_any(unified.domain_context),
            "ambiguity_detected": sanitize_any(
                unified.ambiguity_detected.model_dump()
            ),
            "task_type": "listing",
            "document_category": unified.document_category,
        }