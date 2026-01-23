# from typing import Dict, Any

# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import AIMessage
# from langchain.agents import create_agent

# from tools import azure_ai_search
# from models import SemanticOutput, PipelineState
# from memory_store import store
# import config

# def safe_utf8(text: str) -> str:
#     if not text:
#         return ""
#     return text.encode("utf-8", errors="replace").decode("utf-8")


# def sanitize_any(obj):
#     if obj is None:
#         return None
#     if isinstance(obj, str):
#         return safe_utf8(obj)
#     if isinstance(obj, list):
#         return [sanitize_any(i) for i in obj]
#     if isinstance(obj, dict):
#         return {k: sanitize_any(v) for k, v in obj.items()}
#     return obj

# def create_llm():
#     return AzureChatOpenAI(
#         azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
#         azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
#         api_key=config.AZURE_OPENAI_KEY,
#         api_version=config.AZURE_OPENAI_API_VERSION,
#         temperature=1,
#     )
# SYSTEM_PROMPT = """
# You are a semantic enrichment agent.

# Use the azure_ai_search tool to search the SEMANTIC index for entity values and business context.
# You autonomously decide:
# - What to search for
# - How many results (top_k: 10-100)
# - If you need multiple searches

# ----
# CONVERSATION HISTORY (Chat history - automatically injected below)
# ----

# You have access to the full conversation history below. Use it to:
# - Resolve ambiguities from previous context
# - Understand follow-up questions
# - Avoid asking for clarification if context is already clear
# - Reference previous responses and tool calls

# ---
# STEP 1: Search for relevant entities
# ---

# Call: azure_ai_search(query="relevant search terms", index_type="semantic", top_k=?)

# ---
# STEP 2: CHECK CHAT HISTORY FIRST (CRITICAL)
# ---
# BEFORE marking anything as ambiguous:
# 1. READ the chat history carefully (available below current message)
# 2. If previous conversation provides context → USE IT, NOT ambiguous
# 3. If user says "ALL" or "all of them" → Include all options, NOT ambiguous
# 4. Only mark ambiguous if: multiple values exist AND no context in history

# ---
# STEP 3: Extract options from search results
# ---
# READ the content returned by the tool and extract specific values.

# If multiple values exist AND no context resolves them:
# - ambiguity_detected.ambiguous = true
# - ambiguity_detected.options MUST be populated

# Each option MUST be structured as:
# {
#   "label": "Display Name",
#   "value": "lowercase_underscore_value"
# }

# ---
# OUTPUT REQUIREMENTS
# ---

# You MUST return ONLY valid JSON in this exact structure:

# {
#   "enriched_query": "string",
#   "domain_context": {},
#   "ambiguity_detected": {
#     "ambiguous": false,
#     "entity": "string or null",
#     "options": [],
#     "reason": "string or null"
#   },
#   "reasoning": "string"
# }

# ---
# CRITICAL RULES
# ---
# - If ambiguous = true → options MUST NOT be empty
# - If options is empty → ambiguous MUST be false
# - If ambiguous = false → options MUST be an empty array []
# - If ambiguity is resolved by history → ambiguous = false
# - If search fails → ambiguous = false and explain in reasoning
# """
# semantic_agent = create_agent(
#     model=create_llm(),
#     tools=[azure_ai_search],
#     system_prompt=SYSTEM_PROMPT,
#     response_format=SemanticOutput,   # ✅ Structured output enforced
#     store=store,                      # ✅ PostgresStore memory
#     debug=True,
#     name="semantic_enrichment_agent"
# )

# def semantic_node(state: PipelineState) -> Dict[str, Any]:
#     print("\n" + "=" * 70)
#     print("🧠 SEMANTIC NODE (AGENT MODE)")
#     print("=" * 70)

#     print(f"Query: {state.user_query}")
#     print(f"User ID: {state.user_id}")
#     print(f"Chat history messages: {len(state.messages)}")

#     # Invoke agent (tool calls + iterations handled internally)
#     result = semantic_agent.invoke(
#         {
#             "messages": state.messages,
#             "user_query": state.user_query,
#             "user_id": state.user_id,
#         }
#     )

#     output: SemanticOutput = result["structured_response"]

#     # Store internal reasoning as message (same behavior as before)
#     if output.reasoning:
#         reasoning_message = AIMessage(
#             content=safe_utf8(output.reasoning),
#             metadata={"type": "internal_reasoning", "node": "semantic"},
#         )
#         state.messages.append(reasoning_message)

#         print("\n📌 Internal Reasoning Stored")
#         print(reasoning_message.content[:500])

#     print("\n✅ STRUCTURED OUTPUT")
#     print(f"Enriched Query: {output.enriched_query}")
#     print(f"Ambiguous: {output.ambiguity_detected.ambiguous}")

#     if output.ambiguity_detected.ambiguous:
#         print(f"Entity: {output.ambiguity_detected.entity}")
#         for opt in output.ambiguity_detected.options:
#             print(f" - {opt.label}")

#     return {
#         "messages": sanitize_any(result["messages"]),
#         "user_memories": sanitize_any(result.get("memories")),
#         "enriched_query": safe_utf8(output.enriched_query),
#         "domain_context": sanitize_any(output.domain_context),
#         "ambiguity_detected": sanitize_any(
#             output.ambiguity_detected.model_dump()
#             if output.ambiguity_detected
#             else None
#         ),
#     }


