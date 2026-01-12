"""Formatter Node - Response polishing and memory extraction"""
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from models import FormatterOutput, PipelineState
import config


def create_llm():
    """Create LLM instance"""
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,  # Updated param name
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0.7
    )


def formatter_node(state: PipelineState) -> Dict[str, Any]:
    """
    Formatter node - polishes response and extracts memories.
    
    Responsibilities:
    1. Format RAG answer for user consumption
    2. Add disclaimer if low confidence
    3. Apply user preferences from memories
    4. Extract and save new user memories
    
    Inputs from state:
    - user_query: Original question
    - rag_answer: Raw answer from RAG
    - confidence_score: Evaluation score
    - iteration_count: Number of iterations
    - user_memories: User preferences
    - messages: Chat history
    - user_id: User identifier
    
    Outputs to state:
    - final_response: Polished response for user
    """
    
    print("\n" + "="*70)
    print("✨ FORMATTER NODE")
    print("="*70)
    
    user_query = state["user_query"]
    user_id = state["user_id"]
    rag_answer = state.get("rag_answer", "")
    confidence_score = state.get("confidence_score", 0.0)
    iteration_count = state.get("iteration_count", 1)
    user_memories = state.get("user_memories", [])
    messages = state.get("messages", [])
    retrieved_docs = state.get("retrieved_docs", [])
    
    print(f"Confidence: {confidence_score:.2f}")
    print(f"Iterations: {iteration_count}")
    print(f"User memories: {len(user_memories)}")
    
    # Step 1: Check user preferences for formatting
    formatting_preferences = []
    for mem in user_memories:
        if mem.get("type") == "preference":
            formatting_preferences.append(mem.get("content", ""))
    
    preferences_text = ""
    if formatting_preferences:
        preferences_text = f"\nUser preferences to apply:\n" + "\n".join(f"- {p}" for p in formatting_preferences)
        print(f"📋 Applying user preferences: {len(formatting_preferences)}")
    
    # Step 2: Determine if disclaimer is needed
    needs_disclaimer = (
        confidence_score < config.CONFIDENCE_MEDIUM and 
        iteration_count >= config.MAX_ITERATIONS
    )
    
    if needs_disclaimer:
        print("⚠️ Adding low-confidence disclaimer")
    
    # Step 3: Format the response
    llm = create_llm()
    
    prompt = f"""Polish and format this response for the user.

═══════════════════════════════════════════════════════════════════════════
USER QUERY
═══════════════════════════════════════════════════════════════════════════

{user_query}

═══════════════════════════════════════════════════════════════════════════
RAG ANSWER
═══════════════════════════════════════════════════════════════════════════

{rag_answer}

═══════════════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════════════

Confidence Score: {confidence_score:.2f}
Iterations: {iteration_count}
Documents Used: {len(retrieved_docs)}
{preferences_text}

═══════════════════════════════════════════════════════════════════════════
FORMATTING TASK
═══════════════════════════════════════════════════════════════════════════

1. Make the response conversational and user-friendly
2. Keep all citations and document references intact
3. Apply any user preferences noted above
4. Maintain professional tone
5. Ensure key insights are highlighted

{"6. ADD THIS DISCLAIMER AT THE END: '⚠️ Note: This response may be incomplete. Consider rephrasing your question or providing more context for better results.'" if needs_disclaimer else ""}

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON)
═══════════════════════════════════════════════════════════════════════════

{{
  "formatted_response": "Your polished response here",
  "metadata": {{
    "confidence": {confidence_score},
    "iterations": {iteration_count},
    "sources_count": {len(retrieved_docs)}
  }}
}}"""
    
    llm_structured = create_llm().with_structured_output(FormatterOutput)
    output: FormatterOutput = llm_structured.invoke(prompt)
    
    print(f"\n✅ Response formatted ({len(output.formatted_response)} chars)")
    
    # Step 4: Extract and save memories (if store available)
    try:
        from store import extract_memories_from_conversation
        print("\n🧠 Extracting memories from conversation...")
        saved_keys = extract_memories_from_conversation(
            user_id=user_id,
            messages=messages,
            rag_answer=rag_answer
        )
        if saved_keys:
            print(f"   Saved {len(saved_keys)} new memories")
    except ImportError:
        print("   ⚠️ Memory store not available")
    except Exception as e:
        print(f"   ⚠️ Memory extraction skipped: {e}")
    
    return {
        "final_response": output.formatted_response,
        "needs_clarification": False,
        "clarification_options": [],
    }