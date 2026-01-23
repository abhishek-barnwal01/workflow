"""Formatter Node - Polishes RAG output using full prompt and structured output"""

from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from models import PipelineState, FormatterOutput
import json

# ----------------- Helpers -----------------
def create_llm():
    """Create AzureChatOpenAI instance"""
    import config
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=1,
    )

def safe_utf8(text: str) -> str:
    if not text:
        return ""
    return text.encode("utf-8", errors="replace").decode("utf-8")

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

# ----------------- Node -----------------
def formatter_node(state: PipelineState) -> Dict[str, Any]:
    """
    Formatter Node:
    - Receives RAG final answer + confidence score
    - Polishes the answer to make it user-friendly
    - Outputs structured FormatterOutput
    """

    print("\n" + "="*70)
    print("✨ FORMATTER NODE")
    print("="*70)

    user_query = state.user_query
    rag_final_answer = state.rag_output.final_answer if state.rag_output else ""
    confidence = state.evaluation.confidence_score if state.evaluation else 0.8

    print(f"\n📝 RAG's answer to format ({len(rag_final_answer)} chars)")
    print(f"🔹 Confidence: {confidence:.2f}")

    # ----------------- Full formatting prompt -----------------
    prompt = f"""
Polish and format the RAG final answer for user consumption.

User Query: {user_query}

RAG's Answer:
{rag_final_answer}

Confidence Score: {confidence:.2f}

Your Job:
- Make the answer conversational, clear, and user-friendly
- Add disclaimers if confidence < 0.85
- Keep core content intact
- Use proper paragraphs and formatting
- Optionally highlight key points

Output JSON schema:
{{
  "formatted_response": "your polished response here",
  "metadata": {{
    "confidence": {confidence:.2f}
  }}
}}
"""

    # ----------------- Invoke LLM -----------------
    llm_structured = create_llm().with_structured_output(FormatterOutput, method="function_calling")
    output: FormatterOutput = llm_structured.invoke(prompt)

    print(f"\n✅ Formatted response ({len(output.formatted_response)} chars)")

    # ----------------- Return -----------------
    return {
        "messages": sanitize_any(state.messages),  # preserve chat history
        "formatted": sanitize_any(output.dict()),  # structured formatted output
    }
