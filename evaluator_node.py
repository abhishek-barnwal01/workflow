"""Evaluator Node - Evaluate RAG output using full prompt and structured output"""

from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from models import PipelineState, RAGOutput, EvaluatorOutput
from memory_store import store  # in case you want to access memory
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
def evaluator_node(state: PipelineState) -> Dict[str, Any]:
    """
    Evaluator Node:
    - Receives RAG output from previous node
    - Uses semantic enrichment and user query for context
    - Outputs structured evaluation
    """

    print("\n" + "="*70)
    print("⚖️ EVALUATOR NODE")
    print("="*70)

    user_query = state.user_query
    enriched_query = getattr(state, "enriched_query", "")
    rag_output_dict = getattr(state, "rag_output", {})
    
    # Reconstruct RAGOutput object from dict if needed
    rag_output = RAGOutput.parse_obj(rag_output_dict) if rag_output_dict else RAGOutput(retrieved_docs=[], final_answer="")

    # Show top retrieved docs for debug
    docs_text = "\n\n".join([
        f"Doc {i+1} (score: {doc.score:.2f}, source: {doc.source}):\n{doc.content[:200]}..."
        for i, doc in enumerate(rag_output.retrieved_docs[:5])
    ])

    print(f"\n📄 Documents being evaluated: {len(rag_output.retrieved_docs)}")
    print(f"📝 RAG's answer being evaluated: {rag_output.final_answer[:100]}...")

    # ----------------- Full old prompt -----------------
    prompt = f"""
Evaluate RAG's final answer in enterprise-grade professional manner.

User's Original Query: {user_query}
Semantic Enriched Query: {enriched_query}

RAG's Final Answer:
{rag_output.final_answer}

Retrieved Documents (for reference):
{docs_text}

Your Job:
- Evaluate if RAG's final answer properly addresses the user's query using the retrieved documents.
- Score dimensions (0-1):
  1. relevance: Does the answer address the question?
  2. completeness: Is all necessary information included?
  3. context_match: Does it match semantic context?
- Explain reasoning for each score
- Identify missing info if any
- Suggest if RAG needs additional search

Output JSON schema:
{{
  "confidence_score": 0.0,
  "confidence_breakdown": {{
    "relevance": 0.0,
    "completeness": 0.0,
    "context_match": 0.0
  }},
  "reasoning": "explain your evaluation of the answer",
  "missing_info": ["what's missing from the answer"],
  "suggestion": "should RAG search more, or is current answer acceptable?"
}}

Be conservative. If answer is incomplete, score low. Provide professional, structured feedback.
"""

    # ----------------- Invoke LLM -----------------
    llm_structured = create_llm().with_structured_output(EvaluatorOutput, method="function_calling")
    output: EvaluatorOutput = llm_structured.invoke(prompt)

    # ----------------- Debug printing -----------------
    print(f"\n✅ Confidence: {output.confidence_score:.2f}")
    print(f"   Relevance: {output.confidence_breakdown.relevance:.2f}")
    print(f"   Completeness: {output.confidence_breakdown.completeness:.2f}")
    print(f"   Context Match: {output.confidence_breakdown.context_match:.2f}")
    print(f"\n💡 Evaluator's Reasoning: {output.reasoning}")
    if output.missing_info:
        print(f"   Missing Info: {output.missing_info}")
    print(f"   Suggestion: {output.suggestion}")

    # ----------------- Return -----------------
    return {
        "messages": sanitize_any(state.messages),  # preserve chat history
        "evaluation": sanitize_any(output.dict()),  # structured evaluation
    }
