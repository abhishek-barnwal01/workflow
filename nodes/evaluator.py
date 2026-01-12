"""Evaluator Node - Confidence scoring and iteration feedback"""
from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from models import EvaluatorOutput, ConfidenceBreakdown, PipelineState
import config


def create_llm():
    """Create LLM instance"""
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,  # Updated param name
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0
    )


def evaluator_node(state: PipelineState) -> Dict[str, Any]:
    """
    Evaluator node - scores RAG answer quality.
    
    Responsibilities:
    1. Evaluate RAG answer against user query
    2. Score multiple dimensions (relevance, completeness, context_match)
    3. Identify missing information
    4. Provide actionable feedback for next iteration
    
    Inputs from state:
    - user_query: Original question
    - enriched_query: Enriched query
    - rag_answer: Generated answer
    - retrieved_docs: Documents used
    - iteration_count: Current iteration
    
    Outputs to state:
    - confidence_score: Overall confidence (0-1)
    - confidence_breakdown: Per-dimension scores
    - evaluator_reasoning: Explanation
    - iteration_feedback: Updated with new suggestions
    """
    
    print("\n" + "="*70)
    print("⚖️ EVALUATOR NODE")
    print("="*70)
    
    user_query = state["user_query"]
    enriched_query = state.get("enriched_query", user_query)
    rag_answer = state.get("rag_answer", "")
    retrieved_docs = state.get("retrieved_docs", [])
    iteration_count = state.get("iteration_count", 1)
    previous_feedback = state.get("iteration_feedback", [])
    
    print(f"Iteration: {iteration_count}")
    print(f"Evaluating answer length: {len(rag_answer)} chars")
    print(f"Documents used: {len(retrieved_docs)}")
    
    # Format documents for evaluation
    docs_text = "\n\n".join([
        f"Doc {i+1} (score: {doc.get('score', 0):.2f}, source: {doc.get('source', 'unknown')}):\n{doc.get('content', '')[:200]}..."
        for i, doc in enumerate(retrieved_docs[:5])
    ])
    
    llm = create_llm()
    
    prompt = f"""Evaluate the RAG agent's answer quality.

═══════════════════════════════════════════════════════════════════════════
USER'S QUESTION
═══════════════════════════════════════════════════════════════════════════

Original: {user_query}
Enriched: {enriched_query}

═══════════════════════════════════════════════════════════════════════════
RAG'S ANSWER
═══════════════════════════════════════════════════════════════════════════

{rag_answer}

═══════════════════════════════════════════════════════════════════════════
RETRIEVED DOCUMENTS
═══════════════════════════════════════════════════════════════════════════

{docs_text}

═══════════════════════════════════════════════════════════════════════════
PREVIOUS FEEDBACK (iteration {iteration_count})
═══════════════════════════════════════════════════════════════════════════

{chr(10).join(previous_feedback) if previous_feedback else "First iteration - no previous feedback."}

═══════════════════════════════════════════════════════════════════════════
EVALUATION TASK
═══════════════════════════════════════════════════════════════════════════

Score these dimensions (0.0 to 1.0):

1. RELEVANCE: Does the answer directly address the user's question?
   - 1.0: Perfectly addresses the question
   - 0.5: Partially addresses it
   - 0.0: Completely off-topic

2. COMPLETENESS: Is all necessary information included?
   - 1.0: Comprehensive, nothing missing
   - 0.5: Key points covered but gaps exist
   - 0.0: Major information missing

3. CONTEXT_MATCH: Does it match the semantic/domain context?
   - 1.0: Perfect domain alignment
   - 0.5: Generally aligned
   - 0.0: Wrong domain/context

OVERALL CONFIDENCE = average of the three scores

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON)
═══════════════════════════════════════════════════════════════════════════

{{
  "confidence_score": 0.0,
  "confidence_breakdown": {{
    "relevance": 0.0,
    "completeness": 0.0,
    "context_match": 0.0
  }},
  "reasoning": "Detailed explanation of scores",
  "missing_info": ["List of specific missing information"],
  "suggestion": "Specific actionable suggestion for next iteration"
}}

IMPORTANT:
- Be conservative in scoring
- If previous feedback wasn't addressed, score lower
- Provide SPECIFIC, ACTIONABLE suggestions
- missing_info should be searchable terms"""
    
    # Get structured evaluation
    llm_structured = create_llm().with_structured_output(EvaluatorOutput)
    output: EvaluatorOutput = llm_structured.invoke(prompt)
    
    print(f"\n✅ Evaluation Complete:")
    print(f"   Overall Confidence: {output.confidence_score:.2f}")
    print(f"   - Relevance: {output.confidence_breakdown.relevance:.2f}")
    print(f"   - Completeness: {output.confidence_breakdown.completeness:.2f}")
    print(f"   - Context Match: {output.confidence_breakdown.context_match:.2f}")
    print(f"\n💡 Reasoning: {output.reasoning[:200]}...")
    
    if output.missing_info:
        print(f"📋 Missing Info: {output.missing_info}")
    
    print(f"💬 Suggestion: {output.suggestion}")
    
    # Update iteration feedback if low confidence
    updated_feedback = list(previous_feedback)
    if output.confidence_score < config.CONFIDENCE_MEDIUM:
        # Add new feedback for next iteration
        if output.suggestion:
            updated_feedback.append(output.suggestion)
        for missing in output.missing_info[:2]:  # Top 2 missing items
            updated_feedback.append(f"Search for: {missing}")
    
    return {
        "confidence_score": output.confidence_score,
        "confidence_breakdown": {
            "relevance": output.confidence_breakdown.relevance,
            "completeness": output.confidence_breakdown.completeness,
            "context_match": output.confidence_breakdown.context_match
        },
        "evaluator_reasoning": output.reasoning,
        "iteration_feedback": updated_feedback,
    }