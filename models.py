"""Pydantic models for structured outputs"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Annotated
from langgraph.graph import add_messages

# -------------------------
# Intent Classification Models
# -------------------------
class IntentClassification(BaseModel):
    model_config = {"extra": "forbid"}
    
    intent_type: Literal["chitchat", "direct", "semantic"]
    reasoning: str
    confidence: float = Field(ge=0, le=1)

# -------------------------
# Semantic Node Models
# -------------------------
class AmbiguityOption(BaseModel):
    model_config = {"extra": "forbid"}

    label: str
    value: str


class AmbiguityInfo(BaseModel):
    model_config = {"extra": "forbid"}

    ambiguous: bool
    entity: Optional[str] = None
    options: List[AmbiguityOption] = Field(default_factory=list)
    reason: Optional[str] = None


class SemanticOutput(BaseModel):
    model_config = {"extra": "forbid"}

    enriched_query: str
    domain_context: Optional[Dict[str, Any]] = None  # optional
    ambiguity_detected: AmbiguityInfo
    reasoning: Optional[str] = None  # optional


# -------------------------
# RAG Node Models
# -------------------------
class RetrievedDoc(BaseModel):
    model_config = {"extra": "forbid"}

    content: str
    score: float
    source: str


class RAGOutput(BaseModel):
    model_config = {"extra": "forbid"}

    retrieved_docs: List[RetrievedDoc]
    final_answer: str  # ← RAG's synthesized answer
    search_strategy: str
    reasoning: str
    total_searches: int


# -------------------------
# Evaluator Node Models
# -------------------------
class ConfidenceBreakdown(BaseModel):
    model_config = {"extra": "forbid"}

    relevance: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    context_match: float = Field(ge=0, le=1)


class EvaluatorOutput(BaseModel):
    model_config = {"extra": "forbid"}

    confidence_score: float = Field(ge=0, le=1)
    confidence_breakdown: ConfidenceBreakdown
    reasoning: str
    missing_info: List[str] = []
    suggestion: str


# -------------------------
# Formatter Node Models
# -------------------------
class FormatterOutput(BaseModel):
    model_config = {"extra": "forbid"}

    formatted_response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Pipeline State
# -------------------------
class PipelineState(BaseModel):
    user_query: str
    user_id: str
    messages: Annotated[
        List[Any], add_messages
    ] = []  # shared conversation with reducer
    enriched_query: Optional[str] = None
    domain_context: Optional[Dict[str, Any]] = None
    clarification_message: Optional[str] = None
    rag_output: Optional[RAGOutput] = None
    evaluation: Optional[EvaluatorOutput] = None
    formatted: Optional[FormatterOutput] = None
    semantic_reasoning: Optional[str] = None
    semantic_chitchat: bool = False  # Flag to end flow immediately for chitchat

    # 🔹 Add retrieval memory to track docs already fetched (changed to list for JSON compatibility)
    retrieval_memory: Dict[str, List[str]] = Field(
        default_factory=lambda: {"semantic": [], "rag": []}
    )
    ambiguity_detected: Optional[AmbiguityInfo] = Field(default_factory=lambda: AmbiguityInfo(ambiguous=False))
