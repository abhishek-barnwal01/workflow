"""Pydantic models for structured outputs"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Annotated
from langgraph.graph import add_messages

# -------------------------
# Intent Classification Models
# -------------------------
class IntentClassification(BaseModel):
    model_config = {"extra": "forbid"}

    intent_type: Literal["chitchat", "direct", "semantic_specific", "semantic_broad"]
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
    task_type: Optional[Literal["summarization", "listing", "content_search", "other"]] = "other"
    document_category: Optional[str] = None  # e.g. "Link Test", "U&A" – best-guess for schema pre-load

class UnifiedSemanticOutput(BaseModel):
    """Combined intent classification + enrichment in a single LLM call.

    Eliminates the separate intent → enrichment round-trip for direct and
    semantic_specific intents (saves ~1-3 s per query).
    """
    model_config = {"extra": "forbid"}

    # Intent fields (from IntentClassification)
    intent_type: Literal["chitchat", "direct", "semantic_specific", "semantic_broad"]
    confidence: float = Field(ge=0, le=1)

    # Enrichment fields (from SemanticOutput) — populated for direct / semantic_specific
    enriched_query: str = ""
    domain_context: Optional[Dict[str, Any]] = None
    ambiguity_detected: AmbiguityInfo = Field(default_factory=lambda: AmbiguityInfo(ambiguous=False))
    reasoning: Optional[str] = None
    task_type: Optional[Literal["summarization", "listing", "content_search", "other"]] = "other"
    document_category: Optional[str] = None

# -------------------------
# RAG Node Models
# -------------------------
class RetrievedDoc(BaseModel):
    model_config = {"extra": "forbid"}

    filename: str
    content_path: str
    score: Optional[float] = None  # may be absent for listing queries
    pages: Optional[str] = None    # may be absent for listing queries
    description: str


class RAGOutput(BaseModel):
    model_config = {"extra": "forbid"}

    retrieved_docs: List[RetrievedDoc]
    final_answer: str  # ← RAG's synthesized answer
    search_strategy: Optional[str] = None
    reasoning: Optional[str] = None
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
    awaiting_clarification: bool = False  # Flag to track if we're waiting for clarification response
    previous_ambiguity: Optional[AmbiguityInfo] = None  # Store previous ambiguity for context

    # Memories loaded once in semantic_node and reused by rag_node (avoids double DB query).
    # Each entry is a plain dict with keys: filename, content_path, description, pages, score.
    user_memories: List[Dict[str, Any]] = Field(default_factory=list)

    ambiguity_detected: Optional[AmbiguityInfo] = Field(default_factory=lambda: AmbiguityInfo(ambiguous=False))

    # Set by semantic_node; consumed by rag_node for deterministic flow control.
    task_type: Optional[str] = None           # "summarization" | "listing" | "content_search" | "other"
    document_category: Optional[str] = None   # e.g. "Link Test", "U&A" – used to pre-load schema

    # When True, formatter_node skips its LLM call; app.py streams the formatter directly.
    skip_formatter: bool = False