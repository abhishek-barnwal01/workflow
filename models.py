"""Pydantic models and state definitions for LangGraph RAG Pipeline"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime


# ============================================================================
# EXISTING MODELS (kept for backward compatibility and structured outputs)
# ============================================================================

class AmbiguityOption(BaseModel):
    label: str
    value: str


class AmbiguityInfo(BaseModel):
    ambiguous: bool = False
    entity: Optional[str] = None
    options: List[AmbiguityOption] = Field(default_factory=list)
    reason: Optional[str] = None


class SemanticOutput(BaseModel):
    enriched_query: str
    domain_context: Dict[str, Any] = Field(default_factory=dict)
    ambiguity_detected: AmbiguityInfo = Field(default_factory=AmbiguityInfo)
    reasoning: str = ""


class RetrievedDoc(BaseModel):
    content: str
    score: float
    source: str
    content_id: Optional[str] = None


class RAGOutput(BaseModel):
    retrieved_docs: List[RetrievedDoc] = Field(default_factory=list)
    final_answer: str = ""
    search_strategy: str = ""
    reasoning: str = ""
    total_searches: int = 0


class ConfidenceBreakdown(BaseModel):
    relevance: float = Field(ge=0, le=1, default=0.0)
    completeness: float = Field(ge=0, le=1, default=0.0)
    context_match: float = Field(ge=0, le=1, default=0.0)


class EvaluatorOutput(BaseModel):
    confidence_score: float = Field(ge=0, le=1, default=0.0)
    confidence_breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    reasoning: str = ""
    missing_info: List[str] = Field(default_factory=list)
    suggestion: str = ""


class FormatterOutput(BaseModel):
    formatted_response: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# NEW MODELS FOR LANGGRAPH MEMORY
# ============================================================================

class SearchRecord(BaseModel):
    """Record of a search performed by RAG agent"""
    query: str
    index_type: str  # "main_data" or "semantic"
    top_k: int
    result_count: int
    doc_ids: List[str] = Field(default_factory=list)
    avg_score: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class UserMemory(BaseModel):
    """Long-term memory about a user"""
    type: str  # "preference", "fact", "pattern"
    content: str
    source: str = "automatic"  # "automatic" or "explicit"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence: float = 1.0


# ============================================================================
# LANGGRAPH PIPELINE STATE
# ============================================================================

class PipelineState(TypedDict):
    """
    Complete state for the RAG pipeline.
    Persisted by MongoDBSaver (checkpointer) across messages in a session.
    """
    
    # -------------------------------------------------------------------------
    # CHAT HISTORY (managed by LangGraph with add_messages reducer)
    # -------------------------------------------------------------------------
    messages: Annotated[List[BaseMessage], add_messages]
    
    # -------------------------------------------------------------------------
    # REQUEST CONTEXT
    # -------------------------------------------------------------------------
    user_id: str
    session_id: str
    user_query: str
    
    # -------------------------------------------------------------------------
    # USER MEMORIES (loaded from MongoDBStore at start)
    # -------------------------------------------------------------------------
    user_memories: List[Dict[str, Any]]
    
    # -------------------------------------------------------------------------
    # SEMANTIC NODE OUTPUTS
    # -------------------------------------------------------------------------
    enriched_query: str
    domain_context: Dict[str, Any]
    ambiguity_detected: Optional[Dict[str, Any]]  # Serialized AmbiguityInfo
    
    # -------------------------------------------------------------------------
    # AGENT SCRATCHPAD (shared across iterations within request)
    # -------------------------------------------------------------------------
    search_history: List[Dict[str, Any]]  # Serialized SearchRecord list
    retrieved_doc_ids: List[str]          # For deduplication
    iteration_feedback: List[str]         # Evaluator suggestions
    iteration_count: int
    
    # -------------------------------------------------------------------------
    # RAG NODE OUTPUTS
    # -------------------------------------------------------------------------
    retrieved_docs: List[Dict[str, Any]]  # Serialized RetrievedDoc list
    rag_answer: str
    search_strategy: str
    
    # -------------------------------------------------------------------------
    # EVALUATOR NODE OUTPUTS
    # -------------------------------------------------------------------------
    confidence_score: float
    confidence_breakdown: Dict[str, float]
    evaluator_reasoning: str
    
    # -------------------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------------------
    final_response: str
    needs_clarification: bool
    clarification_options: List[Dict[str, str]]  # Serialized AmbiguityOption list


def get_initial_state(user_query: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Create initial state for a new request"""
    return {
        # Request context
        "user_query": user_query,
        "user_id": user_id,
        "session_id": session_id,
        
        # Will be populated by nodes
        "user_memories": [],
        "enriched_query": "",
        "domain_context": {},
        "ambiguity_detected": None,
        
        # Scratchpad (reset per request)
        "search_history": [],
        "retrieved_doc_ids": [],
        "iteration_feedback": [],
        "iteration_count": 0,
        
        # RAG outputs
        "retrieved_docs": [],
        "rag_answer": "",
        "search_strategy": "",
        
        # Evaluator outputs
        "confidence_score": 0.0,
        "confidence_breakdown": {"relevance": 0.0, "completeness": 0.0, "context_match": 0.0},
        "evaluator_reasoning": "",
        
        # Final output
        "final_response": "",
        "needs_clarification": False,
        "clarification_options": [],
    }