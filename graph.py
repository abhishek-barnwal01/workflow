"""LangGraph StateGraph definition for RAG Pipeline"""
from typing import Literal
from langgraph.graph import StateGraph, START, END
from models import PipelineState
# Import nodes from flat structure
from nodes.semantic import semantic_node
from nodes.rag import rag_node
from nodes.evaluator import evaluator_node
from nodes.formatter import formatter_node
from nodes.clarifier import clarifier_node
from memory.checkpointer import get_checkpointer
from memory.store import get_store
import config


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_semantic(state: PipelineState) -> Literal["clarifier", "rag"]:
    """
    Route after semantic node based on ambiguity detection.
    
    Returns:
        "clarifier" if ambiguity detected
        "rag" if query is clear
    """
    ambiguity = state.get("ambiguity_detected")
    
    if ambiguity and isinstance(ambiguity, dict):
        is_ambiguous = ambiguity.get("ambiguous", False)
        has_options = len(ambiguity.get("options", [])) > 0
        
        if is_ambiguous and has_options:
            print("🔀 Routing: semantic → clarifier (ambiguity detected)")
            return "clarifier"
    
    print("🔀 Routing: semantic → rag (query clear)")
    return "rag"


def route_after_evaluator(state: PipelineState) -> Literal["formatter", "rag"]:
    """
    Route after evaluator based on confidence and iteration count.
    
    Returns:
        "formatter" if confidence sufficient OR max iterations reached
        "rag" if low confidence AND more iterations allowed
    """
    confidence = state.get("confidence_score", 0.0)
    iteration_count = state.get("iteration_count", 1)
    
    # Check if we should proceed to formatter
    if confidence >= config.CONFIDENCE_MEDIUM:
        print(f"🔀 Routing: evaluator → formatter (confidence {confidence:.2f} >= {config.CONFIDENCE_MEDIUM})")
        return "formatter"
    
    if iteration_count >= config.MAX_ITERATIONS:
        print(f"🔀 Routing: evaluator → formatter (max iterations {iteration_count} reached)")
        return "formatter"
    
    # Need more iterations
    print(f"🔀 Routing: evaluator → rag (confidence {confidence:.2f} < {config.CONFIDENCE_MEDIUM}, iteration {iteration_count}/{config.MAX_ITERATIONS})")
    return "rag"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_graph() -> StateGraph:
    """
    Build the RAG pipeline graph.
    
    Flow:
        START → semantic → [ambiguous?] → clarifier → END
                                       ↘
                                         rag ←──────────────┐
                                          ↓                 │
                                       evaluator            │
                                          ↓                 │
                                    [confident?] → NO ──────┘
                                          ↓
                                         YES
                                          ↓
                                      formatter → END
    """
    
    # Create graph with state schema
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("semantic", semantic_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("formatter", formatter_node)
    workflow.add_node("clarifier", clarifier_node)
    
    # Add edges
    
    # START → semantic
    workflow.add_edge(START, "semantic")
    
    # semantic → conditional (clarifier or rag)
    workflow.add_conditional_edges(
        "semantic",
        route_after_semantic,
        {
            "clarifier": "clarifier",
            "rag": "rag"
        }
    )
    
    # clarifier → END
    workflow.add_edge("clarifier", END)
    
    # rag → evaluator
    workflow.add_edge("rag", "evaluator")
    
    # evaluator → conditional (formatter or rag)
    workflow.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "formatter": "formatter",
            "rag": "rag"
        }
    )
    
    # formatter → END
    workflow.add_edge("formatter", END)
    
    return workflow


def compile_graph():
    """
    Compile the graph with checkpointer and store.
    
    Returns:
        Compiled graph ready for invocation
    """
    workflow = build_graph()
    
    # Get memory components
    checkpointer = get_checkpointer()
    store = get_store()
    
    # Compile with memory
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    print("✅ Graph compiled with MongoDB checkpointer and store")
    
    return graph


# ============================================================================
# SINGLETON GRAPH INSTANCE
# ============================================================================

_compiled_graph = None


def get_graph():
    """Get or create compiled graph singleton"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_graph()
    return _compiled_graph