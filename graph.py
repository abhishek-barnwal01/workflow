# from langgraph.graph import StateGraph, END
# from models import PipelineState
# from semantic_node import semantic_node
# from persistence import checkpointer
# from clarification_node import clarification_node
# from rag_node import rag_node
# from memory_store import store
# from evaluator_node import evaluator_node
# from formatter_node import formatter_node


# def build_graph():

#     builder = StateGraph(PipelineState)

#     builder.add_node("semantic", semantic_node)
#     builder.add_node("clarification", clarification_node)
#     builder.add_node("rag", rag_node)
#     builder.add_node("evaluator", evaluator_node)
#     builder.add_node("formatter", formatter_node)

#     builder.set_entry_point("semantic")
#     builder.add_edge("semantic", "clarification")
#     builder.add_edge("clarification", "rag")
#     #builder.add_edge("semantic", "rag")
#     #builder.add_edge("rag", END)
#     builder.add_edge("rag", "evaluator")
#     builder.add_edge("evaluator", "formatter")
#     builder.add_edge("formatter", END)

#     app = builder.compile(checkpointer=checkpointer,store=store )
# #     return app

#     return app

    
from langgraph.graph import StateGraph, END
from models import PipelineState
from semantic_node import semantic_node
from clarification_node import clarification_node
from rag_node import rag_node
from evaluator_node import evaluator_node
from formatter_node import formatter_node
from document_retriever_node import document_retriever_node
from persistence import checkpointer
from memory_store import store


def semantic_router(state: PipelineState):
    """Route from semantic node:
    - If chitchat response generated → END
    - If document_listing intent → document_retriever node
    - If direct answer from history → formatter
    - Otherwise → clarification node
    """
    # Check if semantic_chitchat flag is set (chitchat response already generated)
    if hasattr(state, 'semantic_chitchat') and state.semantic_chitchat:
        return END
    
    # Document listing queries → fast SQL path (skips RAG + formatter)
    if state.task_type == "listing":
        return "document_retriever"
    
    if state.clarification_message and (not state.ambiguity_detected or not state.ambiguity_detected.ambiguous):
        return "formatter"

    return "clarification"


def clarification_router(state: PipelineState):
    """Route from clarification node:
    - If ambiguity detected → clarification_message set → END
    - Otherwise → continue to RAG
    """
    if state.clarification_message:
        return END
    return "rag"


def build_graph():

    builder = StateGraph(PipelineState)

    builder.add_node("semantic", semantic_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("rag", rag_node)
    # builder.add_node("evaluator", evaluator_node)
    builder.add_node("formatter", formatter_node)
    builder.add_node("document_retriever", document_retriever_node)

    builder.set_entry_point("semantic")
    
    # From semantic: route to END if chitchat, else to clarification
    builder.add_conditional_edges(
        "semantic",
        semantic_router,
        {
            "clarification": "clarification",
            "formatter": "formatter",
            "document_retriever": "document_retriever",
            END: END,
        },
    )

    # From clarification: route to END if clarification needed, else to RAG
    builder.add_conditional_edges(
        "clarification",
        clarification_router,
        {
            "rag": "rag",
            END: END,
        },
    )

    # builder.add_edge("rag", "evaluator")
    # builder.add_edge("evaluator", "formatter")
    builder.add_edge("rag", "formatter")
    builder.add_edge("formatter", END)

    # Document retriever goes directly to END (inline formatting, no formatter needed)
    builder.add_edge("document_retriever", END)
    
    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
    
