# clarification_node.py
from models import PipelineState
from typing import Dict, Any
from langchain_core.messages import AIMessage

_MAX_OPTIONS = 5  # cap options shown to user to avoid overwhelming UI


def clarification_node(state: PipelineState) -> Dict[str, Any]:
    ambiguity = state.ambiguity_detected

    # If no ambiguity, pass through to RAG
    if not ambiguity or not ambiguity.ambiguous:
        return {
            "messages": state.messages,
            "clarification_message": None,
            "awaiting_clarification": False,
            "previous_ambiguity": None,
            "enriched_query": state.enriched_query,
            "ambiguity_detected": state.ambiguity_detected,
        }

    # Limit options so UI doesn't show hundreds of choices
    options = ambiguity.options[:_MAX_OPTIONS]
    options_text = "\n".join([f"{i+1}. {opt.label}" for i, opt in enumerate(options)])
    truncation_note = (
        f"\n_(showing {_MAX_OPTIONS} of {len(ambiguity.options)} — try a more specific query to narrow results)_"
        if len(ambiguity.options) > _MAX_OPTIONS else ""
    )
    message = (
        f"I found multiple {ambiguity.entity}. Please clarify which one you mean:\n"
        f"{options_text}{truncation_note}\n"
        f"You can also reply 'ALL' to select all options."
    )

    print(f"\n📌 Clarification Node sending message:")
    print(f"   Entity: {ambiguity.entity}")
    print(f"   Options shown: {len(options)} of {len(ambiguity.options)}")
    for i, opt in enumerate(options):
        print(f"      {i+1}. {opt.label}")
    print(f"\n   Setting awaiting_clarification = True")
    print(f"   Storing previous_ambiguity for next turn")

    # Use AIMessage (not a plain dict) so LangGraph/LangChain serialization works correctly
    messages = state.messages + [AIMessage(content=message)]

    return {
        "messages": messages,
        "clarification_message": message,
        "awaiting_clarification": True,
        "previous_ambiguity": ambiguity,
        "enriched_query": state.enriched_query,
        "ambiguity_detected": state.ambiguity_detected,
    }
