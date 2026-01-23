# clarification_node.py
from models import PipelineState
from typing import Dict, Any
from models import PipelineState

def clarification_node(state: PipelineState) -> Dict[str, Any]:
    ambiguity = state.ambiguity_detected

    # If no ambiguity, do nothing
    if not ambiguity or not ambiguity.ambiguous:
        return {
            "messages": state.messages,  # keep history
            "clarification_message": None,
            "enriched_query": state.enriched_query,
            "ambiguity_detected": state.ambiguity_detected,
        }

    # Build the clarification message
    options_text = "\n".join([f"{i+1}. {opt.label}" for i, opt in enumerate(ambiguity.options)])
    message = f"I found multiple {ambiguity.entity}. Please clarify which one you mean:\n{options_text}\nYou can also reply 'ALL' to select all options."

    print(f"\n📌 Clarification Node sending message:\n{message}")

    # Append AI clarification to messages so it gets stored
    messages = state.messages + [{"type": "assistant", "content": message}]

    return {
        "messages": messages,
        "clarification_message": message,
        "enriched_query": state.enriched_query,
        "ambiguity_detected": state.ambiguity_detected,
    }
