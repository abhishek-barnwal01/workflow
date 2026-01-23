# clarification_node.py
from models import PipelineState
from typing import Dict, Any

def clarification_node(state: PipelineState) -> Dict[str, Any]:
    ambiguity = state.ambiguity_detected

    # If no ambiguity, do nothing
    if not ambiguity or not ambiguity.ambiguous:
        return {
            "messages": state.messages,  # keep history
            "clarification_message": None,
            "awaiting_clarification": False,  # Clear flag
            "previous_ambiguity": None,  # Clear previous ambiguity
            "enriched_query": state.enriched_query,
            "ambiguity_detected": state.ambiguity_detected,
        }

    # Build the clarification message
    options_text = "\n".join([f"{i+1}. {opt.label}" for i, opt in enumerate(ambiguity.options)])
    message = f"I found multiple {ambiguity.entity}. Please clarify which one you mean:\n{options_text}\nYou can also reply 'ALL' to select all options."

    print(f"\n📌 Clarification Node sending message:")
    print(f"   Entity: {ambiguity.entity}")
    print(f"   Options: {len(ambiguity.options)}")
    for i, opt in enumerate(ambiguity.options[:5]):
        print(f"      {i+1}. {opt.label}")
    if len(ambiguity.options) > 5:
        print(f"      ... and {len(ambiguity.options) - 5} more")
    print(f"\n   Setting awaiting_clarification = True")
    print(f"   Storing previous_ambiguity for next turn")

    # Append AI clarification to messages so it gets stored
    messages = state.messages + [{"type": "assistant", "content": message}]

    return {
        "messages": messages,
        "clarification_message": message,
        "awaiting_clarification": True,  # Set flag to track clarification state
        "previous_ambiguity": ambiguity,  # Store ambiguity for next turn
        "enriched_query": state.enriched_query,
        "ambiguity_detected": state.ambiguity_detected,
    }
