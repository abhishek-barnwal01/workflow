"""Clarifier Node - Handles ambiguous queries"""
from typing import Dict, Any
from models import PipelineState, AmbiguityInfo


def clarifier_node(state: PipelineState) -> Dict[str, Any]:
    """
    Clarifier node - formats clarification request for user.
    
    Responsibilities:
    1. Format ambiguity options for user
    2. Create user-friendly clarification message
    3. Set needs_clarification flag
    
    Inputs from state:
    - ambiguity_detected: Ambiguity info from semantic node
    - user_query: Original query
    
    Outputs to state:
    - final_response: Clarification question
    - needs_clarification: True
    - clarification_options: List of options
    """
    
    print("\n" + "="*70)
    print("❓ CLARIFIER NODE")
    print("="*70)
    
    user_query = state["user_query"]
    ambiguity_data = state.get("ambiguity_detected", {})
    
    # Parse ambiguity info
    if isinstance(ambiguity_data, dict):
        ambiguity = AmbiguityInfo(**ambiguity_data)
    else:
        ambiguity = ambiguity_data
    
    entity = ambiguity.entity or "items"
    options = ambiguity.options or []
    reason = ambiguity.reason or "Multiple options were found."
    
    print(f"Query: {user_query}")
    print(f"Ambiguous Entity: {entity}")
    print(f"Options Found: {len(options)}")
    
    # Format clarification message
    clarification = f"I found multiple {entity} that could match your query. Which one are you interested in?\n\n"
    
    for i, opt in enumerate(options, 1):
        label = opt.label if hasattr(opt, 'label') else opt.get('label', f'Option {i}')
        clarification += f"{i}. {label}\n"
    
    clarification += "\nYou can also say **'ALL'** to include all options, or provide more context to help me understand."
    
    print(f"\n📋 Clarification Message:")
    print(clarification)
    
    # Format options for response
    options_list = []
    for opt in options:
        if hasattr(opt, 'model_dump'):
            options_list.append(opt.model_dump())
        elif isinstance(opt, dict):
            options_list.append(opt)
        else:
            options_list.append({"label": str(opt), "value": str(opt).lower().replace(" ", "_")})
    
    return {
        "final_response": clarification,
        "needs_clarification": True,
        "clarification_options": options_list,
    }