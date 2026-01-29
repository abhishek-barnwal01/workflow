from models import PipelineState
from typing import Dict, Any
def clarification_node(state: PipelineState) -> Dict[str, Any]:
   """
   Clarification node - handles ambiguity detection and user clarification
   Two modes:
   1. Fresh ambiguity - show options from semantic search
   2. Cached variants - show user their previous choices
   """
   print("\n" + "="*70)
   print("🔀 CLARIFICATION NODE")
   print("="*70)
   ambiguity = state.ambiguity_detected
   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   # CASE 1: NO AMBIGUITY - Pass through
   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   if not ambiguity or not ambiguity.ambiguous:
       print("   ℹ️  No ambiguity detected - passing through")
       return {
           "messages": state.messages,
           "clarification_message": None,
           "awaiting_clarification": False,
           "previous_ambiguity": None,
           "enriched_query": state.enriched_query,
           "ambiguity_detected": state.ambiguity_detected,
           # ✅ Pass cache state through
           "answer_cache": getattr(state, "answer_cache", {}),
           "clarification_cache": getattr(state, "clarification_cache", {}),
       }
   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   # CASE 2: USER IS RESPONDING TO PREVIOUS CLARIFICATION
   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   if state.awaiting_clarification and state.previous_ambiguity:
       print("\n   🔄 Processing user's clarification response")
       user_response = state.user_query.strip()
       previous_options = state.previous_ambiguity.options
       # Parse user's choice
       chosen_option = None
       # Try numeric input (e.g., "1", "2")
       if user_response.isdigit():
           idx = int(user_response) - 1
           if 0 <= idx < len(previous_options):
               chosen_option = previous_options[idx]
               print(f"   ✅ User selected option #{user_response}: {chosen_option.label}")
       # Try text match (e.g., "RVT", "Lux")
       else:
           user_lower = user_response.lower()
           for opt in previous_options:
               if opt.label.lower() == user_lower or opt.value.lower() == user_lower:
                   chosen_option = opt
                   print(f"   ✅ User selected by name: {chosen_option.label}")
                   break
       # Handle "ALL" selection
       if user_response.upper() == "ALL":
           print("   ✅ User selected ALL options")
           chosen_option = "ALL"
       if not chosen_option:
           print(f"   ⚠️ Could not parse user response: '{user_response}'")
           print("   ℹ️ Defaulting to first option")
           chosen_option = previous_options[0] if previous_options else None
       # ✅ Track what was chosen
       return {
           "messages": state.messages,
           "clarification_message": None,
           "awaiting_clarification": False,
           "previous_ambiguity": None,
           # ✅ Store clarification choice for RAG node
           "clarification_chosen": chosen_option.value if chosen_option != "ALL" and chosen_option else "ALL",
           "clarification_entity": state.previous_ambiguity.entity,
           "enriched_query": state.enriched_query,
           "ambiguity_detected": state.ambiguity_detected,
           # ✅ Pass cache state through
           "answer_cache": getattr(state, "answer_cache", {}),
           "clarification_cache": getattr(state, "clarification_cache", {}),
       }
   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   # CASE 3: CHECK FOR CACHED CLARIFICATION VARIANTS
   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   # ✅ NEW: Check if user has answered this before
   cached_variants = getattr(state, "cached_clarification_options", [])
   if cached_variants and len(cached_variants) > 0:
       print(f"\n   📚 Found {len(cached_variants)} cached variants")
       print("   ℹ️ User has answered this question before")
       # Build message with previous choices
       message = f"You've asked about {ambiguity.entity} before! Which answer would you like?\n\n"
       for i, variant in enumerate(cached_variants, 1):
           label = variant.get("label", "Unknown")
           preview = variant.get("preview", "")[:80]
           message += f"**{i}. {label}**\n   _{preview}_...\n\n"
       # Add option for new clarification
       message += f"**{len(cached_variants) + 1}. Choose a different {ambiguity.entity}**\n"
       print(f"\n   📝 Clarification message with history:")
       print(f"   {message[:200]}...")
       # Append to messages
       messages = state.messages + [{"type": "assistant", "content": message}]
       return {
           "messages": messages,
           "clarification_message": message,
           "awaiting_clarification": True,
           "previous_ambiguity": ambiguity,
           "enriched_query": state.enriched_query,
           "ambiguity_detected": state.ambiguity_detected,
           # ✅ Track that we're showing cached options
           "showing_cached_variants": True,
           "cached_clarification_options": cached_variants,
           # ✅ Pass cache state through
           "answer_cache": getattr(state, "answer_cache", {}),
           "clarification_cache": getattr(state, "clarification_cache", {}),
       }
   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   # CASE 4: FRESH CLARIFICATION - Show options from semantic search
   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   print(f"\n   ❓ Fresh clarification needed")
   print(f"   Entity: {ambiguity.entity}")
   print(f"   Options: {len(ambiguity.options)}")
   # Build the clarification message
   options_text = "\n".join([f"{i+1}. {opt.label}" for i, opt in enumerate(ambiguity.options)])
   message = f"I found multiple {ambiguity.entity}. Please clarify which one you mean:\n\n{options_text}\n\nYou can also reply 'ALL' to select all options."
   print(f"\n   📝 Clarification options:")
   for i, opt in enumerate(ambiguity.options[:5]):
       print(f"      {i+1}. {opt.label}")
   if len(ambiguity.options) > 5:
       print(f"      ... and {len(ambiguity.options) - 5} more")
   print(f"\n   ℹ️ Setting awaiting_clarification = True")
   print(f"   ℹ️ Storing previous_ambiguity for next turn")
   # Append AI clarification to messages
   messages = state.messages + [{"type": "assistant", "content": message}]
   return {
       "messages": messages,
       "clarification_message": message,
       "awaiting_clarification": True,
       "previous_ambiguity": ambiguity,
       "enriched_query": state.enriched_query,
       "ambiguity_detected": state.ambiguity_detected,
       # ✅ Pass cache state through
       "answer_cache": getattr(state, "answer_cache", {}),
       "clarification_cache": getattr(state, "clarification_cache", {}),
   }

# # clarification_node.py
# from models import PipelineState
# from typing import Dict, Any

# def clarification_node(state: PipelineState) -> Dict[str, Any]:
#     ambiguity = state.ambiguity_detected

#     # If no ambiguity, do nothing
#     if not ambiguity or not ambiguity.ambiguous:
#         return {
#             "messages": state.messages,  # keep history
#             "clarification_message": None,
#             "awaiting_clarification": False,  # Clear flag
#             "previous_ambiguity": None,  # Clear previous ambiguity
#             "enriched_query": state.enriched_query,
#             "ambiguity_detected": state.ambiguity_detected,
#         }

#     # Build the clarification message
#     options_text = "\n".join([f"{i+1}. {opt.label}" for i, opt in enumerate(ambiguity.options)])
#     message = f"I found multiple {ambiguity.entity}. Please clarify which one you mean:\n{options_text}\nYou can also reply 'ALL' to select all options."

#     print(f"\n📌 Clarification Node sending message:")
#     print(f"   Entity: {ambiguity.entity}")
#     print(f"   Options: {len(ambiguity.options)}")
#     for i, opt in enumerate(ambiguity.options[:5]):
#         print(f"      {i+1}. {opt.label}")
#     if len(ambiguity.options) > 5:
#         print(f"      ... and {len(ambiguity.options) - 5} more")
#     print(f"\n   Setting awaiting_clarification = True")
#     print(f"   Storing previous_ambiguity for next turn")

#     # Append AI clarification to messages so it gets stored
#     messages = state.messages + [{"type": "assistant", "content": message}]

#     return {
#         "messages": messages,
#         "clarification_message": message,
#         "awaiting_clarification": True,  # Set flag to track clarification state
#         "previous_ambiguity": ambiguity,  # Store ambiguity for next turn
#         "enriched_query": state.enriched_query,
#         "ambiguity_detected": state.ambiguity_detected,
#     }
