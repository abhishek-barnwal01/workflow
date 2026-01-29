from typing import Any, Dict
from models import PipelineState


def cache_orchestrator_node(state: PipelineState) -> Dict[str, Any]:
   """
   Entry point for all requests - handles intelligent cache routing
   """
   print("\n" + "="*70)
   print("🎯 CACHE ORCHESTRATOR")
   print("="*70)
   from cache_service import CacheService
   from tools import get_embedding
   user_query = state.user_query
   thread_id = getattr(state, "thread_id", "default")
   # Initialize cache service
   cache_service = CacheService(
       answer_cache=state.answer_cache,
       clarification_cache=state.clarification_cache,
       thread_id=thread_id
   )
   # Get embedding
   query_embedding = get_embedding(user_query)
   if not query_embedding:
       return {"cache_hit": False}
   # Perform intelligent lookup
   result = cache_service.lookup(user_query, query_embedding)
   # ━━━━ CACHE HIT ━━━━
   if result.cache_hit:
       from models import RAGOutput, RetrievedDoc, FormatterOutput
       # Reconstruct minimal output for formatter
       rag_output = RAGOutput(
           retrieved_docs=[],
           final_answer=result.answer,
           search_strategy=f"Cache hit ({result.source})",
           reasoning=f"Retrieved from {result.source} cache",
           total_searches=0
       )
       return {
           "cache_hit": True,
           "rag_output": rag_output.dict(),
           "answer_cache": state.answer_cache,
           "clarification_cache": state.clarification_cache
       }
   # ━━━━ NEEDS CLARIFICATION (WITH HISTORY) ━━━━
   if result.needs_clarification:
       if result.clarification_type == "previous_choices":
           # User asked this before - show their previous answers
           message = "You've asked this before! Which answer would you like?\n\n"
           for i, opt in enumerate(result.clarification_options, 1):
               message += f"**{i}. {opt['label']}**\n"
               message += f"   _{opt['preview']}..._\n\n"
           if result.allow_new_clarification:
               message += f"**{len(result.clarification_options)+1}. Choose a different option**"
           return {
               "cache_hit": False,
               "needs_clarification_from_cache": True,
               "clarification_message": message,
               "cached_clarification_options": result.clarification_options,
               "answer_cache": state.answer_cache,
               "clarification_cache": state.clarification_cache
           }
       elif result.clarification_type == "new_clarification":
           # We know it's ambiguous, but no answers cached yet
           # Skip semantic search, go directly to clarification
           return {
               "cache_hit": False,
               "needs_clarification_from_cache": True,
               "enriched_query": result.cached_enrichment,
               "ambiguity_detected": result.clarification_options,  # From cache
               "answer_cache": state.answer_cache,
               "clarification_cache": state.clarification_cache
           }
       
       if result.cache_hit:
        print("\n🚨 DEBUG: About to return cache_hit=True")
        return_dict = {
            "cache_hit": True,
            "rag_output": rag_output.dict(),
            "answer_cache": cache_service.answer_cache,
            "clarification_cache": cache_service.clarification_cache
        }
        print(f"   Return dict keys: {return_dict.keys()}")
        print(f"   cache_hit value: {return_dict['cache_hit']}")
        return return_dict
   # ━━━━ CACHE MISS - PROCEED TO PIPELINE ━━━━
   return {
       "cache_hit": False,
       "needs_clarification_from_cache": False,
       "answer_cache": state.answer_cache,
       "clarification_cache": state.clarification_cache
   }