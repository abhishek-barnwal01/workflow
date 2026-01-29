from typing import Dict, List, Optional, Any
from datetime import datetime
from cache_utils import cosine_similarity, create_fingerprint
from tools import get_embedding

class CacheService:
   """
   Centralized cache management with intelligent routing
   """
   def __init__(self, answer_cache: Dict, clarification_cache: Dict, thread_id: str):
       self.answer_cache = answer_cache
       self.clarification_cache = clarification_cache
       self.thread_id = thread_id
   def lookup(self, query: str, query_embedding: List[float]):
       """
       Single entry point for all cache lookups
       Returns:
           CacheLookupResult with cache_hit, answer, needs_clarification, etc.
       """
       result = CacheLookupResult(cache_hit=False)
       # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       # PHASE 1: Exact Answer Match (High Confidence)
       # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       exact_match = self._find_exact_answer(query, query_embedding, threshold=0.75)
       if exact_match:
           print(f"   ⚡ TIER 1 HIT: Exact answer match (similarity: {exact_match.similarity:.3f})")
           result.cache_hit = True
           result.answer = exact_match.entry.get("final_answer", "")
           result.source = "exact_match"
           return result
       # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       # PHASE 2: Check Clarification History
       # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       clarification_cached = self._find_clarification_info(query, query_embedding)
       if clarification_cached and clarification_cached.entry.get("ambiguity_detected", {}).get("ambiguous"):
           print(f"   🔀 TIER 2 HIT: Known ambiguous query")
           # Find all previously answered variants
           variants = self._find_answer_variants(query, query_embedding)
           if variants:
               print(f"   📚 Found {len(variants)} previous answers to this question")
               # Show user their previous choices
               result.needs_clarification = True
               result.clarification_type = "previous_choices"
               result.clarification_options = [
                   {
                       "label": self._extract_clarification_label(v.entry.get("clarification_path")),
                       "preview": v.entry.get("final_answer", "")[:100],
                       "cached_entry": v.entry
                   }
                   for v in variants
               ]
               result.allow_new_clarification = True
               return result
           else:
               # Ambiguity detected but no answers cached yet
               print(f"   🔀 Using cached ambiguity detection")
               result.needs_clarification = True
               result.clarification_type = "new_clarification"
               result.clarification_options = clarification_cached.entry.get("ambiguity_detected", {})
               result.cached_enrichment = clarification_cached.entry.get("enriched_query", "")
               return result
       # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       # PHASE 3: Fuzzy Answer Match (Lower Confidence)
       # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       if not clarification_cached or not clarification_cached.entry.get("ambiguity_detected", {}).get("ambiguous"):
           fuzzy_match = self._find_exact_answer(query, query_embedding, threshold=0.60)
           if fuzzy_match:
               print(f"   ⚡ TIER 1 HIT: Fuzzy answer match (similarity: {fuzzy_match.similarity:.3f})")
               result.cache_hit = True
               result.answer = fuzzy_match.entry.get("final_answer", "")
               result.source = "fuzzy_match"
               result.confidence = fuzzy_match.similarity
               return result
       # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       # PHASE 4: Cache Miss
       # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       print(f"   ❌ CACHE MISS: Proceeding to full pipeline")
       return result
   def store_answer(
       self,
       query: str,
       original_query: str,
       query_embedding: List[float],
       enriched_query: str,
       final_answer: str,
       retrieved_docs: List[Dict],
       clarification_path: Optional[List[str]] = None
   ):
       """Store complete answer"""
       fingerprint = create_fingerprint(enriched_query)
       entry = {
           "query_text": enriched_query,
           "original_query": original_query,
           "query_embedding": query_embedding,
           "enriched_query": enriched_query,
           "final_answer": final_answer,
           "retrieved_docs": retrieved_docs,
           "clarification_path": clarification_path,
           "timestamp": datetime.now().isoformat(),
           "thread_id": self.thread_id,
           "hit_count": 0
       }
       self.answer_cache[fingerprint] = entry
       # Also store under original query fingerprint if different
       original_fingerprint = create_fingerprint(original_query)
       if original_fingerprint != fingerprint:
           self.answer_cache[original_fingerprint] = entry.copy()
           print(f"   💾 Cached under both original and enriched queries")
       else:
           print(f"   💾 Cached under single query")
       self._evict_if_needed(self.answer_cache)
   def store_clarification(
       self,
       query: str,
       query_embedding: List[float],
       enriched_query: str,
       ambiguity_detected: Any,
       domain_context: Optional[Dict]
   ):
       """Store clarification/ambiguity info"""
       fingerprint = create_fingerprint(query)
       entry = {
           "query_text": query,
           "original_query": query,
           "query_embedding": query_embedding,
           "enriched_query": enriched_query,
           "ambiguity_detected": ambiguity_detected.model_dump() if hasattr(ambiguity_detected, 'model_dump') else ambiguity_detected,
           "domain_context": domain_context,
           "timestamp": datetime.now().isoformat(),
           "thread_id": self.thread_id,
           "hit_count": 0
       }
       self.clarification_cache[fingerprint] = entry
       self._evict_if_needed(self.clarification_cache)
       print(f"   💾 Stored clarification info")
   def _find_exact_answer(self, query: str, query_embedding: List[float], threshold: float):
       """Find answer with similarity above threshold"""
       best_match = None
       best_score = 0.0
       for fingerprint, entry in self.answer_cache.items():
           if entry.get("thread_id") != self.thread_id:
               continue
           cached_embedding = entry.get("query_embedding", [])
           if not cached_embedding:
               continue
           similarity = cosine_similarity(query_embedding, cached_embedding)
           if similarity > best_score and similarity >= threshold:
               best_score = similarity
               best_match = entry.copy()
               best_match["similarity_score"] = similarity
               best_match["matched_query"] = entry.get("query_text", "")
       if best_match:
           return CacheMatch(entry=best_match, similarity=best_score)
       return None
   def _find_clarification_info(self, query: str, query_embedding: List[float]):
       """Check if we have clarification info for this query"""
       fingerprint = create_fingerprint(query)
       if fingerprint in self.clarification_cache:
           entry = self.clarification_cache[fingerprint]
           if entry.get("thread_id") == self.thread_id:
               return CacheMatch(entry=entry, similarity=1.0)
       # Fuzzy match on clarification
       for fp, entry in self.clarification_cache.items():
           if entry.get("thread_id") != self.thread_id:
               continue
           cached_embedding = entry.get("query_embedding", [])
           if not cached_embedding:
               continue
           similarity = cosine_similarity(query_embedding, cached_embedding)
           if similarity > 0.85:
               return CacheMatch(entry=entry, similarity=similarity)
       return None
   def _find_answer_variants(self, query: str, query_embedding: List[float]):
       """Find all cached answers that stem from same original query"""
       variants = []
       for fingerprint, entry in self.answer_cache.items():
           if entry.get("thread_id") != self.thread_id:
               continue
           original_query = entry.get("original_query", "")
           if not original_query:
               continue
           original_embedding = get_embedding(original_query)
           if not original_embedding:
               continue
           original_similarity = cosine_similarity(query_embedding, original_embedding)
           if original_similarity > 0.85:
               variants.append(CacheMatch(entry=entry, similarity=original_similarity))
       return variants
   def _extract_clarification_label(self, clarification_path: Optional[List[str]]) -> str:
       """Extract human-readable label from clarification path"""
       if not clarification_path:
           return "Unknown"
       # clarification_path format: ["brand:RVT", "period:2022"]
       labels = [path.split(":")[-1] for path in clarification_path]
       return " - ".join(labels)
   def _evict_if_needed(self, cache: Dict, max_size: int = 100):
       """LRU eviction"""
       if len(cache) > max_size:
           # Remove oldest with lowest hit count
           entries = [(k, v) for k, v in cache.items()]
           if entries:
               least_used = min(entries, key=lambda x: (x[1].get("hit_count", 0), x[1].get("timestamp", "")))
               del cache[least_used[0]]
               print(f"   🗑️ Evicted cache entry: {least_used[0][:8]}...")

class CacheLookupResult:
   """Result from cache lookup"""
   def __init__(self, cache_hit: bool = False):
       self.cache_hit = cache_hit
       self.answer: Optional[str] = None
       self.needs_clarification: bool = False
       self.clarification_type: Optional[str] = None
       self.clarification_options: Optional[List] = None
       self.allow_new_clarification: bool = False
       self.cached_enrichment: Optional[str] = None
       self.source: Optional[str] = None
       self.confidence: float = 1.0

class CacheMatch:
   """Simple cache match wrapper"""
   def __init__(self, entry: Dict, similarity: float):
       self.entry = entry
       self.similarity = similarity