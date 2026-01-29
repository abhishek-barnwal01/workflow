"""Cache utilities for RAG query caching with semantic similarity"""
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

def create_fingerprint(query: str, domain: str = "") -> str:
   """Create deterministic hash for query"""
   normalized = f"{query.lower().strip()}|{domain}".encode('utf-8')
   return hashlib.sha256(normalized).hexdigest()[:16]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
   """Calculate cosine similarity between two embeddings"""
   if not vec1 or not vec2:
       return 0.0
   v1 = np.array(vec1)
   v2 = np.array(vec2)
   dot = np.dot(v1, v2)
   norm = np.linalg.norm(v1) * np.linalg.norm(v2)
   return float(dot / norm) if norm > 0 else 0.0

def cache_lookup(
   query: str,
   query_embedding: List[float],
   cache: Dict[str, Dict],
   threshold: float = 0.80
) -> Optional[Dict]:
   """Search cache for similar queries using semantic similarity"""
   if not cache or not query_embedding:
       return None
   best_match = None
   best_score = 0.0
   for fingerprint, cached in cache.items():
       cached_embedding = cached.get("query_embedding", [])
       if not cached_embedding:
           continue
       score = cosine_similarity(query_embedding, cached_embedding)
       if score > best_score and score >= threshold:
           best_score = score
           best_match = cached.copy()
           best_match["similarity_score"] = score
   if best_match:
       print(f"   ✅ CACHE HIT (similarity: {best_score:.3f})")
       best_match["hit_count"] = best_match.get("hit_count", 0) + 1
   else:
       print(f"   ❌ CACHE MISS (best: {best_score:.3f}, threshold: {threshold})")
   return best_match

def cache_store(
   query: str,
   query_embedding: List[float],
   retrieved_docs: List[Dict],
   final_answer: str,
   cache: Dict[str, Dict],
   max_size: int = 100
) -> Dict[str, Dict]:
   """Store new result in cache with LRU eviction"""
   if not query_embedding:
       print("   ⚠️ Skipping cache store - no embedding")
       return cache
   fingerprint = create_fingerprint(query)
   # Create new cache entry
   cache[fingerprint] = {
       "query_fingerprint": fingerprint,
       "query_text": query,
       "query_embedding": query_embedding,
       "retrieved_docs": retrieved_docs,
       "final_answer": final_answer,
       "timestamp": datetime.now().isoformat(),
       "hit_count": 0,
       "similarity_score": 1.0
   }
   # LRU eviction if cache exceeds max size
   if len(cache) > max_size:
       # Remove entry with lowest hit count
       least_used = min(cache.items(), key=lambda x: (x[1].get("hit_count", 0), x[1].get("timestamp", "")))
       del cache[least_used[0]]
       print(f"   🗑️ Evicted cache entry: {least_used[0][:8]}... (hits: {least_used[1].get('hit_count', 0)})")
   print(f"   💾 Stored in cache (total: {len(cache)} entries)")
   return cache

def should_bypass_cache(query: str) -> bool:
   """Check if query contains keywords that should force fresh retrieval"""
   bypass_keywords = ["latest", "update", "refresh", "current", "recent", "new", "today"]
   query_lower = query.lower()
   for keyword in bypass_keywords:
       if keyword in query_lower:
           print(f"   🔄 Bypassing cache due to keyword: '{keyword}'")
           return True
   return False