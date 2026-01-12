"""MongoDB Store for long-term memory (cross-session user facts)"""
from langgraph.store.base import BaseStore, Item
from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid
import config
from models import UserMemory


class MongoDBUserStore(BaseStore):
    """
    Custom MongoDB store for user memories.
    
    Stores long-term facts about users that persist across sessions:
    - Preferences: "User prefers detailed answers"
    - Facts: "User works in Marketing team"
    - Patterns: "User frequently asks about soap products"
    
    Namespace format: ("user", user_id)
    """
    
    def __init__(self, collection: Collection):
        self.collection = collection
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create indexes for efficient queries"""
        self.collection.create_index([("namespace", 1), ("key", 1)], unique=True)
        self.collection.create_index([("namespace", 1), ("value.type", 1)])
        self.collection.create_index([("namespace", 1), ("value.timestamp", -1)])
    
    def _namespace_to_str(self, namespace: Tuple[str, ...]) -> str:
        """Convert namespace tuple to string for storage"""
        return ":".join(namespace)
    
    def put(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any]) -> None:
        """Store a memory"""
        ns_str = self._namespace_to_str(namespace)
        self.collection.update_one(
            {"namespace": ns_str, "key": key},
            {
                "$set": {
                    "namespace": ns_str,
                    "key": key,
                    "value": value,
                    "updated_at": datetime.utcnow().isoformat()
                },
                "$setOnInsert": {
                    "created_at": datetime.utcnow().isoformat()
                }
            },
            upsert=True
        )
    
    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Item]:
        """Retrieve a specific memory"""
        ns_str = self._namespace_to_str(namespace)
        doc = self.collection.find_one({"namespace": ns_str, "key": key})
        if doc:
            return Item(
                namespace=namespace,
                key=doc["key"],
                value=doc["value"],
                created_at=doc.get("created_at"),
                updated_at=doc.get("updated_at")
            )
        return None
    
    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a memory"""
        ns_str = self._namespace_to_str(namespace)
        self.collection.delete_one({"namespace": ns_str, "key": key})
    
    def search(
        self,
        namespace: Tuple[str, ...],
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Item]:
        """Search memories in a namespace"""
        ns_str = self._namespace_to_str(namespace)
        
        mongo_filter = {"namespace": ns_str}
        
        # Add custom filters
        if filter:
            for k, v in filter.items():
                mongo_filter[f"value.{k}"] = v
        
        # Simple text search if query provided
        if query:
            mongo_filter["value.content"] = {"$regex": query, "$options": "i"}
        
        cursor = self.collection.find(mongo_filter).limit(limit).sort("value.timestamp", -1)
        
        items = []
        for doc in cursor:
            items.append(Item(
                namespace=namespace,
                key=doc["key"],
                value=doc["value"],
                created_at=doc.get("created_at"),
                updated_at=doc.get("updated_at")
            ))
        return items
    
    def list_namespaces(self, prefix: Optional[Tuple[str, ...]] = None) -> List[Tuple[str, ...]]:
        """List all namespaces"""
        pipeline = [{"$group": {"_id": "$namespace"}}]
        if prefix:
            prefix_str = self._namespace_to_str(prefix)
            pipeline.insert(0, {"$match": {"namespace": {"$regex": f"^{prefix_str}"}}})
        
        results = self.collection.aggregate(pipeline)
        return [tuple(doc["_id"].split(":")) for doc in results]
    
    def batch(self, operations: List[Tuple[str, Tuple[str, ...], str, Optional[Dict[str, Any]]]]) -> None:
        """Execute batch operations"""
        for op_type, namespace, key, value in operations:
            if op_type == "put":
                self.put(namespace, key, value)
            elif op_type == "delete":
                self.delete(namespace, key)


# Singleton instance
_store: Optional[MongoDBUserStore] = None


def get_store() -> MongoDBUserStore:
    """Get or create MongoDBUserStore singleton"""
    global _store
    if _store is None:
        from memory.checkpointer import get_mongo_client
        client = get_mongo_client()
        collection = client[config.MONGODB_DB_NAME][config.USER_MEMORIES_COLLECTION]
        _store = MongoDBUserStore(collection)
    return _store


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_user_namespace(user_id: str) -> Tuple[str, str]:
    """Get the namespace tuple for a user's memories"""
    return ("user", user_id)


def save_user_memory(
    user_id: str,
    memory_type: str,
    content: str,
    source: str = "automatic",
    confidence: float = 1.0
) -> str:
    """
    Save a memory about a user.
    
    Args:
        user_id: User identifier
        memory_type: "preference", "fact", or "pattern"
        content: The memory content
        source: "automatic" or "explicit"
        confidence: Confidence score (0-1)
    
    Returns:
        The key of the saved memory
    """
    store = get_store()
    namespace = get_user_namespace(user_id)
    key = str(uuid.uuid4())
    
    memory = UserMemory(
        type=memory_type,
        content=content,
        source=source,
        confidence=confidence
    )
    
    store.put(namespace, key, memory.model_dump())
    return key


def get_user_memories(
    user_id: str,
    memory_type: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Retrieve memories for a user.
    
    Args:
        user_id: User identifier
        memory_type: Optional filter by type
        limit: Maximum number of memories to return
    
    Returns:
        List of memory dictionaries
    """
    store = get_store()
    namespace = get_user_namespace(user_id)
    
    filter_dict = {"type": memory_type} if memory_type else None
    items = store.search(namespace, filter=filter_dict, limit=limit)
    
    return [item.value for item in items]


def extract_memories_from_conversation(
    user_id: str,
    messages: List[Any],
    rag_answer: str
) -> List[str]:
    """
    Extract and save memories from a conversation.
    
    This implements the "hybrid" approach:
    1. Explicit: User says "remember that..." or "my name is..."
    2. Automatic: Extract facts and preferences from conversation
    
    Args:
        user_id: User identifier
        messages: Conversation messages
        rag_answer: The final answer given
    
    Returns:
        List of memory keys that were saved
    """
    from langchain_openai import AzureChatOpenAI
    import config
    import json
    
    saved_keys = []
    
    # Get last few messages for context
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    
    # Convert messages to text
    conversation_text = ""
    for msg in recent_messages:
        role = getattr(msg, 'type', 'unknown')
        content = getattr(msg, 'content', str(msg))
        conversation_text += f"{role}: {content}\n"
    
    # Use LLM to extract memories
    llm = AzureChatOpenAI(
        deployment_name=config.AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0
    )
    
    extraction_prompt = f"""Analyze this conversation and extract any facts worth remembering about the user.

Conversation:
{conversation_text}

Extract ONLY if explicitly stated or strongly implied:
1. Preferences (how they like responses, topics they prefer)
2. Facts (their role, team, projects, name)
3. Patterns (topics they frequently ask about)

Return JSON array. If nothing to extract, return empty array [].

Format:
[
    {{"type": "preference", "content": "User prefers concise answers"}},
    {{"type": "fact", "content": "User works in Marketing team"}},
    {{"type": "pattern", "content": "User frequently asks about soap products"}}
]

Only include high-confidence extractions. Be conservative."""

    try:
        response = llm.invoke(extraction_prompt)
        content = response.content.strip()
        
        # Parse JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        memories = json.loads(content)
        
        for mem in memories:
            if isinstance(mem, dict) and "type" in mem and "content" in mem:
                key = save_user_memory(
                    user_id=user_id,
                    memory_type=mem["type"],
                    content=mem["content"],
                    source="automatic"
                )
                saved_keys.append(key)
                print(f"💾 Saved memory: [{mem['type']}] {mem['content']}")
    
    except Exception as e:
        print(f"⚠️ Memory extraction failed: {e}")
    
    return saved_keys


def handle_explicit_memory_command(user_id: str, user_message: str) -> Optional[str]:
    """
    Check if user is explicitly asking to remember something.
    
    Patterns detected:
    - "Remember that..."
    - "My name is..."
    - "I work at..."
    - "I prefer..."
    
    Returns:
        Memory key if saved, None otherwise
    """
    message_lower = user_message.lower()
    
    # Check for explicit remember commands
    remember_patterns = [
        ("remember that ", "fact"),
        ("remember:", "fact"),
        ("my name is ", "fact"),
        ("i work at ", "fact"),
        ("i work in ", "fact"),
        ("i am a ", "fact"),
        ("i'm a ", "fact"),
        ("i prefer ", "preference"),
        ("i like ", "preference"),
        ("i don't like ", "preference"),
    ]
    
    for pattern, mem_type in remember_patterns:
        if pattern in message_lower:
            # Extract the content after the pattern
            idx = message_lower.find(pattern)
            content = user_message[idx + len(pattern):].strip()
            
            # Clean up content
            if content.endswith("."):
                content = content[:-1]
            
            if len(content) > 5:  # Minimum meaningful content
                key = save_user_memory(
                    user_id=user_id,
                    memory_type=mem_type,
                    content=f"User: {content}" if not content.lower().startswith("user") else content,
                    source="explicit",
                    confidence=1.0
                )
                print(f"💾 Explicit memory saved: [{mem_type}] {content}")
                return key
    
    return None