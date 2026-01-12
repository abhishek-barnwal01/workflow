"""MongoDB Checkpointer for short-term memory (session state)"""
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from typing import Optional
import config


_client: Optional[MongoClient] = None
_checkpointer: Optional[MongoDBSaver] = None


def get_mongo_client() -> MongoClient:
    """Get or create MongoDB client singleton"""
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI)
    return _client


def get_checkpointer() -> MongoDBSaver:
    """
    Get or create MongoDBSaver singleton.
    
    The checkpointer automatically persists:
    - Chat history (messages)
    - Agent scratchpad (search_history, retrieved_doc_ids, iteration_feedback)
    - All other PipelineState fields
    
    Data is keyed by thread_id (session_id from LibreChat).
    """
    global _checkpointer
    if _checkpointer is None:
        client = get_mongo_client()
        _checkpointer = MongoDBSaver(
            client,
            db_name=config.MONGODB_DB_NAME,
            collection_name=config.LANGGRAPH_CHECKPOINTS_COLLECTION
        )
    return _checkpointer


def close_connections():
    """Close MongoDB connections (call on app shutdown)"""
    global _client, _checkpointer
    if _client is not None:
        _client.close()
        _client = None
        _checkpointer = None