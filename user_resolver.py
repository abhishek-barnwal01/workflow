"""Resolve LibreChat user ObjectId → dataAccess via MongoDB.

Queries MongoDB directly on every call — no caching — so access changes
made through the LibreChat admin UI take effect on the very next request,
identical to how the built-in AzureAISearch tool works.
"""

from typing import Optional

_client = None
_db = None


def _get_collection():
    global _client, _db
    if _client is None:
        from config import MONGO_URI
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure
        try:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            db_name = MONGO_URI.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "LibreChat"
            _db = _client[db_name]
        except ConnectionFailure as e:
            print(f"⚠️  MongoDB connection failed: {e}")
            _client = None
    return _db["users"] if _db is not None else None


def resolve_data_access(user_id: str) -> Optional[dict]:
    """Return the dataAccess dict for a LibreChat user ObjectId.

    Queries MongoDB on every call so admin changes take effect immediately.
    Returns None if the user has unrestricted access (no dataAccess field set).
    """
    if not user_id or user_id == "anonymous":
        return None  # anonymous = full access

    try:
        from bson import ObjectId
        from bson.errors import InvalidId
        col = _get_collection()
        if col is None:
            return None
        try:
            oid = ObjectId(user_id)
            doc = col.find_one({"_id": oid}, {"dataAccess": 1})
        except InvalidId:
            doc = col.find_one({"email": user_id.lower()}, {"dataAccess": 1})
        if doc:
            result = doc.get("dataAccess") or None
            print(f"🔒 dataAccess for '{user_id}': {result}")
            return result
    except Exception as e:
        print(f"⚠️  Could not resolve dataAccess for '{user_id}': {e}")

    return None
