"""Configuration for RAG Pipeline with LangGraph Memory"""
import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Azure AI Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
MAIN_DATA_INDEX_NAME = os.getenv("MAIN_DATA_INDEX_NAME")
SEMANTIC_INDEX_NAME = os.getenv("SEMANTIC_INDEX_NAME")

# Azure OpenAI Embedding
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

# Agent Config
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
CONFIDENCE_HIGH = float(os.getenv("CONFIDENCE_HIGH", "0.85"))
CONFIDENCE_MEDIUM = float(os.getenv("CONFIDENCE_MEDIUM", "0.65"))

# MongoDB (LangGraph Memory)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "rag_pipeline")

# LangGraph Collections
LANGGRAPH_CHECKPOINTS_COLLECTION = os.getenv("LANGGRAPH_CHECKPOINTS_COLLECTION", "langgraph_checkpoints")
LANGGRAPH_WRITES_COLLECTION = os.getenv("LANGGRAPH_WRITES_COLLECTION", "langgraph_writes")
USER_MEMORIES_COLLECTION = os.getenv("USER_MEMORIES_COLLECTION", "user_memories")