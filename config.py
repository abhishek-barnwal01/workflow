"""Simple configuration"""
import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Azure AI Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
MAIN_DATA_INDEX_NAME = os.getenv("MAIN_DATA_INDEX_NAME")
SEMANTIC_INDEX_NAME = os.getenv("SEMANTIC_INDEX_NAME")

# Agent Config
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
CONFIDENCE_HIGH = float(os.getenv("CONFIDENCE_HIGH", "0.85"))
CONFIDENCE_MEDIUM = float(os.getenv("CONFIDENCE_MEDIUM", "0.65"))

# Azure OpenAI Embedding
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "qt328pp")

# Data Source — "local" uses PostgreSQL (metadata_gcpl), "dev" uses Databricks (silver_deterministic_data)
DATA_SOURCE = os.getenv("DATA_SOURCE", "local")

# Databricks (required when DATA_SOURCE="dev")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").replace("https://", "").replace("http://", "")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_TABLE = os.getenv("DATABRICKS_TABLE", "silver_deterministic_data")