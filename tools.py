"""Azure AI Search tool with hybrid search"""
from langchain_core.tools import tool  # Updated import for LangChain 1.x
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import config
import json


def get_embedding(text: str) -> list:
    """Get embedding for text using Azure OpenAI"""
    try:
        client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )
        
        response = client.embeddings.create(
            input=text,
            model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        
        return response.data[0].embedding
    except Exception as e:
        print(f"   ⚠️ Embedding error: {e}")
        return None


@tool
def azure_ai_search(query: str, index_type: str, top_k: int) -> str:
    """
    Search Azure AI Search indexes using hybrid search (keyword + vector).
    
    Args:
        query: Search query
        index_type: "main_data" or "semantic"
        top_k: Number of results (1-50)
    
    Returns:
        JSON with docs and metadata including content_id for deduplication
    """
    
    index_name = (
        config.MAIN_DATA_INDEX_NAME if index_type == "main_data"
        else config.SEMANTIC_INDEX_NAME
    )
    
    print(f"\n🔍 TOOL CALL: azure_ai_search (HYBRID)")
    print(f"   Query: {query}")
    print(f"   Index: {index_type} ({index_name})")
    print(f"   Top K: {top_k}")
    
    try:
        client = SearchClient(
            endpoint=config.AZURE_SEARCH_ENDPOINT,
            index_name=index_name,
            credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
        )
        
        # Get query embedding for vector search
        query_embedding = get_embedding(query)
        
        # Build search parameters
        search_kwargs = {
            "search_text": query,
            "top": min(top_k, 50),
            "select": ["content_text", "document_title", "content_path", "content_id"]
        }
        
        # Add vector search if embedding succeeded
        if query_embedding:
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=min(top_k, 50),
                fields="content_embedding"
            )
            search_kwargs["vector_queries"] = [vector_query]
            print(f"   ✓ Using hybrid search (keyword + vector)")
        else:
            print(f"   ⚠️ Using keyword-only search (embedding failed)")
        
        # Perform search
        results = client.search(**search_kwargs)
        
        docs = []
        scores = []
        for result in results:
            content = result.get("content_text", "")
            title = result.get("document_title", "")
            source = result.get("content_path", result.get("document_title", "unknown"))
            
            # Combine title and content
            full_content = f"{title}\n\n{content}" if title else content
            
            doc = {
                "content_id": result.get("content_id"),  # For deduplication
                "content": full_content[:1000],
                "score": result.get("@search.score", 0.0),
                "source": source
            }
            docs.append(doc)
            scores.append(doc["score"])
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        print(f"   ✓ Found {len(docs)} docs (avg score: {avg_score:.2f})")
        if docs:
            print(f"   📄 Top result: {docs[0]['source']} (score: {docs[0]['score']:.2f})")
            print(f"   📝 Preview: {docs[0]['content'][:100]}...")
        
        return json.dumps({
            "docs": docs,
            "metadata": {
                "returned": len(docs),
                "avg_score": round(avg_score, 2),
                "index": index_name,
                "query": query,
                "search_type": "hybrid" if query_embedding else "keyword",
                "doc_ids": [d["content_id"] for d in docs if d["content_id"]]  # For scratchpad
            }
        })
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "docs": [],
            "metadata": {
                "error": str(e),
                "index": index_name,
                "query": query
            }
        })