"""Azure AI Search tool with hybrid search and advanced filtering/faceting"""
from langchain.tools import tool
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import config
import json
from typing import Optional


# Fields to select per index type
MAIN_DATA_SELECT_FIELDS = [
    "content_id",
    "text_document_id",
    "document_title",
    "image_document_id",
    "content_text",
    "content_path",
    "locationMetadata",
    "file_category_ai",
    "product_category_ai",
    "brand_ai",
    "sub_brand_ai",
]

SEMANTIC_SELECT_FIELDS = [
    "content_text",
    "document_title",
    "content_path",
    "content_id",
]


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
def azure_ai_search(
    query: str,
    index_type: str,
    top_k: int,
    filter: Optional[str] = None,
    facets: Optional[list] = None,
    skip: Optional[int] = None,
    select_fields: Optional[str] = None,
) -> str:
    """
    Search Azure AI Search indexes using hybrid search (keyword + vector).
    Supports filtering, faceting, and pagination for advanced document discovery.

    Args:
        query: Search query text. Use "*" to match all documents when using filters/facets only.
        index_type: "main_data" or "semantic". main_data has extra fields like locationMetadata, file_category_ai, brand_ai etc.
        top_k: Number of results (1-50).
        filter: Optional OData filter expression. Examples:
            - "file_category_ai eq 'Usage/Attitude (U&A)'"
            - "locationMetadata/pageNumber eq 6"
            - "document_title eq 'Report.pdf' and locationMetadata/pageNumber eq 6"
            - "brand_ai eq 'Lux'"
            - "product_category_ai eq 'Soaps'"
            - Combine with "and" / "or"
        facets: Optional list of facetable field names for aggregation/counting. Examples:
            - ["document_title,count:1000"] to list/count unique documents
            - ["file_category_ai,count:100"] to list categories
            - ["brand_ai,count:100"] to list brands
            - ["product_category_ai,count:100"] to list product categories
            - Add ",count:N" to get up to N unique values (default is only 10)
        skip: Optional number of results to skip for pagination (default: 0).
        select_fields: Optional comma-separated fields to return. Overrides default field selection.
            Example: "document_title,text_document_id,content_path"

    Returns:
        JSON with docs, facets, pagination info, and metadata.
    """

    index_name = (
        config.MAIN_DATA_INDEX_NAME if index_type == "main_data"
        else config.SEMANTIC_INDEX_NAME
    )

    print(f"\n🔍 TOOL CALL: azure_ai_search (HYBRID)")
    print(f"   Query: {query}")
    print(f"   Index: {index_type} ({index_name})")
    print(f"   Top K: {top_k}")
    if filter:
        print(f"   Filter: {filter}")
    if facets:
        print(f"   Facets: {facets}")
    if skip:
        print(f"   Skip: {skip}")
    if select_fields:
        print(f"   Select Fields: {select_fields}")

    try:
        client = SearchClient(
            endpoint=config.AZURE_SEARCH_ENDPOINT,
            index_name=index_name,
            credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
        )

        # Determine select fields
        if select_fields:
            selected = [f.strip() for f in select_fields.split(",")]
        elif index_type == "main_data":
            selected = MAIN_DATA_SELECT_FIELDS
        else:
            selected = SEMANTIC_SELECT_FIELDS

        # Build search parameters
        search_kwargs = {
            "search_text": query,
            "top": min(top_k, 50),
            "select": selected,
            "include_total_count": True,
        }

        # Add optional parameters
        if filter:
            search_kwargs["filter"] = filter
        if facets and isinstance(facets, list) and len(facets) > 0:
            search_kwargs["facets"] = facets
        if skip and isinstance(skip, int):
            search_kwargs["skip"] = skip

        # Get query embedding for vector search (skip for wildcard queries)
        query_embedding = None
        if query != "*":
            query_embedding = get_embedding(query)

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
            if query == "*":
                print(f"   ✓ Using wildcard search with filters/facets")
            else:
                print(f"   ⚠️ Using keyword-only search (embedding failed)")

        # Perform search
        results = client.search(**search_kwargs)

        # Build enhanced response
        response = {
            "docs": [],
            "totalCount": 0,
            "returnedCount": 0,
            "skip": skip or 0,
            "hasMoreResults": False,
        }

        docs = []
        scores = []
        unique_doc_ids = set()
        unique_doc_titles = set()

        for result in results:
            content = result.get("content_text", "")
            title = result.get("document_title", "")
            source = result.get("content_path", result.get("document_title", "unknown"))

            # Combine title and content
            full_content = f"{title}\n\n{content}" if title else content

            doc = {
                "id": result.get("content_id"),
                "content": full_content[:1000],
                "score": result.get("@search.score", 0.0),
                "source": source,
            }

            # Add main_data index extra fields
            if index_type == "main_data":
                doc["document_title"] = title
                doc["text_document_id"] = result.get("text_document_id", "")
                doc["image_document_id"] = result.get("image_document_id", "")
                doc["file_category_ai"] = result.get("file_category_ai", "")
                doc["product_category_ai"] = result.get("product_category_ai", "")
                doc["brand_ai"] = result.get("brand_ai", "")
                doc["sub_brand_ai"] = result.get("sub_brand_ai", "")

                # Extract locationMetadata (complex type with pageNumber and boundingPolygon)
                location_metadata = result.get("locationMetadata")
                if location_metadata:
                    doc["pageNumber"] = location_metadata.get("pageNumber")
                    doc["boundingPolygon"] = location_metadata.get("boundingPolygon", "")
                else:
                    doc["pageNumber"] = None
                    doc["boundingPolygon"] = ""

                # Track unique documents
                if doc.get("text_document_id"):
                    unique_doc_ids.add(doc["text_document_id"])
                if title:
                    unique_doc_titles.add(title)

            docs.append(doc)
            scores.append(doc["score"])

        response["docs"] = docs
        response["returnedCount"] = len(docs)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Get total count
        if hasattr(results, 'get_count') and results.get_count() is not None:
            response["totalCount"] = results.get_count()
            current_position = (skip or 0) + len(docs)
            response["hasMoreResults"] = current_position < response["totalCount"]
        elif results.get_count() is not None:
            response["totalCount"] = results.get_count()
            current_position = (skip or 0) + len(docs)
            response["hasMoreResults"] = current_position < response["totalCount"]

        # Add unique document tracking for main_data index
        if index_type == "main_data" and (unique_doc_ids or unique_doc_titles):
            response["uniqueDocumentsInBatch"] = len(unique_doc_ids) or len(unique_doc_titles)
            response["documentTitles"] = list(unique_doc_titles)

        # Extract facets if requested
        if facets and hasattr(results, 'get_facets'):
            facet_results = results.get_facets()
            if facet_results:
                response["facets"] = {}
                for facet_name, facet_items in facet_results.items():
                    is_document_facet = facet_name in ("document_title", "text_document_id")
                    response["facets"][facet_name] = [
                        {"value": item.get("value", item.get("additional_properties", {}).get("value", "")),
                         "count": item.get("count", 0)}
                        for item in facet_items
                    ]
                    if is_document_facet:
                        response["facets"][f"{facet_name}_note"] = (
                            f"Number of items in this facet ({len(facet_items)}) = "
                            f"number of unique documents. 'count' shows chunks per document."
                        )
                        response["uniqueDocumentCount"] = len(facet_items)
                    else:
                        response["facets"][f"{facet_name}_note"] = (
                            f"This facet shows categories/values. "
                            f"'count' represents chunks, not unique documents."
                        )

        # Add pagination guidance
        if response["hasMoreResults"]:
            response["nextSkip"] = (skip or 0) + len(docs)
            response["remainingChunks"] = response["totalCount"] - response["nextSkip"]

        # Add notes about chunk vs document counting
        if response["totalCount"] > 0 and "uniqueDocumentCount" not in response:
            response["important_note"] = (
                "totalCount represents CHUNKS, not documents. "
                "To count unique documents, use facets: ['document_title,count:1000']."
            )

        if "uniqueDocumentCount" in response:
            response["summary"] = (
                f"Found {response['uniqueDocumentCount']} unique documents "
                f"(out of {response['totalCount']} total chunks)."
            )

        # Add standard metadata
        response["metadata"] = {
            "returned": len(docs),
            "avg_score": round(avg_score, 2),
            "index": index_name,
            "query": query,
            "search_type": "hybrid" if query_embedding else "keyword",
        }

        print(f"   ✓ Found {len(docs)} docs (avg score: {avg_score:.2f})")
        if docs:
            print(f"   📄 Top result: {docs[0]['source']} (score: {docs[0]['score']:.2f})")
            if index_type == "main_data" and docs[0].get("pageNumber") is not None:
                print(f"      Page: {docs[0]['pageNumber']}")
            print(f"   📝 Preview: {docs[0]['content'][:100]}...")

        # Show top 3 results for visibility
        if len(docs) > 1:
            print(f"\n   📋 Top {min(3, len(docs))} Results:")
            for i, doc in enumerate(docs[:3]):
                page_info = f" (page {doc.get('pageNumber')})" if doc.get("pageNumber") is not None else ""
                print(f"      {i+1}. {doc['source']}{page_info} (score: {doc['score']:.2f})")
                print(f"         {doc['content'][:80]}...")

        return json.dumps(response)

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
