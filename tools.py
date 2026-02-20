"""Azure AI Search tool with hybrid search, filters, facets, and pagination"""
from langchain.tools import tool
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import config
import json
from typing import Optional, List
import hashlib

# 🚀 EMBEDDING CACHE - Avoid regenerating embeddings for same queries
_embedding_cache = {}

def get_embedding(text: str) -> list:
    """Get embedding for text using Azure OpenAI with caching"""
    try:
        # Cache key based on text hash
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Return cached embedding if available
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]
        
        client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )

        response = client.embeddings.create(
            input=text,
            model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )

        embedding = response.data[0].embedding
        _embedding_cache[cache_key] = embedding  # Cache for future use
        return embedding
    except Exception as e:
        print(f"   ⚠️ Embedding error: {e}")
        return None


# Default fields for main_data index (all retrievable fields except content_embedding)
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
    "file_time_period_ai",
    "country_ai",
]

# Default fields for semantic index (keep existing)
SEMANTIC_SELECT_FIELDS = [
    "content_text",
    "document_title",
    "content_path",
    "content_id",
]


@tool
def azure_ai_search(
    query: str,
    index_type: str,
    top_k: int,
    filter: Optional[str] = None,
    facets: Optional[List[str]] = None,
    skip: Optional[int] = None,
    select_fields: Optional[str] = None,
) -> str:
    """Search documents in Azure AI Search. The index stores document CHUNKS, not whole documents.

PARAMETERS:
- query: Search text, or "*" for wildcard (when using filters/facets only)
- index_type: "main_data" (supports all params) or "semantic" (only query, index_type, top_k)
- top_k: Number of results (1-100)
- filter: OData filter (main_data only, CASE-SENSITIVE). ALWAYS add "and text_document_id ne ''" when listing documents to get PDF paths instead of image paths.
- facets: Facetable fields for counting/listing (main_data only). Add ",count:1000" to get up to 1000 values.
  Facetable fields: document_title, text_document_id, content_path, file_category_ai, country_ai
- skip: Pagination offset (main_data only)
- select_fields: Comma-separated fields to return (main_data only).
  VALID fields ONLY: document_title, content_path, content_text, content_id, text_document_id, image_document_id, locationMetadata, file_category_ai, product_category_ai, brand_ai, file_time_period_ai, country_ai
  MUST include "content_text" when you need to READ document content. Omit it only for listing/counting.

EXACT file_category_ai values (case-sensitive):
"Analysis", "Annual presentation", "Brand equity", "Brand Health track", "Concept testing", "Home panel", "Link testing", "Media Optimization", "Miscellaneous", "Needscope", "Post Launch Evaluation", "Product acceptance testing", "Product Performance Evaluation", "Retail audit", "Usage/Attitude (U&A)"

--- FEW-SHOT EXAMPLES ---

Example 1 - Count U&A reports (1 call):
  query="*", index_type="main_data", top_k=1, filter="file_category_ai eq 'Usage/Attitude (U&A)' and text_document_id ne ''", facets=["document_title,count:1000"]
  → Number of facet items = number of unique documents

Example 2 - List all U&A reports with clickable links (1 call):
  query="*", index_type="main_data", top_k=100, filter="file_category_ai eq 'Usage/Attitude (U&A)' and text_document_id ne ''", facets=["document_title,count:1000"], select_fields="document_title,content_path"
  → Facet values = all unique document names; docs array has content_path for links

Example 3 - List brand equity reports with links (1 call):
  query="*", index_type="main_data", top_k=100, filter="file_category_ai eq 'Brand equity' and text_document_id ne ''", facets=["document_title,count:1000"], select_fields="document_title,content_path"

Example 4 - Content search (find insights):
  query="Godrej growth 2022", index_type="main_data", top_k=20
  → Returns full content with page_number for citations. NO facets needed.

Example 5 - Content search with specific fields:
  query="product likability", index_type="main_data", top_k=20, select_fields="content_text,document_title,content_path,locationMetadata"
  → MUST include content_text to read actual text

Example 6 - Specific page of a document:
  query="*", index_type="main_data", top_k=10, filter="locationMetadata/pageNumber eq 6 and document_title eq 'Presentation.pptx'"

Example 7 - List all categories:
  query="*", index_type="main_data", top_k=1, facets=["file_category_ai,count:100"]

Example 8 - Summarize a document (read ALL pages with pagination):
  First call: query="*", index_type="main_data", top_k=100, filter="document_title eq 'Report.pdf'", select_fields="content_text,document_title,content_path,locationMetadata"
  → Check totalCount. If <= 300: paginate to read all (skip=100, skip=200).
  → If > 300: sample middle (skip=totalCount/2, top_k=100) and end (skip=totalCount-100, top_k=100). Max 4 calls total.

--- RULES ---
- USE facets for listing/counting queries (1 call). DO NOT loop per document.
- DO NOT use facets for content/keyword searches.
- ALWAYS add "and text_document_id ne ''" to filter when listing documents.
- WITHOUT ",count:1000" on facets you only get 10 values max.
- DO NOT invent fields (e.g., metadata_storage_last_modified, author, owner do NOT exist).
- For semantic index: ONLY use query, index_type, top_k (no filter/facets/skip/select_fields).
- DEFAULT GEOGRAPHY: if the query has no country/region, add country_ai eq 'India' to filter.
- DEFAULT TIME PERIOD: if no period is specified or latest is mentioned, check file_time_period_ai in results and use the most recent one (2026 > 2025 > 2024 > …).
    """

    index_name = (
        config.MAIN_DATA_INDEX_NAME if index_type == "main_data"
        else config.SEMANTIC_INDEX_NAME
    )

    # For semantic index, ignore advanced parameters (they're not supported)
    if index_type != "main_data":
        if filter or facets or skip or select_fields:
            print(f"   ⚠️ Ignoring filter/facets/skip/select_fields for semantic index (not supported)")
        filter = None
        facets = None
        skip = None
        select_fields = None

    print(f"\n🔍 azure_ai_search | index={index_type} | query={query[:60]}{'...' if len(query)>60 else ''} | top_k={top_k} | filter={filter or 'none'} | facets={facets or 'none'} | skip={skip or 0}")

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
            "top": min(top_k, 100),
            "select": selected,
            "include_total_count": True,
        }

        # Add optional filter
        if filter:
            search_kwargs["filter"] = filter

        # Add optional facets
        if facets:
            search_kwargs["facets"] = facets

        # Add optional skip for pagination
        if skip is not None and skip > 0:
            search_kwargs["skip"] = skip

        # Add vector search (skip for wildcard queries)
        is_wildcard = query.strip() == "*"
        if not is_wildcard:
            query_embedding = get_embedding(query)
            if query_embedding:
                vector_query = VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=min(top_k, 50),
                    fields="content_embedding"
                )
                search_kwargs["vector_queries"] = [vector_query]

        # Perform search
        results = client.search(**search_kwargs)

        # Get total count
        total_count = getattr(results, 'get_count', lambda: None)()

        docs = []
        scores = []
        for result in results:
            content = result.get("content_text", "")
            title = result.get("document_title", "")
            source = result.get("content_path", result.get("document_title", "unknown"))

            if index_type == "main_data":
                # Extract locationMetadata (nested complex type)
                location_metadata = result.get("locationMetadata", {})
                page_number = None
                bounding_polygon = None
                if location_metadata:
                    page_number = location_metadata.get("pageNumber")
                    bounding_polygon = location_metadata.get("boundingPolygon")

                doc = {
                    "id": result.get("content_id"),
                    "text_document_id": result.get("text_document_id", ""),
                    "document_title": title,
                    "image_document_id": result.get("image_document_id", ""),
                    "content_text": content,
                    "content_path": source,
                    "page_number": page_number,
                    "bounding_polygon": bounding_polygon,
                    "file_category_ai": result.get("file_category_ai", ""),
                    "product_category_ai": result.get("product_category_ai", ""),
                    "brand_ai": result.get("brand_ai", ""),
                    "file_time_period_ai": result.get("file_time_period_ai", ""),
                    "country_ai": result.get("country_ai", ""),
                    "score": result.get("@search.score", 0.0),
                }
            else:
                # Semantic index - keep existing format
                full_content = f"{title}\n\n{content}" if title else content
                doc = {
                    "id": result.get("content_id"),
                    "content_text": full_content,
                    "score": result.get("@search.score", 0.0),
                    "source": source,
                }

            docs.append(doc)
            scores.append(doc["score"])

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Build response (like LibreChat's AzureAISearch.js)
        response_data = {
            "documents": docs,
            "totalCount": total_count,
            "returnedCount": len(docs),
            "skip": skip or 0,
            "hasMoreResults": (total_count is not None and (skip or 0) + len(docs) < total_count),
        }

        # Add pagination helper
        if response_data["hasMoreResults"]:
            response_data["nextSkip"] = (skip or 0) + len(docs)
            response_data["remainingChunks"] = total_count - ((skip or 0) + len(docs))

        # Unique doc tracking
        unique_titles = set()
        for d in docs:
            t = d.get("document_title", d.get("source", ""))
            if t:
                unique_titles.add(t)
        response_data["uniqueDocumentsInBatch"] = len(unique_titles)
        response_data["documentTitles"] = list(unique_titles)

        # Add facet results if facets were requested
        if facets:
            try:
                facet_results = results.get_facets()
                if facet_results:
                    formatted_facets = {}
                    for field_name, facet_values in facet_results.items():
                        is_doc_facet = field_name in ("document_title", "text_document_id")
                        formatted_facets[field_name] = [
                            {"value": fv["value"], "count": fv["count"]}
                            for fv in facet_values
                        ]
                        if is_doc_facet:
                            formatted_facets[f"{field_name}_note"] = f"Number of items ({len(facet_values)}) = number of unique documents."
                            response_data["uniqueDocumentCount"] = len(facet_values)

                    response_data["facets"] = formatted_facets

                    # Document list with URLs for easy LLM parsing
                    if "document_title" in formatted_facets:
                        document_list = []
                        for facet in formatted_facets["document_title"]:
                            doc_title = facet["value"]
                            matching_doc = next((d for d in docs if d.get("document_title") == doc_title), None)
                            if matching_doc:
                                document_list.append({
                                    "title": doc_title,
                                    "url": matching_doc.get("content_path", ""),
                                    "count": facet["count"],
                                })
                        response_data["document_list"] = document_list

            except Exception as facet_err:
                print(f"   ⚠️ Facet extraction error: {facet_err}")

        # Add summary note
        if response_data.get("uniqueDocumentCount"):
            response_data["summary"] = f"Found {response_data['uniqueDocumentCount']} unique documents (out of {total_count} total chunks)."
        elif total_count:
            response_data["important_note"] = "totalCount represents CHUNKS, not documents. Use facets to count unique documents."

        return json.dumps(response_data)

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
