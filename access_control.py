"""Role-based access control for Azure AI Search and SQL queries.

Access rules are read from MongoDB users.dataAccess (the same field used
by the LibreChat agent's AzureAISearch tool), so a single admin action
in the LibreChat UI controls access for BOTH the LibreChat agent and
the GCPL RAG custom endpoint (port 5001).

MongoDB dataAccess schema:
  {
    "productCategories": ["Soaps", "Detergents"] | null,  # null = unrestricted
    "countries":         ["India", "UK"]          | null,  # null = unrestricted
  }

null = unrestricted (full access to that dimension).
"""

from typing import Optional, Dict, Any
from user_resolver import resolve_data_access


def get_access_rules(user_id: str) -> Dict[str, Any]:
    """Return access rules for a user.

    user_id is the LibreChat MongoDB ObjectId received from X-User-Id header.
    Reads dataAccess directly from MongoDB users collection — the same source
    used by the LibreChat AzureAISearch tool.

    Returns a dict with:
      - search_categories: list of product_category_ai values for OData filter (or None)
      - sql_categories:    list of product_category_det values for SQL WHERE (or None)
      - countries:         list of country values shared between both (or None)
    """
    data_access = resolve_data_access(user_id)

    if data_access is None:
        # No dataAccess field → full access
        rules = {"search_categories": None, "sql_categories": None, "countries": None}
        print(f"🔒 Access rules for user '{user_id}': full access (no dataAccess set)")
        return rules

    product_categories = data_access.get("productCategories") or None
    countries = data_access.get("countries") or None

    rules = {
        "search_categories": product_categories,  # used for Azure AI Search OData filter
        "sql_categories": product_categories,      # used for SQL WHERE clause
        "countries": countries,
    }
    print(f"🔒 Access rules for user '{user_id}': "
          f"categories={product_categories} | countries={countries}")
    return rules


def build_odata_filter(rules: Dict[str, Any]) -> Optional[str]:
    """OData filter string for Azure AI Search (product_category_ai, country_ai).
    Returns None if the user has full access (no restriction needed).
    """
    parts = []
    cats = rules.get("search_categories")
    countries = rules.get("countries")
    if cats:
        cat_parts = " or ".join(f"product_category_ai eq '{c}'" for c in cats)
        parts.append(f"({cat_parts})")
    if countries:
        country_parts = " or ".join(f"country_ai eq '{c}'" for c in countries)
        parts.append(f"({country_parts})")
    return " and ".join(parts) if parts else None


def build_sql_filter(rules: Dict[str, Any]) -> Optional[str]:
    """SQL WHERE fragment for the Postgres metadata table (product_category_det, country_det).
    Returns None if the user has full access (no restriction needed).
    """
    parts = []
    cats = rules.get("sql_categories")
    countries = rules.get("countries")
    if cats:
        quoted = ", ".join(f"'{c}'" for c in cats)
        parts.append(f"product_category_det IN ({quoted})")
    if countries:
        quoted = ", ".join(f"'{c}'" for c in countries)
        parts.append(f"country_det IN ({quoted})")
    return " AND ".join(parts) if parts else None
