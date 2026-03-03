# persistence.py
import os
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

DB_BACKEND = os.getenv("DB_BACKEND", "postgres")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
    "options": "-c client_encoding=UTF8"  # Set encoding here
}

pool = ConnectionPool(
    conninfo=f"postgres://{os.getenv('POSTGRES_USER','postgres')}:{os.getenv('POSTGRES_PASSWORD','admin')}@{os.getenv('POSTGRES_HOST','localhost')}:{os.getenv('POSTGRES_PORT',5432)}/{os.getenv('POSTGRES_DB','qt328pp')}",
    max_size=20,
    kwargs=connection_kwargs
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()


# ---------------------------------------------------------------------------
# Databricks SQL connector (lazy-loaded, only when DB_BACKEND=databricks)
# ---------------------------------------------------------------------------
_databricks_connection = None


def get_databricks_connection():
    """Return a reusable Databricks SQL connection (created on first call)."""
    global _databricks_connection
    if _databricks_connection is None:
        from databricks import sql as databricks_sql
        _databricks_connection = databricks_sql.connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME", ""),
            http_path=os.getenv("DATABRICKS_HTTP_PATH", ""),
            access_token=os.getenv("DATABRICKS_ACCESS_TOKEN", ""),
        )
    return _databricks_connection


def execute_query(query: str, max_rows: int = 200):
    """Execute a read-only SQL query against the active backend.

    Returns (columns: list[str], rows: list[dict]).
    Works with both PostgreSQL (local) and Databricks (development).
    """
    if DB_BACKEND == "databricks":
        conn = get_databricks_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            raw_rows = cursor.fetchmany(max_rows)
            rows = [dict(zip(columns, row)) for row in raw_rows]
            total = cursor.rowcount if cursor.rowcount and cursor.rowcount >= 0 else len(rows)
            return columns, rows, total
        finally:
            cursor.close()
    else:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                raw_rows = cur.fetchmany(max_rows)
                if raw_rows and isinstance(raw_rows[0], dict):
                    rows = list(raw_rows)
                else:
                    rows = [dict(zip(columns, row)) for row in raw_rows]
                total = cur.rowcount if cur.rowcount >= 0 else len(rows)
                return columns, rows, total


# # persistence.py
# import os
# from dotenv import load_dotenv
# from psycopg_pool import ConnectionPool
# from langgraph.checkpoint.postgres import PostgresSaver

# load_dotenv()

# # ---------- ADD THIS FUNCTION ----------
# def sanitize_any(obj):
#     if obj is None:
#         return None
#     if isinstance(obj, str):
#         return obj.encode("utf-8", errors="replace").decode("utf-8")
#     if isinstance(obj, list):
#         return [sanitize_any(i) for i in obj]
#     if isinstance(obj, dict):
#         return {k: sanitize_any(v) for k, v in obj.items()}
#     return obj
# # -------------------------------------


# pool = ConnectionPool(
#     conninfo=f"""
#         host={os.getenv('POSTGRES_HOST','localhost')}
#         port={os.getenv('POSTGRES_PORT',5432)}
#         user={os.getenv('POSTGRES_USER','postgres')}
#         password={os.getenv('POSTGRES_PASSWORD','postgres')}
#         dbname={os.getenv('POSTGRES_DB','qt328pp')}
#         client_encoding=UTF8
#     """,
#     kwargs={"autocommit": True},
# )


# # ---------- REPLACE PostgresSaver ----------
# class SafePost(PostgresSaver):
#     def put(self, *args, **kwargs):
#         """
#         Sanitize checkpoint and metadata before saving.
#         args layout (LangGraph internal):
#         args[0] = config
#         args[1] = checkpoint
#         args[2] = metadata
#         args[3] = new_versions (optional)
#         """
#         args = list(args)

#         # Sanitize checkpoint and metadata
#         if len(args) > 1:
#             args[1] = sanitize_any(args[1])  # checkpoint
#         if len(args) > 2:
#             args[2] = sanitize_any(args[2])  # metadata

#         return super().put(*args, **kwargs)

#     def get_tuple(self, config):
#         """
#         Return a wrapped object with checkpoint & metadata,
#         safely handling old rows or unexpected extra values.
#         """
#         data = super().get_tuple(config)
#         if not data:
#             return data

#         # Safely get first two elements
#         checkpoint = data[0] if len(data) > 0 else None
#         metadata = data[1] if len(data) > 1 else None

#         # Wrap in object expected by LangGraph
#         class CheckpointWrapper:
#             def __init__(self, checkpoint, metadata):
#                 self.checkpoint = sanitize_any(checkpoint)
#                 self.metadata = sanitize_any(metadata)

#         return CheckpointWrapper(checkpoint, metadata)





# checkpointer = SafePost(conn=pool)

# # Run only once – safe to keep
# checkpointer.setup()
