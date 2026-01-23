# store.py
import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langgraph.store.postgres import PostgresStore

load_dotenv()

pool = ConnectionPool(
    conninfo=f"postgres://{os.getenv('POSTGRES_USER','postgres')}:"
             f"{os.getenv('POSTGRES_PASSWORD','admin')}@"
             f"{os.getenv('POSTGRES_HOST','localhost')}:"
             f"{os.getenv('POSTGRES_PORT',5432)}/"
             f"{os.getenv('POSTGRES_DB','qt328pp')}",
    kwargs={"autocommit": True},
    max_size=10,
)

store = PostgresStore(pool)

# Run once (safe to keep)
store.setup()
