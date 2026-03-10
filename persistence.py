# persistence.py
import asyncio
import os
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()


connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
    "options": "-c client_encoding=UTF8"
}

pool = ConnectionPool(
    conninfo=f"postgres://{os.getenv('POSTGRES_USER','postgres')}:{os.getenv('POSTGRES_PASSWORD','admin')}@{os.getenv('POSTGRES_HOST','localhost')}:{os.getenv('POSTGRES_PORT',5432)}/{os.getenv('POSTGRES_DB','qt328pp')}",
    max_size=20,
    kwargs=connection_kwargs
)


class _AsyncPostgresSaver(PostgresSaver):
    """Async wrapper around the synchronous PostgresSaver.

    graph.astream_events() uses the async Pregel loop which calls
    aget_tuple / aput / aput_writes on the checkpointer. The base
    PostgresSaver only has sync implementations and raises
    NotImplementedError for the async variants. This subclass wraps
    each sync method in asyncio.to_thread so the event loop is never
    blocked and the async Pregel loop works correctly.
    """

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item


checkpointer = _AsyncPostgresSaver(pool)
checkpointer.setup()
