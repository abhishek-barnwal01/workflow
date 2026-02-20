"""Shared utilities for all pipeline nodes.

Single source of truth for: safe_utf8, sanitize_any, create_llm,
filter_sensitive_content, and tool execution helpers.
Previously each node had its own copy — 4x duplication.
"""

import re
import json
import asyncio
import concurrent.futures
from typing import Any

import config
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import ToolMessage


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def safe_utf8(text: str) -> str:
    """Sanitize a string for PostgreSQL JSON storage.

    Replaces invalid UTF-8 bytes and removes null bytes, which PostgreSQL
    cannot store in JSON columns.
    """
    if not text:
        return ""
    cleaned = text.encode("utf-8", errors="replace").decode("utf-8")
    return cleaned.replace("\x00", "")


def sanitize_any(obj: Any) -> Any:
    """Recursively sanitize all strings in any nested structure."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return safe_utf8(obj)
    if isinstance(obj, list):
        return [sanitize_any(i) for i in obj]
    if isinstance(obj, dict):
        return {k: sanitize_any(v) for k, v in obj.items()}
    return obj


def filter_sensitive_content(text: str) -> str:
    """Replace patterns that may trigger Azure OpenAI content filters."""
    if not text:
        return ""
    patterns = {
        r"(?i)content.*filter": "content validation",
        r"(?i)harmful": "inappropriate",
        r"(?i)violence|violent": "aggressive",
        r"(?i)hate|hateful": "prejudiced",
        r"(?i)abuse|abusive": "harmful behavior",
    }
    filtered = text
    for pattern, replacement in patterns.items():
        try:
            filtered = re.sub(pattern, replacement, filtered)
        except Exception:
            pass
    return filtered


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def create_llm(timeout: float = 120.0, max_retries: int = 2) -> AzureChatOpenAI:
    """Create a configured AzureChatOpenAI instance."""
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=1,
        timeout=timeout,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------

def _is_event_loop_running() -> bool:
    try:
        return asyncio.get_event_loop().is_running()
    except RuntimeError:
        return False


async def _invoke_single_tool_async(tool_call, tools_map: dict) -> ToolMessage:
    """Execute a single tool call asynchronously."""
    if hasattr(tool_call, "name"):
        tool_name, tool_args, tool_id = tool_call.name, tool_call.args, tool_call.id
    else:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")

    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            return ToolMessage(
                content=json.dumps({"error": f"Invalid JSON in tool args: {tool_args}"}),
                tool_call_id=tool_id,
            )

    if tool_name in tools_map:
        try:
            result = await asyncio.to_thread(tools_map[tool_name].invoke, tool_args)
            return ToolMessage(content=result, tool_call_id=tool_id)
        except Exception as e:
            return ToolMessage(content=json.dumps({"error": str(e)}), tool_call_id=tool_id)
    return ToolMessage(
        content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
        tool_call_id=tool_id,
    )


def _invoke_single_tool_sync(tool_call, tools_map: dict) -> ToolMessage:
    """Synchronous fallback for a single tool call."""
    if hasattr(tool_call, "name"):
        tool_name, tool_args, tool_id = tool_call.name, tool_call.args, tool_call.id
    else:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")

    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            return ToolMessage(
                content=json.dumps({"error": f"Invalid JSON in tool args: {tool_args}"}),
                tool_call_id=tool_id,
            )

    if tool_name in tools_map:
        try:
            result = tools_map[tool_name].invoke(tool_args)
            return ToolMessage(content=result, tool_call_id=tool_id)
        except Exception as e:
            return ToolMessage(content=json.dumps({"error": str(e)}), tool_call_id=tool_id)
    return ToolMessage(
        content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
        tool_call_id=tool_id,
    )


def execute_tool_calls(tool_calls: list, tools_map: dict) -> list:
    """Execute multiple tool calls, parallelising where possible.

    If an event loop is already running (e.g. inside FastAPI's async context),
    falls back to a thread pool to avoid nest_asyncio issues.
    """
    if not tool_calls:
        return []
    if _is_event_loop_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = [
                pool.submit(_invoke_single_tool_sync, tc, tools_map)
                for tc in tool_calls
            ]
            return [f.result() for f in futures]
    else:
        async def _run_all():
            return await asyncio.gather(
                *[_invoke_single_tool_async(tc, tools_map) for tc in tool_calls]
            )
        return asyncio.run(_run_all())
