"""FastAPI server with LangGraph + Postgres persistence.

Async endpoints allow concurrent request handling — multiple users
can hit the server simultaneously without blocking each other.
Graph nodes remain synchronous and are offloaded to a thread pool
via asyncio.to_thread so the event loop stays free.
"""

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from graph import build_graph


# ---------- Build LangGraph Flow (sync, runs once at startup) ----------
graph = build_graph()


# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Starting LangGraph RAG Server (FastAPI + async)...")
    print("💡 POST → http://localhost:5001/chat")
    print('   {"question": "your question", "session_id": "user123"}')
    print("\n💡 POST → http://localhost:5001/v1/chat/completions (LibreChat)")
    print("   OpenAI-compatible endpoint\n")
    yield
    print("Shutting down...")


app = FastAPI(title="LangGraph RAG Server", lifespan=lifespan)


# ---------- Helpers ----------
def append_sas_to_blob_urls(markdown_text: str) -> str:
    """Finds all Azure Blob Storage URLs in markdown and appends SAS token."""
    sas_token = os.getenv("AZURE_BLOB_SAS_TOKEN", "")

    if not sas_token:
        print("⚠️ WARNING: AZURE_BLOB_SAS_TOKEN not set")
        return markdown_text

    blob_pattern = re.compile(
        r"(https://[a-zA-Z0-9]+\.blob\.core\.windows\.net/[^\s\)]+?)(?=[\s\)\]]|$)"
    )

    def add_sas(match):
        url = match.group(1)
        if "sv=" in url or "sig=" in url:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{sas_token}"

    return blob_pattern.sub(add_sas, markdown_text)


# ---------- Chat Endpoint ----------
@app.post("/chat")
async def chat(request: Request):
    """
    Simple chat endpoint.
    Body: {"question": "What is market share?", "session_id": "user123"}
    """
    data = await request.json()
    user_query = data.get("question")
    thread_id = data.get("session_id", "default")

    if not user_query:
        return JSONResponse({"error": "Question required"}, status_code=400)

    try:
        result = await asyncio.to_thread(
            graph.invoke,
            {"user_query": user_query, "user_id": "abhishek"},
            config={"configurable": {"thread_id": thread_id}},
        )

        clarification_msg = result.get("clarification_message")
        if clarification_msg:
            return {
                "response": clarification_msg,
                "needs_clarification": True,
                "session_id": thread_id,
            }

        formatter_result = result.get("formatted", {})
        eval_result = result.get("evaluation", {})
        rag_result = result.get("rag_output", {})
        iteration = result.get("iteration", 0)

        return {
            "response": formatter_result.get("formatted_response", ""),
            "metadata": {
                "confidence": eval_result.get("confidence_score", 0.0),
                "confidence_breakdown": {
                    "relevance": eval_result.get("confidence_breakdown", {}).get("relevance", 0.0),
                    "completeness": eval_result.get("confidence_breakdown", {}).get("completeness", 0.0),
                    "context_match": eval_result.get("confidence_breakdown", {}).get("context_match", 0.0),
                },
                "sources": len(rag_result.get("retrieved_docs", [])),
                "iterations": iteration + 1,
                "enriched_query": result.get("enriched_query", ""),
                "evaluator_reasoning": eval_result.get("reasoning", ""),
            },
            "session_id": thread_id,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e), "session_id": thread_id}, status_code=500)


# ---------- History Endpoint ----------
@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    from persistence import checkpointer

    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = await asyncio.to_thread(checkpointer.get, config)

    if not checkpoint:
        return {"session_id": thread_id, "messages": []}

    channel_values = checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])

    serialized_messages = []
    for msg in messages:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            serialized_messages.append({"role": msg.type, "content": msg.content})
        elif isinstance(msg, dict):
            serialized_messages.append({
                "role": msg.get("type", "unknown"),
                "content": msg.get("content", ""),
            })
        else:
            serialized_messages.append({"role": "unknown", "content": str(msg)})

    return {"session_id": thread_id, "messages": serialized_messages}


# ---------- OpenAI-Compatible Endpoint ----------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    LibreChat / OpenAI-compatible endpoint.
    Converts OpenAI message format to LangGraph pipeline format.
    """
    data = await request.json()
    print(data)
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    model = data.get("model", "rag-pipeline")

    # Extract user/session info from headers (LibreChat sends these)
    user_id = request.headers.get("X-User-Id", "anonymous")
    session_id = request.headers.get("X-Conversation-Id", str(uuid.uuid4()))

    if not messages:
        return JSONResponse(
            {"error": {"message": "No messages provided", "type": "invalid_request_error"}},
            status_code=400,
        )

    # Extract user_query from last user message (skip system messages)
    user_query = None
    for msg in reversed(messages):
        if msg.get("role", "").lower() == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        return JSONResponse(
            {"error": {"message": "No user message found", "type": "invalid_request_error"}},
            status_code=400,
        )

    # Convert OpenAI format to LangChain format (exclude last user message)
    langchain_messages = []
    for msg in messages[:-1]:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "system":
            continue
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))

    # Streaming
    if stream:
        return StreamingResponse(
            generate_stream(user_query, langchain_messages, user_id, session_id, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming
    try:
        result = await asyncio.to_thread(
            graph.invoke,
            {"user_query": user_query, "messages": langchain_messages, "user_id": user_id},
            config={"configurable": {"thread_id": session_id}},
        )

        clarification_msg = result.get("clarification_message")
        if clarification_msg:
            final_response = clarification_msg
        else:
            final_response = result.get("formatted", {}).get(
                "formatted_response", "I couldn't generate a response."
            )

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(final_response.split()),
                "total_tokens": sum(len(m.get("content", "").split()) for m in messages)
                + len(final_response.split()),
            },
        }

        return JSONResponse(
            response,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": {"message": str(e), "type": "internal_error", "code": "internal_error"}},
            status_code=500,
        )


# ---------- Streaming generator for LibreChat ----------
async def generate_stream(user_query, langchain_messages, user_id, session_id, model):
    """
    Async generator that streams the response as OpenAI SSE chunks.

    KEY FIXES:
    1. Send role: "assistant" in first chunk (critical for markdown rendering)
    2. Stream entire response at once (preserves markdown formatting)
    """
    try:
        result = await asyncio.to_thread(
            graph.invoke,
            {"user_query": user_query, "messages": langchain_messages, "user_id": user_id},
            config={"configurable": {"thread_id": session_id}},
        )

        print(f"\n🔍 DEBUG generate_stream result keys: {list(result.keys())}")
        print(f"📌 clarification_message: {result.get('clarification_message')}")
        print(f"📌 semantic_chitchat: {result.get('semantic_chitchat')}")

        clarification_msg = result.get("clarification_message")
        if clarification_msg:
            print(f"✅ Clarification detected, returning: {clarification_msg[:100]}...")
            final_response = clarification_msg
        else:
            print("📄 No clarification, using formatted response")
            final_response = result.get("formatted", {}).get("formatted_response", "")
            final_response = append_sas_to_blob_urls(final_response)
            if not final_response:
                print(f"⚠️ No formatted response, result keys: {result.keys()}")

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        # CRITICAL: Send role first — LibreChat needs this for markdown rendering
        role_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        # Send complete markdown in one chunk to preserve formatting
        content_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": final_response}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

        # Final stop chunk
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        role_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        error_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n❌ Error: {str(e)}"},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# ---------- Run Server ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
