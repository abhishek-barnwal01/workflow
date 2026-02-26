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
from formatter_node import build_formatter_prompt
from utils import create_llm


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

        formatter_result = result.get("formatted", {})
        eval_result = result.get("evaluation", {})
        rag_result = result.get("rag_output", {})
        iteration = result.get("iteration", 0)

        return {
            "response": append_sas_to_blob_urls(formatter_result.get("formatted_response", "")),
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

        formatted_response = result.get("formatted", {}).get("formatted_response", "")
        clarification_msg = result.get("clarification_message", "")
        # Document listing output (SQL agent, pre-formatted)
        doc_listing = result.get("document_listing_output") or {}
        doc_listing_response = (
            doc_listing.get("formatted_response", "") if isinstance(doc_listing, dict)
            else getattr(doc_listing, "formatted_response", "")
        )
        # Fallback to RAG final answer if formatter is skipped or empty (e.g., chat title requests)
        rag_out = result.get("rag_output") or {}
        rag_answer = (
            rag_out.get("final_answer", "") if isinstance(rag_out, dict)
            else getattr(rag_out, "final_answer", "")
        )
        final_response = formatted_response or doc_listing_response or clarification_msg or rag_answer or "I couldn't generate a response."
        # Ensure blob links include SAS before sending
        final_response = append_sas_to_blob_urls(final_response)

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


# ---------- SSE chunk helpers ----------
def _sse_chunk(chunk_id: str, created_time: int, model: str, *, delta: dict, finish_reason=None) -> str:
    return f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})}\n\n"


# ---------- Streaming generator for LibreChat ----------
async def generate_stream(user_query, langchain_messages, user_id, session_id, model):
    """
    Two-phase streaming:
      Phase 1 — Run the graph with skip_formatter=True (RAG + semantic nodes,
                 no formatter LLM call). Graph finishes fast; formatter is skipped.
      Phase 2 — Stream the formatter LLM token-by-token via llm.astream().
                 Tokens are piped directly to SSE so the client sees text appear
                 progressively rather than waiting for the full response.

    Short responses (chitchat, clarification questions) skip Phase 2 and are
    sent as a single chunk since they require no formatting.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    try:
        # ── Phase 1: run graph, skip formatter ──────────────────────────────
        result = await asyncio.to_thread(
            graph.invoke,
            {"user_query": user_query, "messages": langchain_messages, "user_id": user_id,
             "skip_formatter": True},
            config={"configurable": {"thread_id": session_id}},
        )

        print(f"\n🔍 DEBUG generate_stream result keys: {list(result.keys())}")
        print(f"📌 clarification_message: {result.get('clarification_message')}")
        print(f"📌 semantic_chitchat: {result.get('semantic_chitchat')}")

        semantic_chitchat: bool = result.get("semantic_chitchat", False)
        clarification_msg: str = result.get("clarification_message") or ""
        awaiting_clarification: bool = result.get("awaiting_clarification", False)

        rag_output = result.get("rag_output") or {}
        rag_answer: str = (
            rag_output.get("final_answer", "") if isinstance(rag_output, dict)
            else getattr(rag_output, "final_answer", "")
        )

        evaluation = result.get("evaluation") or {}
        confidence: float = (
            evaluation.get("confidence_score", 0.8) if isinstance(evaluation, dict)
            else getattr(evaluation, "confidence_score", 0.8)
        )

        # ── Determine content to format ──────────────────────────────────────
        # Chitchat, clarification questions, and document listings need no formatting — send as-is.
        doc_listing = result.get("document_listing_output") or {}
        doc_listing_response: str = (
            doc_listing.get("formatted_response", "") if isinstance(doc_listing, dict)
            else getattr(doc_listing, "formatted_response", "")
        )

        if semantic_chitchat or (clarification_msg and awaiting_clarification) or doc_listing_response:
            print("📄 Streaming chitchat/clarification/listing directly (no formatter)")
            final_response = doc_listing_response or clarification_msg
            final_response = append_sas_to_blob_urls(final_response)
            yield _sse_chunk(chunk_id, created_time, model, delta={"role": "assistant"})
            yield _sse_chunk(chunk_id, created_time, model, delta={"content": final_response})
            yield _sse_chunk(chunk_id, created_time, model, delta={}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        # Direct answer from semantic node (e.g. answered from history) or RAG answer.
        text_to_format = (
            clarification_msg if (clarification_msg and not awaiting_clarification)
            else rag_answer
        )
        if not text_to_format:
            print("⚠️ No content to format")
            yield _sse_chunk(chunk_id, created_time, model, delta={"role": "assistant"})
            yield _sse_chunk(chunk_id, created_time, model, delta={}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        # Ensure any blob links in the raw text already carry SAS before formatting
        text_to_format = append_sas_to_blob_urls(text_to_format)

        # ── Phase 2: stream formatter LLM token-by-token ─────────────────────
        print(f"📄 Streaming formatter output ({len(text_to_format)} chars to format)")
        prompt = build_formatter_prompt(user_query, text_to_format, confidence)
        llm = create_llm()

        # CRITICAL: send role first — LibreChat needs this for markdown rendering
        yield _sse_chunk(chunk_id, created_time, model, delta={"role": "assistant"})

        async for chunk in llm.astream(prompt):
            token: str = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield _sse_chunk(chunk_id, created_time, model, delta={"content": token})

        yield _sse_chunk(chunk_id, created_time, model, delta={}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()

        yield _sse_chunk(chunk_id, created_time, model, delta={"role": "assistant"})

        # Sanitize error message — never echo filter-related keywords back into
        # chat history (they would trigger Azure's content filter on every
        # subsequent request, creating a self-perpetuating loop).
        raw_err = str(e).lower()
        if any(kw in raw_err for kw in ("jailbreak", "content_filter", "content filter", "responsibleai")):
            safe_error = "I wasn't able to format that response due to a content policy check. Please try rephrasing your question."
        else:
            safe_error = "Something went wrong while processing your request. Please try again."

        yield _sse_chunk(chunk_id, created_time, model, delta={"content": safe_error}, finish_reason="stop")
        yield "data: [DONE]\n\n"


@app.get("/heartbeat")
async def heartbeat():
    """Basic health check endpoint."""
    return {"status": "OK", "message": "Server is running"}

# ---------- Run Server ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
