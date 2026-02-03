"""Flask API with LangGraph + Postgres persistence"""

from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from graph import build_graph
import uuid
import time
import json
import os
from flask import Flask, request, jsonify, Response, stream_with_context

app = Flask(__name__)

# ---------- Build LangGraph Flow ----------
graph = build_graph()  # this must return COMPILED graph


# ---------- Chat Endpoint ----------
@app.route("/chat", methods=["POST"])

# chat completion 
# username, 
def chat():
    """
    Body:
    {
        "question": "What is market share?",
        "session_id": "user123"
    }
    """
    data = request.json or {}
    user_query = data.get("question")
    thread_id = data.get("session_id", "default")

    if not user_query:
        return jsonify({"error": "Question required"}), 400

    try:
        # Only pass new input - let checkpoint restore the rest
        result = graph.invoke(
            {"user_query": user_query, "user_id": "abhishek"},
            config={"configurable": {"thread_id": thread_id}},
        )

        clarification_msg = result.get("clarification_message")
        if clarification_msg:
            # Return clarification to user without running RAG
            return jsonify({
                "response": clarification_msg,
                "needs_clarification": True,
                "session_id": thread_id
            })

        # return jsonify(
        #     {
        #         # "response": result["rag_output"]["final_answer"],
        #         "response": result["formatted"]["formatted_response"],
        #         "rag_output": result["rag_output"],
        #         "evaluation": result.get("evaluation"),
        #         "session_id": thread_id,
        #     }
        # )

        # Prepare pieces safely
        formatter_result = result.get("formatted", {})
        eval_result = result.get("evaluation", {})
        rag_result = result.get("rag_output", {})
        semantic_result = result  # contains enriched_query etc.
        iteration = result.get("iteration", 0)  # optional, if you track iterations

        return jsonify({
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
                "enriched_query": semantic_result.get("enriched_query", ""),
                "evaluator_reasoning": eval_result.get("reasoning", ""),
            },
            "session_id": thread_id,
        })


    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "session_id": thread_id}), 500

# ---------- History Endpoint ----------
@app.route("/history/<thread_id>", methods=["GET"])
def get_history(thread_id):
    from persistence import checkpointer

    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = checkpointer.get(config)

    if not checkpoint:
        return jsonify({"session_id": thread_id, "messages": []})

    # ✅ Messages are stored in channel_values["messages"]
    channel_values = checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])

    serialized_messages = []
    for msg in messages:
        # msg can be a dict or LangChain message object
        if hasattr(msg, "type") and hasattr(msg, "content"):
            serialized_messages.append({"role": msg.type, "content": msg.content})
        elif isinstance(msg, dict):
            serialized_messages.append({
                "role": msg.get("type", "unknown"),
                "content": msg.get("content", "")
            })
        else:
            serialized_messages.append({"role": "unknown", "content": str(msg)})

    return jsonify({"session_id": thread_id, "messages": serialized_messages})


# ---------- OpenAI-Compatible Endpoint ----------
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    LibreChat/OpenAI-compatible endpoint.
    Converts OpenAI format to LangGraph pipeline format.
    """
    data = request.json or {}
    print(data)
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    model = data.get("model", "rag-pipeline")

    # Extract user/session info from headers (LibreChat sends these)
    user_id = request.headers.get("X-User-Id", "anonymous")
    session_id = request.headers.get("X-Conversation-Id", str(uuid.uuid4()))
    print(request)

    if not messages:
        return jsonify({
            "error": {"message": "No messages provided", "type": "invalid_request_error"}
        }), 400

    # Extract user_query from last user message (skip system messages)
    user_query = None
    for msg in reversed(messages):
        role = msg.get("role", "").lower()
        if role == "user":
            user_query = msg.get("content", "")
            break
    
    if not user_query:
        return jsonify({
            "error": {"message": "No user message found", "type": "invalid_request_error"}
        }), 400

    # Convert OpenAI format (role: user/assistant/system) to LangChain format
    # Filter out system messages as they are handled by the LLM prompts
    langchain_messages = []
    for msg in messages[:-1]:  # Exclude last user message (it's now user_query)
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        # Skip system messages - they shouldn't be part of chat history
        if role == "system":
            continue
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))

    # For clarification, don't stream - return immediately
    # Check if this will result in clarification by doing a quick graph check
    # Actually, we should just handle streaming but check for clarification in generate_stream
    
    # For clarification, don't stream - return immediately
    # Check if this will result in clarification by doing a quick graph check
    # Actually, we should just handle streaming but check for clarification in generate_stream
    
    # Streaming version
    if stream:
        return Response(
            stream_with_context(generate_stream(user_query, langchain_messages, user_id, session_id, model)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )

    # Non-streaming version
    try:
        result = graph.invoke(
            {
                "user_query": user_query,
                "messages": langchain_messages,
                "user_id": user_id
            },
            config={"configurable": {"thread_id": session_id}}
        )

        # Handle clarification if needed
        clarification_msg = result.get("clarification_message")
        if clarification_msg:
            final_response = clarification_msg
        else:
            final_response = result.get("formatted", {}).get("formatted_response", "I couldn't generate a response.")

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_response},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(final_response.split()),
                "total_tokens": sum(len(m.get("content", "").split()) for m in messages) + len(final_response.split())
            }
        }

        # Return with explicit charset to handle markdown rendering
        resp = jsonify(response)
        resp.headers['Content-Type'] = 'application/json; charset=utf-8'
        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "internal_error"
            }
        }), 500

import re

def append_sas_to_blob_urls(markdown_text: str) -> str:
    """
    Finds all Azure Blob Storage URLs in markdown and appends SAS token.
    """
    sas_token = os.getenv('AZURE_BLOB_SAS_TOKEN', '')
    
    if not sas_token:
        print("⚠️ WARNING: AZURE_BLOB_SAS_TOKEN not set")
        return markdown_text
    
    # Pattern to match blob URLs
    blob_pattern = re.compile(
        r'(https://[a-zA-Z0-9]+\.blob\.core\.windows\.net/[^\s\)]+?)(?=[\s\)\]]|$)'
    )
    
    def add_sas(match):
        url = match.group(1)
        
        # Skip if SAS already present
        if 'sv=' in url or 'sig=' in url:
            return url
        
        # Append SAS token
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}{sas_token}"
    
    return blob_pattern.sub(add_sas, markdown_text)

# ---------- Streaming generator for LibreChat ----------
def generate_stream(user_query, langchain_messages, user_id, session_id, model):
    """
    Streams the response with proper OpenAI SSE format for LibreChat markdown rendering.
    
    KEY FIXES:
    1. Send role: "assistant" in first chunk (critical for markdown rendering)
    2. Stream entire response at once (preserves markdown formatting)
    """
    try:
        result = graph.invoke(
            {
                "user_query": user_query,
                "messages": langchain_messages,
                "user_id": user_id
            },
            config={"configurable": {"thread_id": session_id}}
        )

        # Debug: print what we got back
        print(f"\n🔍 DEBUG generate_stream result keys: {list(result.keys())}")
        print(f"📌 clarification_message: {result.get('clarification_message')}")
        print(f"📌 semantic_chitchat: {result.get('semantic_chitchat')}")

        # Handle clarification if needed
        clarification_msg = result.get("clarification_message")
        if clarification_msg:
            print(f"✅ Clarification detected, returning: {clarification_msg[:100]}...")
            final_response = clarification_msg
        else:
            print(f"📄 No clarification, using formatted response")
            final_response = result.get("formatted", {}).get("formatted_response", "")
            final_response = append_sas_to_blob_urls(final_response)
            if not final_response:
                print(f"⚠️ No formatted response, result keys: {result.keys()}")

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        # ==================== FIX 1: Send role first ====================
        # CRITICAL: LibreChat needs this to know it's an assistant message
        # Without this, it treats the message as plain text instead of markdown
        role_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        # ==================== FIX 2: Stream entire response at once ====================
        # Streaming word-by-word breaks markdown syntax:
        # "**Bold text**" split into ["**Bold", "text**"] renders incorrectly
        # Solution: Send the complete markdown in one chunk
        content_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": final_response},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

        # Final chunk to indicate completion
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        
        # Send role first even in error case
        role_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"
        
        # Then send error message
        error_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n❌ Error: {str(e)}"},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

# ---------- Run Flask ----------
if __name__ == "__main__":
    print("\n🚀 Starting LangGraph RAG Server...")
    print("💡 POST → http://localhost:5001/chat")
    print('   {"question": "your question", "session_id": "user123"}')
    print("\n💡 POST → http://localhost:5001/v1/chat/completions (LibreChat)")
    print('   OpenAI-compatible endpoint\n')
    app.run(debug=False, port=5001, host='0.0.0.0')
