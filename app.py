"""Flask API - OpenAI-compatible endpoints for LibreChat integration"""
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from graph import get_graph
from models import get_initial_state
from memory.checkpointer import close_connections
import config
import uuid
import time

app = Flask(__name__)


# ============================================================================
# OPENAI-COMPATIBLE ENDPOINTS
# ============================================================================

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint.
    
    LibreChat sends:
    - model: Model name (ignored, we use our pipeline)
    - messages: List of {role, content} messages
    
    Headers (configured in LibreChat):
    - X-User-Id: User identifier
    - X-Conversation-Id: Session/conversation identifier
    """
    
    data = request.json
    messages = data.get("messages", [])
    
    # Extract user and session IDs from headers
    user_id = request.headers.get("X-User-Id", "anonymous")
    session_id = request.headers.get("X-Conversation-Id", str(uuid.uuid4()))
    
    # Get the latest user message
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if not user_message:
        return jsonify({
            "error": {
                "message": "No user message found",
                "type": "invalid_request_error"
            }
        }), 400
    
    print("\n" + "🚀"*35)
    print(f"NEW REQUEST - OpenAI Compatible")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print(f"Message: {user_message[:100]}...")
    print("🚀"*35)
    
    try:
        # Get compiled graph
        graph = get_graph()
        
        # Build config for LangGraph
        langgraph_config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id
            }
        }
        
        # Build initial state
        initial_state = get_initial_state(
            user_query=user_message,
            user_id=user_id,
            session_id=session_id
        )
        
        # Add the user message to messages
        initial_state["messages"] = [HumanMessage(content=user_message)]
        
        # Invoke graph
        result = graph.invoke(initial_state, langgraph_config)
        
        # Extract response
        final_response = result.get("final_response", "I couldn't generate a response.")
        needs_clarification = result.get("needs_clarification", False)
        clarification_options = result.get("clarification_options", [])
        
        print("\n" + "✅"*35)
        print("RESPONSE COMPLETE")
        print(f"Clarification needed: {needs_clarification}")
        print(f"Response length: {len(final_response)}")
        print("✅"*35 + "\n")
        
        # Build OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", "rag-pipeline"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(final_response.split()),
                "total_tokens": len(user_message.split()) + len(final_response.split())
            }
        }
        
        # Add metadata if clarification needed
        if needs_clarification:
            response["choices"][0]["message"]["metadata"] = {
                "needs_clarification": True,
                "options": clarification_options
            }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    """
    OpenAI-compatible models endpoint.
    Lists available models for LibreChat.
    """
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "rag-pipeline",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom",
                "permission": [],
                "root": "rag-pipeline",
                "parent": None
            }
        ]
    })


# ============================================================================
# LEGACY ENDPOINTS (backward compatibility)
# ============================================================================

@app.route("/chat", methods=["POST"])
def chat_legacy():
    """
    Legacy chat endpoint for backward compatibility.
    
    Body:
    {
        "question": "What is market share?",
        "session_id": "user123",
        "user_id": "user123"
    }
    """
    
    data = request.json
    question = data.get("question", "")
    session_id = data.get("session_id", str(uuid.uuid4()))
    user_id = data.get("user_id", "anonymous")
    
    if not question:
        return jsonify({"error": "Question required"}), 400
    
    print("\n" + "🚀"*35)
    print(f"NEW REQUEST - Legacy Endpoint")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print(f"Question: {question}")
    print("🚀"*35)
    
    try:
        # Get compiled graph
        graph = get_graph()
        
        # Build config
        langgraph_config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id
            }
        }
        
        # Build initial state
        initial_state = get_initial_state(
            user_query=question,
            user_id=user_id,
            session_id=session_id
        )
        initial_state["messages"] = [HumanMessage(content=question)]
        
        # Invoke graph
        result = graph.invoke(initial_state, langgraph_config)
        
        # Build response
        response = {
            "response": result.get("final_response", ""),
            "needs_clarification": result.get("needs_clarification", False),
            "options": result.get("clarification_options", []),
            "metadata": {
                "confidence": result.get("confidence_score", 0.0),
                "confidence_breakdown": result.get("confidence_breakdown", {}),
                "sources": len(result.get("retrieved_docs", [])),
                "iterations": result.get("iteration_count", 0),
                "enriched_query": result.get("enriched_query", ""),
                "evaluator_reasoning": result.get("evaluator_reasoning", "")
            },
            "session_id": session_id
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": str(e),
            "session_id": session_id
        }), 500


@app.route("/history/<session_id>", methods=["GET"])
def get_history(session_id):
    """
    Get chat history for a session.
    
    Note: With LangGraph, history is stored in MongoDB checkpoints.
    This endpoint retrieves it from there.
    """
    try:
        from memory.checkpointer import get_checkpointer
        
        checkpointer = get_checkpointer()
        config = {"configurable": {"thread_id": session_id}}
        
        # Get the latest checkpoint
        checkpoint = checkpointer.get(config)
        
        if checkpoint:
            messages = checkpoint.get("channel_values", {}).get("messages", [])
            history = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                history.append({"role": role, "content": content})
            
            return jsonify({
                "session_id": session_id,
                "history": history
            })
        else:
            return jsonify({
                "session_id": session_id,
                "history": []
            })
    
    except Exception as e:
        return jsonify({
            "session_id": session_id,
            "history": [],
            "error": str(e)
        })


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "langgraph": True,
            "mongodb_memory": True,
            "openai_compatible": True
        }
    })


# ============================================================================
# SHUTDOWN HANDLER
# ============================================================================

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Clean up on shutdown"""
    pass  # Connection cleanup handled by atexit


import atexit
atexit.register(close_connections)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting RAG Pipeline Server v2.0")
    print(f"   Max Iterations: {config.MAX_ITERATIONS}")
    print(f"   Confidence Thresholds: {config.CONFIDENCE_HIGH} / {config.CONFIDENCE_MEDIUM}")
    print(f"   MongoDB: {config.MONGODB_URI}")
    print(f"   Database: {config.MONGODB_DB_NAME}")
    print("\n📡 Endpoints:")
    print("   POST /v1/chat/completions  (OpenAI-compatible)")
    print("   GET  /v1/models            (OpenAI-compatible)")
    print("   POST /chat                 (Legacy)")
    print("   GET  /history/<session_id> (Legacy)")
    print("   GET  /health               (Health check)")
    print()
    
    app.run(debug=True, port=5001)