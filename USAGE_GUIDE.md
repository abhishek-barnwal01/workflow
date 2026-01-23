# RAG Workflow Usage Guide

## Quick Start

### Running the Server

```bash
# Default (INFO logging)
python app.py

# Debug mode (verbose logging)
LOG_LEVEL=DEBUG python app.py

# Quiet mode (only warnings/errors)
LOG_LEVEL=WARNING python app.py
```

The server will start on `http://localhost:5001`

## How It Works

### Intent Detection Flow

When you ask a question, the system automatically detects the intent:

```
┌─────────────────────────────────────────┐
│         User Question                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     Semantic Node                       │
│  • Intent Classification                │
│  • Query Enrichment                     │
│  • Ambiguity Detection                  │
└──────────────┬──────────────────────────┘
               │
               ▼
        ┌──────┴──────┐
        │             │
    chitchat      direct/semantic
        │             │
        ▼             ▼
      END      Clarification Node
                     │
              ┌──────┴──────┐
              │             │
          ambiguous     clear
              │             │
              ▼             ▼
        Ask User        RAG Node
              │             │
              ▼             ▼
        User Reply    Formatter
              │             │
              └─────►END◄───┘
```

### Intent Types

#### 1. **Chitchat** 💬
Simple greetings and casual conversation.

**Examples**:
- "hi"
- "hello"
- "thanks"
- "bye"
- "how are you"

**Response**: Friendly message, no document search.

```
User: "hey"
Assistant: "Hello! I'm here to help you with document-related questions. What would you like to know?"
```

#### 2. **Direct** 📖
General knowledge questions that don't require company-specific data.

**Examples**:
- "what is GDP"
- "define market share"
- "explain EBITDA"
- "what does ROI mean"

**Response**: Definition or explanation from general knowledge or documents.

```
User: "what is market share"
Assistant: "Market share is the percentage of total sales in a market captured by a particular company or product..."
```

#### 3. **Semantic** 🔍
Domain-specific questions requiring company document search.

**Examples**:
- "what is OUR market share"
- "show Q3 sales data"
- "compare Lux and Dettol performance"
- "how many reports are there"

**Response**: Answer from company documents, may ask for clarification if ambiguous.

```
User: "what is market share of soap"
Assistant: "I found multiple soap brands. Please clarify which one you mean:
1. Lux
2. Dettol
3. Lifebuoy
..."

User: "Lux"
Assistant: "Lux soap has a market share of 24.5% in the urban segment as of Q3 2023..."
```

## Clarification Handling

### When Does Clarification Happen?

The system asks for clarification when:
1. Multiple values exist for an entity (e.g., multiple brands, categories, regions)
2. The query is ambiguous
3. No context from chat history resolves the ambiguity

### How to Respond to Clarification

You can respond in three ways:

1. **By number**: "1" or "2"
2. **By name**: "Lux" or "Dipstick report"
3. **All options**: "ALL" or "all of them"

**Example**:

```
System: "I found multiple u_and_a_report_category. Please clarify:
1. Dipstick report
2. Brand equity report
3. Concept testing report
You can also reply 'ALL' to select all options."

✅ Valid responses:
- "1" → Selects Dipstick report
- "Dipstick report" → Selects Dipstick report
- "ALL" → Selects all categories
```

### What Happens After Clarification?

1. Your response is captured
2. System **skips** intent re-classification
3. Goes directly to semantic enrichment
4. Enriches the original query with your selection
5. Proceeds to document retrieval
6. Returns the answer

**Technical Flow**:
```
Original query: "how many u&A reports are there?"
Clarification: User selects "1" (Dipstick report)
Enriched query: "Count all Dipstick reports in the repository"
Result: "2 Dipstick reports found..."
```

## Logging Levels

### INFO (Default)
Shows main operations without overwhelming detail.

**Output**:
```
======================================================================
🧠 SEMANTIC NODE
======================================================================
Query: how many u&A reports are there?
----------------------------------------------------------------------
STEP 1: Intent Classification
----------------------------------------------------------------------
✅ Intent: semantic (confidence: 0.86)
----------------------------------------------------------------------
STEP 2C: Semantic Enrichment
----------------------------------------------------------------------
✅ Enriched: Count of Usage & Attitude (U&A) reports...
Ambiguous: True
Entity: u_and_a_report_category
Options: 8
📌 Clarification requested: u_and_a_report_category (8 options)
```

### DEBUG
Shows all details including tool calls, iterations, reasoning.

**Output**:
```
[All INFO output plus:]
🔍 Loaded 0 user memories
📜 Chat History: 10 messages available
🔍 Iteration 1
🔧 Tool calls: 1
🔍 azure_ai_search: semantic, query='U&A reports Usage...', top_k=50
✓ Using hybrid search (keyword + vector)
✓ Found 39 docs (avg score: 0.02)
📄 Top result: Semantic_file_document_taxonomy.md (score: 0.03)
[Reasoning messages, internal thoughts, etc.]
```

### WARNING
Only shows warnings and errors.

**Output**:
```
⚠️ Could not load memories: Connection timeout
⚠️ Embedding error: Rate limit exceeded
```

### ERROR
Only critical errors.

**Output**:
```
❌ Failed to retrieve documents: Index not found
```

## API Endpoints

### 1. `/chat` - Simple Chat Endpoint

**Request**:
```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "how many u&A reports are there?",
    "session_id": "user123"
  }'
```

**Response**:
```json
{
  "response": "I found multiple u_and_a_report_category...",
  "needs_clarification": true,
  "session_id": "user123"
}
```

### 2. `/v1/chat/completions` - OpenAI-Compatible Endpoint

**Request**:
```bash
curl -X POST http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag",
    "messages": [
      {"role": "user", "content": "how many u&A reports are there?"}
    ],
    "stream": false
  }'
```

**Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1705987200,
  "model": "rag",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I found multiple u_and_a_report_category..."
    },
    "finish_reason": "stop"
  }]
}
```

**Streaming** (set `"stream": true`):
```bash
# Returns Server-Sent Events (SSE)
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",...}
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",...}
data: [DONE]
```

## Best Practices

### 1. Use Session IDs
Always provide a `session_id` to maintain conversation context:
```json
{
  "question": "What about Lux?",
  "session_id": "user123"  // ← Maintains context
}
```

### 2. Be Specific
Provide context in your questions to avoid clarification:
```
❌ "what is market share of soap"
✅ "what is market share of Lux soap in urban India"
```

### 3. Handle Clarification Gracefully
When you receive `needs_clarification: true`, present options to the user and send their selection back with the same `session_id`.

### 4. Set Appropriate Log Levels
- **Development**: `LOG_LEVEL=DEBUG`
- **Production**: `LOG_LEVEL=INFO`
- **Troubleshooting**: `LOG_LEVEL=DEBUG`
- **Performance testing**: `LOG_LEVEL=WARNING`

### 5. Monitor Ambiguity Patterns
If certain queries always require clarification, consider:
- Adding more context to the semantic index
- Pre-filtering common ambiguities
- Storing user preferences

## Troubleshooting

### Issue: "Intent keeps changing"
**Cause**: Chat history might be influencing classification.
**Solution**: Check the chat history in DEBUG mode:
```bash
LOG_LEVEL=DEBUG python app.py
```

### Issue: "Clarification loop not working"
**Cause**: Session state might be lost.
**Solution**:
1. Verify you're using the same `session_id`
2. Check PostgreSQL connection
3. Verify `awaiting_clarification` flag in logs

### Issue: "Too much logging"
**Cause**: Running in DEBUG mode.
**Solution**:
```bash
LOG_LEVEL=INFO python app.py
```

### Issue: "Not enough logging"
**Cause**: Running in WARNING or ERROR mode.
**Solution**:
```bash
LOG_LEVEL=DEBUG python app.py
```

### Issue: "Ambiguity not detected"
**Cause**: Semantic search might not be finding multiple options.
**Solution**:
1. Check semantic index has the data
2. Increase `top_k` in search
3. Verify entity extraction logic

## Examples

### Example 1: Complete Clarification Flow

```
User: "how many u&A reports are there?"

System: 🧠 SEMANTIC NODE
        Intent: semantic (confidence: 0.86)
        Enriched: Count of U&A reports...
        Ambiguous: True

Response: "I found multiple u_and_a_report_category. Please clarify:
           1. Dipstick report
           2. Brand equity report
           3. Concept testing report
           ..."

User: "1"

System: 🧠 SEMANTIC NODE
        🔄 CLARIFICATION RESPONSE DETECTED
        Response: '1'
        Previous ambiguity: u_and_a_report_category
        → Skipping intent classification
        Enriched: Count all Dipstick reports...
        Ambiguous: False

Response: "Short answer: 2 Dipstick (U&A-aligned) reports were found..."
```

### Example 2: Direct Question

```
User: "what is EBITDA"

System: 🧠 SEMANTIC NODE
        Intent: direct (confidence: 0.85)
        Enriched: Define EBITDA...

Response: "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization..."
```

### Example 3: Chitchat

```
User: "hey"

System: 🧠 SEMANTIC NODE
        Intent: chitchat (confidence: 0.92)

Response: "Hello! I'm here to help you with document-related questions. What would you like to know?"
```

## Advanced Usage

### Custom Log Level in Code

```python
from logging_config import set_log_level

# Change log level dynamically
set_log_level("DEBUG")
```

### Check Clarification State

```python
# In your client
response = requests.post("/chat", json={
    "question": "1",
    "session_id": "user123"
})

# Check if system is in clarification mode
if response.json().get("needs_clarification"):
    # Handle clarification UI
    pass
```

### Access Conversation State

```python
# The state is automatically persisted in PostgreSQL
# Access it via LangGraph's checkpointer
from persistence import checkpointer

state = checkpointer.get({"thread_id": "user123"})
print(state.get("awaiting_clarification"))
print(state.get("previous_ambiguity"))
```

---

**Need Help?**
- Check logs with `LOG_LEVEL=DEBUG`
- Review `CHANGES.md` for technical details
- Verify session IDs are consistent
- Check PostgreSQL connection

**Last Updated**: 2026-01-23
