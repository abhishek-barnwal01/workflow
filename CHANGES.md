# Workflow Changes - Intent Detection & Clarification Routing

## Summary

This document describes the improvements made to the RAG workflow to fix intent detection, clarification response routing, and logging verbosity.

## Changes Made

### 1. **Clarification Response Routing** ✅

**Problem**: When users responded to clarification questions (e.g., "1" or "Lux"), the system would re-run intent classification with full chat history, causing confusion and inefficiency.

**Solution**: Implemented a clarification state tracking system that skips intent re-classification and routes directly to semantic enrichment.

#### Implementation Details:

- **Added to `PipelineState` (`models.py`)**:
  ```python
  awaiting_clarification: bool = False  # Track if waiting for clarification
  previous_ambiguity: Optional[AmbiguityInfo] = None  # Store previous ambiguity
  ```

- **Modified `semantic_node.py`**:
  - Detects when `awaiting_clarification = True`
  - Skips intent classification entirely
  - Goes directly to Step 2C (semantic enrichment)
  - Uses clarification context to enrich the original query
  - Clears flags after processing

- **Modified `clarification_node.py`**:
  - Sets `awaiting_clarification = True` when ambiguity detected
  - Stores `previous_ambiguity` for context
  - Clears flags when no ambiguity

#### Flow Example:

**User**: "what is market share of soap"
→ Semantic detects ambiguity: multiple soap brands
→ System asks: "I found multiple soap brands. Please clarify: 1. Lux 2. Dettol..."
→ Sets `awaiting_clarification = True`

**User**: "Lux"
→ Semantic detects `awaiting_clarification = True`
→ **SKIPS** intent classification
→ Goes directly to semantic enrichment (Step 2C)
→ Enriches query: "what is market share of Lux soap"
→ Clears `awaiting_clarification` flag
→ Proceeds to RAG node

### 2. **Logging Configuration** ✅

**Problem**: Terminal logs were extremely verbose with all tool calls, iterations, and debugging information displayed.

**Solution**: Implemented a proper logging system with configurable log levels.

#### Implementation Details:

- **Created `logging_config.py`**:
  - Configurable log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Environment variable: `LOG_LEVEL` (default: INFO)
  - Module-specific loggers: semantic, rag, tools, clarification, formatter
  - Helper functions: `log_section`, `log_subsection`, `log_info`, `log_debug`, etc.

- **Updated all modules** to use logging:
  - `semantic_node.py`: Uses `semantic_logger`
  - `tools.py`: Uses `tools_logger`
  - `clarification_node.py`: Uses `clarification_logger`
  - `app.py`: Uses `workflow_logger`

#### Usage:

```bash
# Default (INFO level - shows main operations)
python app.py

# Debug mode (shows all details including tool calls)
LOG_LEVEL=DEBUG python app.py

# Quiet mode (only warnings and errors)
LOG_LEVEL=WARNING python app.py
```

#### Log Output Comparison:

**Before** (Always verbose):
```
======================================================================
🧠 SEMANTIC NODE - TWO STEP APPROACH
======================================================================
Query: how many u&A reports are there?
User ID: 691c041a052a1a153880f7b3
📚 Loaded 0 user memories
📜 Chat History: 10 messages available
----------------------------------------------------------------------
STEP 1: Intent Classification
----------------------------------------------------------------------
✅ Intent: semantic
   Confidence: 0.86
   Reasoning: The question asks for the number of "u&A reports"...
----------------------------------------------------------------------
STEP 2C: Semantic Enrichment - Full Tool Loop
----------------------------------------------------------------------
--- Iteration 1 ---
🔧 Tool calls: 1
🔍 TOOL CALL: azure_ai_search (HYBRID)
   Query: U&A reports Usage & Attitude reports count...
   Index: semantic (semantic-rag-1767864275169)
   Top K: 50
   ✓ Using hybrid search (keyword + vector)
   ✓ Found 39 docs (avg score: 0.02)
   📄 Top result: None (score: 0.03)
   📝 Preview: Semantic_file_document_taxonomy.md...
```

**After** (INFO level - concise):
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
✅ Enriched: Count of Usage & Attitude (U&A) consumer insight reports...
Ambiguous: True
Entity: u_and_a_report_category
Options: 8
```

**After** (DEBUG level - detailed):
```
[Shows everything including tool calls, iterations, reasoning, etc.]
```

### 3. **Intent Detection Logic** ✅

**Current Behavior**: Intent classification works correctly with three types:

1. **chitchat**: Greetings, thanks, farewells
   - Examples: "hi", "hello", "thanks", "bye"
   - Action: Generate friendly response → END

2. **direct**: General knowledge questions
   - Examples: "what is GDP", "define market share"
   - Action: Light enrichment → RAG with `ambiguity=false`

3. **semantic**: Domain-specific questions
   - Examples: "our market share", "Q3 sales", "all products"
   - Action: Full semantic enrichment with AI search tool → detect ambiguity

**Clarification Response**: Now treated as **semantic** but skips intent classification entirely.

### 4. **Workflow Routing** ✅

The workflow now correctly handles the full lifecycle:

```
Entry: semantic_node
├─ semantic_router
│  ├─ chitchat → END
│  └─ not chitchat → clarification_node
│
├─ clarification_node
│  └─ clarification_router
│     ├─ ambiguous → sets awaiting_clarification=True → END (ask user)
│     └─ not ambiguous → rag_node
│
├─ rag_node (retrieves documents)
├─ formatter_node (formats response)
└─ END
```

**Clarification Loop**:
```
User asks → Ambiguity detected → Ask clarification → User responds
→ semantic_node (with awaiting_clarification=True)
→ Skips intent classification
→ Enriches with clarification context
→ Clears awaiting_clarification
→ Continues to RAG
```

## Testing

### Test Case 1: U&A Reports with Clarification

**Input 1**: "how many u&A reports are there?"

**Expected Output 1**:
```
I found multiple u_and_a_report_category. Please clarify which one you mean:
1. Dipstick report
2. Brand equity report
3. Concept testing report
...
You can also reply 'ALL' to select all options.
```

**Input 2**: "1"

**Expected Behavior**:
- Skips intent classification
- Enriches query: "Count all Dipstick reports..."
- Proceeds to RAG
- Returns: "Short answer: 2 Dipstick (U&A-aligned) reports were found..."

### Test Case 2: Market Share with Brand Clarification

**Input 1**: "what is market share of soap"

**Expected Output 1**:
```
I found multiple soap brands. Please clarify which one you mean:
1. Lux
2. Dettol
...
```

**Input 2**: "Lux"

**Expected Behavior**:
- Skips intent classification
- Enriches query: "what is market share of Lux soap"
- Proceeds to RAG
- Returns answer about Lux market share

### Test Case 3: Chitchat

**Input**: "hey"

**Expected Behavior**:
- Intent: chitchat
- Generates friendly response
- Ends immediately (no RAG)

### Test Case 4: Direct Question

**Input**: "what is EBITDA"

**Expected Behavior**:
- Intent: direct
- Light enrichment (no tools)
- Passes to RAG with ambiguity=false
- Returns definition from documents or general knowledge

## Configuration

### Environment Variables

```bash
# Logging level (default: INFO)
export LOG_LEVEL=DEBUG  # or INFO, WARNING, ERROR, CRITICAL

# Existing Azure configs
export AZURE_OPENAI_DEPLOYMENT=...
export AZURE_OPENAI_ENDPOINT=...
# ... etc
```

### Log Levels

- **DEBUG**: Shows all details (tool calls, iterations, reasoning, embeddings, etc.)
- **INFO**: Shows main operations (intent, enrichment, clarification, RAG results)
- **WARNING**: Shows only warnings and errors
- **ERROR**: Shows only errors
- **CRITICAL**: Shows only critical failures

## Files Modified

### Core Changes:
1. `models.py` - Added `awaiting_clarification` and `previous_ambiguity` to `PipelineState`
2. `semantic_node.py` - Added clarification detection and skip logic
3. `clarification_node.py` - Set/clear clarification flags
4. `logging_config.py` - **NEW FILE** - Logging configuration
5. `tools.py` - Updated to use logging
6. `app.py` - Added logging configuration and log level environment variable

### No Changes Required:
- `graph.py` - Routing logic already works correctly
- `rag_node.py` - No changes needed
- `formatter_node.py` - No changes needed
- `config.py` - No changes needed

## Migration Notes

### Breaking Changes
**None** - All changes are backward compatible.

### New Features
1. Clarification responses now skip intent re-classification
2. Configurable logging levels via `LOG_LEVEL` environment variable
3. Cleaner terminal output by default (INFO level)

### Recommended Actions
1. Set `LOG_LEVEL=INFO` in production for clean logs
2. Set `LOG_LEVEL=DEBUG` during development for detailed debugging
3. Test clarification flows with your specific use cases

## Future Improvements

1. **Multi-turn clarification**: Handle nested clarifications (e.g., "which Lux? Lux Soap or Lux Shampoo?")
2. **Clarification history**: Store clarification decisions for future queries
3. **Smart defaults**: Learn user preferences to avoid repeated clarifications
4. **Structured logging**: Add JSON logging for better log aggregation
5. **Performance metrics**: Log query processing time, token usage, etc.

## Questions & Support

For issues or questions:
1. Check the logs with `LOG_LEVEL=DEBUG`
2. Verify the state flags: `awaiting_clarification`, `previous_ambiguity`
3. Check the message history to understand the flow
4. Review this document for expected behavior

---

**Last Updated**: 2026-01-23
**Version**: 1.0
**Author**: Claude Code
