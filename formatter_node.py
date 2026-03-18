"""Formatter Node - Polishes RAG output using full prompt and structured output"""

import json
from typing import Dict, Any
from models import PipelineState, FormatterOutput
from utils import safe_utf8, sanitize_any, create_llm


# ---------------------------------------------------------------------------
# Shared prompt builder — used by formatter_node (non-streaming) and
# app.py generate_stream (streaming). Single source of truth.
# ---------------------------------------------------------------------------

def build_chart_only_prompt(user_query: str, rag_answer: str, enriched_query: str = "") -> str:
    """
    Prompt for the RAG → formatter path.
    The RAG answer was already streamed live to the user.
    Formatter must analyse the user's intent and output ONLY 1-2 focused chart(s).
    """
    enriched_section = f'\nEnriched/interpreted query: "{enriched_query}"' if enriched_query else ""
    return f"""You are a chart generator for an enterprise RAG system.

The user asked: "{user_query}"{enriched_section}

The following answer has ALREADY been displayed to the user word-by-word:
---
{rag_answer}
---

TASK: Analyse what the user is asking to visualise, then generate ONLY the charts that best answer their specific request.

STEP 1 — UNDERSTAND THE USER'S INTENT:
Think carefully:
- What specific data or metric did the user ask to chart?
- Are they asking for a comparison (e.g. across edits/brands/periods), a summary of key KPIs, a trend over time, or a distribution?
- Identify the single most important dataset in the RAG answer that directly answers the user's visualisation request.
- Do NOT chart every numeric value in the answer. Select only the data most relevant to the user's intent.

STEP 2 — GENERATE FOCUSED CHART:
- If the user asked for a summary → chart the top-level KPI comparison (the one headline metric that summarises the answer).
- If the user asked for a specific metric → chart only that metric.
- If two complementary views are genuinely needed (e.g. raw scores + percentiles), output 2 charts; otherwise output 1.
- Each chart must be self-contained and directly answer the user's question.

OUTPUT RULES:
- Do NOT repeat, reformat, or summarise any text from the answer.
- Do NOT add headers, bullets, explanations, or prose — charts only.
- Output each chart wrapped in the artifact block below.
- If there is genuinely no numeric data relevant to the user's request, output nothing at all.

CHART FORMAT:
• Labels with spaces or special characters MUST use quotes: ["Label with spaces"]
• Avoid special chars like %, +, &, $ in labels (use words instead: "16.6% growth" → ["16.6 percent growth"])
• Use --> for arrows (not => or ->)
• Graph types: graph TD (top-down), graph LR (left-right)

CORRECT FORMAT:
    :::artifact{{type="application/vnd.mermaid" title="Market Analysis"}}
    graph TD
        A["Market Overview"] --> B["Brand A"]
        A --> C["Brand B"]
        B --> D["Growth: 16.6 percent YoY"]
        C --> E["Penetration: 41.6 percent"]
    :::

- BAR CHARTS (comparing metrics across categories):
    :::artifact{{type="application/vnd.mermaid" title="Sales Comparison"}}
    %%{{init: {{'theme':'base'}}}}%%
    xychart-beta
        title "Brand Sales Growth (YoY)"
        x-axis ["GN1", "Lux", "Lifebuoy", "Dove", "Santoor"]
        y-axis "Growth Percent" 0 --> 20
        bar [16.6, 8.2, 12.4, 5.7, 10.1]
    :::

- LINE CHARTS (trends over time):
    :::artifact{{type="application/vnd.mermaid" title="Market Share Trend"}}
    %%{{init: {{'theme':'base'}}}}%%
    xychart-beta
        title "GN1 Market Share Trend"
        x-axis ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        y-axis "Market Share Percent" 0 --> 20
        line [12.5, 13.2, 13.8, 14.5, 15.1, 16.6]
    :::

- PIE CHARTS (showing proportions):
    :::artifact{{type="application/vnd.mermaid" title="Category Share"}}
    %%{{init: {{'theme':'base'}}}}%%
    pie title Market Share by Brand
        "GN1" : 16.6
        "Lux" : 41.6
        "Lifebuoy" : 18.5
        "Others" : 23.3
    :::

- MULTIPLE DATA SERIES (comparing trends):
    :::artifact{{type="application/vnd.mermaid" title="Brand Performance"}}
    %%{{init: {{'theme':'base'}}}}%%
    xychart-beta
        title "Sales vs Penetration Trends"
        x-axis ["Q1", "Q2", "Q3", "Q4"]
        y-axis "Percent" 0 --> 50
        line [10, 12, 15, 16.6]
        line [35, 38, 40, 41.6]
    :::

Output the chart artifact(s) only. Begin immediately — no preamble.
"""


def build_formatter_prompt(user_query: str, rag_answer: str, confidence: float) -> str:
    return f"""You are a professional content formatter for an enterprise RAG system.

Your mission: Transform the RAG answer into a PRODUCTION-GRADE, beautifully formatted response using MARKDOWN.

---
USER'S QUESTION
---
{user_query}

---
RAG'S RAW ANSWER (Unformatted)
---
{rag_answer}

---
CONFIDENCE SCORE: {confidence:.2f} / 1.00
---

---
FORMATTING INSTRUCTIONS
---

Transform the RAW answer above into a POLISHED, PROFESSIONAL response with these guidelines:

1. STRUCTURE & SECTIONS
   - Start with a brief, direct answer to the question (1-2 sentences)
   - Use clear section headers with ### for different topics
   - Separate distinct concepts into logical sections
   - Add blank lines between sections for readability
   → For REPORT SUMMARIES: Render a clean business style report using the same schema.
    - The RAW answer already uses a business report structure, do NOT prepend a one-line answer; preserve the existing sections, headings, tables, bullets, and citations.
    - Avoid single-line slot responses in reports; write multi-sentence paragraphs within each section.
    - Include quantitative tables where applicable (e.g., metrics vs norms).

2. KEY INFORMATION FORMATTING
   - Use **bold** for important numbers, metrics, and key findings
   - Use bullet points (-) for lists of items
   - Use numbered lists (1.) for sequential information or steps
   - Use *italics* for source names, document titles, and time periods

3. DATA PRESENTATION
   - Format percentages clearly: **+16.6% YoY** or **41.6% penetration**
   - Format comparisons: **Brand A** vs **Brand B**
   - Create markdown tables when comparing multiple data points
   - Example table:
     | Metric | Value | Change |
     |--------|-------|--------|
     | Sales  | $10M  | +15%   |
   - Highlight trends: 📈 for growth, 📉 for decline (when appropriate)
   - Use visual data relationships, including Mermaid diagrams when only when explicitly requested or when it significantly enhances understanding (avoid overuse)

4. CITATIONS & SOURCES
   - At the end, add a "### Sources" section
   - List all referenced documents/reports as bullet points
   - Format: - Always use 📄 [filename](content_path)
   - Always include page number when available: 📄 [filename](content_path) (Page N)
   - Example: - *Soaps Annual Presentation 2022 - Nielsen IQ RMS* (Page 5)
   - Extract cleaned filename by removing UUID prefix
   - Format as markdown links
   - For LISTING queries (user asked to "list all X", "show all X documents"):
     Do NOT add a separate "### Sources" section — the document list with links IS the answer.
     Adding Sources would duplicate the same documents at the bottom. Just present the list cleanly in [filename1](url). If url is not available, just list the filename without link.

5. CLARITY & READABILITY
   - Use short paragraphs (2-4 sentences max)
   - Break up long walls of text
   - Use line breaks generously
   - Make it scannable - readers should quickly find what they need

6. CONFIDENCE DISCLAIMERS
   - If confidence < 0.65: Add a note at the bottom
   - Format: > Note: This answer has moderate confidence. Please verify critical details from the original sources.
   - If confidence < 0.35: Be more explicit about uncertainty
   - Format: > Disclaimer: The confidence in this answer is low. Please review the source documents for accurate information.

7. CONVERSATIONAL TONE
   - Make it friendly but professional
   - Use "Based on the data..." or "According to..."
   - Maintain natural language, not robotic

8. PRESERVE ACCURACY
   - Keep ALL numbers, dates, and facts EXACTLY as provided
   - Add only information that was in the RAW answer
   - Retain all important details

9. ADVANCED FORMATTING OPTIONS
   - Code Blocks: Use triple backticks with language for examples
     ```python
     def calculate_growth(old, new):
         return ((new - old) / old) * 100
     ```
   - Mermaid Diagrams: For visualizing relationships, flows, or hierarchies
     CRITICAL SYNTAX RULES:
     • Labels with spaces or special characters MUST use quotes: ["Label with spaces"]
     • Avoid special chars like %, +, &, $ in labels (use words instead: "16.6% growth" → ["16.6 percent growth"])
     • Use --> for arrows (not => or ->)
     • Graph types: graph TD (top-down), graph LR (left-right)

     CORRECT FORMAT:
     :::artifact{{type="application/vnd.mermaid" title="Market Analysis"}}
     graph TD
         A["Market Overview"] --> B["Brand A"]
         A --> C["Brand B"]
         B --> D["Growth: 16.6 percent YoY"]
         C --> E["Penetration: 41.6 percent"]
     :::

    - BAR CHARTS (comparing metrics across categories):
     :::artifact{{type="application/vnd.mermaid" title="Sales Comparison"}}
     %%{{init: {{'theme':'base'}}}}%%
     xychart-beta
         title "Brand Sales Growth (YoY)"
         x-axis ["GN1", "Lux", "Lifebuoy", "Dove", "Santoor"]
         y-axis "Growth Percent" 0 --> 20
         bar [16.6, 8.2, 12.4, 5.7, 10.1]
     :::

    - LINE CHARTS (trends over time):
     :::artifact{{type="application/vnd.mermaid" title="Market Share Trend"}}
     %%{{init: {{'theme':'base'}}}}%%
     xychart-beta
         title "GN1 Market Share Trend"
         x-axis ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
         y-axis "Market Share Percent" 0 --> 20
         line [12.5, 13.2, 13.8, 14.5, 15.1, 16.6]
     :::

    - PIE CHARTS (showing proportions):
     :::artifact{{type="application/vnd.mermaid" title="Category Share"}}
     %%{{init: {{'theme':'base'}}}}%%
     pie title Market Share by Brand
         "GN1" : 16.6
         "Lux" : 41.6
         "Lifebuoy" : 18.5
         "Others" : 23.3
     :::

    - MULTIPLE DATA SERIES (comparing trends):
     :::artifact{{type="application/vnd.mermaid" title="Brand Performance"}}
     %%{{init: {{'theme':'base'}}}}%%
     xychart-beta
         title "Sales vs Penetration Trends"
         x-axis ["Q1", "Q2", "Q3", "Q4"]
         y-axis "Percent" 0 --> 50
         line [10, 12, 15, 16.6]
         line [35, 38, 40, 41.6]
     :::

   - Math Equations: Use LaTeX syntax for formulas
     $$Growth Rate = (New - Old) / Old * 100$$
   - Blockquotes: Use > for important notes or disclaimers
   - Horizontal Rules: Use --- to separate major sections

CRITICAL RULES:
• Bar/Line charts: Use "xychart-beta" keyword
• Always include: title, x-axis, y-axis, data
• NO special chars in labels (%, +, &, $) - spell out "percent", "dollars"
• Y-axis range: "0 --> maxValue" (use arrows)
• Multiple bars/lines: add multiple "bar [...]" or "line [...]" rows
• Pie charts: Use "pie title" syntax
• All charts work in artifacts: :::artifact{{type="application/vnd.mermaid" title="..."}}...:::

WHEN TO USE EACH CHART TYPE:
• **Flow diagrams**: Show relationships, hierarchies, processes
• **Bar charts**: Compare values across categories (sales, growth, market share)
• **Line charts**: Show trends over time (quarterly performance, historical data)
• **Pie charts**: Show composition/proportions (market share distribution)
• **Multiple series**: Compare two metrics side-by-side (sales vs penetration)

==================================================
EXAMPLE OUTPUT FORMAT
==================================================

Based on Nielsen IQ RMS data, **Godrej No. 1 achieved +16.6% sales value growth** compared to the previous year in MAT Dec'22.

### Key Insights

- **Sales Performance:** GN1 grew **+16.6% in sales value** year-over-year
- **Market Position:** Despite Lux having higher penetration (41.6%), GN1 demonstrated stronger volume performance
- **Consumer Behavior:** GN1 buyers showed higher consumption intensity compared to Lux users

### Market Context

The Kantar household panel data shows complementary insights:
- **Penetration:** 41.6% (with **+15% YoY growth**)
- **Average Consumption:** 1.62 gms/HH/month (**+11% YoY growth**)

### Comparative Analysis

| Brand | Sales Growth | Penetration | Volume Intensity |
|-------|-------------|-------------|------------------|
| **GN1** | +16.6% YoY | 41.6% | High |
| **Lux** | - | Higher | Lower |

### Market Dynamics Visualization

:::artifact{{type="application/vnd.mermaid" title="GN1 Growth vs PY (Kantar Panel, MAT Dec'22)"}}
graph TD
    A["Toilet Soaps Market 2022"] --> B["Godrej No.1"]
    B --> C["Value Growth: Up 16.6 percent YoY"]
    B --> D["ASP and Mix Improvement"]
    B --> E["Penetration: 41.6 percent"]
    A --> F["Category Volume Pressure"]
:::

### Sources

- *Soaps Annual Presentation 2022 - Nielsen IQ RMS* (Page 5)
- *Toilet Soap Annual Presentation 2022 - Kantar (Feb 22, 2023)* (Page 12)

> Note: Data represents MAT Dec'22 period. For the most current figures, please refer to the latest quarterly reports.

==================================================
FORMAT THE PROVIDED ANSWER
==================================================

Provide your formatted response as markdown directly. Output pure markdown format without wrapping in JSON or code blocks.
Do NOT include any preface like "Here's the polished..." or "Formatted response:". Begin directly with the content.
Avoid meta commentary such as "Below is" or "Here is".

Your response should include:
- Use markdown formatting extensively (headers, bold, italic, tables, code blocks, mermaid diagrams)
- Professional chat application appearance
- Confidence disclaimer if confidence < 0.85
- Accurate facts maintained
- Scannable and easy to read structure
- Advanced formatting (tables, mermaid, code blocks) when it is explicitly requested by user or when it enhances understanding
"""


# ----------------- Node -----------------
def formatter_node(state: PipelineState) -> Dict[str, Any]:
    """
    Formatter Node — only runs when rag_node sets needs_formatter=True.
    Triggered when the user explicitly requests charts or graphs.

    Uses plain llm.invoke() (not with_structured_output) so tokens are
    emitted as content chunks and captured by graph.astream_events(),
    letting chart code stream live to the client right after the RAG answer.
    """

    print("\n" + "="*70)
    print("✨ FORMATTER NODE")
    print("="*70)

    user_query = state.user_query

    # DEBUG: Print the full RAG output
    print("\n" + "-"*70)
    print("🐛 DEBUG: RAG OUTPUT RECEIVED")
    print("-"*70)
    if state.rag_output:
        print(f"RAG Output Type: {type(state.rag_output)}")
        print(f"RAG Output Keys: {state.rag_output.keys() if hasattr(state.rag_output, 'keys') else 'N/A (not dict)'}")
        print(f"\nFull RAG Output:\n{json.dumps(state.rag_output, indent=2, default=str)}")
    else:
        print("⚠️ RAG Output is None!")
    print("-"*70)

    # ✅ PRIORITY 1: Clarification question - return as-is (no formatting needed)
    if state.clarification_message and state.awaiting_clarification:
        print("📝 Returning clarification question")
        return {
            "messages": sanitize_any(state.messages),
            "formatted": sanitize_any({
                "formatted_response": safe_utf8(state.clarification_message),
                "metadata": {"source": "clarification", "confidence": 1.0}
            })
        }

    # ✅ PRIORITY 2: Direct answer from semantic (no prior RAG streaming) — full reformat
    if state.clarification_message and not state.awaiting_clarification:
        print("📝 Formatting direct answer from semantic node (full reformat)")
        rag_final_answer = state.clarification_message
        confidence = 1.0
        prompt = build_formatter_prompt(user_query, rag_final_answer, confidence)
        print(f"\n📝 Direct answer to format ({len(rag_final_answer)} chars)")
        print(f"🔹 Confidence: {confidence:.2f}")
    else:
        # RAG already streamed its full answer live — only append charts, no repeat text
        rag_final_answer = state.rag_output.final_answer if state.rag_output else ""
        confidence = state.evaluation.confidence_score if state.evaluation else 0.8
        prompt = build_chart_only_prompt(user_query, rag_final_answer, state.enriched_query or "")
        print(f"\n📊 Chart-only formatter ({len(rag_final_answer)} chars RAG input)")
        print(f"🔹 Confidence: {confidence:.2f}")

    # Plain invoke (not with_structured_output) so tokens are emitted as
    # content chunks and captured by graph.astream_events() for live streaming.
    max_retries = 2
    formatted_text = ""
    for attempt in range(max_retries):
        try:
            response = create_llm().invoke(prompt)
            formatted_text = response.content if hasattr(response, "content") else str(response)
            print(f"\n✅ Formatted response ({len(formatted_text)} chars)")
            break
        except Exception as e:
            error_msg = str(e).lower()
            if "jailbreak" in error_msg or "content filter" in error_msg or "content_filter" in error_msg or "responsibleai" in error_msg or "400" in error_msg:
                print(f"\n⚠️ Azure filter triggered (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                else:
                    formatted_text = f"## Answer\n\n{rag_final_answer}\n\n### Sources\nRefer to original documents."
                    break
            else:
                raise
    output = FormatterOutput(formatted_response=formatted_text, metadata={"source": "formatter"})

    # ----------------- Return -----------------
    return {
        "messages": sanitize_any(state.messages),  # preserve chat history
        "formatted": sanitize_any(output.dict()),   # structured formatted output
    }
