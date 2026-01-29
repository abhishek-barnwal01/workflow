"""Formatter Node - Polishes RAG output using full prompt and structured output"""

from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from models import PipelineState, FormatterOutput
import json

# ----------------- Helpers -----------------
def create_llm():
    """Create AzureChatOpenAI instance"""
    import config
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=1,
        timeout=30.0,
        max_retries=2,
    )

def safe_utf8(text: str) -> str:
    if not text:
        return ""
    return text.encode("utf-8", errors="replace").decode("utf-8")

def sanitize_any(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        return safe_utf8(obj)
    if isinstance(obj, list):
        return [sanitize_any(i) for i in obj]
    if isinstance(obj, dict):
        return {k: sanitize_any(v) for k, v in obj.items()}
    return obj

# ----------------- Node -----------------
def formatter_node(state: PipelineState) -> Dict[str, Any]:
    """
    Formatter Node:
    - Receives RAG final answer + confidence score
    - Polishes the answer to make it user-friendly
    - Outputs structured FormatterOutput
    """

    print("\n" + "="*70)
    print("✨ FORMATTER NODE")
    print("="*70)

    user_query = state.user_query
    rag_final_answer = state.rag_output.final_answer if state.rag_output else ""
    confidence = state.evaluation.confidence_score if state.evaluation else 0.8

    print(f"\n📝 RAG's answer to format ({len(rag_final_answer)} chars)")
    print(f"🔹 Confidence: {confidence:.2f}")

    # Production-grade formatting prompt with all features
    prompt = f"""You are a professional content formatter for an enterprise RAG system.

Your mission: Transform the RAG answer into a PRODUCTION-GRADE, beautifully formatted response using MARKDOWN.

==================================================
USER'S QUESTION
==================================================
{user_query}

==================================================
RAG'S RAW ANSWER (Unformatted)
==================================================
{rag_final_answer}

==================================================
CONFIDENCE SCORE: {confidence:.2f} / 1.00
==================================================

==================================================
FORMATTING INSTRUCTIONS
==================================================

Transform the RAW answer above into a POLISHED, PROFESSIONAL response with these guidelines:

1. STRUCTURE & SECTIONS
   - Start with a brief, direct answer to the question (1-2 sentences)
   - Use clear section headers with ### for different topics
   - Separate distinct concepts into logical sections
   - Add blank lines between sections for readability

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
   - Use visual data relationships, including Mermaid diagrams when appropriate

4. CITATIONS & SOURCES
   - At the end, add a "### Sources" section
   - List all referenced documents/reports as bullet points
   - Format: - Always use 📄 [filename](content_path)
   - Example: - *Soaps Annual Presentation 2022 - Nielsen IQ RMS*
   - Extract cleaned filename by removing UUID prefix
   - Format as markdown links

5. CLARITY & READABILITY
   - Use short paragraphs (2-4 sentences max)
   - Break up long walls of text
   - Use line breaks generously
   - Make it scannable - readers should quickly find what they need

6. CONFIDENCE DISCLAIMERS
   - If confidence < 0.85: Add a note at the top or bottom
   - Format: > Note: This answer has moderate confidence. Please verify critical details from the original sources.
   - If confidence < 0.65: Be more explicit about uncertainty
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

- *Soaps Annual Presentation 2022 - Nielsen IQ RMS*
- *Toilet Soap Annual Presentation 2022 - Kantar (Feb 22, 2023)*

> Note: Data represents MAT Dec'22 period. For the most current figures, please refer to the latest quarterly reports.

==================================================
FORMAT THE PROVIDED ANSWER
==================================================

Provide your formatted response as markdown directly. Output pure markdown format without wrapping in JSON or code blocks.

Your response should include:
- Use markdown formatting extensively (headers, bold, italic, tables, code blocks, mermaid diagrams)
- Professional chat application appearance
- Confidence disclaimer if confidence < 0.85
- Accurate facts maintained
- Scannable and easy to read structure
- Advanced formatting (tables, mermaid, code blocks) when it enhances understanding
"""

    # Invoke LLM with retry logic for jailbreak detection
    max_retries = 2
    for attempt in range(max_retries):
        try:
            llm_structured = create_llm().with_structured_output(FormatterOutput, method="function_calling")
            output: FormatterOutput = llm_structured.invoke(prompt)
            print(f"\n✅ Formatted response ({len(output.formatted_response)} chars)")
            break
        except ValueError as e:
            error_msg = str(e).lower()
            if "jailbreak" in error_msg or "content filter" in error_msg:
                print(f"\n⚠️ Azure filter triggered (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    # Retry with slightly modified prompt
                    print(f"   Retrying with adjusted prompt...")
                    import time
                    time.sleep(1)
                    continue
                else:
                    # Final fallback
                    print(f"   Using fallback formatter")
                    output = FormatterOutput(
                        formatted_response=f"## Answer\n\n{rag_final_answer}\n\n### Sources\nRefer to original documents.",
                        metadata={"fallback": True, "reason": "jailbreak_filter"}
                    )
                    break
            else:
                raise

    # ----------------- Return -----------------
    return {
        "messages": sanitize_any(state.messages),  # preserve chat history
        "formatted": sanitize_any(output.dict()),  # structured formatted output
    }
