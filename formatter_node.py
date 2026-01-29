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

    # ----------------- Production-grade formatting prompt -----------------
    prompt = f"""
You are a professional content formatter for an enterprise RAG system.

Your mission: Transform the RAG answer into a PRODUCTION-GRADE, beautifully formatted response using MARKDOWN.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER'S QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG'S RAW ANSWER (Unformatted)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rag_final_answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE SCORE: {confidence:.2f} / 1.00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR FORMATTING INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Transform the RAW answer above into a POLISHED, PROFESSIONAL response using these guidelines:

1. **STRUCTURE & SECTIONS**
   - Start with a brief, direct answer to the question (1-2 sentences)
   - Use clear section headers with `###` for different topics
   - Separate distinct concepts into logical sections
   - Add blank lines between sections for readability

2. **KEY INFORMATION FORMATTING**
   - **Bold** important numbers, metrics, and key findings
   - Use bullet points (`-`) for lists of items
   - Use numbered lists (`1.`) for sequential information or steps
   - *Italicize* source names, document titles, and time periods

3. **DATA PRESENTATION**
   - Format percentages clearly: **+16.6% YoY** or **41.6% penetration**
   - Format comparisons: **Brand A** vs **Brand B**
   - Use markdown tables when comparing multiple data points:
     ```
     | Metric | Value | Change |
     |--------|-------|--------|
     | Sales  | $10M  | +15%   |
     ```
   - Highlight trends: 📈 for growth, 📉 for decline (optional, only if appropriate)
   - For visual data relationships, use Mermaid diagrams when appropriate

4. **CITATIONS & SOURCES**
   - At the end, add a "### Sources" section
   - List all referenced documents/reports as bullet points
   - Format: `- *Source Name* (Date or Context)`
   - Example: `- *Soaps Annual Presentation 2022 - Nielsen IQ RMS*`

5. **CLARITY & READABILITY**
   - Use short paragraphs (2-4 sentences max)
   - Break up long walls of text
   - Use line breaks generously
   - Make it scannable - readers should quickly find what they need

6. **CONFIDENCE DISCLAIMERS**
   - If confidence < 0.85: Add a note at the top or bottom
   - Format: `> ℹ️ **Note:** This answer has moderate confidence. Please verify critical details from the original sources.`
   - If confidence < 0.65: Be more explicit about uncertainty
   - Format: `> ⚠️ **Disclaimer:** The confidence in this answer is low. Please review the source documents for accurate information.`

7. **CONVERSATIONAL TONE**
   - Make it friendly but professional
   - Use "Based on the data..." or "According to..."
   - Avoid robotic language - be natural

8. **PRESERVE ACCURACY**
   - Keep ALL numbers, dates, and facts EXACTLY as provided
   - Don't add information that wasn't in the RAW answer
   - Don't remove important details

9. **ADVANCED FORMATTING** (Use when appropriate)
   - **Code Blocks**: Use triple backticks with language for code examples
     ```python
     def calculate_growth(old, new):
         return ((new - old) / old) * 100
     ```
   - **Mermaid Diagrams**: For visualizing relationships, flows, or hierarchies
     ```mermaid
     graph TD
         A[Market Share] --> B[GN1: 16.6%]
         A --> C[Lux: 41.6%]
     ```
   - **Math Equations**: Use LaTeX syntax for formulas when needed
     $$\\text{{Growth Rate}} = \\frac{{\\text{{New}} - \\text{{Old}}}}{{\\text{{Old}}}} \\times 100$$
   - **Blockquotes**: Use `>` for important notes or disclaimers
   - **Horizontal Rules**: Use `---` to separate major sections

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

**Visualization of market dynamics:**

```mermaid
graph LR
    A[Soap Market] --> B[GN1: Strong Growth]
    A --> C[Lux: High Penetration]
    B --> D[+16.6% Sales Value]
    C --> E[41.6% Market Share]
```

### Sources

- *Soaps Annual Presentation 2022 - Nielsen IQ RMS*
- *Toilet Soap Annual Presentation 2022 - Kantar (Feb 22, 2023)*

> ℹ️ **Note:** Data represents MAT Dec'22 period. For the most current figures, please refer to the latest quarterly reports.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOW FORMAT THE RAW ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output your formatted response as JSON:
{{
  "formatted_response": "your beautifully formatted markdown response here",
  "metadata": {{
    "confidence": {confidence:.2f}
  }}
}}

REMEMBER:
- Use markdown formatting extensively (tables, code blocks, mermaid diagrams)
- Make it look like a professional chat application response
- Add confidence disclaimer if needed
- Keep all facts accurate
- Make it scannable and easy to read
- Use advanced formatting (tables, mermaid, code blocks) when it ENHANCES understanding
- Don't force advanced formatting if simple bullets/text work better
- Mermaid diagrams are great for: comparisons, hierarchies, flows, relationships
- Tables are great for: comparing metrics across entities, showing data series
- Code blocks are for: formulas, calculations, examples
"""

    # ----------------- Invoke LLM -----------------
    llm_structured = create_llm().with_structured_output(FormatterOutput, method="function_calling")
    output: FormatterOutput = llm_structured.invoke(prompt)

    print(f"\n✅ Formatted response ({len(output.formatted_response)} chars)")

    # ----------------- Return -----------------
    return {
        "messages": sanitize_any(state.messages),  # preserve chat history
        "formatted": sanitize_any(output.dict()),  # structured formatted output
    }
