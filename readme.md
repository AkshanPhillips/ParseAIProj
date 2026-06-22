# Parse.ai Loan Portfolio Analytics Dashboard

An AI-powered, agentic Streamlit application designed to dynamically ingest raw loan tape data, automatically map unknown schemas to a canonical data model, and generate deterministic credit portfolio metrics and visualisations.

Built as a submission for the **Parse.ai Engineering Case Study**.

---

##  Architecture Overview

The application is built using an **Agentic Architecture** orchestrated by [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/). Instead of relying on hard-coded rules, specialized AI agents handle different stages of the data pipeline, ensuring robustness against unknown or changing input schemas.

### The Agents
1. **Schema Discovery Agent (LLM-Powered)**
   - **Tool**: Groq API (Llama-3.1-8b-instant).
   - **Role**: Ingests raw CSVs, samples the columns, and intelligently maps them to the system's canonical schema using strict JSON constraints. Automatically adapts to completely different file structures (e.g., Loan Tape 1 vs. Loan Tape 2).
2. **Data Validation Agent**
   - **Role**: Cleanses the data, standardizes date formats, coerces numeric types, calculates dynamic fields (e.g., standard DPD buckets: `Current`, `1-30`, `31-60`, `61-90`, `90+`), and generates a data quality report.
3. **Metric Computation Agent**
   - **Role**: Deterministically calculates portfolio metrics (e.g., Principal Outstanding, Collection Efficiency, Transition flows) and safely stores them in the graph's state for the UI layer.
4. **Visualisation Agent**
   - **Role**: A Streamlit-based UI that renders interactive Plotly charts. Applies the strict visualisation guidelines (directional color palettes, time-window controls) and allows slicing by Product, Region, and City.
5. **Interaction Agent (LLM-Powered)**
   - **Tool**: Groq API (Llama-3.1-8b-instant) + Chat Interface.
   - **Role**: Acts as a senior credit risk analyst. It reads the computed state/metrics and allows the user to ask natural language questions about the specific portfolio data currently rendered on the screen.

---

##  Key Features

- **Schema-Agnostic Ingestion**: Drop any loan tape CSV into the target directory; the AI dynamically figures out what the columns mean.
- **Portfolio KPIs**: Real-time calculation of Active Loans, Total POS, Interest Outstanding, WA Interest Rate, and WA Remaining Tenor.
- **Transition Matrices**: Interactive $N \times N$ heatmaps tracking absolute and percentage-based POS/Loan Count migration across DPD buckets over user-defined time windows.
- **Collections Efficiency**: Time-series analysis and DPD-bucket breakdowns showing EMI Due vs. Amount Collected.
- **Vintage Curves**: Principal repayment tracking by disbursement cohort (Month on Book).
- **Interactive Chat Q&A**: Ask questions directly to the dashboard to uncover risk signals and portfolio insights.

---



# Install dependencies
pip install pandas numpy streamlit plotly groq langgraph