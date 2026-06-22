# -*- coding: utf-8 -*-
"""
Parse.ai Loan Portfolio Analytics — Agentic Streamlit App
Architecture: LangGraph orchestrator with 5 specialised agents
  1. SchemaDiscoveryAgent   — maps raw CSV columns → canonical schema (via Groq LLM)
  2. DataValidationAgent    — checks completeness, types, referential integrity
  3. MetricComputationAgent — computes all 9 required metrics (serialisable outputs only)
  4. VisualisationAgent     — renders charts/tables in Streamlit
  5. InteractionAgent       — answers user questions about the data (via Groq LLM)
  Orchestrator (LangGraph StateGraph) — coordinates agents 1-3; 4-5 run in Streamlit
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from langgraph.graph import END, StateGraph

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/akshanphillipsoffice/Desktop/parseai/data/Large data set" 
    # Change this to the Large dataset folder to test!


GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "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"
)# set via env var — never hard-code
CANONICAL_SCHEMA = [
    "loan_id", "disbursement_date", "disbursed_principal", "interest_rate",
    "maturity_date", "loan_status", "snapshot_date", "dpd",
    "principal_outstanding", "interest_outstanding", "expected_emi_amount",
    "actual_principal_collected", "actual_interest_collected",
    "Product", "Region", "City",
]

DPD_BUCKETS = ["Current (0 DPD)", "DPD 1–30", "DPD 31–60", "DPD 61–90", "DPD 90+"]

COLOR_SCHEME = {
    "Current (0 DPD)": "#2ca02c",
    "DPD 1–30":        "#ffbf00",
    "DPD 31–60":       "#ff7f0e",
    "DPD 61–90":       "#d62728",
    "DPD 90+":         "#8b0000",
}

# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH STATE
# NOTE: metrics dict must only contain pickle-safe types (DataFrames, lists,
# dicts, scalars). No local functions / closures — those break st.cache_data.
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    data_dir: str
    raw_dfs: Dict[str, pd.DataFrame]
    ai_mappings: Dict[str, dict]
    merged_df: Optional[pd.DataFrame]
    active_df: Optional[pd.DataFrame]   # active-loan panel, stored for on-demand use
    validation_report: Dict[str, Any]
    metrics: Dict[str, Any]
    errors: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# GROQ HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _groq_chat(system: str, user: str, json_mode: bool = False) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", GROQ_API_KEY))
    kwargs: dict = dict(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — SCHEMA DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def schema_discovery_agent(state: AgentState) -> AgentState:
    """Reads CSVs, maps columns to canonical schema via Groq LLM."""
    data_dir = state["data_dir"]
    errors: List[str] = list(state.get("errors", []))
    raw_dfs: Dict[str, pd.DataFrame] = {}
    ai_mappings: Dict[str, dict] = {}

    try:
        all_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".csv")
        ]
        if not all_files:
            errors.append(f"No CSV files found in {data_dir}")
            return {**state, "errors": errors}

        for filepath in all_files:
            fname = os.path.basename(filepath)
            sample = pd.read_csv(filepath, nrows=10, low_memory=False)
            raw_cols = sample.columns.tolist()

            system_prompt = textwrap.dedent(f"""
                You are a financial data engineer. Map raw loan-tape column names to our canonical schema.
                Canonical columns: {CANONICAL_SCHEMA}
                Rules:
                1. Output ONLY valid JSON: {{"raw_col": "canonical_col", ...}}
                2. Keys MUST be raw column names. Values MUST be canonical column names.
                3. Map each canonical column at most once (best match only).
                4. 'actual_principal_collected' and 'actual_interest_collected' map to
                   monthly collected amounts — NOT cumulative totals.
                5. Skip columns with no good match.
            """)
            user_prompt = f"Raw columns for '{fname}': {raw_cols}"

            mapping: dict = json.loads(_groq_chat(system_prompt, user_prompt, json_mode=True))

            # Auto-fix if LLM returned {canonical→raw} instead of {raw→canonical}
            if sum(1 for k in mapping if k in CANONICAL_SCHEMA) > len(mapping) / 2:
                mapping = {v: k for k, v in mapping.items()}

            ai_mappings[fname] = mapping

            full_df = pd.read_csv(filepath, low_memory=False).rename(columns=mapping)
            keep = [c for c in CANONICAL_SCHEMA if c in full_df.columns]
            if keep:
                raw_dfs[fname] = full_df[keep]

    except Exception as exc:
        errors.append(f"SchemaDiscoveryAgent error: {exc}")

    return {**state, "raw_dfs": raw_dfs, "ai_mappings": ai_mappings, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# MERGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate canonical column names and always return a clean RangeIndex."""
    df = df.reset_index(drop=True)
    if not df.columns.duplicated().any():
        return df
    result = {}
    for col in df.columns.unique():
        block = df.loc[:, df.columns == col]
        result[col] = block.bfill(axis=1).iloc[:, 0] if block.shape[1] > 1 else block.iloc[:, 0]
    return pd.DataFrame(result).reset_index(drop=True)


def _build_merged_df(raw_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    dfs = sorted(raw_dfs.values(), key=len, reverse=True)
    if not dfs:
        return pd.DataFrame()
    base = _collapse_duplicates(dfs[0])
    for other in dfs[1:]:
        other = _collapse_duplicates(other)
        if "loan_id" in base.columns and "loan_id" in other.columns:
            new_cols = [c for c in other.columns if c != "loan_id" and c not in base.columns]
            if new_cols:
                static = other.drop_duplicates(subset=["loan_id"], keep="last")
                base = base.merge(
                    static[["loan_id"] + new_cols], on="loan_id", how="left"
                ).reset_index(drop=True)
    return _collapse_duplicates(base)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — DATA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def data_validation_agent(state: AgentState) -> AgentState:
    """Parses dates, coerces numerics, assigns DPD buckets, builds validation report."""
    errors = list(state.get("errors", []))
    raw_dfs = state.get("raw_dfs", {})
    report: Dict[str, Any] = {}

    # Support being called with a pre-merged df (filter re-run path)
    if state.get("merged_df") is not None and not raw_dfs:
        df = state["merged_df"].copy()
    elif raw_dfs:
        df = _build_merged_df(raw_dfs)
    else:
        errors.append("DataValidationAgent: no data to validate.")
        return {**state, "validation_report": report, "errors": errors}

    # Ensure clean integer RangeIndex before any boolean masking
    df = df.reset_index(drop=True)

    for col, default in [("Product", "Unknown"), ("Region", "Unknown"), ("City", "Unknown")]:
        if col not in df.columns:
            df[col] = default

    for dcol in ("snapshot_date", "disbursement_date", "maturity_date"):
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    num_cols = [
        "dpd", "principal_outstanding", "interest_outstanding",
        "disbursed_principal", "interest_rate",
        "expected_emi_amount", "actual_principal_collected", "actual_interest_collected",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "dpd" in df.columns:
        # Reset index so boolean masks are perfectly aligned with the DataFrame rows
        df = df.reset_index(drop=True)
        dpd = pd.to_numeric(df["dpd"], errors="coerce").fillna(0)
        df["dpd_bucket"] = "Unknown"
        df.loc[dpd == 0,                   "dpd_bucket"] = "Current (0 DPD)"
        df.loc[(dpd >= 1) & (dpd <= 30),   "dpd_bucket"] = "DPD 1–30"
        df.loc[(dpd >= 31) & (dpd <= 60),  "dpd_bucket"] = "DPD 31–60"
        df.loc[(dpd >= 61) & (dpd <= 90),  "dpd_bucket"] = "DPD 61–90"
        df.loc[dpd > 90,                   "dpd_bucket"] = "DPD 90+"

    if "maturity_date" in df.columns and "snapshot_date" in df.columns:
        df["remaining_tenor_months"] = (
            (df["maturity_date"] - df["snapshot_date"]).dt.days / 30
        ).clip(lower=0)

    if "actual_principal_collected" in df.columns and "actual_interest_collected" in df.columns:
        df["amount_collected_this_month"] = (
            df["actual_principal_collected"].fillna(0) +
            df["actual_interest_collected"].fillna(0)
        )
    elif "actual_principal_collected" in df.columns:
        df["amount_collected_this_month"] = df["actual_principal_collected"].fillna(0)

    critical = ["loan_id", "snapshot_date", "dpd", "principal_outstanding"]
    report["missing_critical"] = {c: int(df[c].isna().sum()) for c in critical if c in df.columns}
    report["row_count"] = len(df)
    report["columns_mapped"] = list(df.columns)
    if "snapshot_date" in df.columns:
        dates = df["snapshot_date"].dropna()
        if not dates.empty:
            report["date_range"] = {"min": str(dates.min().date()), "max": str(dates.max().date())}

    return {**state, "merged_df": df, "validation_report": report, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — METRIC COMPUTATION
# Only pickle-safe objects stored in metrics dict (DataFrames, lists, scalars).
# ─────────────────────────────────────────────────────────────────────────────
def metric_computation_agent(state: AgentState) -> AgentState:
    """Computes all 9 portfolio metrics. Stores active_df separately in state."""
    errors = list(state.get("errors", []))
    df: Optional[pd.DataFrame] = state.get("merged_df")
    metrics: Dict[str, Any] = {}

    if df is None or df.empty:
        errors.append("MetricComputationAgent: no data.")
        return {**state, "metrics": metrics, "errors": errors}

    # Ensure clean RangeIndex before slicing
    df = df.reset_index(drop=True)

    # ── Active loan filter ────────────────────────────────────────────────────
    if "loan_status" in df.columns:
        closed = {"closed", "settled", "written off", "write-off", "foreclosed"}
        active_df = df[~df["loan_status"].astype(str).str.lower().str.strip().isin(closed)].copy().reset_index(drop=True)
    elif "principal_outstanding" in df.columns:
        active_df = df[df["principal_outstanding"].fillna(0) > 0].copy().reset_index(drop=True)
    else:
        active_df = df.copy().reset_index(drop=True)

    # ── Latest snapshot per loan ──────────────────────────────────────────────
    if "snapshot_date" in active_df.columns:
        active_df = active_df.sort_values("snapshot_date").reset_index(drop=True)
        latest_df = active_df.groupby("loan_id", as_index=False).last().reset_index(drop=True)
        latest_snapshot = active_df["snapshot_date"].max()
    else:
        latest_df = active_df.copy()
        latest_snapshot = None

    metrics["latest_snapshot"] = str(latest_snapshot.date()) if latest_snapshot else "N/A"

    # ── METRIC 1 ──────────────────────────────────────────────────────────────
    pos_total = latest_df["principal_outstanding"].sum() if "principal_outstanding" in latest_df.columns else 0
    int_total = latest_df["interest_outstanding"].sum() if "interest_outstanding" in latest_df.columns else 0
    active_count = latest_df["loan_id"].nunique() if "loan_id" in latest_df.columns else len(latest_df)

    wa_rate = 0.0
    if "interest_rate" in latest_df.columns and pos_total > 0:
        wa_rate = (latest_df["interest_rate"].fillna(0) * latest_df["principal_outstanding"].fillna(0)).sum() / pos_total

    wa_tenor = 0.0
    if "remaining_tenor_months" in latest_df.columns and pos_total > 0:
        wa_tenor = (latest_df["remaining_tenor_months"].fillna(0) * latest_df["principal_outstanding"].fillna(0)).sum() / pos_total

    metrics["m1_kpi"] = {
        "pos_cr":          round(pos_total / 1e7, 2),
        "int_cr":          round(int_total / 1e7, 2),
        "active_loans":    active_count,
        "wa_rate_raw":     round(wa_rate, 6),   # stored raw; display logic in viz
        "wa_tenor_months": round(wa_tenor, 1),
    }

    # # ── METRIC 2 ──────────────────────────────────────────────────────────────
    # if "dpd_bucket" in latest_df.columns and "principal_outstanding" in latest_df.columns:
    #     m2 = (
    #         latest_df.groupby("dpd_bucket")["principal_outstanding"]
    #         .sum().reindex(DPD_BUCKETS, fill_value=0).reset_index()
    #     )
    #     m2.columns = ["dpd_bucket", "pos_raw"]
    #     m2["pos_cr"] = m2["pos_raw"] / 1e7
    #     total_pos = m2["pos_cr"].sum()
    #     m2["pct"] = (m2["pos_cr"] / total_pos * 100).where(total_pos > 0, 0)
    #     metrics["m2_pos_dist"] = m2

# ── METRIC 2 ──────────────────────────────────────────────────────────────
    if "dpd_bucket" in latest_df.columns and "principal_outstanding" in latest_df.columns:
        m2 = (
            latest_df.groupby("dpd_bucket")["principal_outstanding"]
            .sum().reindex(DPD_BUCKETS, fill_value=0).reset_index()
        )
        m2.columns = ["dpd_bucket", "pos_raw"]
        m2["pos_cr"] = m2["pos_raw"] / 1e7
        total_pos = m2["pos_cr"].sum()
        
        # --- FIX: Use standard if/else instead of .where() for a scalar condition ---
        if total_pos > 0:
            m2["pct"] = (m2["pos_cr"] / total_pos) * 100
        else:
            m2["pct"] = 0.0
            
        metrics["m2_pos_dist"] = m2

    # ── METRIC 3 ──────────────────────────────────────────────────────────────
    if (
        "snapshot_date" in active_df.columns
        and "expected_emi_amount" in active_df.columns
        and "amount_collected_this_month" in active_df.columns
    ):
        def _agg_ts(freq: str) -> pd.DataFrame:
            ts = (
                active_df.groupby(active_df["snapshot_date"].dt.to_period(freq))
                .agg(emi_due=("expected_emi_amount", "sum"),
                     collected=("amount_collected_this_month", "sum"))
                .reset_index()
            )
            ts["snapshot_date"] = ts["snapshot_date"].dt.to_timestamp()
            ts["efficiency_pct"] = (ts["collected"] / ts["emi_due"].replace(0, np.nan) * 100).fillna(0)
            ts["emi_due_cr"] = ts["emi_due"] / 1e7
            ts["collected_cr"] = ts["collected"] / 1e7
            return ts

        metrics["m3_collections_ts"]   = _agg_ts("M")
        metrics["m3_collections_ts_q"] = _agg_ts("Q")

    # ── METRIC 4 — available months list only (compute happens in viz layer) ──
    if "snapshot_date" in active_df.columns and "dpd_bucket" in active_df.columns:
        sorted_months = sorted(active_df["snapshot_date"].dt.to_period("M").unique())
        metrics["m4_available_months"] = [str(m) for m in sorted_months]

    # ── METRICS 5-8 — available dates list only ───────────────────────────────
    if "snapshot_date" in active_df.columns and "dpd_bucket" in active_df.columns:
        metrics["transition_available_dates"] = sorted(active_df["snapshot_date"].dt.date.unique())

    # ── METRIC 9 ──────────────────────────────────────────────────────────────
    if "disbursement_date" in latest_df.columns and "principal_outstanding" in latest_df.columns:
        v = latest_df.copy()
        v["vintage_year"] = v["disbursement_date"].dt.year
        m9 = v.groupby("vintage_year")["principal_outstanding"].sum().reset_index()
        m9["pos_cr"] = m9["principal_outstanding"] / 1e7
        metrics["m9_vintage"] = m9

    # Store active_df in state (NOT inside metrics to keep metrics pickle-safe)
    return {**state, "metrics": metrics, "active_df": active_df, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# ON-DEMAND COMPUTE FUNCTIONS (called by viz agent — NOT stored in state)
# ─────────────────────────────────────────────────────────────────────────────
def compute_m4(active_df: pd.DataFrame, selected_month_str: str) -> pd.DataFrame:
    """Collections efficiency by DPD bucket for a selected month."""
    sel_period   = pd.Period(selected_month_str, freq="M")
    prior_period = sel_period - 1

    prior_df = active_df[active_df["snapshot_date"].dt.to_period("M") == prior_period]
    sel_df   = active_df[active_df["snapshot_date"].dt.to_period("M") == sel_period]

    if prior_df.empty:
        prior_df = sel_df   # fallback: use same month

    dpd_start = prior_df.groupby("loan_id")["dpd_bucket"].last()
    sel_agg = sel_df.groupby("loan_id").agg(
        emi_due=("expected_emi_amount", "sum"),
        collected=("amount_collected_this_month", "sum"),
    )
    merged = sel_agg.join(dpd_start, how="left")
    merged["dpd_bucket"] = merged["dpd_bucket"].fillna("Unknown")

    eff = (
        merged.groupby("dpd_bucket")
        .agg(emi_due=("emi_due", "sum"), collected=("collected", "sum"))
        .reindex(DPD_BUCKETS, fill_value=0)
        .reset_index()
    )
    eff["efficiency_pct"] = (eff["collected"] / eff["emi_due"].replace(0, np.nan) * 100).fillna(0)
    return eff


def compute_transition(active_df: pd.DataFrame, start_date, end_date):
    """Returns (mat_pos_cr, mat_pos_pct, mat_count, mat_count_pct) DataFrames."""
    def _snap(date):
        return (
            active_df[active_df["snapshot_date"].dt.date == date]
            .sort_values("snapshot_date")
            .groupby("loan_id", as_index=False)
            .last()
            .reset_index(drop=True)
        )

    df_s = _snap(start_date).rename(columns={"dpd_bucket": "Start_DPD", "principal_outstanding": "Start_POS"})
    df_e = _snap(end_date).rename(columns={"dpd_bucket": "End_DPD",   "principal_outstanding": "End_POS"})
    trans = pd.merge(
        df_s[["loan_id", "Start_DPD", "Start_POS"]],
        df_e[["loan_id", "End_DPD",   "End_POS"]],
        on="loan_id", how="inner"
    )

    mat_pos = pd.crosstab(
        trans["Start_DPD"], trans["End_DPD"],
        values=trans["Start_POS"], aggfunc="sum"
    ).reindex(index=DPD_BUCKETS, columns=DPD_BUCKETS, fill_value=0) / 1e7

    mat_pos_pct   = mat_pos.div(mat_pos.sum(axis=1), axis=0).fillna(0) * 100
    mat_count     = pd.crosstab(trans["Start_DPD"], trans["End_DPD"]).reindex(
                        index=DPD_BUCKETS, columns=DPD_BUCKETS, fill_value=0)
    mat_count_pct = mat_count.div(mat_count.sum(axis=1), axis=0).fillna(0) * 100

    return mat_pos, mat_pos_pct, mat_count, mat_count_pct


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def visualisation_agent(metrics: Dict[str, Any], active_df: pd.DataFrame) -> None:
    """Renders all metric charts. active_df passed directly (not from pickle)."""

    snapshot_label = metrics.get("latest_snapshot", "N/A")
    st.title("Parse.ai — Loan Portfolio Analytics Dashboard")
    st.caption(f"Latest Snapshot: {snapshot_label}")

    # ── METRIC 1: KPI Cards ───────────────────────────────────────────────────
    kpi = metrics.get("m1_kpi", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Principal Outstanding", f"₹{kpi.get('pos_cr', 0):,.2f} Cr")
    c2.metric("Interest Outstanding",  f"₹{kpi.get('int_cr', 0):,.2f} Cr")
    c3.metric("Active Loans",           f"{kpi.get('active_loans', 0):,}")

    raw_rate = kpi.get("wa_rate_raw", 0)
    display_rate = raw_rate * 100 if 0 < raw_rate < 1 else raw_rate
    c4.metric("WA Interest Rate",      f"{display_rate:.2f}%")
    c5.metric("WA Remaining Tenor",    f"{kpi.get('wa_tenor_months', 0):.1f} months")

    st.divider()

    # ── METRIC 2 & 4 ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Metric 2: POS Distribution by DPD Bucket")
        m2 = metrics.get("m2_pos_dist")
        if m2 is not None:
            fig2 = px.bar(
                m2, x="dpd_bucket", y="pos_cr",
                color="dpd_bucket", color_discrete_map=COLOR_SCHEME,
                text="pos_cr",
                category_orders={"dpd_bucket": DPD_BUCKETS},
                labels={"pos_cr": "POS (₹ Cr)", "dpd_bucket": ""},
                title=f"POS by DPD Bucket — {snapshot_label}",
            )
            fig2.update_traces(
                texttemplate="%{text:.2f} Cr<br>(%{customdata[0]:.1f}%)",
                customdata=m2[["pct"]].values,
                textposition="outside",
            )
            fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("DPD / POS data unavailable.")

    with col_right:
        st.subheader("Metric 4: Collections Efficiency by DPD Bucket")
        m4_months = metrics.get("m4_available_months", [])
        has_m4_data = (
            active_df is not None
            and "expected_emi_amount" in active_df.columns
            and "amount_collected_this_month" in active_df.columns
            and m4_months
        )
        if has_m4_data:
            sel_month = st.selectbox(
                "Select Month", options=m4_months, index=len(m4_months) - 1, key="m4_sel"
            )
            m4_df = compute_m4(active_df, sel_month)
            fig4 = px.bar(
                m4_df, x="efficiency_pct", y="dpd_bucket",
                orientation="h", color="dpd_bucket",
                color_discrete_map=COLOR_SCHEME, text="efficiency_pct",
                category_orders={"dpd_bucket": list(reversed(DPD_BUCKETS))},
                labels={"efficiency_pct": "Efficiency (%)", "dpd_bucket": ""},
                title=f"Collections Efficiency — {sel_month}",
            )
            fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig4.add_vline(x=100, line_dash="dash", line_color="white",
                           annotation_text="100%", annotation_position="top right")
            fig4.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("EMI / collections data unavailable for Metric 4.")

    st.divider()

    # ── METRIC 3: Collections Efficiency Time Series ──────────────────────────
    st.subheader("Metric 3: Overall Collections Efficiency — Time Series")
    ts_m = metrics.get("m3_collections_ts")
    ts_q = metrics.get("m3_collections_ts_q")

    if ts_m is not None:
        def _draw_ts(ts_df: pd.DataFrame, period_label: str) -> go.Figure:
            fig = go.Figure()
            fig.add_bar(x=ts_df["snapshot_date"], y=ts_df["emi_due_cr"],
                        name="EMI Due (Cr)", marker_color="steelblue")
            fig.add_bar(x=ts_df["snapshot_date"], y=ts_df["collected_cr"],
                        name="Collected (Cr)", marker_color="teal")
            fig.add_scatter(
                x=ts_df["snapshot_date"], y=ts_df["efficiency_pct"],
                name="Efficiency %", yaxis="y2",
                mode="lines+markers+text",
                text=ts_df["efficiency_pct"].round(1).astype(str) + "%",
                textposition="top center",
                line=dict(color="gold", width=2),
                marker=dict(color="gold", size=6),
            )
            fig.update_layout(
                title=f"Overall Collections Efficiency — {period_label}",
                xaxis_title="Period",
                yaxis=dict(title="Amount (₹ Crores)"),
                yaxis2=dict(title="Collection Efficiency (%)", overlaying="y",
                            side="right", range=[0, 120]),
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            return fig

        tab_m, tab_q = st.tabs(["Monthly View", "Quarterly View"])
        with tab_m:
            st.plotly_chart(_draw_ts(ts_m, "Monthly"), use_container_width=True)
        with tab_q:
            if ts_q is not None:
                st.plotly_chart(_draw_ts(ts_q, "Quarterly"), use_container_width=True)
    else:
        st.info("Collections time-series unavailable (need EMI & collected columns).")

    st.divider()

    # ── METRICS 5–8: Transition Matrices ─────────────────────────────────────
    st.subheader("Transition Matrices (Metrics 5–8)")
    avail_dates = metrics.get("transition_available_dates", [])
    has_trans = active_df is not None and "dpd_bucket" in active_df.columns and len(avail_dates) >= 2

    if has_trans:
        tc1, tc2 = st.columns(2)
        with tc1:
            start_date = st.selectbox(
                "Start State (Prior Month-End)",
                options=avail_dates[:-1], index=len(avail_dates) - 2, key="tm_start"
            )
        with tc2:
            end_date = st.selectbox(
                "End State (Current Month-End)",
                options=avail_dates, index=len(avail_dates) - 1, key="tm_end"
            )

        mat_pos, mat_pos_pct, mat_count, mat_count_pct = compute_transition(
            active_df, start_date, end_date
        )

        def _heatmap(matrix: pd.DataFrame, fmt: str, title: str) -> go.Figure:
            return px.imshow(
                matrix, text_auto=fmt, color_continuous_scale="RdYlGn_r",
                aspect="auto", title=title,
                labels={"x": "Ending DPD Bucket", "y": "Starting DPD Bucket"},
            )

        tab5, tab6, tab7, tab8 = st.tabs([
            "Metric 5: POS Flow (INR Cr)",
            "Metric 6: POS Flow (%)",
            "Metric 7: Loan Count",
            "Metric 8: Loan Count (%)",
        ])
        with tab5:
            st.plotly_chart(_heatmap(mat_pos,       ".2f", "POS Flow (INR Crores)"),  use_container_width=True)
        with tab6:
            st.plotly_chart(_heatmap(mat_pos_pct,   ".1f", "POS Flow (%)"),           use_container_width=True)
        with tab7:
            st.plotly_chart(_heatmap(mat_count,     "d",   "Loan Count Flow"),        use_container_width=True)
        with tab8:
            st.plotly_chart(_heatmap(mat_count_pct, ".1f", "Loan Count Flow (%)"),    use_container_width=True)
    else:
        st.info("Need at least two snapshot dates for Transition Matrices.")

    st.divider()

    # ── METRIC 9: POS by Vintage ──────────────────────────────────────────────
    st.subheader("Metric 9: Principal Outstanding by Vintage (Cohort)")
    m9 = metrics.get("m9_vintage")
    if m9 is not None and not m9.empty:
        fig9 = px.bar(
            m9, x="vintage_year", y="pos_cr",
            labels={"vintage_year": "Vintage Year", "pos_cr": "Outstanding (₹ Cr)"},
            title="Principal Outstanding by Vintage (Cohort)",
            text="pos_cr",
        )
        fig9.update_traces(texttemplate="%{text:.2f} Cr", textposition="outside")
        fig9.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.info("Disbursement date data unavailable for Vintage chart.")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — INTERACTION (Chat Q&A)
# ─────────────────────────────────────────────────────────────────────────────
def interaction_agent_ui(metrics: Dict[str, Any], validation_report: Dict[str, Any]) -> None:
    st.divider()
    st.subheader("💬 Ask About Your Portfolio")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    kpi = metrics.get("m1_kpi", {})
    raw_rate = kpi.get("wa_rate_raw", 0)
    display_rate = raw_rate * 100 if 0 < raw_rate < 1 else raw_rate
    m2 = metrics.get("m2_pos_dist")
    m2_summary = m2[["dpd_bucket", "pos_cr", "pct"]].to_string(index=False) if m2 is not None else "N/A"

    data_context = textwrap.dedent(f"""
        Portfolio snapshot: {metrics.get('latest_snapshot', 'N/A')}
        Principal Outstanding: ₹{kpi.get('pos_cr', 0):,.2f} Cr
        Interest Outstanding:  ₹{kpi.get('int_cr', 0):,.2f} Cr
        Active Loans:          {kpi.get('active_loans', 0):,}
        WA Interest Rate:      {display_rate:.2f}%
        WA Remaining Tenor:    {kpi.get('wa_tenor_months', 0):.1f} months

        POS by DPD Bucket:
        {m2_summary}

        Validation report: {json.dumps(validation_report, indent=2)}
    """)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask anything about the portfolio data…")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        system_prompt = textwrap.dedent(f"""
            You are a senior credit risk analyst reviewing a loan portfolio dashboard.
            Answer questions based ONLY on the data provided. Be concise and precise.
            Use INR Crore as the currency unit. Highlight any risk signals you notice.
            --- PORTFOLIO DATA ---
            {data_context}
        """)
        try:
            answer = _groq_chat(system_prompt, user_q)
        except Exception as e:
            answer = f"(LLM error: {e})"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(AgentState)
    graph.add_node("schema_discovery",   schema_discovery_agent)
    graph.add_node("data_validation",    data_validation_agent)
    graph.add_node("metric_computation", metric_computation_agent)
    graph.set_entry_point("schema_discovery")
    graph.add_edge("schema_discovery",   "data_validation")
    graph.add_edge("data_validation",    "metric_computation")
    graph.add_edge("metric_computation", END)
    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Parse.ai Loan Analytics", layout="wide")

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")
    data_dir = st.sidebar.text_input("Data Directory", value=DATA_DIR)
    groq_key = st.sidebar.text_input("Groq API Key", value=GROQ_API_KEY, type="password")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    else:
        st.sidebar.warning("Set your Groq API key to enable schema discovery and chat.")

    st.sidebar.divider()
    st.sidebar.title("Dashboard Filters")

    # ── RUN PIPELINE (cached by session_state, not st.cache_data) ────────────
    # We use session_state for caching because the pipeline returns DataFrames
    # (which are not hashable for st.cache_data) and we need the active_df
    # reference alive for interactive compute functions.
    cache_key = f"{data_dir}::{groq_key}"
    if "pipeline_cache_key" not in st.session_state or st.session_state.pipeline_cache_key != cache_key:
        with st.spinner("🤖 Agentic pipeline: Schema Discovery → Validation → Metric Computation…"):
            try:
                pipeline = build_pipeline()
                initial_state: AgentState = {
                    "data_dir":          data_dir,
                    "raw_dfs":           {},
                    "ai_mappings":       {},
                    "merged_df":         None,
                    "active_df":         None,
                    "validation_report": {},
                    "metrics":           {},
                    "errors":            [],
                }
                final_state = pipeline.invoke(initial_state)
                st.session_state.pipeline_cache_key = cache_key
                st.session_state.final_state = final_state
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

    final_state = st.session_state.final_state

    # Pipeline errors
    for err in final_state.get("errors", []):
        st.sidebar.error(err)

    with st.sidebar.expander("🔍 AI Schema Mappings"):
        st.json(final_state.get("ai_mappings", {}))
    with st.sidebar.expander("📋 Validation Report"):
        st.json(final_state.get("validation_report", {}))

    # ── FILTERS ───────────────────────────────────────────────────────────────
    merged_df: Optional[pd.DataFrame] = final_state.get("merged_df")
    if merged_df is None or merged_df.empty:
        st.error("No data loaded. Check your data directory and CSV files.")
        st.stop()

    products = sorted(merged_df["Product"].dropna().astype(str).unique())
    regions  = sorted(merged_df["Region"].dropna().astype(str).unique())
    cities   = sorted(merged_df["City"].dropna().astype(str).unique())

    sel_product = st.sidebar.multiselect("Product", products, default=products)
    sel_region  = st.sidebar.multiselect("Region",  regions,  default=regions)
    sel_city    = st.sidebar.multiselect("City",    cities,   default=cities)

    filtered_df = merged_df[
        merged_df["Product"].isin(sel_product) &
        merged_df["Region"].isin(sel_region)   &
        merged_df["City"].isin(sel_city)
    ]

    # ── RE-COMPUTE METRICS ON FILTERED SLICE (if filters applied) ─────────────
    if len(filtered_df) < len(merged_df):
        filt_state: AgentState = {
            "data_dir":          data_dir,
            "raw_dfs":           {},
            "ai_mappings":       {},
            "merged_df":         filtered_df,
            "active_df":         None,
            "validation_report": final_state.get("validation_report", {}),
            "metrics":           {},
            "errors":            [],
        }
        filt_state = data_validation_agent(filt_state)
        filt_state = metric_computation_agent(filt_state)
        metrics        = filt_state.get("metrics", {})
        active_df      = filt_state.get("active_df")
        validation_rep = filt_state.get("validation_report", {})
    else:
        metrics        = final_state.get("metrics", {})
        active_df      = final_state.get("active_df")
        validation_rep = final_state.get("validation_report", {})

    # ── RENDER ────────────────────────────────────────────────────────────────
    visualisation_agent(metrics, active_df)
    interaction_agent_ui(metrics, validation_rep)


if __name__ == "__main__":
    main()