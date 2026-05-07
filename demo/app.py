"""
demo/app.py — Sepsis Atlas Streamlit demo.

Pages:
  1. Evidence Query  — natural language → LLM generates SQL → results from DB
  2. Extract Paper   — upload PDF, run pipeline, see structured output
  3. Browse Database — full cohorts table + paper details
  4. Export          — download CSV
"""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from openai import OpenAI
from sqlalchemy import text

import config
from pipeline.pdf_reader import extract_text, get_relevant_chunk
from pipeline.extractor import extract_paper
from pipeline.validator import validate_paper, filter_low_confidence_fields
from utils.db import (
    init_db, save_paper, save_cohorts,
    load_all_papers, load_cohorts_with_paper_meta, export_to_csv,
)
from utils.logger import get_logger
from sqlalchemy import create_engine

logger = get_logger(__name__)

st.set_page_config(page_title="Sepsis Atlas", page_icon="🧬", layout="wide")
init_db()

st.sidebar.title("🧬 Sepsis Atlas")
page = st.sidebar.radio("Navigate", ["Evidence Query", "Extract Paper", "Browse Database", "Export"])

_llm = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)


def confidence_badge(conf: float) -> str:
    if conf >= 0.8:   return f"🟢 {conf:.0%}"
    elif conf >= 0.5: return f"🟡 {conf:.0%}"
    else:             return f"🔴 {conf:.0%}"


# ── DB schema description passed to LLM for SQL generation ────────────────────

DB_SCHEMA = """
SQLite database with two tables:

TABLE papers (
    paper_id TEXT PRIMARY KEY,
    meta_title TEXT,             -- title of the paper
    meta_year TEXT,              -- publication year
    meta_journal TEXT,
    meta_study_design TEXT,      -- e.g. RCT, retrospective cohort, prospective cohort
    meta_country TEXT,
    sep_definition TEXT,         -- sepsis definition used (Sepsis-1, Sepsis-2, Sepsis-3)
    sep_sofa TEXT,               -- SOFA score reported
    sep_qsofa TEXT,
    sep_lactate TEXT,            -- lactate threshold used
    sep_shock TEXT,              -- septic shock included (yes/no)
    int_primary TEXT,            -- primary intervention
    int_comparison TEXT,
    int_antibiotics TEXT,
    int_fluids TEXT,
    int_vasopressors TEXT,
    overall_confidence REAL
)

TABLE cohorts (
    cohort_id TEXT PRIMARY KEY,
    paper_id TEXT,               -- FK to papers.paper_id
    cohort_name TEXT,            -- e.g. "KPNC cohort", "UPMC derivation", "Overall cohort"
    sample_size TEXT,
    mean_age TEXT,
    percent_male TEXT,
    clinical_setting TEXT,       -- ICU, non-ICU, ED, mixed
    inclusion_criteria TEXT,
    mortality_rate TEXT,         -- e.g. "28.3%"
    mortality_timepoint TEXT,    -- e.g. "28-day", "hospital", "ICU"
    icu_length_of_stay TEXT,
    primary_outcome TEXT,
    confidence REAL
)

To get cohort rows with paper metadata, JOIN on cohorts.paper_id = papers.paper_id.
"""


def nl_to_sql(user_query: str) -> str:
    prompt = f"""You are an expert SQLite query writer for a sepsis research database.

{DB_SCHEMA}

User question: {user_query}

Write a single SQLite SELECT query that answers this question.
- Use JOIN when paper metadata is needed alongside cohort data
- Cast numeric-looking text columns (sample_size, mortality_rate) with CAST(... AS REAL) only when doing comparisons
- Return only the SQL query, no explanation, no markdown fences."""

    resp = _llm.chat.completions.create(
        model=config.MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = resp.choices[0].message.content.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"\s*```$", "", sql.strip(), flags=re.MULTILINE)
    return sql


def run_sql(sql: str) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ═══════════════════════════════════════════════════════════════
# Page 1: Evidence Query  (NL → LLM → SQL → results)
# ═══════════════════════════════════════════════════════════════
if page == "Evidence Query":
    st.title("Evidence Query")
    st.markdown(
        "Ask any clinical question in plain English. "
        "The LLM translates it into a SQL query and retrieves matching cohorts from the database."
    )

    query = st.text_input(
        "Clinical question",
        placeholder="e.g. Show all ICU cohorts with mortality rate above 25% using Sepsis-3",
    )

    if st.button("Search", type="primary") and query:
        df_check = load_all_papers()
        if df_check.empty:
            st.warning("No papers extracted yet. Go to 'Extract Paper' first.")
        else:
            with st.spinner("Generating SQL query..."):
                try:
                    sql = nl_to_sql(query)
                except Exception as e:
                    st.error(f"LLM error: {e}")
                    st.stop()

            with st.expander("Generated SQL", expanded=True):
                st.code(sql, language="sql")

            try:
                results = run_sql(sql)
            except Exception as e:
                st.error(f"SQL execution error: {e}")
                st.stop()

            if results.empty:
                st.info("No matching records found.")
            else:
                st.success(f"{len(results)} row(s) found")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "⬇ Download results (CSV)",
                    results.to_csv(index=False),
                    file_name="query_results.csv",
                    mime="text/csv",
                )


# ═══════════════════════════════════════════════════════════════
# Page 2: Extract a new paper
# ═══════════════════════════════════════════════════════════════
elif page == "Extract Paper":
    st.title("Extract a Sepsis Paper")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        tmp_path = config.PAPERS_DIR / uploaded.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        if st.button("Run extraction", type="primary"):
            with st.spinner("Extracting text..."):
                full_text = extract_text(tmp_path)
            st.success(f"Extracted {len(full_text):,} characters")

            with st.expander("Raw text (first 2000 chars)"):
                st.text(full_text[:2000])

            with st.spinner("Sending to LLM for structured extraction..."):
                llm_text = get_relevant_chunk(full_text, max_chars=config.CHUNK_SIZE)
                paper = extract_paper(llm_text, uploaded.name)
                is_valid, warnings = validate_paper(paper)
                paper = filter_low_confidence_fields(paper, min_confidence=config.MIN_CONFIDENCE)
                paper.compute_overall_confidence()
                save_paper(paper)
                save_cohorts(paper)

            if warnings:
                for w in warnings:
                    st.warning(w)

            st.subheader(f"Extraction complete — confidence: {paper.overall_confidence:.0%}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Title", (paper.metadata.title.value or "N/A")[:40])
            c2.metric("Cohorts found", len(paper.cohorts))
            c3.metric("Overall confidence", f"{paper.overall_confidence:.0%}")

            if paper.cohorts:
                st.markdown("### Cohorts")
                cohort_rows = [c.model_dump() for c in paper.cohorts]
                st.dataframe(pd.DataFrame(cohort_rows), use_container_width=True)

            st.markdown("### Study metadata")
            for label, ef in [
                ("Title", paper.metadata.title),
                ("Year", paper.metadata.year),
                ("Journal", paper.metadata.journal),
                ("Study design", paper.metadata.study_design),
                ("Country", paper.metadata.country_or_region),
            ]:
                if ef.value:
                    with st.expander(f"**{label}** — {ef.value[:80]}"):
                        st.write(ef.value)
                        if ef.source_sentence:
                            st.markdown(f"> *\"{ef.source_sentence}\"*")
                        st.caption(f"Confidence: {confidence_badge(ef.confidence)}")

            if paper.prognostic_findings:
                st.markdown("### Prognostic findings")
                st.dataframe(pd.DataFrame([f.model_dump() for f in paper.prognostic_findings]), use_container_width=True)

            with st.expander("Full JSON"):
                st.json(json.loads(paper.model_dump_json()))


# ═══════════════════════════════════════════════════════════════
# Page 3: Browse Database
# ═══════════════════════════════════════════════════════════════
elif page == "Browse Database":
    st.title("Database")

    tab1, tab2 = st.tabs(["Cohorts view", "Papers view"])

    with tab1:
        df = load_cohorts_with_paper_meta()
        if df.empty:
            st.info("No data yet.")
        else:
            st.metric("Total cohorts", len(df))
            st.dataframe(df, use_container_width=True)

    with tab2:
        df = load_all_papers()
        if df.empty:
            st.info("No data yet.")
        else:
            st.metric("Total papers", len(df))
            display_cols = {
                "pdf_filename": "File", "meta_title": "Title", "meta_year": "Year",
                "meta_study_design": "Design", "sep_definition": "Sepsis def.", "overall_confidence": "Conf.",
            }
            show_df = df[list(display_cols.keys())].rename(columns=display_cols)
            show_df["Conf."] = show_df["Conf."].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
            st.dataframe(show_df, use_container_width=True)

            selected = st.selectbox("Paper details", df["pdf_filename"].tolist())
            if selected:
                row = df[df["pdf_filename"] == selected].iloc[0]
                st.subheader(row.get("meta_title") or selected)
                from utils.db import load_cohorts
                cohort_df = load_cohorts(row["paper_id"])
                if not cohort_df.empty:
                    st.markdown("#### Cohorts")
                    st.dataframe(cohort_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Page 4: Export
# ═══════════════════════════════════════════════════════════════
elif page == "Export":
    st.title("Export Data")
    df = load_cohorts_with_paper_meta()
    st.metric("Cohort rows ready to export", len(df))
    if not df.empty:
        st.download_button(
            "⬇ Download cohorts CSV",
            df.to_csv(index=False),
            file_name="sepsis_atlas_cohorts.csv",
            mime="text/csv",
        )
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("No data to export yet.")
