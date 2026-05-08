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

import fitz  # PyMuPDF
from streamlit_pdf_viewer import pdf_viewer
from streamlit_option_menu import option_menu

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
    init_db, save_paper, save_cohorts, save_findings,
    load_all_papers, load_cohorts_with_paper_meta, export_to_csv, load_all_findings
)
from utils.logger import get_logger
from sqlalchemy import create_engine
from usecase3_ranking import generate_ranking

logger = get_logger(__name__)

st.set_page_config(page_title="Sepsis Atlas", page_icon="🧬", layout="wide")
init_db()

with st.sidebar:
    page = option_menu(
        menu_title="Sepsis Atlas",
        menu_icon="activity",
        options=["Evidence Query", "Use Cases", "Extract Paper", "Browse Database", "Export"],
        icons=["search", "lightbulb", "file-earmark-text", "database", "download"],
        default_index=0,
    )

_llm = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)


# ── Reusable helpers ──────────────────────────────────────────────────────────

def confidence_badge(conf: float) -> str:
    if conf >= 0.8:   return f"🟢 {conf:.0%}"
    elif conf >= 0.5: return f"🟡 {conf:.0%}"
    else:             return f"🔴 {conf:.0%}"

def find_sentence_annotations(pdf_path: str, sentence: str):
    """Search PDF for a sentence and return highlight annotations + the first page it appears on."""
    sentence = re.split(r"\.{3}|…", sentence)[0].strip()
    if not sentence:
        return [], None

    annotations = []
    target_page = None
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            rects = page.search_for(sentence)
            if not rects and len(sentence) > 60:
                rects = page.search_for(sentence[:60])
            for rect in rects:
                annotations.append({
                    "page": page_num,
                    "x": rect.x0, "y": rect.y0,
                    "width": rect.width, "height": rect.height,
                    "color": "rgba(255, 235, 59, 0.7)",
                })
                if target_page is None:
                    target_page = page_num
        doc.close()
    except Exception as e:
        logger.warning(f"PDF highlight search failed: {e}")
    return annotations, target_page

def show_source_trace(source_sentence: str, paper_id: str,
                       predictor: str = None, outcome: str = None,
                       effect_size: str = None, confidence: float = None):
    """
    Reusable component for showing source traceability.
    Shows the exact sentence from the paper that supports a claim.
    """
    if not source_sentence:
        return

    label = f"📎 {paper_id}"
    if predictor:
        label += f" — {predictor}"
    if outcome:
        label += f" → {outcome}"

    with st.expander(label):
        col1, col2 = st.columns([3, 1])
        with col1:
            if effect_size:
                st.markdown(f"**Effect size:** {effect_size}")
            st.markdown(f"> *\"{source_sentence}\"*")
        with col2:
            if confidence is not None:
                st.metric("Confidence", confidence_badge(confidence))
            st.caption(f"Source: {paper_id}")


def show_source_traces_from_df(df: pd.DataFrame):
    """Show source traces for all rows in a DataFrame that have source_sentence."""
    rows_with_sources = df[df["source_sentence"].notna()] if "source_sentence" in df.columns else pd.DataFrame()
    if rows_with_sources.empty:
        st.caption("No source sentences available for these results.")
        return

    for _, row in rows_with_sources.iterrows():
        show_source_trace(
            source_sentence=row.get("source_sentence"),
            paper_id=row.get("paper_id", row.get("paper", "")),
            predictor=row.get("predictor"),
            outcome=row.get("outcome"),
            effect_size=row.get("effect_size"),
            confidence=float(row["confidence"]) if pd.notna(row.get("confidence")) else None
        )


# ── DB schema ─────────────────────────────────────────────────────────────────

DB_SCHEMA = """
SQLite database with three tables:

TABLE papers (
    paper_id TEXT PRIMARY KEY,
    meta_title TEXT,
    meta_year TEXT,
    meta_journal TEXT,
    meta_study_design TEXT,
    meta_country TEXT,
    sep_definition TEXT,
    sep_sofa TEXT,
    sep_qsofa TEXT,
    sep_lactate TEXT,
    sep_shock TEXT,
    int_primary TEXT,
    int_comparison TEXT,
    int_antibiotics TEXT,
    int_fluids TEXT,
    int_vasopressors TEXT,
    overall_confidence REAL
)

TABLE cohorts (
    cohort_id TEXT PRIMARY KEY,
    paper_id TEXT,
    cohort_name TEXT,
    sample_size TEXT,
    mean_age TEXT,
    percent_male TEXT,
    clinical_setting TEXT,
    inclusion_criteria TEXT,
    mortality_rate TEXT,
    mortality_timepoint TEXT,
    icu_length_of_stay TEXT,
    primary_outcome TEXT,
    source_sentence TEXT,
    confidence REAL
)

TABLE findings (
    finding_id TEXT PRIMARY KEY,
    paper_id TEXT,
    predictor TEXT,
    outcome TEXT,
    timing TEXT,
    method TEXT,
    effect_size TEXT,
    performance TEXT,
    notes TEXT,
    source_sentence TEXT,
    confidence REAL
)

Join tables on paper_id.
For predictor/outcome questions use the findings table.
For population/mortality questions use the cohorts table joined with papers.
"""


def nl_to_sql(user_query: str) -> str:
    prompt = f"""You are an expert SQLite query writer for a sepsis research database.

{DB_SCHEMA}

User question: {user_query}

Write a single SQLite SELECT query that answers this question.
- ALWAYS include source_sentence AND paper_id in the SELECT if querying findings or cohorts
- When searching for biomarkers or predictors, use multiple LIKE conditions with OR to catch synonyms
- Example: predictor LIKE '%lactate%' OR predictor LIKE '%lactic%' OR notes LIKE '%lactate%'
- Also search in notes and effect_size columns, not just predictor
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

def summarize_evidence(user_query: str, results: pd.DataFrame) -> str:
    # Drop bulky columns and cap rows to keep the prompt small
    compact = results.drop(columns=["source_sentence"], errors="ignore").head(20)
    table_str = compact.to_csv(index=False)

    prompt = f"""You are summarizing sepsis research evidence for a clinician.

Question asked: {user_query}

Evidence retrieved (CSV, up to 20 rows):
{table_str}

Write a 2-3 sentence plain-English summary of what this evidence shows in answer to the question.
- Be concrete: mention specific predictors, effect sizes, or numbers when relevant.
- If studies disagree or evidence is thin, say so.
- Do NOT invent details that are not in the table.
- No preamble, no "Based on the evidence..." — just the summary."""

    resp = _llm.chat.completions.create(
        model=config.MODEL,
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# ═══════════════════════════════════════════════════════════════
# Page 1: Evidence Query
# ═══════════════════════════════════════════════════════════════
if page == "Evidence Query":
    st.title("Evidence Query")
    st.markdown(
        "Ask any clinical question in plain English. "
        "The LLM translates it into a SQL query and retrieves matching results from the database."
    )

    # Example queries
    st.markdown("**Example queries:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Lactate and 28-day mortality"):
            st.session_state["query"] = "What predictors of 28-day mortality involve lactate?"
    with col2:
        if st.button("ICU cohorts with high mortality"):
            st.session_state["query"] = "Show all ICU cohorts with mortality rate above 30%"
    with col3:
        if st.button("SOFA score performance"):
            st.session_state["query"] = "Which studies report AUC for SOFA score predicting mortality?"

    query = st.text_input(
        "Clinical question",
        value=st.session_state.get("query", ""),
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

            with st.expander("Generated SQL", expanded=False):
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

                # Show table without source_sentence column (shown below)
                display_cols = [c for c in results.columns if c != "source_sentence"]
                st.dataframe(results[display_cols], use_container_width=True)

                # ── Summary (NEW) ──
                with st.spinner("Summarizing evidence..."):
                    try:
                        summary = summarize_evidence(query, results)
                        st.markdown("### Summary")
                        st.info(summary)
                    except Exception as e:
                        st.warning(f"Could not generate summary: {e}")

                st.download_button(
                    "⬇ Download results (CSV)",
                    results.to_csv(index=False),
                    file_name="query_results.csv",
                    mime="text/csv",
                )

                # Source traces
                st.divider()
                st.subheader("Source traces")
                show_source_traces_from_df(results)

elif page == "Use Cases":
    st.title("Use Cases")

    tab1, tab2, tab3 = st.tabs([
        "Use Case 1 — Mortality Predictors",
        "Use Case 2 — Phenotypes",
        "Use Case 3 — Biomarker Ranking"
    ])

    # ── Use Case 1 ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Evidence Table")
        st.markdown("Predictor → outcome associations across studies with full traceability.")

        engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
        evidence_df = pd.read_sql("""
            SELECT 
                f.paper_id          AS study,
                c.clinical_setting  AS population,
                c.sample_size       AS sample_size,
                f.predictor,
                f.timing            AS measurement_timing,
                f.outcome,
                f.method,
                f.effect_size,
                f.performance,
                f.notes,
                f.source_sentence,
                f.confidence
            FROM findings f
            LEFT JOIN cohorts c ON f.paper_id = c.paper_id
            WHERE f.outcome LIKE '%mortality%'
               OR f.outcome LIKE '%death%'
               OR f.outcome LIKE '%survival%'
            ORDER BY f.paper_id
        """, engine)

        if evidence_df.empty:
            st.info("No findings yet. Run the extraction pipeline first.")
        else:
            st.metric("Total findings", len(evidence_df))

            display_cols = [c for c in evidence_df.columns if c != "source_sentence"]
            st.dataframe(evidence_df[display_cols], use_container_width=True)

            st.download_button(
                "⬇ Download evidence table (CSV)",
                evidence_df.to_csv(index=False),
                file_name="usecase1_evidence_table.csv",
                mime="text/csv",
            )

            st.divider()
            st.subheader("Source traces")
            show_source_traces_from_df(evidence_df)

    # ── Use Case 2 ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Sepsis Phenotypes")
        st.markdown("Patient clusters identified across studies.")
        st.info("Phenotype extraction will be available after re-running the pipeline with the updated schema.")

    # ── Use Case 3 ────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Evidence Table")
        st.markdown("All biomarker and score findings relevant to mortality risk stratification.")

        engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
        uc3_evidence_df = pd.read_sql("""
            SELECT 
                f.paper_id          AS study,
                c.clinical_setting  AS population,
                c.sample_size       AS sample_size,
                f.predictor,
                f.timing            AS measurement_timing,
                f.outcome,
                f.method,
                f.effect_size,
                f.performance,
                f.notes,
                f.source_sentence,
                f.confidence
            FROM findings f
            LEFT JOIN cohorts c ON f.paper_id = c.paper_id
            WHERE (
                f.outcome LIKE '%mortality%'
                OR f.outcome LIKE '%death%'
                OR f.outcome LIKE '%survival%'
            )
            AND f.effect_size IS NOT NULL
            ORDER BY f.paper_id
        """, engine)

        if not uc3_evidence_df.empty:
            display_cols = [c for c in uc3_evidence_df.columns if c != "source_sentence"]
            st.dataframe(uc3_evidence_df[display_cols], use_container_width=True)
            st.download_button(
                "⬇ Download evidence table (CSV)",
                uc3_evidence_df.to_csv(index=False),
                file_name="usecase3_evidence_table.csv",
                mime="text/csv",
            )

        st.divider()
        st.markdown("### Ranked Predictors")
        st.markdown("Biomarkers and scores ranked by strongest prognostic metric.")

        ranked_df = generate_ranking()

        if ranked_df.empty:
            st.info("No ranked findings available yet.")
        else:
            # Add relevance column via LLM
            if st.button("Generate relevance assessment", type="secondary"):
                with st.spinner("Assessing relevance to target population..."):
                    try:
                        resp = _llm.chat.completions.create(
                            model=config.MODEL,
                            max_tokens=800,
                            messages=[{"role": "user", "content": f"""
For each predictor below, rate its relevance to ICU sepsis patients 
(High/Medium/Low) and give a one-sentence clinical note.
Return ONLY a JSON array:
[{{"predictor": "...", "relevance": "High/Medium/Low", "note": "..."}}]

Predictors: {ranked_df["predictor"].tolist()}
                            """}]
                        )
                        raw = resp.choices[0].message.content
                        raw = re.sub(r"```json|```", "", raw).strip()
                        relevance_data = json.loads(raw)
                        relevance_df = pd.DataFrame(relevance_data)
                        ranked_df = ranked_df.merge(relevance_df, on="predictor", how="left")
                    except Exception as e:
                        st.warning(f"Could not generate relevance: {e}")

            show_cols = ["predictor", "best_metric", "value", "effect_size", "study"]
            if "relevance" in ranked_df.columns:
                show_cols = ["predictor", "best_metric", "value", "effect_size", "study", "relevance", "note"]
            show_cols = [c for c in show_cols if c in ranked_df.columns]

            st.dataframe(ranked_df[show_cols], use_container_width=True)

            st.download_button(
                "⬇ Download ranked predictors (CSV)",
                ranked_df.to_csv(index=False),
                file_name="usecase3_ranked_predictors.csv",
                mime="text/csv",
            )

            st.divider()
            st.subheader("Source traces")
            show_source_traces_from_df(ranked_df.rename(columns={"study": "paper_id"}))
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
            for k in ("paper", "pdf_path", "full_text", "warnings",
                      "selected_sentence", "_last_findings_idx"):
                st.session_state.pop(k, None)

            with st.spinner("Extracting text..."):
                full_text = extract_text(tmp_path)

            with st.spinner("Sending to LLM for structured extraction..."):
                llm_text = get_relevant_chunk(full_text, max_chars=config.CHUNK_SIZE)
                paper = extract_paper(llm_text, uploaded.name)
                is_valid, warnings = validate_paper(paper)
                paper = filter_low_confidence_fields(paper, min_confidence=config.MIN_CONFIDENCE)
                paper.compute_overall_confidence()
                save_paper(paper)
                save_cohorts(paper)
                save_findings(paper)   # ← keep colleague's findings save

            st.session_state.paper = paper
            st.session_state.pdf_path = str(tmp_path)
            st.session_state.full_text = full_text
            st.session_state.warnings = warnings
            st.session_state.selected_sentence = None

    if "paper" in st.session_state:
        paper = st.session_state.paper
        full_text = st.session_state.get("full_text", "")
        warnings = st.session_state.get("warnings", [])

        for w in warnings:
            st.warning(w)

        st.success(f"Extracted {len(full_text):,} characters")
        c1, c2, c3 = st.columns(3)
        c1.metric("Title", (paper.metadata.title.value or "N/A")[:40])
        c2.metric("Cohorts found", len(paper.cohorts))
        c3.metric("Overall confidence", f"{paper.overall_confidence:.0%}")

        st.divider()

        col_data, col_pdf = st.columns([2, 1], gap="large")

        with col_data:
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
                            if st.button("📍 Show in PDF", key=f"hl_{label}"):
                                st.session_state.selected_sentence = ef.source_sentence
                                st.rerun()
                        st.caption(f"Confidence: {confidence_badge(ef.confidence)}")

            if paper.prognostic_findings:
                st.markdown("### Prognostic findings")
                st.caption("Click a row to highlight its source sentence in the PDF.")
                findings_df = pd.DataFrame([f.model_dump() for f in paper.prognostic_findings])
                selection = st.dataframe(
                    findings_df,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="findings_table",
                )
                if selection.selection.rows:
                    idx = selection.selection.rows[0]
                    if idx != st.session_state.get("_last_findings_idx"):
                        sentence = paper.prognostic_findings[idx].source_sentence
                        if sentence:
                            st.session_state.selected_sentence = sentence
                        st.session_state._last_findings_idx = idx
                else:
                    st.session_state._last_findings_idx = None

            with st.expander("Raw text (first 2000 chars)"):
                st.text(full_text[:2000])

            with st.expander("Full JSON"):
                st.json(json.loads(paper.model_dump_json()))

        with col_pdf:
            st.markdown("### PDF")
            selected = st.session_state.get("selected_sentence")
            if selected:
                preview = selected if len(selected) <= 120 else selected[:120] + "..."
                st.info(f"Highlighting: \"{preview}\"")
                annotations, target_page = find_sentence_annotations(
                    st.session_state.pdf_path, selected
                )
                if not annotations:
                    st.warning("Couldn't locate that sentence in the PDF.")
                pdf_viewer(
                    input=st.session_state.pdf_path,
                    width=350, height=700,
                    annotations=annotations,
                    annotation_outline_size=2,
                    resolution_boost=2,
                    scroll_to_annotation=1 if annotations else None,
                )
            else:
                st.caption("Click 📍 on a metadata field or click a finding row to highlight here.")
                pdf_viewer(input=st.session_state.pdf_path, width=350, height=700, resolution_boost=2)

# ═══════════════════════════════════════════════════════════════
# Page 3: Browse Database
# ═══════════════════════════════════════════════════════════════
elif page == "Browse Database":
    st.title("Database")

    tab1, tab2, tab3 = st.tabs(["Cohorts", "Findings", "Papers"])

    with tab1:
        df = load_cohorts_with_paper_meta()
        if df.empty:
            st.info("No data yet.")
        else:
            st.metric("Total cohorts", len(df))
            display_cols = [c for c in df.columns if c != "source_sentence"]
            st.dataframe(df[display_cols], use_container_width=True)

            st.divider()
            st.subheader("Source traces")

            # Cohort-level sources
            for _, cohort_row in df.iterrows():
                if cohort_row.get("source_sentence"):
                    show_source_trace(
                        source_sentence=cohort_row["source_sentence"],
                        paper_id=cohort_row["paper_id"],
                        predictor=f"Cohort: {cohort_row.get('cohort_name', '')}",
                        outcome=f"Mortality: {cohort_row.get('mortality_rate', 'N/A')}",
                        confidence=float(cohort_row["confidence"]) if pd.notna(cohort_row.get("confidence")) else None
                    )

            # Field-level sources only for papers in this cohort list
            paper_ids_in_cohorts = df["paper_id"].unique().tolist()
            papers_df = load_all_papers()
            papers_df = papers_df[papers_df["paper_id"].isin(paper_ids_in_cohorts)]

            for _, paper_row in papers_df.iterrows():
                src_cols = [c for c in paper_row.index if c.endswith("_src") and paper_row.get(c)]
                for src_col in src_cols:
                    value_col = src_col.replace("_src", "")
                    conf_col = src_col.replace("_src", "_conf")
                    value = paper_row.get(value_col)
                    source = paper_row.get(src_col)
                    conf = paper_row.get(conf_col)
                    if value and source:
                        show_source_trace(
                            source_sentence=source,
                            paper_id=paper_row["paper_id"],
                            predictor=value_col.replace("_", " ").title(),
                            outcome=str(value)[:80],
                            confidence=float(conf) if conf else None
                        )

    with tab2:
        df_f = load_all_findings()
        if df_f.empty:
            st.info("No findings yet.")
        else:
            st.metric("Total findings", len(df_f))

            col1, col2 = st.columns(2)
            with col1:
                papers = ["All"] + sorted(df_f["paper_id"].dropna().unique().tolist())
                selected_paper = st.selectbox("Filter by paper", papers)
            with col2:
                predictors = ["All"] + sorted(df_f["predictor"].dropna().unique().tolist())
                selected_predictor = st.selectbox("Filter by predictor", predictors)

            filtered = df_f.copy()
            if selected_paper != "All":
                filtered = filtered[filtered["paper_id"] == selected_paper]
            if selected_predictor != "All":
                filtered = filtered[filtered["predictor"] == selected_predictor]

            show_cols = ["paper_id", "predictor", "outcome", "effect_size", "method", "confidence"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            st.dataframe(filtered[show_cols], use_container_width=True)

            st.divider()
            st.subheader("Source traces")
            show_source_traces_from_df(filtered)

    with tab3:
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

                # Field-level source traces
                st.markdown("#### Field sources")
                # Dynamically find all _src columns and show their sources
                src_cols = [c for c in row.index if c.endswith("_src") and row.get(c)]
                for src_col in src_cols:
                    value_col = src_col.replace("_src", "")
                    conf_col = src_col.replace("_src", "_conf")
                    value = row.get(value_col)
                    source = row.get(src_col)
                    conf = row.get(conf_col)
                    if value and source:
                        show_source_trace(
                            source_sentence=source,
                            paper_id=row["paper_id"],
                            predictor=value_col.replace("_", " ").title(),
                            outcome=str(value)[:80],
                            confidence=float(conf) if conf else None
                        )

        


# ═══════════════════════════════════════════════════════════════
# Page 4: Export
# ═══════════════════════════════════════════════════════════════
elif page == "Export":
    st.title("Export Data")

    tab1, tab2, tab3 = st.tabs(["Cohorts", "Findings", "Papers"])

    with tab1:
        df = load_cohorts_with_paper_meta()
        st.metric("Cohort rows", len(df))
        if not df.empty:
            st.download_button(
                "⬇ Download cohorts CSV",
                df.to_csv(index=False),
                file_name="sepsis_atlas_cohorts.csv",
                mime="text/csv",
            )
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.info("No data yet.")

    with tab2:
        df_f = load_all_findings()
        st.metric("Findings rows", len(df_f))
        if not df_f.empty:
            st.download_button(
                "⬇ Download findings CSV",
                df_f.to_csv(index=False),
                file_name="sepsis_atlas_findings.csv",
                mime="text/csv",
            )
            st.dataframe(df_f.head(10), use_container_width=True)
        else:
            st.info("No findings yet.")

    with tab3:
        df_p = load_all_papers()
        st.metric("Papers", len(df_p))
        if not df_p.empty:
            st.download_button(
                "⬇ Download papers CSV",
                df_p.to_csv(index=False),
                file_name="sepsis_atlas_papers.csv",
                mime="text/csv",
            )
            st.dataframe(df_p.head(10), use_container_width=True)
        else:
            st.info("No papers yet.")