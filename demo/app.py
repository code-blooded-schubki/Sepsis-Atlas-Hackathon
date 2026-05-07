import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

import config
from pipeline.pdf_reader import extract_text, get_relevant_chunk, parse_sections, chunk_paper
from pipeline.extractor import extract_paper
from pipeline.validator import validate_paper, filter_low_confidence_fields
from utils.db import init_db, save_paper, save_sections, save_chunks, load_all_papers, search_chunks

st.set_page_config(page_title="Sepsis Atlas", page_icon="🧬", layout="wide")

init_db()

st.sidebar.title("🧬 Sepsis Atlas")
page = st.sidebar.radio("Navigate", ["Evidence Query", "Extract Paper", "Browse Database", "Export"])


def confidence_badge(conf: float) -> str:
    if conf >= 0.8:
        return f"🟢 {conf:.0%}"
    elif conf >= 0.5:
        return f"🟡 {conf:.0%}"
    else:
        return f"🔴 {conf:.0%}"


def render_extracted_field(label: str, ef) -> None:
    if ef.value:
        with st.expander(f"**{label}** — {ef.value[:80]}{'…' if len(str(ef.value)) > 80 else ''}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(ef.value)
                if ef.source_sentence:
                    st.markdown(f"> *\"{ef.source_sentence}\"*")
                else:
                    st.caption("No source sentence available")
            with col2:
                st.metric("Confidence", confidence_badge(ef.confidence))
    else:
        st.caption(f"**{label}**: *not reported*")


# ═══════════════════════════════════════════════════════════════
# Page 1: Evidence Query  (semantic search via ChromaDB)
# ═══════════════════════════════════════════════════════════════
if page == "Evidence Query":
    st.title("Evidence Query")
    st.markdown("Ask anything about the sepsis literature — results are retrieved semantically from the full text of all papers.")

    col_q, col_sec = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Clinical question",
            placeholder="e.g. What is the relationship between lactate and 28-day mortality in septic shock?",
        )
    with col_sec:
        section_filter = st.selectbox(
            "Section filter",
            ["All sections", "abstract", "introduction", "methods", "results", "discussion", "conclusion"],
        )

    n_results = st.slider("Max results", min_value=3, max_value=20, value=8)

    if st.button("Search", type="primary") and query:
        section = None if section_filter == "All sections" else section_filter
        with st.spinner("Searching..."):
            results = search_chunks(query, n_results=n_results, section_name=section)

        if not results:
            st.warning("No results found. Make sure the pipeline has been run and chunks are stored.")
        else:
            st.success(f"{len(results)} matching passage(s) found across the literature")

            # Summary table
            summary_rows = []
            for r in results:
                summary_rows.append({
                    "Score": r["score"],
                    "Paper": r.get("title") or r["paper_id"],
                    "Year": r.get("year") or "—",
                    "Section": r["section_name"],
                    "Design": r.get("study_design") or "—",
                    "N": r.get("sample_size") or "—",
                    "Mortality": r.get("mortality") or "—",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            st.markdown("---")
            st.subheader("Matching passages")
            for r in results:
                title = r.get("title") or r["paper_id"]
                year  = r.get("year") or ""
                score = r["score"]
                section_name = r["section_name"]

                badge = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
                with st.expander(f"{badge} **{title}** ({year}) — {section_name} — score {score}"):
                    st.markdown(f"> {r['chunk_text']}")
                    st.divider()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Study design", r.get("study_design") or "—")
                    c2.metric("N patients",   r.get("sample_size") or "—")
                    c3.metric("Mortality",    r.get("mortality") or "—")
                    c4.metric("Sepsis def.",  r.get("sepsis_definition") or "—")

            # Download
            dl_df = pd.DataFrame([{
                "paper_id": r["paper_id"],
                "title": r.get("title") or "",
                "year": r.get("year") or "",
                "section": r["section_name"],
                "score": r["score"],
                "passage": r["chunk_text"],
            } for r in results])
            st.download_button(
                "⬇ Download results (CSV)",
                dl_df.to_csv(index=False),
                file_name="semantic_search_results.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════
# Page 2: Extract a new paper
# ═══════════════════════════════════════════════════════════════
elif page == "Extract Paper":
    st.title("Extract a Sepsis Paper")
    st.markdown("Upload a PDF and the pipeline will extract structured clinical data from it.")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        tmp_path = config.PAPERS_DIR / uploaded.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        if st.button("Run extraction", type="primary"):
            with st.spinner("Extracting text from PDF..."):
                full_text = extract_text(tmp_path)

            st.success(f"Extracted {len(full_text):,} characters from PDF")

            with st.expander("Raw extracted text (first 2000 chars)", expanded=False):
                st.text(full_text[:2000])

            paper_id = tmp_path.stem
            with st.spinner("Parsing sections and saving chunks..."):
                sections = parse_sections(full_text, paper_id)
                save_sections(sections)
                chunks = chunk_paper(sections, paper_id)
                save_chunks(chunks)
            st.info(f"Saved {len(sections)} sections and {len(chunks)} chunks to database")

            with st.spinner("Sending to LLM for structured extraction..."):
                llm_text = get_relevant_chunk(full_text, max_chars=config.CHUNK_SIZE)
                paper = extract_paper(llm_text, uploaded.name)
                is_valid, warnings = validate_paper(paper)
                paper = filter_low_confidence_fields(paper, min_confidence=config.MIN_CONFIDENCE)
                paper.compute_overall_confidence()
                save_paper(paper)

            if warnings:
                for w in warnings:
                    st.warning(w)

            st.markdown("---")
            st.subheader(f"Extraction complete — overall confidence: {paper.overall_confidence:.0%}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Title", paper.metadata.title.value or "N/A")
            col2.metric("Patients", paper.population.sample_size.value or "N/A")
            col3.metric("Mortality", paper.outcomes.mortality_rate.value or "N/A")

            st.markdown("### Study metadata")
            for label, ef in [
                ("Title", paper.metadata.title),
                ("Year", paper.metadata.year),
                ("Journal", paper.metadata.journal),
                ("Study design", paper.metadata.study_design),
                ("Country/Region", paper.metadata.country_or_region),
            ]:
                render_extracted_field(label, ef)

            st.markdown("### Patient population")
            for label, ef in [
                ("Sample size", paper.population.sample_size),
                ("Mean age", paper.population.mean_age),
                ("% Male", paper.population.percent_male),
                ("Clinical setting", paper.population.clinical_setting),
                ("Inclusion criteria", paper.population.inclusion_criteria_summary),
            ]:
                render_extracted_field(label, ef)

            st.markdown("### Sepsis definition")
            for label, ef in [
                ("Definition used", paper.sepsis_definition.definition_used),
                ("SOFA score", paper.sepsis_definition.sofa_score_reported),
                ("qSOFA", paper.sepsis_definition.qsofa_reported),
                ("Lactate threshold", paper.sepsis_definition.lactate_threshold),
                ("Septic shock included", paper.sepsis_definition.septic_shock_included),
            ]:
                render_extracted_field(label, ef)

            st.markdown("### Interventions")
            for label, ef in [
                ("Primary intervention", paper.interventions.primary_intervention),
                ("Comparison group", paper.interventions.comparison_group),
                ("Antibiotic protocol", paper.interventions.antibiotic_protocol),
                ("Fluid resuscitation", paper.interventions.fluid_resuscitation),
                ("Vasopressor use", paper.interventions.vasopressor_use),
            ]:
                render_extracted_field(label, ef)

            st.markdown("### Outcomes")
            for label, ef in [
                ("Primary outcome", paper.outcomes.primary_outcome),
                ("Mortality rate", paper.outcomes.mortality_rate),
                ("Mortality timepoint", paper.outcomes.mortality_timepoint),
                ("ICU length of stay", paper.outcomes.icu_length_of_stay),
                ("Secondary outcomes", paper.outcomes.secondary_outcomes_summary),
            ]:
                render_extracted_field(label, ef)

            if paper.prognostic_findings:
                st.markdown("### Prognostic findings")
                rows = [f.model_dump() for f in paper.prognostic_findings]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            if paper.extraction_notes:
                st.info(f"Extraction notes: {paper.extraction_notes}")

            with st.expander("Full JSON output"):
                st.json(json.loads(paper.model_dump_json()))


# ═══════════════════════════════════════════════════════════════
# Page 3: Browse database
# ═══════════════════════════════════════════════════════════════
elif page == "Browse Database":
    st.title("Extracted Papers Database")

    df = load_all_papers()

    if df.empty:
        st.info("No papers extracted yet. Go to 'Extract Paper' to get started.")
    else:
        st.metric("Total papers", len(df))

        display_cols = {
            "pdf_filename": "File",
            "meta_title": "Title",
            "meta_year": "Year",
            "meta_study_design": "Design",
            "pop_sample_size": "N patients",
            "sep_definition": "Sepsis def.",
            "out_mortality": "Mortality",
            "overall_confidence": "Confidence",
        }
        show_df = df[list(display_cols.keys())].rename(columns=display_cols)
        show_df["Confidence"] = show_df["Confidence"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
        st.dataframe(show_df, use_container_width=True)

        st.markdown("---")
        selected = st.selectbox("View paper details", options=df["pdf_filename"].tolist())
        if selected:
            row = df[df["pdf_filename"] == selected].iloc[0]
            st.subheader(row.get("meta_title") or selected)

            c1, c2, c3 = st.columns(3)
            c1.metric("Year", row.get("meta_year") or "N/A")
            c2.metric("N patients", row.get("pop_sample_size") or "N/A")
            c3.metric("Mortality", row.get("out_mortality") or "N/A")

            st.markdown("#### Source traces")
            source_fields = [
                ("Sepsis definition", "sep_definition", "sep_definition_src", "sep_definition_conf"),
                ("Mortality rate", "out_mortality", "out_mortality_src", "out_mortality_conf"),
                ("Sample size", "pop_sample_size", "pop_sample_size_src", "pop_sample_size_conf"),
                ("Primary intervention", "int_primary", "int_primary_src", "int_primary_conf"),
            ]
            for label, val_col, src_col, conf_col in source_fields:
                val = row.get(val_col)
                src = row.get(src_col)
                conf = row.get(conf_col, 0.0)
                if val:
                    with st.expander(f"**{label}**: {val}"):
                        if src:
                            st.markdown(f"> *\"{src}\"*")
                        st.caption(f"Confidence: {confidence_badge(conf)}")

            raw = row.get("prog_findings")
            if raw:
                try:
                    findings = json.loads(raw)
                    if findings:
                        st.markdown("#### Prognostic findings")
                        st.dataframe(pd.DataFrame(findings), use_container_width=True)
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════
# Page 4: Export
# ═══════════════════════════════════════════════════════════════
elif page == "Export":
    st.title("Export Data")

    df = load_all_papers()
    st.metric("Papers ready to export", len(df))

    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇ Download CSV",
            data=csv,
            file_name="sepsis_atlas_extracted.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Data preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("No data to export yet.")
