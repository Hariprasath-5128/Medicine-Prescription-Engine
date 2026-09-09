"""Streamlit UI for the Medicine Prescription Engine.

    streamlit run src/ui/app.py

Deliberately plain. The one piece of real interaction design is the citation
expander: each numbered citation opens to show the exact retrieved chunk, its
section and relevance, and a link to the original source - so a reader can
check the evidence rather than trusting the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Medicine Prescription Engine", page_icon="+",
                   layout="wide")


@st.cache_resource
def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource
def load_retriever():
    from src.rag.retriever import EvidenceRetriever

    return EvidenceRetriever(load_config())


@st.cache_resource
def load_pipeline():
    """Returns (pipeline, error) so an untrained model degrades to search-only."""
    from src.rag.pipeline import PrescriptionPipeline

    try:
        return PrescriptionPipeline(load_config()), None
    except Exception as exc:
        return None, str(exc)


cfg = load_config()

st.title("Medicine Prescription Engine")
st.caption("Evidence-grounded medication decision support - research use only")

st.warning(cfg["safety"]["disclaimer"].strip(), icon="!")


def render_evidence(evidence_list, key_prefix: str) -> None:
    """Numbered, expandable citations: chunk text + provenance + source link."""
    if not evidence_list:
        st.error(
            "No supporting evidence passed the relevance threshold. "
            "This suggestion is UNVERIFIED - treat it as a hypothesis only.",
            icon="!",
        )
        return

    st.subheader("Supporting evidence")
    for i, ev in enumerate(evidence_list, 1):
        get = ev.get if isinstance(ev, dict) else lambda k, d=None: getattr(ev, k, d)
        focus = (get("question_focus") or get("document_id") or "source")
        section = (get("section") or "").title()
        label = f"[{i}] {str(focus).title()}" + (f" - {section}" if section else "")

        with st.expander(label, expanded=(i == 1)):
            st.markdown(f"> {get('text', '')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Relevance", get("relevance", "-"))
            c2.markdown(f"**Section**\n\n{section or '-'}")
            c3.markdown(f"**Semantic group**\n\n{get('umls_semantic_group') or '-'}")

            st.caption(f"chunk `{get('chunk_id', '')}` | document "
                       f"`{get('document_id', '')}`")
            url = get("source_url")
            if url:
                st.markdown(f"[View original source]({url})")


tab_rec, tab_eve, tab_about = st.tabs(["Recommendation", "Evidence search", "About"])

with tab_rec:
    pipeline, error = load_pipeline()
    if error:
        st.info(
            "The recommender is not trained yet, so this tab is inactive. "
            "Run `python scripts/03_train_recommender.py`, then reload.\n\n"
            "The **Evidence search** tab works now - it only needs the index.",
            icon="i",
        )
        st.caption(f"details: {error}")
    else:
        text = st.text_area(
            "Describe the symptoms",
            placeholder="e.g. I have had a persistent cough with mucus for two weeks",
            height=110,
        )
        top_k = st.slider("Number of medications to suggest", 1, 10, 5)

        if st.button("Get recommendation", type="primary") and text.strip():
            with st.spinner("Analysing and retrieving evidence..."):
                rec = pipeline.run(text, top_k_drugs=top_k)

            if rec.red_flag:
                st.error(
                    f"**URGENT - SEEK IMMEDIATE MEDICAL ATTENTION**\n\n"
                    f"Your description mentions: **{', '.join(rec.red_flag_terms)}**. "
                    f"These can signal a medical emergency. Contact emergency "
                    f"services or attend an emergency department now.",
                    icon="!",
                )

            left, right = st.columns([1, 1])
            with left:
                st.subheader("Assessment")
                st.metric("Likely condition", rec.condition,
                          f"{rec.condition_confidence:.1%} confidence")
                if rec.alternatives:
                    st.caption("Also considered")
                    for name, score in rec.alternatives:
                        st.write(f"- {name} ({score:.1%})")
            with right:
                st.subheader("Associated medications")
                st.caption("Ranked by association in patient-reported data - "
                           "NOT a dosage or a prescription.")
                for i, (name, score) in enumerate(rec.drugs, 1):
                    st.write(f"**{i}. {name}**")
                    st.progress(min(float(score), 1.0))

            st.divider()
            render_evidence(rec.evidence, "rec")

with tab_eve:
    st.caption(f"Searching {cfg['evidence']['collection']} "
               f"(bge-m3 embeddings, {cfg['evidence']['top_k']} results by default)")
    query = st.text_input("Search the medical evidence corpus",
                          placeholder="e.g. what causes keratoderma with woolly hair")
    col_a, col_b = st.columns(2)
    k = col_a.slider("Results", 1, 15, 5)
    max_dist = col_b.slider("Max distance (lower = stricter)", 0.1, 1.0,
                            float(cfg["evidence"]["max_distance"]), 0.05)

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Embedding query and searching..."):
            result = load_retriever().retrieve(query, k, max_dist)
        if result.filtered_all:
            st.warning("Neighbours were found but all exceeded the distance "
                       "threshold. Loosen the slider to inspect them.", icon="!")
        render_evidence(result.evidence, "search")

with tab_about:
    st.markdown(f"""
### How this works

    symptoms
      -> safety triage (red-flag terms escalate immediately)
      -> recommender    condition + ranked drugs (PubMedBERT, trained on
                        95,456 deduplicated patient reviews)
      -> retriever      evidence chunks from {cfg['evidence']['collection']}
                        (bge-m3, cosine distance <= {cfg['evidence']['max_distance']})
      -> citations      each with section, relevance and a source URL

### What it is not

The medication list reflects **statistical association in patient-reported
reviews**, not a clinical prescribing decision. It carries no dose, route,
frequency, duration, contraindication or interaction check. The evidence corpus
is disease-information text, so it can confirm that a *condition* is described
as stated; it does not constitute drug-efficacy evidence.

Every recommendation requires review by a qualified clinician.
""")
    st.caption(cfg["safety"]["disclaimer"].strip())
