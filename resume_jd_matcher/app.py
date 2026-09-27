"""
Streamlit dashboard: paste a resume + job description, get an instant match
score and the top missing keywords.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from matcher import ResumeJDMatcher, fit_on_sample_data, DEFAULT_MODEL_PATH
from sample_data import RESUMES, JOB_DESCRIPTIONS

st.set_page_config(page_title="Resume-JD Matcher", page_icon="📄", layout="centered")
st.title("📄 Resume ↔ Job Description Matcher")
st.caption(
    "Scores how well a resume matches a job description using TF-IDF + cosine "
    "similarity, and highlights the most important JD keywords missing from the resume."
)


@st.cache_resource
def load_matcher() -> ResumeJDMatcher:
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            return ResumeJDMatcher.load(DEFAULT_MODEL_PATH)
        except Exception:
            pass
    # Fall back to fitting fresh on the sample corpus if no saved model exists yet
    matcher = fit_on_sample_data()
    matcher.save()
    return matcher


matcher = load_matcher()

with st.sidebar:
    st.subheader("Load a sample pair")
    st.caption("Populates the text boxes below with a bundled synthetic resume + JD.")
    sample_resume_id = st.selectbox("Sample resume", list(RESUMES.keys()), index=0)
    sample_jd_id = st.selectbox("Sample job description", list(JOB_DESCRIPTIONS.keys()), index=0)
    load_sample = st.button("Load sample pair")

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

if load_sample:
    st.session_state.resume_text = " ".join(RESUMES[sample_resume_id].split())
    st.session_state.jd_text = " ".join(JOB_DESCRIPTIONS[sample_jd_id].split())

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area(
        "Resume text", value=st.session_state.resume_text, height=280,
        placeholder="Paste resume text here...",
    )
with col2:
    jd_text = st.text_area(
        "Job description text", value=st.session_state.jd_text, height=280,
        placeholder="Paste job description text here...",
    )

if st.button("Score match", type="primary", disabled=not (resume_text.strip() and jd_text.strip())):
    score = matcher.score(resume_text, jd_text)
    missing = matcher.missing_keywords(resume_text, jd_text, top_n=10)

    st.subheader("Match score")
    st.progress(min(score / 100, 1.0))
    st.metric("Similarity", f"{score:.1f}%")

    if score >= 60:
        st.success("Strong match.")
    elif score >= 35:
        st.warning("Partial match — consider addressing the gaps below.")
    else:
        st.error("Weak match — this resume likely needs significant tailoring for this JD.")

    st.subheader("Top missing keywords from the JD")
    if missing:
        st.write(
            "These are the highest-weighted terms in the job description that "
            "don't appear anywhere in the resume:"
        )
        st.write(", ".join(f"`{kw}`" for kw in missing))
    else:
        st.write("No significant keyword gaps found — the resume covers the JD's key terms well.")

st.divider()
st.caption(
    "This is a lexical (keyword-based) matcher, not a semantic one — it won't recognize "
    "that 'ML' and 'machine learning' mean the same thing. See the README for extension ideas."
)
