"""
app.py
------
Streamlit interface: paste a resume and a job description, get an
instant match score and a list of important JD keywords missing from
the resume.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from matcher import ResumeJDMatcher, build_and_save_default_matcher  # noqa: E402
MODEL_PATH = BASE / "models" / "matcher.joblib"
DATA_DIR = BASE / "data"

st.set_page_config(page_title="Resume ↔ JD Matcher", layout="centered")


@st.cache_resource
def get_matcher() -> ResumeJDMatcher:
    if MODEL_PATH.exists():
        return ResumeJDMatcher.load(MODEL_PATH)
    return build_and_save_default_matcher()


@st.cache_data
def get_sample_texts():
    resumes = pd.read_csv(DATA_DIR / "resumes.csv")
    jds = pd.read_csv(DATA_DIR / "job_descriptions.csv")
    return resumes, jds


st.title("📄 Resume ↔ Job Description Matcher")
st.caption(
    "TF-IDF + cosine similarity match scoring with missing-keyword "
    "suggestions. Paste your own text, or load a sample pair below."
)

matcher = get_matcher()
resumes_df, jds_df = get_sample_texts()

with st.expander("Load a sample resume / JD pair"):
    col_a, col_b = st.columns(2)
    with col_a:
        sample_resume = st.selectbox(
            "Sample resume", ["(none)"] + list(resumes_df["candidate"])
        )
    with col_b:
        sample_jd = st.selectbox(
            "Sample job description", ["(none)"] + list(jds_df["title"])
        )

default_resume = ""
if sample_resume != "(none)":
    default_resume = resumes_df.loc[resumes_df["candidate"] == sample_resume, "text"].iloc[0].strip()

default_jd = ""
if sample_jd != "(none)":
    default_jd = jds_df.loc[jds_df["title"] == sample_jd, "text"].iloc[0].strip()

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("Resume text", value=default_resume, height=280,
                                placeholder="Paste resume text here...")
with col2:
    jd_text = st.text_area("Job description text", value=default_jd, height=280,
                            placeholder="Paste job description text here...")

if st.button("Score match", type="primary", use_container_width=True):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please provide both resume and job description text.")
    else:
        score = matcher.score(resume_text, jd_text)
        missing = matcher.missing_keywords(resume_text, jd_text, top_n=10)

        st.divider()
        st.subheader("Results")

        if score >= 40:
            st.success(f"Match score: **{score}%** — strong overlap")
        elif score >= 20:
            st.info(f"Match score: **{score}%** — partial overlap")
        else:
            st.warning(f"Match score: **{score}%** — low overlap")

        st.progress(min(int(score), 100))

        st.markdown("**Top JD keywords missing from this resume:**")
        if missing:
            st.write(", ".join(f"`{m}`" for m in missing))
        else:
            st.write("No significant gaps found — great coverage!")

        st.caption(
            "Score is computed on the fitted TF-IDF vocabulary from this "
            "project's sample corpus; scores shift as more resumes/JDs are "
            "added to the training corpus (see `src/matcher.py`)."
        )

st.divider()
st.caption(
    "Model: TF-IDF (unigrams) + cosine similarity, scikit-learn. "
    "Evaluated against 20 manually labeled resume-JD pairs — see "
    "`reports/evaluation_report.txt` for metrics."
)
