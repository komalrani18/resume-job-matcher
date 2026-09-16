"""
matcher.py
----------
Core matching logic: fits a TF-IDF vectorizer over resume + JD text,
scores resume-to-JD fit with cosine similarity, and surfaces
important JD keywords missing from the resume.

Run:
    python src/matcher.py
(prints example scores for the sample data)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_preprocessing import preprocess_for_vectorizer, tokenize

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"


class ResumeJDMatcher:
    """TF-IDF + cosine-similarity resume/job-description matcher."""

    def __init__(self, max_features: int = 3000, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self._is_fit = False

    def fit(self, corpus: list[str]):
        """Fit the TF-IDF vocabulary on a corpus of resume + JD texts."""
        cleaned = [preprocess_for_vectorizer(t) for t in corpus]
        self.vectorizer.fit(cleaned)
        self._is_fit = True
        return self

    def score(self, resume_text: str, jd_text: str) -> float:
        """Cosine similarity between a single resume and JD, as a 0-100 score."""
        if not self._is_fit:
            raise RuntimeError("Call .fit() with a corpus before scoring.")
        vecs = self.vectorizer.transform([
            preprocess_for_vectorizer(resume_text),
            preprocess_for_vectorizer(jd_text),
        ])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return round(float(sim) * 100, 1)

    def missing_keywords(self, resume_text: str, jd_text: str, top_n: int = 10) -> list[str]:
        """Top TF-IDF-weighted JD terms that don't appear in the resume."""
        if not self._is_fit:
            raise RuntimeError("Call .fit() with a corpus before scoring.")

        jd_clean = preprocess_for_vectorizer(jd_text)
        jd_vec = self.vectorizer.transform([jd_clean])
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        jd_scores = jd_vec.toarray()[0]
        nonzero_idx = jd_scores.nonzero()[0]
        ranked_idx = nonzero_idx[np.argsort(-jd_scores[nonzero_idx])]
        ranked_terms = feature_names[ranked_idx]

        resume_tokens = set(tokenize(resume_text))
        # also match multi-word phrases against the resume's raw token set
        missing = []
        for term in ranked_terms:
            term_tokens = term.split()
            if not all(tok in resume_tokens for tok in term_tokens):
                missing.append(term)
            if len(missing) >= top_n:
                break
        return missing

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ResumeJDMatcher":
        return joblib.load(path)


def build_and_save_default_matcher() -> ResumeJDMatcher:
    """Fits the matcher on the bundled sample resumes + JDs and saves it,
    so the Streamlit app has a sensible default vocabulary to start from."""
    resumes = pd.read_csv(BASE / "data" / "resumes.csv")
    jds = pd.read_csv(BASE / "data" / "job_descriptions.csv")
    corpus = list(resumes["text"]) + list(jds["text"])

    matcher = ResumeJDMatcher().fit(corpus)
    MODEL_DIR.mkdir(exist_ok=True)
    matcher.save(MODEL_DIR / "matcher.joblib")
    return matcher


if __name__ == "__main__":
    matcher = build_and_save_default_matcher()

    resumes = pd.read_csv(BASE / "data" / "resumes.csv")
    jds = pd.read_csv(BASE / "data" / "job_descriptions.csv")

    r = resumes[resumes.resume_id == "R1"].iloc[0]
    j = jds[jds.jd_id == "J1"].iloc[0]

    print(f"{r['candidate']} vs '{j['title']}'")
    print(f"Match score: {matcher.score(r['text'], j['text'])}%")
    print(f"Missing keywords: {matcher.missing_keywords(r['text'], j['text'])}")
