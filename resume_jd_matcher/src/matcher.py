"""
Core matching logic: fits a TF-IDF vectorizer over a resume+JD corpus, scores
any resume/JD pair by cosine similarity, and extracts the top JD keywords
missing from a given resume.

Run directly to fit the matcher on the sample corpus and save it:
    python src/matcher.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_preprocessing import preprocess_for_tfidf

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "matcher.joblib"
)


class ResumeJDMatcher:
    """Fits one shared TF-IDF vocabulary/IDF weighting across a resume+JD corpus,
    then scores arbitrary resume/JD text pairs by cosine similarity."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, corpus_texts: list):
        """corpus_texts: list of raw (uncleaned) resume + JD strings to build the
        shared vocabulary/IDF weights from."""
        preprocessed = [preprocess_for_tfidf(t) for t in corpus_texts]
        self.vectorizer.fit(preprocessed)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Matcher has not been fit yet. Call .fit() or load a saved model.")

    def score(self, resume_text: str, jd_text: str) -> float:
        """Returns a 0-100 match score between one resume and one JD."""
        self._check_fitted()
        resume_clean = preprocess_for_tfidf(resume_text)
        jd_clean = preprocess_for_tfidf(jd_text)
        vectors = self.vectorizer.transform([resume_clean, jd_clean])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        return round(float(sim) * 100, 2)

    def missing_keywords(self, resume_text: str, jd_text: str, top_n: int = 10) -> list:
        """Ranks the JD's TF-IDF terms by weight and returns the top ones that do
        not appear (as a token) anywhere in the resume."""
        self._check_fitted()
        resume_tokens = set(preprocess_for_tfidf(resume_text).split())
        jd_clean = preprocess_for_tfidf(jd_text)

        jd_vector = self.vectorizer.transform([jd_clean])
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        jd_scores = jd_vector.toarray()[0]

        # Rank JD terms by TF-IDF weight, descending
        ranked_idx = np.argsort(jd_scores)[::-1]

        missing = []
        for idx in ranked_idx:
            if jd_scores[idx] <= 0:
                break
            term = feature_names[idx]
            # For multi-word n-grams, check if ALL constituent words are present;
            # for single words, a direct membership check.
            term_words = term.split()
            present = all(word in resume_tokens for word in term_words)
            if not present:
                missing.append(term)
            if len(missing) >= top_n:
                break
        return missing

    def save(self, path: str = DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str = DEFAULT_MODEL_PATH) -> "ResumeJDMatcher":
        return joblib.load(path)


def fit_on_sample_data(data_dir: str = None) -> ResumeJDMatcher:
    """Convenience function: loads the generated sample CSVs and fits a matcher
    on the combined resume + JD corpus."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(project_root, "data")

    resumes_path = os.path.join(data_dir, "resumes.csv")
    jds_path = os.path.join(data_dir, "job_descriptions.csv")

    if not (os.path.exists(resumes_path) and os.path.exists(jds_path)):
        raise FileNotFoundError(
            f"Sample data not found in '{data_dir}'. Run `python src/sample_data.py` first."
        )

    resumes_df = pd.read_csv(resumes_path)
    jds_df = pd.read_csv(jds_path)

    corpus = list(resumes_df["text"]) + list(jds_df["text"])
    matcher = ResumeJDMatcher().fit(corpus)
    return matcher


if __name__ == "__main__":
    matcher = fit_on_sample_data()
    saved_path = matcher.save()
    print(f"Fitted TF-IDF matcher on the sample corpus and saved to '{saved_path}'.")

    # Quick sanity check
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resumes_df = pd.read_csv(os.path.join(project_root, "data", "resumes.csv"))
    jds_df = pd.read_csv(os.path.join(project_root, "data", "job_descriptions.csv"))

    r1 = resumes_df[resumes_df.resume_id == "R1_data_scientist"].text.iloc[0]
    j1 = jds_df[jds_df.jd_id == "JD1_data_scientist"].text.iloc[0]
    j2 = jds_df[jds_df.jd_id == "JD4_devops_engineer"].text.iloc[0]

    print(f"\nSanity check — Data Scientist resume vs Data Scientist JD: {matcher.score(r1, j1)}%")
    print(f"Sanity check — Data Scientist resume vs DevOps JD:         {matcher.score(r1, j2)}%")
    print(f"Missing keywords (DS resume vs DS JD): {matcher.missing_keywords(r1, j1, top_n=8)}")
