# 📄 Resume ↔ Job Description Matcher

A tool that scores how well a resume matches a job description using
TF-IDF vectorization and cosine similarity, and surfaces the most
important JD keywords missing from the resume — with a Streamlit
interface for real-time scoring.

## What it does

- **Text preprocessing** — lightweight cleaning/tokenization with no
  external downloads required (`src/text_preprocessing.py`)
- **Matching** — fits a TF-IDF vectorizer over a resume+JD corpus and
  scores any resume/JD pair via cosine similarity, plus extracts the
  top JD keywords missing from a given resume (`src/matcher.py`)
- **Evaluation** — benchmarks match scores against 20 manually labeled
  resume-JD pairs, reporting correlation, ranking accuracy, and
  precision/recall/F1 at a chosen threshold (`src/evaluate.py`)
- **Dashboard** — a Streamlit app to paste in a resume and JD (or load
  a bundled sample pair) and get an instant score + keyword gaps
  (`app.py`)

## Project structure

```
resume-jd-matcher/
├── app.py                      # Streamlit app (final deliverable)
├── requirements.txt
├── src/
│   ├── sample_data.py          # synthetic resumes, JDs, labeled pairs
│   ├── text_preprocessing.py   # cleaning + tokenization
│   ├── matcher.py              # TF-IDF + cosine similarity matcher
│   └── evaluate.py             # evaluation against labeled pairs
├── data/                       # generated CSVs
├── models/                     # saved matcher (.joblib)
└── reports/
    └── evaluation_report.txt
```

## Getting started

```bash
# 1. Clone and install
git clone https://github.com/yourhandle/resume-jd-matcher.git
cd resume-jd-matcher
pip install -r requirements.txt

# 2. Generate sample data and fit the matcher
python src/sample_data.py
python src/matcher.py

# 3. Evaluate against the labeled pairs
python src/evaluate.py

# 4. Launch the app
streamlit run app.py
```

## Results (on the sample dataset)

| Metric | Value |
|---|---|
| Point-biserial correlation (score vs. human label) | 0.94 |
| Top-1 ranking accuracy | 100% |
| Precision / Recall / F1 @ 20% threshold | 1.00 / 1.00 / 1.00 |

See `reports/evaluation_report.txt` for the full per-pair breakdown.
Metrics are computed on a small, hand-labeled demo set (20 pairs across
5 resumes and 4 job descriptions) — treat them as a sanity check on the
approach, not a large-scale benchmark.

## How it works

1. All resume and JD text in the corpus is cleaned and tokenized
   (lowercased, punctuation-normalized, stopwords removed).
2. A `TfidfVectorizer` is fit across the combined resume + JD corpus
   to build a shared vocabulary and IDF weighting.
3. For a given resume/JD pair, both texts are vectorized and compared
   with cosine similarity, scaled to a 0–100% score.
4. Missing keywords are found by ranking the JD's TF-IDF terms by
   weight and reporting the top ones that don't appear in the resume.

## Possible extensions

- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`)
  for semantic rather than lexical matching
- Add spaCy-based lemmatization/NER to catch skill synonyms (e.g.
  "ML" vs "machine learning")
- Expand the labeled pair set for a more robust evaluation
- Support PDF/DOCX resume upload instead of pasted text

## Tech stack

Python · scikit-learn (TF-IDF, cosine similarity) · Pandas · Streamlit

## Author

Komal Rani — [github.com/yourhandle](https://github.com/yourhandle)
