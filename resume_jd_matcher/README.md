# Resume ↔ JD Matcher

A tool that scores how well a resume matches a job description using TF-IDF
vectorization and cosine similarity, and surfaces the most important JD keywords
missing from the resume — with a Streamlit interface for real-time scoring.

## What it does

- **Text preprocessing** — lightweight cleaning/tokenization with no external
  downloads required (`src/text_preprocessing.py`)
- **Matching** — fits a TF-IDF vectorizer over a resume+JD corpus and scores any
  resume/JD pair via cosine similarity, plus extracts the top JD keywords missing
  from a given resume (`src/matcher.py`)
- **Evaluation** — benchmarks match scores against 20 manually labeled resume-JD
  pairs, reporting correlation, ranking accuracy, and precision/recall/F1 at a
  chosen threshold (`src/evaluate.py`)
- **Dashboard** — a Streamlit app to paste in a resume and JD (or load a bundled
  sample pair) and get an instant score + keyword gaps (`app.py`)

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
# 1. Install
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

These numbers come from actually running `src/evaluate.py` against the bundled
synthetic dataset (5 resumes × 4 job descriptions = 20 labeled pairs) — they are
not hypothetical.

| Metric | Value |
|---|---|
| Point-biserial correlation (score vs. human label) | 0.96 |
| Top-1 ranking accuracy | 100% |
| Precision / Recall / F1 @ top-20% threshold | 1.00 / 1.00 / 1.00 |

See `reports/evaluation_report.txt` for the full per-pair breakdown. **Metrics are
computed on a small, hand-labeled demo set (20 pairs across 5 resumes and 4 job
descriptions) — treat them as a sanity check on the approach, not a large-scale
benchmark.** The dataset was constructed so each resume's true role matches
exactly one JD's role (or none, for the product-manager resume, which has no
corresponding JD in this small set) — this makes the "correct" labels unambiguous,
but also means the task is easier than messy real-world resumes/JDs would be.

## How it works

1. All resume and JD text in the corpus is cleaned and tokenized (lowercased,
   punctuation-normalized, a curated stopword list removed — including common
   job-posting boilerplate like "required"/"looking"/"responsibilities" so the
   missing-keyword output stays focused on actual skills).
2. A `TfidfVectorizer` (unigrams + bigrams) is fit across the combined resume +
   JD corpus to build a shared vocabulary and IDF weighting.
3. For a given resume/JD pair, both texts are vectorized and compared with cosine
   similarity, scaled to a 0–100% score.
4. Missing keywords are found by ranking the JD's TF-IDF terms by weight and
   reporting the top ones that don't appear (as a token) in the resume.

## Known limitations

- **Lexical, not semantic**: this is pure keyword/n-gram matching. It won't
  recognize that "ML" and "machine learning" refer to the same skill, or that
  "Node.js" and "JavaScript backend" overlap conceptually. See "Possible
  extensions" below.
- **Bigram artifacts**: because stopwords are stripped *before* n-grams are
  generated, some reported "missing keyword" bigrams bridge across a removed
  word and read a little oddly (e.g. "present findings to non-technical
  stakeholders" can surface as "findings non"). The underlying skill/topic
  signal is usually still legible, but it's not publication-quality phrasing.
- **Small evaluation set**: 20 hand-built pairs is enough to sanity-check that
  the scoring behaves sensibly (matching pairs score much higher than
  non-matching ones), but far too small to treat the correlation/F1 numbers as
  a rigorous benchmark.

## Possible extensions

- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`) for
  semantic rather than lexical matching.
- Add spaCy-based lemmatization/NER to catch skill synonyms (e.g. "ML" vs
  "machine learning").
- Expand the labeled pair set for a more robust evaluation.
- Support PDF/DOCX resume upload instead of pasted text.

## Tech stack

Python · scikit-learn (TF-IDF, cosine similarity) · pandas · SciPy (point-biserial
correlation) · joblib (model persistence) · Streamlit
