"""
text_preprocessing.py
----------------------
Lightweight text cleaning shared by the matcher and evaluator.
Deliberately avoids external downloads (no NLTK/spaCy corpora) so the
project runs offline out of the box; swap in spaCy lemmatization later
if desired (see README).
"""

import re

# Small, hand-picked English stopword list (kept local -> no downloads).
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "for", "with", "by", "is", "are", "was", "were",
    "be", "been", "being", "as", "it", "this", "that", "these", "those",
    "from", "into", "than", "we", "you", "your", "our", "will", "would",
    "can", "could", "should", "has", "have", "had", "not", "no", "do",
    "does", "did", "up", "down", "out", "about", "over", "under", "again",
    "further", "here", "there", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "too",
    "very", "just", "also", "using", "used", "use", "including", "required",
    "preferred", "experience", "plus", "role", "looking", "seeking",
    "help", "big", "desired", "strong", "hands", "familiarity", "skills",
    "skilled", "proficient", "comfortable", "build", "building", "built",
    "work", "working", "worked", "years", "year", "well", "big", "you'll",
    "hiring", "reporting", "support", "knowledge",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#]*(?:\.[a-zA-Z0-9+#]+)*")


def clean_text(text: str) -> str:
    """Lowercase, strip stray whitespace/newlines, normalize punctuation
    spacing. Keeps tokens like 'c++' and 'ci/cd' intact for tokenization."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    text = clean_text(text)
    tokens = TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def preprocess_for_vectorizer(text: str) -> str:
    """Returns a cleaned, space-joined token string ready for TfidfVectorizer."""
    return " ".join(tokenize(text))
