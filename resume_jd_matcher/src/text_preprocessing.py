"""
Lightweight text cleaning/tokenization for resumes and job descriptions.

Deliberately dependency-free (no NLTK/spaCy downloads) so this runs anywhere
`pip install -r requirements.txt` works, with no extra corpus/model downloads.
"""
import re

# A small, hand-picked English stopword list — enough to strip filler words from
# resumes/JDs without needing to download NLTK's corpus.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "once", "here", "there",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "of", "as", "it", "its", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "them", "their", "our", "your", "his", "her",
    "who", "whom", "which", "what", "would", "could", "may", "might", "must",
    "shall", "using", "used", "use", "etc", "per",
    # Common job-posting boilerplate that survives basic stopword removal but
    # carries little signal as a "missing skill" — filtering these keeps the
    # missing-keyword output focused on actual skills/tools rather than
    # generic posting language.
    "required", "looking", "skills", "experience", "strong", "join", "team",
    "responsibilities", "plus", "including", "years", "seeking", "candidate",
    "role", "work", "working", "ability", "preferred", "requirements",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#./-]*")


def clean_text(text: str) -> str:
    """Lowercase, normalize whitespace, strip most punctuation while preserving
    tokens like 'c++', 'node.js', 'ci/cd' that carry meaning in tech resumes."""
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    """Extracts word-like tokens, optionally removing stopwords. Keeps tokens
    with internal +/#/./- so skills like 'c++', 'c#', 'ci/cd' survive intact."""
    cleaned = clean_text(text)
    tokens = _WORD_RE.findall(cleaned)
    tokens = [t.strip(".-") for t in tokens if t.strip(".-")]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    """Returns a cleaned, stopword-filtered string ready to hand to TfidfVectorizer
    (which does its own final tokenization) — this step does the domain-specific
    cleaning TfidfVectorizer's default tokenizer wouldn't know to do."""
    return " ".join(tokenize(text))
