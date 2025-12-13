"""data_preparation.py

Utilities for:
- text cleaning and normalization
- simple tokenization helpers
- basic feature engineering

This module focuses on Arabic text used in the abstract-classification project.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Make sure NLTK stopwords are available at runtime
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------
def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel) from the input text."""
    arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    return re.sub(arabic_diacritics, "", text)


def normalize_arabic(text: str) -> str:
    """Normalize common Arabic character variants to a unified form."""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    # Remove non-Arabic characters (keep spaces)
    text = re.sub("[^؀-ۿ ]+", " ", text)
    return text


# Stopwords + Stemmer
ARABIC_STOPWORDS = set(stopwords.words("arabic"))
ARABIC_STEMMER = ISRIStemmer()

# ---------------------------------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------------------------------
def preprocess_text(text) -> str:
    """Full preprocessing pipeline for a single text value."""
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)

    tokens = text.split()
    tokens = [w for w in tokens if w not in ARABIC_STOPWORDS]
    tokens = [ARABIC_STEMMER.stem(w) for w in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------
# TOKENIZATION UTILITIES
# ---------------------------------------------------------
try:
    # Use re2 if available for better performance
    import re2  # type: ignore
except Exception:
    re2 = re


def simple_word_tokenize(text: str):
    """Tokenize into Arabic words, Latin words, and punctuation."""
    return re2.findall(r"\p{Arabic}+|\w+|[^\s\w]", text, flags=re2.VERSION1)


def sentence_tokenize(text: str):
    """Split a text into sentences using Arabic and English punctuation."""
    parts = re.split(r"(?<=[\.\?\!\u061F\u061B])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def paragraph_tokenize(text):
    """Split a text into paragraphs based on blank lines."""
    if not isinstance(text, str):
        return []
    paragraphs = re.split(r"\s*\n\s*\n\s*", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
ARABIC_TATWEEL_CHAR = "ـ"
# Backwards-compatible alias for any existing code
TATWEEL = ARABIC_TATWEEL_CHAR


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all engineered features and token-level structures.

    Expects the input DataFrame to contain a ``clean_text`` column.
    This function adds new columns in-place and also returns the DataFrame.
    """

    # Tokenization-level structures
    df["tokens"] = df["clean_text"].apply(
        lambda t: [tok for tok in simple_word_tokenize(t) if tok.strip()]
    )

    df["words"] = df["tokens"].apply(
        lambda token_list: [tok for tok in token_list if re.search(r"\w", tok)]
    )

    df["sentences"] = df["clean_text"].apply(sentence_tokenize)
    df["paragraphs"] = df["clean_text"].apply(paragraph_tokenize)

    # F1 — Total number of characters
    df["f001_total_chars"] = df["clean_text"].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )

    # F2 — Letters / Characters
    df["f002_letters_over_C"] = df["clean_text"].apply(
        lambda t: len(re2.findall(r"\p{L}", str(t), flags=re2.VERSION1)) / len(str(t))
        if len(str(t)) > 0
        else 0
    )

    # F3 — Digits / Characters
    df["f003_digits_over_C"] = df["clean_text"].apply(
        lambda t: len(re.findall(r"\d", str(t))) / len(str(t))
        if len(str(t)) > 0
        else 0
    )

    # F4 — Spaces / Characters
    df["f004_spaces_over_C"] = df["clean_text"].apply(
        lambda t: str(t).count(" ") / len(str(t)) if len(str(t)) > 0 else 0
    )

    # F5 — Elongations (tatweel + repeated chars)
    df["f005_num_elongations"] = df["clean_text"].apply(
        lambda t: str(t).count(ARABIC_TATWEEL_CHAR)
        + len(re.findall(r"(.)\1{2,}", str(t)))
    )

    return df
