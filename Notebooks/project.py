# -*- coding: utf-8 -*-
"""
Main script for the Arabic abstract classification project.

This script trains and evaluates several models
(Logistic Regression, SVM, Random Forest, XGBoost, FFNN)
to detect AI-generated Arabic abstracts.
"""


# !pip install datasets
# !pip install python-dotenv

from dotenv import load_dotenv
import os
import sys

# Base directories assuming the following layout:
# - notebooks/project.py
# - scripts/   (helper Python modules)
# - models/    (saved model files: .pkl, .h5)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure the scripts directory is on sys.path so imports work from notebooks/
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from huggingface_hub import login

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
print(dataset)

# Inspect column names and data types for one split (e.g., 'by_polishing')
print("\nFeatures in 'by_polishing':")
print(dataset['by_polishing'].features)

# Check dataset info (shape, structure, statistics)
print("\nDataset info for 'by_polishing':")
print(dataset['by_polishing'])

# Choose one split (e.g., by_polishing)
split1 = dataset["by_polishing"]

# Count human-written abstracts
num_human = len(split1["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split1["allam_generated_abstract"]) \
       + len(split1["jais_generated_abstract"]) \
       + len(split1["llama_generated_abstract"]) \
       + len(split1["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))

import pandas as pd
# Convert to pandas for easier checks
df = pd.DataFrame(split1)

# 1. Missing values
print("Missing values per column:")
print(df.isnull().sum())
print("_________________________________________")

# 2. Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Also check duplicates in each column separately
for col in df.columns:
    print(f"Duplicates in column {col}: {df[col].duplicated().sum()}")
print("_________________________________________")


# 3. Inconsistencies: empty strings or only spaces
for col in df.columns:
    empty_count = df[col].apply(lambda x: str(x).strip() == "").sum()
    print(f"Empty/blank values in column {col}: {empty_count}")

split2 = dataset["from_title"]

# Count human-written abstracts
num_human = len(split2["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split2["allam_generated_abstract"]) \
       + len(split2["jais_generated_abstract"]) \
       + len(split2["llama_generated_abstract"]) \
       + len(split2["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))

import pandas as pd
# Convert to pandas for easier checks
df = pd.DataFrame(split2)

# 1. Missing values
print("Missing values per column:")
print(df.isnull().sum())
print("_________________________________________")

# 2. Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Also check duplicates in each column separately
for col in df.columns:
    print(f"Duplicates in column {col}: {df[col].duplicated().sum()}")
print("_________________________________________")


# 3. Inconsistencies: empty strings or only spaces
for col in df.columns:
    empty_count = df[col].apply(lambda x: str(x).strip() == "").sum()
    print(f"Empty/blank values in column {col}: {empty_count}")

split3 = dataset["from_title_and_content"]

# Count human-written abstracts
num_human = len(split3["original_abstract"])

# Count AI-generated abstracts (4 per row)
num_ai = len(split3["allam_generated_abstract"]) \
       + len(split3["jais_generated_abstract"]) \
       + len(split3["llama_generated_abstract"]) \
       + len(split3["openai_generated_abstract"])

print("Number of human abstracts:", num_human)
print("Number of AI-generated abstracts:", num_ai)

# Distribution ratio
total = num_human + num_ai
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))

import pandas as pd
# Convert to pandas for easier checks
df = pd.DataFrame(split3)

# 1. Missing values
print("Missing values per column:")
print(df.isnull().sum())
print("_________________________________________")

# 2. Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Also check duplicates in each column separately
for col in df.columns:
    print(f"Duplicates in column {col}: {df[col].duplicated().sum()}")
print("_________________________________________")


# 3. Inconsistencies: empty strings or only spaces
for col in df.columns:
    empty_count = df[col].apply(lambda x: str(x).strip() == "").sum()
    print(f"Empty/blank values in column {col}: {empty_count}")

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from datasets import load_dataset

# Download required NLTK resources
nltk.download('stopwords')

# Check columns
print(df.head())

#Define Arabic text cleaning functions
# Remove tashkeel (diacritics)
def remove_diacritics(text):
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)

# Normalize Arabic text
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[^؀-ۿ ]+", " ", text)  # remove non-Arabic chars
    return text

# Initialize stopwords and stemmer
arabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()

# Full preprocessing pipeline
def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# Apply preprocessing
text_columns = [
    'original_abstract',
    'allam_generated_abstract',
    'jais_generated_abstract',
    'llama_generated_abstract',
    'openai_generated_abstract'
]
for col in text_columns:
    clean_col = col + "_clean"
    df[clean_col] = df[col].apply(preprocess_text)
print(" Preprocessing complete! Here are the new columns:")
print(df.columns)
df.head(2)

# Task 2.2: Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import numpy as np

# Combine AI abstracts into one column
ai_texts = pd.concat([
    df['allam_generated_abstract_clean'],
    df['jais_generated_abstract_clean'],
    df['llama_generated_abstract_clean'],
    df['openai_generated_abstract_clean']
], axis=0).dropna().tolist()

human_texts = df['original_abstract_clean'].dropna().tolist()

# --- Statistical Analysis ---
def text_stats(texts):
    words = [w for txt in texts for w in txt.split()]
    avg_word_len = np.mean([len(w) for w in words])
    avg_sent_len = np.mean([len(txt.split()) for txt in texts])
    vocab = set(words)
    ttr = len(vocab) / len(words)
    return avg_word_len, avg_sent_len, ttr

stats_human = text_stats(human_texts)
stats_ai = text_stats(ai_texts)

print("\n Statistical Summary:")
print(f"Human-written: Avg word len={stats_human[0]:.2f}, Avg sent len={stats_human[1]:.2f}, TTR={stats_human[2]:.3f}")
print(f"AI-generated : Avg word len={stats_ai[0]:.2f}, Avg sent len={stats_ai[1]:.2f}, TTR={stats_ai[2]:.3f}")

# --- N-gram Frequency ---
def plot_top_ngrams(texts, n=2, top_k=15):
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(ngram_range=(n, n))
    bag = vec.fit_transform(texts)
    sum_words = bag.sum(axis=0)
    freqs = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]
    words, counts = zip(*freqs)
    plt.figure(figsize=(10,4))
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f"Top {top_k} {n}-grams – {n}-grams for {'Human' if texts==human_texts else 'AI'} abstracts")
    plt.show()

print("\n🔤 Top Bigrams for Human-written abstracts:")
plot_top_ngrams(human_texts, n=2)

print("\n🔤 Top Bigrams for AI-generated abstracts:")
plot_top_ngrams(ai_texts, n=2)

import matplotlib.pyplot as plt

df["human_length"] = df["original_abstract"].apply(lambda x: len(x.split()))
df["ai_length"] = df["openai_generated_abstract"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,5))
plt.hist(df["human_length"], bins=30, alpha=0.6, label="Human-written", color='blue')
plt.hist(df["ai_length"], bins=30, alpha=0.6, label="AI-generated", color='orange')
plt.xlabel("Sentence Length (words)")
plt.ylabel("Frequency")
plt.title("Sentence Length Distribution")
plt.legend()
plt.show()

def type_token_ratio(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

df["human_ttr"] = df["original_abstract"].apply(type_token_ratio)
df["ai_ttr"] = df["openai_generated_abstract"].apply(type_token_ratio)

plt.figure(figsize=(6,5))
plt.boxplot([df["human_ttr"], df["ai_ttr"]], labels=["Human", "AI"])
plt.title("Vocabulary Richness (Type–Token Ratio)")
plt.ylabel("TTR Score")
plt.show()

from collections import Counter
import pandas as pd

human_words = " ".join(df["original_abstract"]).split()
ai_words = " ".join(df["openai_generated_abstract"]).split()

human_freq = Counter(human_words)
ai_freq = Counter(ai_words)

common_words = set(list(human_freq.keys())[:100]) & set(list(ai_freq.keys())[:100])

data = []
for w in common_words:
    data.append((w, human_freq[w], ai_freq[w]))

freq_df = pd.DataFrame(data, columns=["word", "human", "ai"]).sort_values("human", ascending=False)[:15]

freq_df.plot(x="word", kind="bar", figsize=(10,5), title="Top Words: Human vs AI", rot=45)
plt.ylabel("Frequency")
plt.show()

# Combine all splits into one df_human
splits = ["by_polishing", "from_title", "from_title_and_content"]

df_human = pd.concat([dataset[s].to_pandas() for s in splits], ignore_index=True)

dfs = []

for split_name in ["by_polishing", "from_title", "from_title_and_content"]:
    split_df = dataset[split_name].to_pandas().copy()
    split_df["source_split"] = split_name   # <-- Create column manually
    dfs.append(split_df)

df_human = pd.concat(dfs, ignore_index=True)

ai_rows = []

for _, row in df_human.iterrows():
    ai_models = [
        ("allam", row["allam_generated_abstract"]),
        ("jais", row["jais_generated_abstract"]),
        ("llama", row["llama_generated_abstract"]),
        ("openai", row["openai_generated_abstract"]),
    ]

    for model_name, text in ai_models:
        ai_rows.append({
            "abstract_text": text,
            "source_split": row["source_split"],     # now this exists
            "generated_by": model_name,
            "label": 0  # AI
        })

# Convert to dataframe
df_ai = pd.DataFrame(ai_rows)

# Create human dataframe
df_h = pd.DataFrame({
    "abstract_text": df_human["original_abstract"],
    "source_split": df_human["source_split"],
    "generated_by": "human",
    "label": 1
})

# Final unified dataset
df = pd.concat([df_h, df_ai], ignore_index=True)

print("Final unified dataset shape:", df.shape)
df.head(10)

print(df.columns)

# Inspect column names and data types for one split (e.g., 'by_polishing')
print("\nFeatures in 'by_polishing':")
print(dataset['by_polishing'].features)

# Check dataset info (shape, structure, statistics)
print("\nDataset info for 'by_polishing':")
print(dataset['by_polishing'])

num_human = df[df["label"] == 1].shape[0]
num_ai = df[df["label"] == 0].shape[0]

total = num_human + num_ai

print("\n===== Target Variable Distribution =====")
print("Human-written abstracts:", num_human)
print("AI-generated abstracts:", num_ai)
print("Human %:", round(num_human / total * 100, 2))
print("AI %:", round(num_ai / total * 100, 2))

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Duplicate Rows =====")
print("Total duplicate rows:", df.duplicated().sum())

print("\n===== Duplicate values per column =====")
for col in df.columns:
    print(f"{col}: {df[col].duplicated().sum()}")

print("\n===== Empty / Blank Values =====")
for col in df.columns:
    empty_count = df[col].apply(lambda x: str(x).strip() == "").sum()
    print(f"{col}: {empty_count}")

# Check columns
print(df.head())

#Define Arabic text cleaning functions
# Remove tashkeel (diacritics)
def remove_diacritics(text):
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)

# Normalize Arabic text
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[^؀-ۿ ]+", " ", text)  # remove non-Arabic chars
    return text

# Initialize stopwords and stemmer
arabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()

# Full preprocessing pipeline
def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# Apply preprocessing
text_columns = [
    'abstract_text',
    'source_split',
    'generated_by',
    'label',
]

# Apply preprocessing on the unified abstract text column
df["abstract_text_clean"] = df["abstract_text"].apply(preprocess_text)

print("Preprocessing complete! Here are the new columns:")
print(df.columns)

df.head(2)

"""**FEATURES:**"""

#important library
import re
import math
import numpy as np
import pandas as pd
import unicodedata
from collections import Counter
from datasets import load_dataset
import regex as re2

#Helper functions


def simple_word_tokenize(text):
    """
    Tokenize text into words / symbols with Arabic support.
    """
    return re2.findall(r"\p{Arabic}+|\w+|[^\s\w]", text, flags=re2.VERSION1)

def sentence_tokenize(text):
    """
    Split text into sentences using Arabic/English punctuation.
    """
    if not isinstance(text, str):
        return []
    parts = re.split(r'(?<=[\.\?\!\u061F\u061B])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def paragraph_tokenize(text):
    """
    Split text into paragraphs based on double newlines.
    """
    if not isinstance(text, str):
        return []
    paragraphs = re.split(r'\s*\n\s*\n\s*|\s*\r\n\s*\r\n\s*', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

# Column names to use
original_text_columns = "abstract_text"
clean_text_columns = "abstract_text_clean"

# =============================
# 1. Tokens (use clean text)
# =============================
df["tokens"] = df[clean_text_columns].apply(
    lambda t: [tok for tok in simple_word_tokenize(t) if tok.strip()] if isinstance(t, str) else []
)

# =============================
# 2. Words (use clean tokens only)
# =============================
df["words"] = df["tokens"].apply(
    lambda toks: [tok for tok in toks if re.search(r'\w', tok)]
)

# =============================
# 3. Sentences (use original_text_columns for accurate sentence boundary detection)
# =============================
df["sentences"] = df[original_text_columns].apply(
    lambda t: sentence_tokenize(t)
)

# =============================
# 4. Paragraphs (use original_text_columns to preserve original structural breaks)
# =============================
df["paragraphs"] = df[original_text_columns].apply(
    lambda t: paragraph_tokenize(t)
)

print("Feature engineering completed! Columns now:")
print(df.columns)
df.head(2)

# Column names to use
original_text_columns = "abstract_text"
clean_text_columns = "abstract_text_clean"

# Feature 12: Number of short words / N
feature_name = f'{clean_text_columns}_f012_num_short_words_over_N'

def _short_word_ratio(words, max_length=2):
    """Calculates the ratio of words shorter than or equal to max_length to the total number of words."""
    total_words = len(words)
    if total_words == 0:
        return 0.0

    # Count words where length is less than or equal to the defined maximum
    short_words_count = sum(1 for w in words if len(w) <= max_length)

    return float(short_words_count) / total_words

df[feature_name] = df["words"].apply(_short_word_ratio)

df.head(5)

# 35. Total number of paragraphs (P)
df['f035_Total_number_of_paragraphs_(P)'] = df["paragraphs"].apply(len)

df.head(5)

# 58. Number of words found in the 100 most frequent positions
# (approximation using global frequency over the clean words column)

from collections import Counter

# حساب الترددات على عمود الكلمات الموحد "words"
all_words = []
for words_list in df["words"]:
    if isinstance(words_list, list):
        all_words.extend(words_list)

counter = Counter(all_words)
top100_words = set([w for w, _ in counter.most_common(100)])  # Top 100 كلمات

# لحساب Feature 58
# ------------------------------
def count_top100_embedding_words(tokens, top_words_set=top100_words):
    if not isinstance(tokens, list):
        return 0
    return sum(1 for tok in tokens if tok in top_words_set)

# تطبيق Feature 58 على عمود الكلمات الموحد
feature_58_name = f'{clean_text_columns}_f058_num_top100_embedding_words'
df[feature_58_name] = df["words"].apply(count_top100_embedding_words)

# لفحص نتائج ميزة 58
df[["words", feature_58_name]].head()

# !pip install transformers -q

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import math

# Feature 81: LogRank (placeholder) - requires LLM token ranks

# تحميل نموذج GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# استخدام GPU إذا متاح
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# دالة لحساب LogRank لكل نص
def compute_logrank(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # loss هو متوسط -log(prob) لكل التوكن
        log_rank_score = outputs.loss.item()  # loss في LM هو متوسط -log(prob)

    return log_rank_score

# تطبيق Feature 81 على العمود النظيف الموحد
feature_81_name = f"{clean_text_columns}_f081_logrank"
df[feature_81_name] = df[clean_text_columns].apply(
    lambda t: compute_logrank(t) if pd.notna(t) else 0.0
)

# عرض النتائج الأولى
df[[feature_81_name]].head()

# 104. Root Pattern Frequency
# يقيس استخدام أنماط الجذور العربية (مثل "فعل"، "مفعول") بطريقة مبسطة.

root_patterns = ["فعل", "افعال", "مفعول", "فاعلة"]  # مثال لأنماط جذور

def root_pattern_frequency(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    count = sum(text.count(pattern) for pattern in root_patterns)
    total_words = len(simple_word_tokenize(text))
    return count / total_words if total_words > 0 else 0.0

feature_104_name = f"{original_text_columns}_f104_root_pattern_freq"
df[feature_104_name] = df[original_text_columns].apply(root_pattern_frequency)

"""#Split the data"""

from sklearn.model_selection import train_test_split

# First split: Train 70%, Temp 30%
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)

# Second split: Temp 30% → 15% Validation, 15% Test
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

# Show sizes
print("TOTAL:", len(df))
print("TRAIN:", len(train_df))
print("VAL:", len(val_df))
print("TEST:", len(test_df))

"""#TF-IDF Features from Cleaned Text

"""

#apply with abstract_text_clean only
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer for Arabic text
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,   # limit vocabulary
    ngram_range=(1,2),   # unigrams + bigrams
    analyzer='word'
)

# Fit only on training set
tfidf_vectorizer.fit(train_df["abstract_text_clean"])

# Transform train/validation/test sets
X_train_tfidf = tfidf_vectorizer.transform(train_df["abstract_text_clean"])
X_val_tfidf   = tfidf_vectorizer.transform(val_df["abstract_text_clean"])
X_test_tfidf  = tfidf_vectorizer.transform(test_df["abstract_text_clean"])

print("TF-IDF shapes:")
print("Train:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)
print("Test:", X_test_tfidf.shape)

"""##Define X and y"""

from scipy.sparse import hstack

#Select numeric features (The generated feature engineering exclude label and text)
EXCLUDED_COLS = ['label', 'abstract_text', 'abstract_text_clean',
                 'tokens', 'words', 'sentences', 'paragraphs', 'abstract_text_pos_tags']
# Select columns that are numeric AND not in the exclusion list>>feature engineering columns
numeric_cols = [
    col for col in train_df.select_dtypes(include=np.number).columns.tolist()
    if col not in EXCLUDED_COLS
]
# Convert the numeric features DataFrames to NumPy arrays (dense matrices)
# We must use the values/to_numpy() method to extract the array for sparse matrix stacking.
X_train_num_array = train_df[numeric_cols].values
X_val_num_array   = val_df[numeric_cols].values
X_test_num_array  = test_df[numeric_cols].values


# Target variable
y_train = train_df["label"]
y_val   = val_df["label"]
y_test  = test_df["label"]

# Features: TF-IDF and the creating feature engineering
X_train = hstack([X_train_tfidf, X_train_num_array]).tocsr()
X_val= hstack([X_val_tfidf, X_val_num_array]).tocsr()
X_test= hstack([X_test_tfidf, X_test_num_array]).tocsr()

print("X and y are ready for ML models.")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

"""#Build Machine learning Models

##1-Baseline Model (Naïve Bayes & Logistic Regression)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train on training set
lr_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred = lr_model.predict(X_val)

# Evaluate on validation set
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))

# Evaluation
# Predict on test set
y_test_pred = lr_model.predict(X_test)

# Evaluate on test set
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))

# Optional: confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

"""##2-Traditional Machine Learning Models ( Support Vector Machine (SVM), Random Forest, XGBoost)using the validation set"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Dictionary to store models and results
models = {}

# -----------------------
#Support Vector Machine (SVM)
# -----------------------
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

y_val_pred_svm = svm_model.predict(X_val)
print("SVM Validation Accuracy:", accuracy_score(y_val, y_val_pred_svm))
print(classification_report(y_val, y_val_pred_svm))

models['SVM'] = svm_model

# -----------------------
#Random Forest
# -----------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_val_pred_rf = rf_model.predict(X_val)
print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

models['RandomForest'] = rf_model

# -----------------------
#XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

y_val_pred_xgb = xgb_model.predict(X_val)
print("XGBoost Validation Accuracy:", accuracy_score(y_val, y_val_pred_xgb))
print(classification_report(y_val, y_val_pred_xgb))

models['XGBoost'] = xgb_model

#Evaluation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# List of models to evaluate
model_names = ['SVM', 'RandomForest', 'XGBoost']

for name in model_names:
    model = models[name]

    # Predict on test set
    y_test_pred = model.predict(X_test)

    print(f"\n===== {name} Test Evaluation =====")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

"""#Build Deep learning Models

Build neural network classifier ( simple Feedforward Network) on top of the extracted BERT embeddings or fine-tune the pre-trained BERT model for the classification task.

#Step1: Extract BERT Embeddings (Sentence-level)
"""

# !pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Load Arabic-compatible BERT model
bert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Convert clean_text into embeddings
X_train_emb = bert_model.encode(train_df["clean_text"].tolist(), convert_to_numpy=True)
X_val_emb   = bert_model.encode(val_df["clean_text"].tolist(), convert_to_numpy=True)
X_test_emb  = bert_model.encode(test_df["clean_text"].tolist(), convert_to_numpy=True)

y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

print("Train embedding shape:", X_train_emb.shape)

"""##Step 2: Build a Feedforward Neural Network"""

#import tensorflow as tf
from tensorflow.keras import layers, models

# Basic feedforward classifier on embeddings
ffnn_model = models.Sequential([
    layers.Input(shape=(X_train_emb.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")   # binary classification
])

ffnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ffnn_model.summary()

"""##Step3: Train the Model"""

history = ffnn_model.fit(
    X_train_emb, y_train,
    validation_data=(X_val_emb, y_val),
    epochs=10,
    batch_size=32
)

"""##Step 4: Evaluate on Test Set

"""

from sklearn.metrics import accuracy_score, classification_report

# Predict
y_test_pred = (ffnn_model.predict(X_test_emb) > 0.5).astype(int)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

"""##Save Models"""

import os
import joblib
from tensorflow.keras.models import Model as KerasModel

def save_all_models(models_dict, save_dir=None):
    """
    Saves all ML/DL models to disk based on their type.

    Parameters:
    -----------
    models_dict : dict
        Example:
            {
                "logistic_regression": log_reg_model,
                "svm": svm_model,
                "random_forest": rf_model,
                "xgboost": xgb_model,
                "ffnn": ffnn_model
            }

    save_dir : str
        Directory where models will be saved.
    """
    if save_dir is None:
        save_dir = MODELS_DIR

    # Create save folder
    os.makedirs(save_dir, exist_ok=True)

    for model_name, model_obj in models_dict.items():

        # Case 1 — Keras deep learning model
        if isinstance(model_obj, KerasModel):
            file_path = os.path.join(save_dir, f"{model_name}.h5")
            model_obj.save(file_path)
            print(f"[Saved] Keras model → {file_path}")

        # Case 2 — All pickle-compatible models (Sklearn, XGBoost)
        else:
            file_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model_obj, file_path)
            print(f"[Saved] Pickle model → {file_path}")

    print("\nAll models saved successfully!")

import os
os.makedirs(MODELS_DIR, exist_ok=True)
models_dict = {
    "lr_model": lr_model,
    "svm": svm_model,
    "random_forest": rf_model,
    "xgboost": xgb_model,
    "ffnn": ffnn_model
}

save_all_models(models_dict)