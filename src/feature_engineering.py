"""
Feature Engineering Script
--------------------------
- Cleans text
- Fits TF-IDF vectorizer
- Saves trained vectorizer for reuse
"""

from pathlib import Path
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# PATH SETUP
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# LOAD DATA
df_train = pd.read_csv(DATA_DIR / "train_final.csv")
df_test  = pd.read_csv(DATA_DIR / "test_final.csv")

# TEXT CLEANING 
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

TEXT_COLUMNS = ["title", "description", "input_format", "output_format"]

for col in TEXT_COLUMNS:
    df_train[col] = df_train[col].fillna("")
    df_test[col] = df_test[col].fillna("")

df_train["combined_text"] = (
    df_train["title"] + " " +
    df_train["description"] + " " +
    df_train["input_format"] + " " +
    df_train["output_format"]
)

df_test["combined_text"] = (
    df_test["title"] + " " +
    df_test["description"] + " " +
    df_test["input_format"] + " " +
    df_test["output_format"]
)

df_train["clean_text"] = df_train["combined_text"].apply(clean_text)
df_test["clean_text"]  = df_test["combined_text"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

vectorizer.fit(df_train["clean_text"])

# SAVE
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

print("Feature engineering complete.")
print("Saved: models/tfidf_vectorizer.pkl")
