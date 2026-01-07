import streamlit as st
import joblib
from pathlib import Path
import re

# ================= PATH SETUP =================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ================= LOAD MODELS ================
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
classifier = joblib.load(MODEL_DIR / "difficulty_classifier.pkl")
regressor = joblib.load(MODEL_DIR / "rating_regressor.pkl")

# ================= TEXT CLEANING ==============
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ================= UI =========================
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("AutoJudge")
st.subheader("Automated Programming Problem Difficulty Prediction")

st.write(
    "Paste a programming problem description below to predict "
    "its difficulty level and estimated rating."
)

user_input = st.text_area(
    "Problem Description",
    height=250,
    placeholder="Enter title, description, input format, output format..."
)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a problem description.")
    else:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])

        difficulty = classifier.predict(features)[0]
        rating = int(regressor.predict(features)[0])

        st.success("Prediction Complete")
        st.write(f"**Predicted Difficulty:** {difficulty}")
        st.write(f"**Predicted Rating:** {rating}")
