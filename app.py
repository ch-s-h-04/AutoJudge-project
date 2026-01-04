import streamlit as st
import pandas as pd
import joblib
import re

# -----------------------------
# Load saved models
# -----------------------------
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
clf = joblib.load("models/difficulty_classifier.pkl")
reg = joblib.load("models/rating_regressor.pkl")

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.markdown(
    """
    Paste the **problem statement details** below and click **Predict**.
    """
)

# Input fields
title = st.text_input("Problem Title")
description = st.text_area("Problem Description", height=200)
input_format = st.text_area("Input Description", height=120)
output_format = st.text_area("Output Description", height=120)

# Predict button
if st.button("Predict Difficulty"):
    if title.strip() == "" or description.strip() == "":
        st.warning("Please provide at least Title and Description.")
    else:
        # Combine text
        combined_text = f"{title} {description} {input_format} {output_format}"
        cleaned = clean_text(combined_text)

        # Vectorize
        X = vectorizer.transform([cleaned])

        # Predictions
        difficulty_pred = clf.predict(X)[0]
        rating_pred = reg.predict(X)[0]

        # Output
        st.success("Prediction Completed")
        st.markdown(f"### 🏷️ Difficulty Class: **{difficulty_pred}**")
        st.markdown(f"### 📊 Difficulty Score: **{int(rating_pred)}**")
