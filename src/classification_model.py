"""
Classification Model Training
-----------------------------
- Loads TF-IDF vectorizer
- Trains multiple classifiers
- Evaluates performance
- Saves final classifier
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# PATH SETUP
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# LOAD DATA 
df_train = pd.read_csv(DATA_DIR / "train_final.csv")
df_test  = pd.read_csv(DATA_DIR / "test_final.csv")

# LOAD VECTORIZER 
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

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

X_train = vectorizer.transform(df_train["combined_text"])
X_test  = vectorizer.transform(df_test["combined_text"])

y_train = df_train["difficulty"]
y_test  = df_test["difficulty"]

# MODELS
results = []

lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)
results.append(("Logistic Regression", accuracy_score(y_test, lr.predict(X_test))))

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)
results.append(("Random Forest", accuracy_score(y_test, rf.predict(X_test))))

svm = LinearSVC(class_weight="balanced")
svm.fit(X_train, y_train)
results.append(("SVM (Linear)", accuracy_score(y_test, svm.predict(X_test))))

# SELECT BEST 
final_classifier = rf

# SAVE 
joblib.dump(final_classifier, MODEL_DIR / "difficulty_classifier.pkl")

print("Classification training complete.")
for name, acc in results:
    print(f"{name}: Accuracy = {acc:.4f}")

print("Saved: models/difficulty_classifier.pkl")
