"""
Regression Model Training
-------------------------
- Loads TF-IDF features
- Trains regression models
- Evaluates MAE and RMSE
- Saves final regressor
"""

from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

y_train = df_train["rating"]
y_test  = df_test["rating"]

# MODELS
results = []

lin = LinearRegression()
lin.fit(X_train, y_train)
lin_pred = lin.predict(X_test)
results.append((
    "Linear Regression",
    mean_absolute_error(y_test, lin_pred),
    np.sqrt(mean_squared_error(y_test, lin_pred))
))

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results.append((
    "Random Forest",
    mean_absolute_error(y_test, rf_pred),
    np.sqrt(mean_squared_error(y_test, rf_pred))
))

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
results.append((
    "Gradient Boosting",
    mean_absolute_error(y_test, gb_pred),
    np.sqrt(mean_squared_error(y_test, gb_pred))
))

# SELECT BEST 
final_regressor = rf

# SAVE
joblib.dump(final_regressor, MODEL_DIR / "rating_regressor.pkl")

print("Regression training complete.")
for name, mae, rmse in results:
    print(f"{name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

print("Saved: models/rating_regressor.pkl")
