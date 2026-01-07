"""
Data Preparation Script
-----------------------
- Loads Codeforces dataset
- Cleans missing values
- Creates combined text field
- Converts rating to difficulty class
- Saves final train and test CSV files
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset

# PATH SETUP 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# LOAD DATASET 
ds = load_dataset("open-r1/codeforces", "default")

df_train = ds["train"].to_pandas()
df_test = ds["test"].to_pandas()


# CLEANING 
# Drop rows without rating (cannot be used for supervised learning)
df_train = df_train.dropna(subset=["rating"])
df_test = df_test.dropna(subset=["rating"])

# Text columns used for NLP
TEXT_COLUMNS = ["title", "description", "input_format", "output_format"]

# Filling missing text fields with empty strings
for col in TEXT_COLUMNS:
    df_train[col] = df_train[col].fillna("")
    df_test[col] = df_test[col].fillna("")

# FEATURE CREATION 
# Combine all text fields
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

# LABEL CREATION 
def rating_to_difficulty(rating: float) -> str:
    if rating < 1200:
        return "Easy"
    elif rating < 1800:
        return "Medium"
    else:
        return "Hard"

df_train["difficulty"] = df_train["rating"].apply(rating_to_difficulty)
df_test["difficulty"] = df_test["rating"].apply(rating_to_difficulty)

# FINAL DATA 
FINAL_COLUMNS = [
    "title",
    "description",
    "input_format",
    "output_format",
    "rating",
    "difficulty"
]

df_train_final = df_train[FINAL_COLUMNS]
df_test_final = df_test[FINAL_COLUMNS]

# SAVE FILES 
df_train_final.to_csv(DATA_DIR / "train_final.csv", index=False)
df_test_final.to_csv(DATA_DIR / "test_final.csv", index=False)

print("Data preparation complete.")
print("Saved files:")
print("- data/train_final.csv")
print("- data/test_final.csv")
