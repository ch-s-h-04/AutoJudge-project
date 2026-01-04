# AutoJudge: Predicting Programming Problem Difficulty

## Project Overview
AutoJudge is a machine learning–based system that automatically predicts the difficulty of programming problems using **only textual information**.

Online competitive programming platforms (such as Codeforces) classify problems into *Easy*, *Medium*, or *Hard* and also assign a numerical difficulty rating. This process is traditionally driven by human judgment, editorial complexity, and user feedback.

The goal of this project is to **automate difficulty prediction** by building:
- A **classification model** to predict difficulty class (*Easy / Medium / Hard*)
- A **regression model** to predict a numerical difficulty score

### Important Constraints Followed
- Only **textual data** is used (problem title, description, input format, output format)
- **No deep learning models** are used, as explicitly stated in the project requirements
- Only **classical machine learning models** are applied

---

## Dataset Used

### Initially Provided Dataset (Not Used)
A reference dataset was initially provided as part of the problem statement. However, after experimentation, it was observed that:
- The trained models showed **poor accuracy**
- Predictions were **heavily biased toward a single class** (mostly *Medium*)
- The dataset failed to generalize well on unseen problems

Due to these limitations, the provided dataset was **not suitable** for reliable difficulty prediction.

### Final Dataset Used
A **Codeforces-based dataset** was used instead, containing real competitive programming problems with richer and more diverse textual content.

The dataset includes:
- Problem title
- Problem description
- Input format
- Output format
- Numerical difficulty rating

This dataset resulted in **significantly improved performance** for both classification and regression tasks.

Dataset source:  
https://huggingface.co/datasets/open-r1/codeforces

The dataset already contained **separate training and testing splits**, so no additional train–test splitting was performed.

---

## Difficulty Label Creation
The original dataset contained only a **numerical problem rating**.

To enable difficulty classification, a new categorical column named **`difficulty`** was created using the following thresholds:

- **Easy**: Rating < 1200  
- **Medium**: 1200 ≤ Rating < 1800  
- **Hard**: Rating ≥ 1800  

These ranges were chosen after analyzing the rating distribution and align with commonly accepted difficulty levels on competitive programming platforms.

---

## Approach and Models Used

### 1. Data Preprocessing
- Selected only relevant columns required for prediction
- Removed unnecessary metadata
- Handled missing values by replacing them with empty strings
- Combined textual fields into a single text representation
- Cleaned text (lowercasing, removing special characters, whitespace normalization)

### 2. Feature Extraction
- Used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical feature vectors
- TF-IDF was chosen due to its efficiency and suitability for classical NLP models

### 3. Classification Models Evaluated
The following models were trained and compared:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (Linear SVM)

**Final Selected Model:** Random Forest Classifier  
Chosen due to higher accuracy and better class-wise balance.

### 4. Regression Models Evaluated
The following models were trained:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Final Selected Model:** Random Forest Regressor  
Chosen due to lowest MAE and RMSE.

---

## Evaluation Metrics

### Classification
- Accuracy
- Confusion Matrix

### Regression
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Final Regression Results

| Model | MAE | RMSE |
|------|-----|------|
| Linear Regression | ~694 | ~869 |
| Random Forest Regressor | **~508** | **~664** |
| Gradient Boosting Regressor | ~525 | ~665 |

---

## Web Interface Explanation
A **Streamlit-based web application** is provided to demonstrate the working of the models.

The interface allows the user to:
1. Enter the problem title
2. Paste the problem description
3. Paste the input format
4. Paste the output format
5. Click the **Predict** button

The application then:
- Processes the input text
- Applies the trained TF-IDF vectorizer
- Predicts:
  - Difficulty class (*Easy / Medium / Hard*)
  - Numerical difficulty score

---

## Steps to Run the Project Locally

```bash
# 1. Clone the repository
git clone https://github.com/ch-s-h-04/AutoJudge.git

# 2. Move into the project directory
cd AutoJudge

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the web application
streamlit run app.py

