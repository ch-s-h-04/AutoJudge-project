# AutoJudge: Automated Programming Problem Difficulty Prediction

## 👨‍💻 Author Details
- **Name:** Chiranshu Sarraf  
- **Project:** AutoJudge – Machine Learning for Problem Classification & Rating  
- **Organization:** ACM Project Submission  
- **Institute:** IIT Roorkee  
- **Python Version:** 3.13.1  

---


## 📖 Project Overview
AutoJudge is a machine learning–based system designed to **automatically assess the difficulty of competitive programming problems** using **only textual information**.

Online platforms such as **Codeforces** classify problems as *Easy*, *Medium*, or *Hard* and also assign a numerical difficulty rating. This process is typically driven by editorial complexity, human judgment, and user feedback.

The goal of this project is to **automate this process** by building:

1. **A classification model** to predict difficulty class (*Easy / Medium / Hard*)
2. **A regression model** to predict a numerical difficulty rating

The project strictly follows a **Classical Machine Learning constraint**, avoiding Deep Learning models as per the problem statement.

---

## 📌 Important Constraints Followed
- Only **textual data** is used:
  - Problem title  
  - Problem description  
  - Input format  
  - Output format  
- **No deep learning models** are used  
- Only **classical ML algorithms** are applied  
- Predictions are based solely on problem statements, not user statistics or metadata  

---

## 📊 Dataset & Labeling

### Initially Provided Dataset (Not Used)
The dataset initially suggested in the problem statement was explored first. However, experimentation revealed that:

- Model accuracy was poor
- Predictions were heavily biased toward a single class (mostly *Medium*)
- The dataset did not generalize well on unseen problems

Due to these issues, the dataset was deemed unsuitable for reliable difficulty prediction.

---

### Final Dataset Used
A **Codeforces-based dataset** was selected instead, sourced from Hugging Face:

🔗 **Dataset Link:**  
https://huggingface.co/datasets/open-r1/codeforces

**Dataset characteristics:**
- Real-world competitive programming problems
- Rich and diverse textual descriptions
- Numerical difficulty ratings
- Predefined training and testing splits (no manual split required)

**Fields used:**
- Problem title  
- Problem description  
- Input format  
- Output format  
- Numerical difficulty rating  

---

## 🧩 Difficulty Label Creation
The original dataset provided **only numerical difficulty ratings**.

To enable classification, a new categorical column `difficulty` was created using the following thresholds:

- **Easy:** Rating < 1200  
- **Medium:** 1200 ≤ Rating < 1800  
- **Hard:** Rating ≥ 1800  

These ranges align with commonly accepted difficulty levels on competitive programming platforms such as Codeforces.

---

## ⚙️ Approach & Models

### 1. Data Preprocessing
All textual fields (title, description, input format, output format) were merged into a single text block. The following preprocessing steps were applied:

- Conversion to lowercase  
- Removal of special characters and newline symbols  
- Whitespace normalization  
- Handling missing values by replacing them with empty strings  

This ensured consistent and meaningful input to the feature extractor.

---

### 2. Feature Extraction
**TF-IDF (Term Frequency–Inverse Document Frequency)** was used to convert text into numerical feature vectors.

**Why TF-IDF?**
- Efficient and lightweight
- Well-suited for classical NLP pipelines
- Highlights important keywords (e.g., *dp*, *graph*, *recursion*) correlated with difficulty

---

### 3. Model Selection

**Classification models evaluated:**
- Logistic Regression  
- Linear Support Vector Machine (SVM)  
- Random Forest Classifier  

**Regression models evaluated:**
- Linear Regression  
- Gradient Boosting Regressor  
- Random Forest Regressor  

After comparison, **Random Forest** models were selected for both tasks due to better generalization and robustness.

---

## 📈 Evaluation Metrics

### Classification Performance

| Model | Accuracy (%) |
|------|-------------|
| Logistic Regression | 60.92 |
| Linear SVM | 62.66 |
| **Random Forest (Selected)** | **66.38** |

---

### Regression Performance

| Model | MAE | RMSE |
|------|-----|------|
| Linear Regression | 694.32 | 869.37 |
| Gradient Boosting | 525.05 | 664.60 |
| **Random Forest (Selected)** | **508.16** | **664.19** |

---

### 💡 Key Insights
- **Stability vs. Speed:** Random Forest took longer to train but produced more reliable predictions.
- **Class Separation:** Confusion matrices showed better separation between *Medium* and *Hard* problems.
- **Non-linearity:** Tree-based models captured complex keyword interactions better than linear models.

---

## 📓 Implementation Note (Important Clarification)
The **entire machine learning pipeline** — including data preprocessing, feature extraction, classification, and regression — is implemented using **Jupyter notebooks**.

This design choice was made to:
- Enable step-by-step experimentation and visualization
- Clearly document model comparisons and intermediate results
- Improve interpretability for academic evaluation

The trained models are then used by the Streamlit application (`app.py`) for inference.  
This notebook-driven workflow is standard practice for academic and exploratory ML projects.

---

## 📦 Saved Models & Git LFS
This repository uses **Git Large File Storage (Git LFS)** to store trained model files.

- **Why Git LFS?**  
  Standard Git is not suitable for large binary files (>100 MB). Git LFS ensures efficient storage and cloning.

- **Tracked model files:**
  - `difficulty_classifier.pkl`
  - `rating_regressor.pkl`
  - `tfidf_vectorizer.pkl`

---

## 🌐 Web Interface
A **Streamlit-based web application** provides an interactive interface for predictions.

**User workflow:**
1. Enter problem title  
2. Paste problem description  
3. Paste input format  
4. Paste output format  
5. Click **Predict**

**Output:**
- Predicted difficulty class (*Easy / Medium / Hard*)
- Predicted numerical difficulty rating

---

## 📽️ Mandatory Demo Video
**[Link to Demo Video (2–3 Minutes) – INSERT LINK HERE]**

The demo video includes:
- A brief project overview
- Explanation of the ML pipeline and model choices
- Live walkthrough of the Streamlit web interface with predictions

---

## 🚀 Steps to Run Locally

### 1. Prerequisites
- Python **3.13.1**
- Git LFS installed

### 2. Setup & Execution
```bash
# Install Git LFS (one-time)
git lfs install

# Clone the repository
git clone https://github.com/ch-s-h-04/AutoJudge-project.git
cd AutoJudge-project

# Pull LFS model files
git lfs pull

# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run app.py
