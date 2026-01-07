# AutoJudge: Automated Programming Problem Difficulty Prediction

## Author Details
- **Name:** Chiranshu Sarraf  
- **Project:** AutoJudge – Machine Learning for Problem Classification & Rating  
- **Submission Type:** ACM Project Submission  
- **Institute:** IIT Roorkee  


---


## Project Overview

AutoJudge is a machine learning system that predicts the difficulty of competitive programming problems using only their textual problem statements.

The system takes as input the problem title, description, input format, and output format, and produces two outputs:
1. A difficulty class prediction (*Easy / Medium / Hard*)
2. A numerical difficulty score estimation

The project is implemented using a classical NLP pipeline, where textual problem statements are transformed into numerical features and used to train both classification and regression models. The trained models are exposed through a Streamlit-based web interface for interactive use.


---

## Important Constraints Followed

- The system uses **only textual problem information**, specifically:
  - Problem title
  - Problem description
  - Input format
  - Output format
- **No deep learning models** are used; the entire pipeline is based on **classical machine learning techniques**
- No user statistics, submission data, or platform-specific metadata are used
- All predictions are derived exclusively from the problem statement text

---

## Dataset & Labeling

### Initially Provided Dataset (Exploratory Analysis)

The dataset initially suggested in the problem statement was first explored during preliminary experimentation. However, initial trials showed limited class separation and poor generalization on unseen samples, with predictions heavily biased toward a single difficulty class.

Given these limitations, the dataset was not used for the final model training, and an alternative dataset with richer textual diversity and clearer difficulty signals was selected.


---

### Final Dataset Used

For the final implementation, a **Codeforces-based dataset** was used, sourced from Hugging Face:

🔗 Dataset: https://huggingface.co/datasets/open-r1/codeforces

This dataset contains real-world competitive programming problems with detailed textual statements and associated numerical difficulty ratings. It provides sufficient diversity in problem types and difficulty levels, making it suitable for both classification and regression tasks.

**Fields used in this project:**
- Problem title
- Problem description
- Input format
- Output format
- Numerical difficulty rating

The dataset includes predefined training and testing splits, eliminating the need for manual data partitioning.

---

### Difficulty Label Creation

The dataset provides numerical difficulty ratings but does not include categorical difficulty labels. To enable classification, a new categorical column `difficulty` was created using the following thresholds:

- **Easy:** Rating < 1200  
- **Medium:** 1200 ≤ Rating < 1800  
- **Hard:** Rating ≥ 1800  

These thresholds are consistent with commonly accepted difficulty ranges on competitive programming platforms such as Codeforces.

---

## Approach & Models

### 1. Data Preprocessing

The raw dataset was preprocessed to retain only fields relevant to text-based difficulty estimation. Rows with missing numerical difficulty ratings were removed, as they cannot be used for supervised learning.

All textual fields (title, description, input format, and output format) were retained in their original form, with missing values replaced by empty strings. These structured text fields serve as the base input for downstream feature extraction and model training.

---

### 2. Feature Extraction

Textual features were extracted using **TF-IDF (Term Frequency–Inverse Document Frequency)**. For each problem, the title, description, input format, and output format were concatenated into a single text representation.

The TF-IDF vectorizer was fitted on the training data only, using unigrams and bigrams with stopword removal. The trained vectorizer was saved and reused across both classification and regression models to ensure consistent feature representation.

---

### 3. Model Selection

For difficulty classification, the following models were evaluated:
- Logistic Regression
- Linear Support Vector Machine (SVM)
- Random Forest Classifier

For numerical difficulty score prediction, the following regression models were evaluated:
- Linear Regression
- Gradient Boosting Regressor
- Random Forest Regressor

Models were compared using standard evaluation metrics on a held-out test set. In both tasks, Random Forest models demonstrated better generalization and robustness, particularly in handling non-linear relationships within high-dimensional TF-IDF features. As a result, Random Forest was selected as the final model for both classification and regression.

---

## Evaluation Metrics


### Classification Performance

The following classification results were obtained by evaluating the trained models on a held-out test set.

| Model | Accuracy (%) |
|------|-------------|
| Logistic Regression | 60.92 |
| Linear SVM | 62.66 |
| **Random Forest (Selected)** | **66.38** |

---

### Regression Performance

Regression performance was evaluated on the test set using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

| Model | MAE | RMSE |
|------|-----|------|
| Linear Regression | 694.32 | 869.37 |
| Gradient Boosting | 525.05 | 664.60 |
| **Random Forest (Selected)** | **508.16** | **664.19** |

---
### Key Insights

- Random Forest models exhibited better stability and generalization compared to linear models, particularly on high-dimensional TF-IDF features.
- Confusion matrix analysis indicated improved separation between *Medium* and *Hard* difficulty classes.
- Tree-based models were more effective at capturing non-linear interactions between textual features than linear baselines.

## Implementation Note

The final machine learning pipeline is implemented as modular Python scripts located in the `src/` directory. These scripts handle data preprocessing, feature extraction, model training, and evaluation in a reproducible and execution-ready manner.

Jupyter notebooks were used during earlier stages of development for exploratory analysis, visualization, and model comparison. Selected notebooks are retained in the repository for reference and interpretability, but they are not required to run the project.

---

## Saved Models

The trained models used for inference are saved in the `models/` directory:

- `tfidf_vectorizer.pkl`
- `difficulty_classifier.pkl`
- `rating_regressor.pkl`

These files are loaded directly by the training scripts and the Streamlit web application. Git Large File Storage (Git LFS) is used to manage these binary files efficiently within the repository.

---

## Web Interface

A Streamlit-based web application provides an interface for running inference using the trained models.

The application loads the pre-trained TF-IDF vectorizer, classification model, and regression model from disk and performs predictions without retraining.

**User workflow:**
1. Enter the problem title  
2. Provide the problem description  
3. Provide the input format  
4. Provide the output format  
5. Click **Predict**

**Output:**
- Predicted difficulty class (*Easy / Medium / Hard*)
- Predicted numerical difficulty rating

---

## Mandatory Demo Video
**[Link to Demo Video (2–3 Minutes) – INSERT LINK HERE]**

The demo video includes:
- A brief project overview
- Explanation of the ML pipeline and model choices
- Live walkthrough of the Streamlit web interface with predictions

---

## Steps to Run Locally

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
