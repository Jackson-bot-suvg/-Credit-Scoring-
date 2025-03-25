# 💳 Credit Scoring Prediction with Machine Learning

This repository presents a comprehensive machine learning project focused on predicting customer credit scores based on lifestyle and financial behaviors. The project explores various classification models to optimize credit scoring accuracy for financial institutions.

---

## 🧠 Project Overview

**Research Question**  
> How do users’ lifestyle and financial behaviors impact banks’ evaluation of customer credit scores?

### 🎯 Objectives

- Improve credit risk assessment using ML models.
- Address class imbalance, feature bias, and encoding issues.
- Compare performance of: **Random Forest**, **SVM**, and **KNN**.

---

## 📁 Dataset

- **Samples**: ~46,000 records  
- **Features**: 28 (age, income, spending, etc.)  
- **Target**: `Credit_Score` (multi-class: Good, Standard, Poor)  
- **Preprocessing**:  
  - SMOTE / downsampling for class imbalance  
  - One-hot / label encoding  
  - Feature scaling (StandardScaler, MinMaxScaler)

---

## 🧪 Models

| Model         | Accuracy | Key Notes                            |
|---------------|----------|---------------------------------------|
| Random Forest | **78%**  | Best performance, robust to noise     |
| SVM           | 67%      | Handles nonlinear but slower training |
| KNN           | 64.8%    | Simple, affected by high dimensions   |

### 📊 Metrics Used

- Accuracy  
- F1 Score (per class)  
- AUC-ROC (macro average)

---

## 🔧 Optimization

Used `RandomizedSearchCV` & `GridSearchCV` for hyperparameter tuning:

- Random Forest AUC improved to **0.91**
- Best F1-scores across all classes

---

## 💻 Tech Stack

- Python, Jupyter Notebook  
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 🏃 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook group.ipynb
