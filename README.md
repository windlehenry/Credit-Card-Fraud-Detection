# Credit Card Fraud Detection

This project tackles the challenge of detecting fraudulent credit card transactions using supervised machine learning. The dataset is highly imbalanced — with frauds making up just 0.17% of all transactions — making this a strong case for applying techniques like SMOTE, regularization, cross-validation, and tree-based modeling.

The work is presented across well-structured Jupyter notebooks to ensure both clarity and depth.

---

## Project Overview

- **Objective**: Predict whether a transaction is fraudulent (binary classification)
- **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Challenges**:
  - Highly imbalanced dataset
  - Features are anonymized (PCA components)
  - Cost of false negatives is high
- **Approach**:
  - Exploratory Data Analysis (EDA)
  - Feature engineering and SMOTE balancing
  - Baseline and regularized logistic regression
  - Random Forest and XGBoost models
  - GridSearchCV-based tuning
  - Final model comparison and interpretation

---

## Repository Structure

```bash
Credit-Card-Fraud-Detection/
│
├── data_exploration.ipynb         # EDA and initial insights
├── feature_engineering.ipynb      # Preprocessing, SMOTE, and train/test split
├── modeling_logistic.ipynb        # Logistic regression + regularization
├── modelling_tree_based.ipynb     # Random Forest & XGBoost models + tuning
├── model_comparison_summary.ipynb # ROC curves, confusion matrices, and final summary
