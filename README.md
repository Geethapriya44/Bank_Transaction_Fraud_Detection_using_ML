# Bank_Transaction_Fraud_Detection_using_ML

# 🏦 Bank Transaction Fraud Detection using Machine Learning

A complete, end-to-end **Data Science / Machine Learning** project that
detects fraudulent bank transactions from behavioural, device, and
session-level signals. Built as a portfolio project to demonstrate the
full ML lifecycle: EDA → preprocessing → handling class imbalance →
model training → evaluation → model persistence.

---

## 📌 Problem Statement

Financial institutions process huge volumes of transactions daily, but
only a small percentage are fraudulent. Manually reviewing every
transaction is impossible, and simple rule-based systems miss evolving
fraud patterns. The goal of this project is to build a **binary
classification model** that predicts whether a transaction is
`Fraud` or `Not Fraud`, using 18 behavioural/device features such as:

- Login attempts, failed transactions in last 30 days
- Device risk score, transaction velocity score, anomaly score
- Session duration, geo-distance from usual location
- Payment channel (ATM / POS / Mobile App / Web Banking)
- Authentication type (OTP / Biometric / 2FA / Password only)
- Suspicious IP flag, international transaction flag, card-present flag

This is a real-world style **imbalanced classification problem**
(only ~12.5% of transactions are fraud), which makes it a great
demonstration of practical data science skills beyond toy datasets.

---

## 🎯 What This Project Demonstrates (for recruiters)

- Exploratory Data Analysis (EDA) on a realistic tabular dataset
- Feature engineering: numeric scaling + one-hot encoding of categoricals
- Handling **class imbalance** using **SMOTE** (Synthetic Minority Oversampling)
- Training & comparing 3 ML models: Logistic Regression, Random Forest, XGBoost
- Proper **imbalanced-classification evaluation**: Precision, Recall,
  F1-score, ROC-AUC, Confusion Matrix, ROC curve comparison (not just accuracy)
- Feature importance analysis to explain *why* a transaction is flagged
- Saving a production-style inference pipeline (`preprocessor.joblib` +
  `best_model.joblib`) using `joblib`

---

## 🧰 Tech Stack / Libraries Used

| Purpose                     | Library                              |
|------------------------------|---------------------------------------|
| Data handling                | `pandas`, `numpy`                     |
| Visualization                | `matplotlib`, `seaborn`               |
| Preprocessing & pipelines    | `scikit-learn` (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`) |
| Class imbalance handling     | `imbalanced-learn` (`SMOTE`)          |
| Models                       | `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier` (xgboost) |
| Evaluation                   | `scikit-learn.metrics`                |
| Model persistence            | `joblib`                              |

---

## 📊 Results

Three models were trained and compared. Metrics below are on a held-out
20% test set (2,000 transactions, stratified split), after training on
SMOTE-balanced data:

| Model               | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC |
|---------------------|----------|--------------------|-----------------|------------|---------|
| Logistic Regression | 0.932    | 0.664              | 0.924           | 0.773      | 0.979   |
| Random Forest       | 0.945    | 0.722              | 0.904           | 0.803      | 0.976   |
| **XGBoost (best)**  | **0.950**| **0.782**          | **0.832**       | **0.806**  | 0.977   |

> ⚠️ Note on interpreting these numbers: in fraud detection, **Recall**
> (catching actual fraud) and **F1-score** matter far more than plain
> accuracy, because the dataset is imbalanced (~87.5% non-fraud). A
> model that predicts "not fraud" every time would already score 87.5%
> accuracy while being useless. That's why this project reports
> Precision/Recall/F1/ROC-AUC and a confusion matrix for every model,
> and selects the best model by **F1-score on the fraud class**, not
> raw accuracy.

Plots generated (see `outputs/` folder after running):
- `class_balance.png` – fraud vs non-fraud distribution
- `correlation_heatmap.png` – feature correlation matrix
- `confusion_matrix_*.png` – per-model confusion matrices
- `roc_curve_comparison.png` – ROC curves of all 3 models
- `feature_importance.png` – top 15 most important features (best model)
- `model_comparison.csv` – metric summary table

---




## ▶️ How to Run


# 1. Clone the repo
git clone https://github.com/<your-username>/fraud-detection-project.git
cd fraud-detection-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python fraud_detection.py

This will print EDA summaries + model metrics to the console and save
all plots + the trained model into an `outputs/` folder.


## 🚀 Possible Future Improvements

- Hyperparameter tuning with `GridSearchCV` / `Optuna`
- Try `LightGBM`, `CatBoost`, or a simple neural network (Keras/PyTorch)
- Deploy the saved model as a REST API (FastAPI/Flask) for real-time scoring
- Add SHAP values for explainable predictions per transaction
- Build a simple Streamlit dashboard to demo live fraud scoring
- Time-based train/test split to simulate real deployment drift

---

## 📄 License

This project is open-sourced for educational/portfolio purposes.
