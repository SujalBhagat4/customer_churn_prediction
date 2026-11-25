# customer_churn_prediction
End-to-end Machine Learning project in Python using Pandas and Random Forest to predict telecom customer churn (~80% accuracy).

# Customer Churn Prediction System

An end-to-end supervised Machine Learning classification project built in Python to identify telecom customers at high risk of canceling their subscription ("churning") based on their billing history, tenure, and service usage.

---

## Project Overview
Retaining existing customers is significantly more cost-effective than acquiring new ones. By predicting which active subscribers are likely to leave, telecom companies can proactively offer targeted discounts, loyalty rewards, or customer support interventions before cancellation happens.

---

## Workflow & Key Steps

1. **Dataset**
   * Used a real-world telecom dataset containing **7,043 customer records** and **21 attributes** (contract type, monthly charges, total charges, tenure, internet service, and technical support).

2. **Data Preprocessing (`Pandas`)**
   * Removed non-predictive identifiers (`customerID`).
   * Converted `TotalCharges` into numeric format and resolved missing values using **median imputation** to prevent outlier distortion.
   * Mapped the target variable (`Churn`: `Yes` → `1`, `No` → `0`).
   * Applied **one-hot encoding** (`pd.get_dummies`) to convert categorical text columns into numerical features.

3. **Model Training & Tuning (`Scikit-Learn`)**
   * Split the cleaned dataset using an **80/20 train-test split**.
   * Trained a **Random Forest Classifier** (100 decision trees) with depth regularization (`max_depth=6`) to prevent overfitting and improve generalization on unseen data.

4. **Evaluation & Business Impact**
   * Achieved approximately **80% accuracy** on unseen test data, validated via a **Confusion Matrix**, **Precision**, **Recall**, and **F1-score**.
   * Implemented probability-based risk scoring (`predict_proba`) to flag high-risk customers for retention outreach.

---

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Environment:** Google Colab / Jupyter Notebook

---

## How to Run
1. Open the `.ipynb` notebook in **Google Colab** or **Jupyter Notebook**.
2. Run all cells sequentially to load the dataset, preprocess features, train the Random Forest model, and view evaluation metrics.

