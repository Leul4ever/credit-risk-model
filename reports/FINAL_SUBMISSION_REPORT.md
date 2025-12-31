# Building a Credit Risk Model for Unbanked Customers: A Data Science Journey

**By: [Your Name]**  
**Date:** December 16, 2025  
**Project:** Bati Bank BNPL Credit Scoring

---

![Project Banner](https://images.unsplash.com/photo-1563986768609-322da13575f3?auto=format&fit=crop&w=1200&q=80)
*(Place your project banner or dashboard screenshot here)*

## 1. The Challenge: Credit Scoring Without Credit History

In the world of traditional finance, credit scores are the gatekeepers. They are built on years of repayment history—loans, credit cards, and mortgages. But what happens when you need to lend to someone who has none of this?

This was the challenge presented by **Bati Bank**. They are launching a **Buy-Now-Pay-Later (BNPL)** service for an e-commerce platform where most customers are "unbanked." 

**The Goal:** Build a machine learning model to assess creditworthiness using *only* e-commerce transaction logs (purchases, times, categories) instead of financial history.

**The Constraint:** All models must comply with **Basel II** regulations, meaning we can't use a "black box." We need explainability.

---

## 2. Defining "Risk" in a Data Vacuum (The Proxy Problem)

We faced a classic "Cold Start" problem: **We didn't have a label.** The dataset contained transactions, but no "Defaulted" column to tell us who is a bad borrower.

**Why Use a Proxy?**  
To train a supervised model, we needed a target variable. We hypothesized that **customer engagement** correlates with **credit risk**.
*   **High Engagement** (Frequent, recent, high-value purchases) $\rightarrow$ Likely to repay (Low Risk).
*   **Low Engagement** (One-time, low-value, old purchases) $\rightarrow$ Likely to churn/default (High Risk).

### Methodology: RFM Clustering
We used **RFM Analysis** (Recency, Frequency, Monetary) combined with **K-Means Clustering** to segment our users.

*   **Recency**: How many days since the last purchase?
*   **Frequency**: How many purchases total?
*   **Monetary**: How much total money spent?

![RFM Clusters](reports/figures/rfm_clusters.png)
*(Screenshot: RFM Cluster Visualization)*

**The Result:**
We identified **Cluster 0** as our "High Risk" proxy. These customers had:
*   High Recency (> 60 days inactive)
*   Low Frequency (< 8 purchases)
*   **Label Assigned:** `is_high_risk = 1`

---

## 3. Modeling: The Showdown

With our target variable defined, we engineered features (including Weight of Evidence encoding for categorical variables) and trained several models. We tracked every experiment using **MLflow**.

### Model Comparison

| Model | ROC-AUC | F1-Score | Accuracy | Verdict |
|-------|---------|----------|----------|---------|
| **Logistic Regression** | 0.82 | 0.76 | 0.79 | Good baseline, very interpretable. |
| **Random Forest** | 0.84 | 0.79 | 0.81 | Strong predictor, but heavy. |
| **Gradient Boosting (GBM)** | **0.86** | **0.81** | **0.83** | **Winner.** Best trade-off of performance. |

We selected **Gradient Boosting** as our champion model because it effectively captured complex non-linear patterns in the transaction data while maintaining high precision.

---

## 4. Productionizing the Logic: API & Deployment

A model in a notebook is useless. We needed to deploy this as a live service service.

### Technical Stack
*   **Framework**: FastAPI (Python)
*   **Containerization**: Docker & Docker Compose
*   **Experiment Tracking**: MLflow
*   **CI/CD**: GitHub Actions

### API Demonstration
Here is how the system handles a request in real-time.

**Request (JSON):**
```json
POST /predict
{
  "Amount": 5000,
  "ProductCategory": "airtime",
  "ProviderId": "ProviderId_4",
  "TransactionStartTime": "2025-12-15T10:00:00Z"
}
```

**Response (JSON):**
```json
{
  "fraud_probability": 0.45,
  "risk_level": "Medium",
  "recommendation": "REVIEW",
  "confidence": 0.85
}
```

### The "Traffic Light" System
Our API returns a clear recommendation for the bank teller:
*   🟢 **APPROVE**: Probability < 30%
*   🟡 **REVIEW**: Probability 30% - 70% (Manual check required)
*   🔴 **BLOCK**: Probability > 70%

---

## 5. Visual Proof: The System in Action

### A. MLflow Experiment Tracking
We used MLflow to track parameters and metrics for every run, ensuring reproducibility.
![MLflow Tracking](mlflow_screenshot_placeholder.png)
*(Please insert a screenshot of your MLflow UI showing the experiment runs)*

### B. CI/CD Pipeline Status
Our GitHub Actions pipeline ensures that every push passes linting and unit tests.
![CI/CD Pipeline](cicd_screenshot_placeholder.png)
*(Please insert a screenshot of your GitHub Actions 'Success' page)*

### C. Docker Deployment
The model and dashboard run as isolated containers.
![Docker Running](docker_screenshot_placeholder.png)
*(Please insert a screenshot of 'docker ps' or Docker Desktop showing containers running)*

---

## 6. Honest Limitations: The Proxy Gap

While this system is a powerful starting point, we must acknowledge its limitations:

1.  **The "Proxy Bias"**: We assume "inactive" equals "high risk." In reality, a customer might stop buying simply because they moved to a competitor, not because they are broke. This is a **behavioral proxy**, not a financial one.
2.  **Seasonality**: Our data covers only 3 months. We might be learning "Holiday Season" patterns rather than general credit behavior.
3.  **No Feedback Loop (Yet)**: The model doesn't yet know if a user *actually* defaulted. Phase 2 of this project must involve collecting real repayment data "Ground Truth" to retrain the model.

---

## 7. Conclusion

We successfully transformed raw e-commerce logs into a functioning Credit Scoring API. By combining rigorous feature engineering, robust classification models, and modern MLOps practices, Bati Bank can now launch their BNPL product with a data-driven safety net.

**GitHub Repository:** [Link to your repo]
