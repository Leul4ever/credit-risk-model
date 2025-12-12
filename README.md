
# Credit Risk Probability Model for Alternative Data

## Overview
This project aims to build an end-to-end Credit Scoring Model for Bati Bank using alternative data from an eCommerce platform. The goal is to enable a buy-now-pay-later service by predicting the likelihood of default for potential borrowers.

## Credit Scoring Business Understanding

### 1. Basel II and Model Interpretability
The Basel II Capital Accord emphasizes robust risk management through the **Internal Ratings-Based (IRB)** approach. This framework allows financial institutions to use their own models to estimate Probability of Default (PD), provided they can demonstrate the model's reliability and transparency.
- **Influence**: This necessitates a model that is not just accurate but also **interpretable and well-documented**. Regulators require a clear audit trail of how risk estimates are derived. A "black box" model that cannot explain *why* a customer risk score changed (e.g., due to a drop in transaction frequency) fails the "use test" and makes validation under Basel II difficult.
- **Adverse Action**: Furthermore, financial regulations often require explaining declining a loan to a customer. Interpretable models facilitate these explanations.

### 2. The Need for a Proxy Variable
In the absence of a historical "default" label (ground truth of borrowers who failed to pay), we must infer creditworthiness from behavioral data.
- **Why Necessary**: We assume that **RfM (Recency, Frequency, Monetary)** patterns correlate with financial stability. A user who is highly active (high frequency) and spends significantly (high monetary) is likely more engaged and financially capable than a dormant user. We use **Clustering** to group these behaviors and assign a "High Risk" label to the least engaged group.
- **Business Risks**:
    - **Proxy Bias**: The proxy predicts *engagement*, not *solvency*. A wealthy user who rarely uses this specific platform could be misclassified as "High Risk" (False Positive), leading to lost business.
    - **Stability**: Behavioral patterns change faster than creditworthiness. A model trained on a "proxy" might be less stable over time compared to one trained on actual repayment history.

### 3. Trade-offs: Interpretability vs. Performance
In a regulated financial context, the choice of model involves significant trade-offs:

| Feature | Logistic Regression (with WoE) | Gradient Boosting (XGBoost/LGBM) |
| :--- | :--- | :--- |
| **Interpretability** | **High**: Coefficients equate to "Scorecard Points". Relationships are monotonic and easy to explain to Risk Officers and Regulators. | **Low**: Complex non-linear decision boundaries. Requires post-hoc explainers (SHAP), which are approximations and harder to validate. |
| **Performance** | **Moderate**: May miss complex, non-linear interactions in alternative data. | **High**: Typically achieves higher accuracy (AUC) by capturing subtle patterns. |
| **Deployment** | **Simple**: Can be implemented as simple SQL rules or a lookup table. | **Complex**: Requires a Python runtime or specialized inference engine. |

**Strategy**: We will start with a robust feature engineering pipeline (WoE) that can support a Logistic Regression baseline, while also training a Champion GBM model to quantify the "cost" of interpretability (i.e., how much accuracy we lose by choosing the simpler model).

## Project Structure
- `data/`: Contains raw and processed data (ignored by git).
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `src/`: Source code for data processing, training, and API.
- `tests/`: Unit tests.
- `.github/workflows/`: CI/CD configurations.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run EDA: `jupyter notebook notebooks/eda.ipynb`
3. Train model: `python src/train.py`
4. Run API: `uvicorn src.api.main:app --reload`
