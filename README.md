
# Credit Risk Probability Model for Alternative Data

## Project Overview
Bati Bank is partnering with an e-commerce company to launch a Buy-Now-Pay-Later (BNPL) service. This project aims to develop a credit scoring model using alternative data (e-commerce transaction history) since traditional credit bureau data is unavailable. The model will assess customer creditworthiness to inform loan approval decisions and terms.

## Business Problem
Traditional credit scoring relies on historical loan repayment data, which is not available for e-commerce customers. We must use behavioral transaction data to:
1. Identify high-risk customers likely to default on BNPL payments
2. Assign risk probabilities to new customers
3. Recommend appropriate credit limits and repayment terms

## Data Source
The dataset contains e-commerce transaction records with features including:
- Customer identifiers (`CustomerId`, `AccountId`)
- Transaction details (`Amount`, `Value`, `TransactionStartTime`)
- Product information (`ProductId`, `ProductCategory`)
- Platform details (`ChannelId`, `ProviderId`)
- Geographic information (`CountryCode`, `CurrencyCode`)
- Fraud indicators (`FraudResult`)

## Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Requirements

The Basel II Accord's emphasis on risk measurement imposes three critical requirements on our model:

**a) Model Interpretability:** Under Pillar 2 (Supervisory Review), regulators must be able to understand and validate our risk assessment methodology. A "black box" model, regardless of its accuracy, would fail regulatory scrutiny because it doesn't allow for meaningful oversight or challenge of individual credit decisions.

**b) Comprehensive Documentation:** The Accord requires detailed documentation of all modeling assumptions, data sources, validation procedures, and performance metrics. This documentation must demonstrate that the model is conceptually sound, empirically validated, and implemented correctly.

**c) Risk Sensitivity:** Basel II moved from standardized risk weights to models that are sensitive to the actual risk profile of borrowers. Our model must accurately discriminate between different risk levels and produce probabilities that can be used for capital allocation decisions.

**Implication for Our Project:** We cannot prioritize predictive accuracy at the expense of transparency. Every modeling decision—from proxy variable definition to feature selection—must be justifiable from both statistical and business perspectives.

### 2. Necessity and Risks of Proxy Variables

**Why a Proxy is Necessary:**
Traditional credit scoring relies on historical repayment data (loans, credit cards). In this BNPL partnership, we lack this direct evidence of credit behavior. The e-commerce transaction data contains behavioral patterns (purchase frequency, amount, recency) that correlate with financial responsibility. A proxy variable allows us to translate these behavioral signals into a credit risk assessment.

**Potential Business Risks:**

**a) Proxy Misalignment Risk:** The core assumption—that "disengaged e-commerce customers" equate to "high credit risk"—may be flawed. A customer could have low engagement due to factors unrelated to creditworthiness (e.g., temporary financial constraints, preference for other platforms, one-time purchasers).

**b) Data Representation Bias:** The e-commerce data represents only one dimension of financial behavior. Customers who are financially responsible but infrequent online shoppers could be incorrectly labeled as high-risk.

**c) Regulatory Scrutiny:** Using alternative data and proxy variables invites closer regulatory examination. We must demonstrate that our proxy is a reasonable, non-discriminatory predictor of actual default risk.

**d) Performance Degradation:** If the correlation between our proxy and actual default is weak, the model's real-world performance will be poor, leading to either excessive loan losses (if too lenient) or missed revenue opportunities (if too conservative).

**Mitigation Strategy:** We must rigorously validate our proxy through:
- Comparative analysis with any available credit performance data
- Sensitivity testing of different proxy definitions
- Clear documentation of assumptions and limitations

### 3. Trade-offs: Simple vs. Complex Models in Regulated Finance

**Logistic Regression with WoE (Simple, Interpretable):**

*Advantages:*
- **Full Transparency:** Each feature's contribution to the final score is explicit and quantifiable through coefficients
- **Regulatory Acceptance:** Well-established methodology that regulators understand and trust
- **Monotonic Relationships:** WoE transformation ensures logical, monotonic relationships between features and risk
- **Easy to Implement in Production:** Simple scoring formula can be deployed in various environments

*Disadvantages:*
- **Limited Complexity:** Assumes linear relationships between transformed variables and the log-odds of default
- **May Sacrifice Predictive Power:** Could underperform on capturing complex, non-linear interactions in the data
- **Manual Feature Engineering:** Requires careful binning and WoE calculation

**Gradient Boosting (Complex, High-Performance):**

*Advantages:*
- **Superior Predictive Accuracy:** Can capture complex, non-linear relationships and feature interactions
- **Robust to Outliers:** Generally more resistant to noisy data
- **Automatic Feature Importance:** Identifies which variables matter most

*Disadvantages:*
- **"Black Box" Nature:** Difficult to explain why a specific prediction was made
- **Regulatory Challenges:** May require additional explainability layers (SHAP, LIME) that add complexity
- **Risk of Overfitting:** Without careful tuning and validation, can memorize noise rather than learn patterns
- **Implementation Complexity:** More challenging to deploy and monitor in production

**Our Approach:** Given the regulatory context and the novelty of using e-commerce data for credit scoring, we will likely adopt a **hybrid strategy**:
1. **Primary Model:** Logistic Regression with WoE for its interpretability and regulatory compliance
2. **Benchmark Model:** Gradient Boosting to establish the "upper bound" of predictive performance
3. **Model Comparison:** If Gradient Boosting shows significantly better performance, we can use it with extensive explainability tools (SHAP values) to bridge the interpretability gap

This approach balances regulatory requirements with performance optimization while maintaining the ability to justify our decisions to both business stakeholders and regulators.

## Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology
1. **Proxy Target Creation:** Use RFM (Recency, Frequency, Monetary) analysis and clustering to identify high-risk customer segments
2. **Feature Engineering:** Transform raw transaction data into predictive features using statistical aggregations and WoE encoding
3. **Model Development:** Train and compare multiple models (Logistic Regression, Random Forest, Gradient Boosting)
4. **Model Deployment:** Containerize the best model and deploy as a REST c;API
5. **CI/CD Pipeline:** Implement automated testing and deployment workflows

## Next Steps
1. Download and explore the dataset (Task 2 - EDA)
2. Implement feature engineering pipeline (Task 3)
3. Create proxy target variable using RFM clustering (Task 4)
4. Train and evaluate models (Task 5)
5. Deploy model API with CI/CD (Task 6)
