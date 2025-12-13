
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
├── docs/                       # Documentation
│   ├── TASK3_FEATURE_ENGINEERING.md  # Task 3 documentation
│   └── TASK4_TARGET_ENGINEERING.md    # Task 4 documentation
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis (Task 2)
├── reports/                    # Analysis reports and figures
│   ├── TASK_2_COMPLETION.md   # Task 2 completion report
│   ├── TASK_4_COMPLETION.md   # Task 4 completion report
│   └── figures/               # EDA and clustering visualization figures
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Data loading utilities
│   ├── feature_engineering.py # Feature engineering pipeline (Task 3)
│   ├── target_engineering.py  # RFM analysis and target creation (Task 4)
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
4. **Model Deployment:** Containerize the best model and deploy as a REST API
5. **CI/CD Pipeline:** Implement automated testing and deployment workflows

## Task 2: Exploratory Data Analysis (EDA) ✅

### Overview
Task 2 focuses on exploring the dataset to uncover patterns, identify data quality issues, and form hypotheses that will guide feature engineering. The EDA was conducted in `notebooks/eda.ipynb` and all visualizations are saved to `reports/figures/`.

### Key Findings

#### 1. Dataset Characteristics
- **95,662 transactions** across **16 features**
- **Time period**: 2018-11-15 to 2019-02-13 (~3 months)
- **Data quality**: ✅ No missing values detected
- **Data types**: 5 numerical, 11 categorical features
- **Currency**: All transactions in UGX (single currency)

#### 2. Highly Skewed Transaction Amounts
- **Amount** and **Value** features show extreme right-skewness (skewness > 51)
- Indicates a small number of very large transactions
- **Action Required**: Log transformation needed for modeling
- Strong correlation between Amount and Value (r = 0.99)

#### 3. Imbalanced Target Variable
- **FraudResult** shows severe class imbalance (skewness = 22.20)
- Fraud cases represent a small minority of transactions
- **Action Required**: Use SMOTE, class weights, or stratified sampling

#### 4. Significant Outlier Presence
- **Amount**: 24,441 outliers (25.5% of data)
- **Value**: 9,021 outliers (9.4% of data)
- May represent legitimate high-value transactions or potential fraud cases
- **Action Required**: Consider robust scaling, capping, or separate treatment

#### 5. Categorical Feature Insights
- **ProductCategory**: "financial_services" dominates (47.5% of dataset)
- **ChannelId**: ChannelId_3 most common (62.2%)
- **ProviderId**: ProviderId_4 accounts for 39.9%
- High cardinality in ID columns suggests rich customer-level patterns

### EDA Deliverables

All required sections have been completed:

1. ✅ **Overview of the Data** - Dataset structure, rows, columns, data types
2. ✅ **Summary Statistics** - Central tendency, dispersion, distribution shape
3. ✅ **Distribution of Numerical Features** - Histograms with KDE for all numerical features
4. ✅ **Distribution of Categorical Features** - Count plots for categorical variables
5. ✅ **Correlation Analysis** - Correlation heatmap of numerical features
6. ✅ **Identifying Missing Values** - Missing value analysis (none found)
7. ✅ **Outlier Detection** - Box plots with IQR statistics
8. ✅ **Top 5 Insights Summary** - Key findings documented

### Running the EDA

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**: `notebooks/eda.ipynb`

3. **Run all cells**: The notebook will automatically:
   - Load and analyze the data
   - Generate all visualizations
   - Save figures to `reports/figures/`

### Generated Figures

All visualizations are saved in `reports/figures/`:
- `01_distribution_*.png` - Distribution plots for numerical features (5 files)
- `02_categorical_*.png` - Distribution plots for categorical features (4 files)
- `03_correlation_matrix.png` - Correlation heatmap
- `04_outlier_detection_boxplots.png` - Outlier analysis box plots

### Documentation

- **Completion Report**: See `reports/TASK_2_COMPLETION.md` for detailed findings
- **Notebook**: `notebooks/eda.ipynb` contains all analysis code and outputs

### Recommendations for Task 3 (Feature Engineering)

Based on EDA findings:
1. Apply log transformation to Amount and Value features
2. Handle outliers using robust scaling or capping
3. Address class imbalance using appropriate techniques
4. Create time-based features from TransactionStartTime
5. Generate customer-level aggregations (by CustomerId, AccountId)
6. Encode categorical variables (one-hot, target encoding, or WoE)

## Task 3: Feature Engineering Pipeline ✅

### Overview
Task 3 implements a comprehensive, automated, and reproducible feature engineering pipeline using `sklearn.pipeline.Pipeline` to transform raw financial transaction data into a model-ready format.

### Key Features Implemented

1. ✅ **Customer Aggregate Features**
   - Total Transaction Amount, Average Transaction Amount
   - Transaction Count, Standard Deviation per customer

2. ✅ **Temporal Feature Extraction**
   - Transaction Hour, Day, Month, Year
   - Day of Week, Is Weekend flag
   - Time of Day categories (night/morning/afternoon/evening)

3. ✅ **Categorical Encoding**
   - One-Hot Encoding for low-cardinality features (≤ 10 unique values)
   - Label Encoding for high-cardinality features (> 10 unique values)
   - Automatic strategy selection based on cardinality

4. ✅ **Missing Value Handling**
   - Simple Imputation (mean, median, mode)
   - KNN Imputation for complex patterns
   - Column removal for > 50% missing values

5. ✅ **Feature Scaling**
   - Standardization (Z-score normalization)
   - Min-Max Scaling (range normalization)

6. ✅ **Weight of Evidence (WoE) and Information Value (IV)**
   - WoE transformation for categorical variables
   - IV calculation for feature selection
   - Automatic filtering by IV threshold

### Implementation

- **Main Module**: `src/feature_engineering.py`
- **Pipeline**: Uses sklearn Pipeline for reproducibility
- **Documentation**: `docs/TASK3_FEATURE_ENGINEERING.md`

### Usage

```python
from src.feature_engineering import FeatureEngineeringPipeline

# Initialize pipeline
pipeline = FeatureEngineeringPipeline()

# Fit and transform
X_processed = pipeline.fit_transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test)
```

## Task 4: Proxy Target Variable Engineering ✅

### Overview
Task 4 creates a credit risk proxy target variable using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to identify "disengaged" customers as high-risk proxies.

### Implementation Steps

1. ✅ **Calculate RFM Metrics**
   - **Recency**: Days since last transaction (from snapshot date)
   - **Frequency**: Total number of transactions per customer
   - **Monetary**: Total transaction value per customer
   - **Results**: 3,742 unique customers analyzed

2. ✅ **Cluster Customers**
   - **Algorithm**: K-Means clustering with 3 clusters
   - **Pre-processing**: StandardScaler applied to RFM features
   - **Random State**: 42 (for reproducibility)
   - **Silhouette Score**: 0.5732 (good clustering quality)

3. ✅ **Define and Assign High-Risk Label**
   - **High-Risk Cluster**: Cluster 0 identified (38.11% of customers)
   - **Characteristics**: High Recency (60.88 days), Low Frequency (7.72), Low Monetary (89,738)
   - **Binary Target**: `is_high_risk` column created (1 = high-risk, 0 = low-risk)

4. ✅ **Integrate Target Variable**
   - Merged `is_high_risk` back to original dataset
   - Output: `data/processed/data_with_risk_target.csv`

### Cluster Profiles

| Cluster | Recency (days) | Frequency | Monetary | Customers | Interpretation |
|---------|----------------|-----------|----------|-----------|----------------|
| **0** 🔴 | 60.88 | 7.72 | 89,738 | 1,426 (38%) | **High-Risk**: Inactive, low engagement |
| 1 | 11.72 | 34.70 | 224,757 | 2,312 (62%) | Low-Risk: Active, moderate engagement |
| 2 | 22.25 | 1,104.50 | 74.9M | 4 (0.1%) | VIP: Very high engagement |

### Implementation

- **Main Module**: `src/target_engineering.py`
- **Classes**: `RFMAnalyzer`, `CustomerSegmentation`
- **Main Function**: `create_proxy_target()` - Complete pipeline
- **Documentation**: `docs/TASK4_TARGET_ENGINEERING.md`
- **Completion Report**: `reports/TASK_4_COMPLETION.md`
- **Visualization**: `reports/figures/rfm_clusters.png`

### Usage

```python
from src.target_engineering import create_proxy_target

# Create proxy target
df_with_target, metadata = create_proxy_target(
    df,
    customer_col='CustomerId',
    date_col='TransactionStartTime',
    value_col='Value',
    n_clusters=3,
    random_state=42,
    save_visualization=True
)

print(f"High-Risk Customers: {metadata['high_risk_count']} ({metadata['high_risk_percentage']:.2f}%)")
```

## Task 5: Model Training and Tracking ✅

### Overview
Task 5 implements a robust, reproducible model training pipeline with experiment tracking using **MLflow**. It systematically trains, tunes, and evaluates multiple machine learning algorithms to predict credit risk.

### Key Components

1. ✅ **Data Preparation**
   - **Splitting**: 80/20 Train/Test split (`random_state=42`)
   - **Balancing**: SMOTE applied to training data
   - **Preprocessing**: Integration with feature engineering pipeline

2. ✅ **Model Selection**
   - **Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
   - **Tuning**: RandomizedSearchCV for hyperparameter optimization

3. ✅ **Experiment Tracking (MLflow)**
   - **Logging**: Parameters, metrics (ROC-AUC, F1, Accuracy), and artifacts
   - **Registry**: Automatic registration of the best performing model

4. ✅ **Unit Testing**
   - Comprehensive tests in `tests/test_data_processing.py` to verify data integrity and processing logic

### Implementation

- **Main Script**: `src/train_mlflow.py`
- **Unit Tests**: `tests/test_data_processing.py`
- **Documentation**: `docs/TASK5_MODEL_TRAINING.md`

### Usage

**Train Models:**
```bash
python src/train_mlflow.py
```

**View MLflow Dashboard:**
```bash
mlflow ui
```

**Run Unit Tests:**
```bash
pytest tests/test_data_processing.py
```

## Next Steps
1. ✅ ~~Download and explore the dataset (Task 2 - EDA)~~ **COMPLETED**
2. ✅ ~~Implement feature engineering pipeline (Task 3)~~ **COMPLETED**
3. ✅ ~~Create proxy target variable using RFM clustering (Task 4)~~ **COMPLETED**
4. ✅ ~~Train and evaluate models (Task 5)~~ **COMPLETED**
5. ✅ ~~Deploy model API with CI/CD (Task 6)~~ **COMPLETED**

## Task 6: Model Deployment and CI/CD ✅

### Overview
Task 6 involved containerizing the credit risk model and deploying it as a REST API with a user-friendly dashboard. We also established a CI/CD pipeline for automated quality checks.

### 1. Application Components
- **API (FastAPI)**: Serves model predictions at port 8000.
    - Features: Real-time inference (`/predict`), Health checks (`/health`), Batch prediction (`/predict/batch`).
    - Robustness: Automatically falls back to a local model if MLflow is unavailable.
- **Dashboard (Streamlit)**: Interactive UI at port 8501.
    - Features: Form input for transaction details, visual risk gauges, and probability displays.
- **Containerization**:
    - `Dockerfile`: Builds the API image (Python 3.10).
    - `Dockerfile.streamlit`: Builds the Dashboard image.
    - `docker-compose.yml`: Orchestrates both services.

### 2. How to Run the Application 🚀

To start the full stack (API + Dashboard), open your terminal in the project root and run:

```bash
docker-compose up --build
```
*(Use `--build` to ensure any code changes are picked up)*

### 3. Accessing the Application

Once the containers are running:

- **Dashboard**: [http://localhost:8501](http://localhost:8501)
    - Open this in your browser to test the model interactively.

- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
    - Swagger UI for testing API endpoints directly.

- **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### 4. CI/CD Pipeline
A GitHub Actions workflow (`.github/workflows/ci.yml`) is configured to run on every push:
- **Linting**: Checks code quality with `flake8`.
- **Testing**: Runs unit tests with `pytest`.
