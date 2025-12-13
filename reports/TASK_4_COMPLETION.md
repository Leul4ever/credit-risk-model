# Task 4 - Proxy Target Variable Engineering - Completion Report

## Status: ✅ COMPLETED

## Objective
Create a credit risk target variable using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to identify "disengaged" customers as high-risk proxies.

## Deliverables Summary

All required sections have been completed in `src/target_engineering.py`:

### ✅ 1. Calculate RFM Metrics
- **Location**: `RFMAnalyzer.calculate_rfm()` method
- **Completed**: Yes
- **Implementation Details**:
  - **Recency**: Days since last transaction (calculated from snapshot date)
  - **Frequency**: Total number of transactions per customer
  - **Monetary**: Total transaction value per customer
  - **Snapshot Date**: Defined consistently (defaults to max transaction date)
- **Results**:
  - 3,742 unique customers analyzed
  - Recency: Mean=30.46 days, Median=24 days, Range=[0-90]
  - Frequency: Mean=25.56, Median=7, Range=[1-4,091]
  - Monetary: Mean=253,103, Median=32,000, Range=[50-104.9M]

### ✅ 2. Cluster Customers
- **Location**: `CustomerSegmentation.fit_kmeans()` method
- **Completed**: Yes
- **Implementation Details**:
  - **Algorithm**: K-Means clustering
  - **Number of Clusters**: 3 (as required)
  - **Pre-processing**: StandardScaler applied to RFM features
  - **Random State**: 42 (for reproducibility) ✅
  - **Silhouette Score**: 0.5732 (good clustering quality)
- **Cluster Profiles**:
  - **Cluster 0** 🔴: 1,426 customers (38.11%) - High-Risk: Inactive, low engagement
    - Recency: 60.88 days, Frequency: 7.72, Monetary: 89,738
  - **Cluster 1**: 2,312 customers (61.89%) - Low-Risk: Active, moderate engagement
    - Recency: 11.72 days, Frequency: 34.70, Monetary: 224,757
  - **Cluster 2**: 4 customers (0.11%) - VIP: Very high engagement
    - Recency: 22.25 days, Frequency: 1,104.50, Monetary: 74.9M

### ✅ 3. Define and Assign High-Risk Label
- **Location**: `CustomerSegmentation._identify_high_risk_cluster()` and `create_risk_labels()` methods
- **Completed**: Yes
- **Implementation Details**:
  - **Risk Score Calculation**: 
    - Risk_Score = (Recency_Score + Frequency_Score + Monetary_Score) / 3
    - Cluster 0: 0.9973 ⚠️ HIGH RISK
    - Cluster 1: 0.7193
    - Cluster 2: 0.1218
  - **High-Risk Cluster Identified**: Cluster 0
    - Characteristics: High Recency (60.88 days), Low Frequency (7.72), Low Monetary (89,738)
  - **Binary Target Column**: `is_high_risk` ✅
    - **Value 1**: Customers in Cluster 0 (high-risk)
    - **Value 0**: Customers in other clusters (low-risk)
- **Distribution**:
  - High Risk (1): 1,426 customers (38.11%)
  - Low Risk (0): 2,316 customers (61.89%)

### ✅ 4. Integrate Target Variable
- **Location**: `create_proxy_target()` function
- **Completed**: Yes
- **Implementation Details**:
  - Merges `is_high_risk` column back to original dataset
  - Uses left join on `CustomerId`
  - Original dataset: 95,662 transactions × 16 features
  - Final dataset: 95,662 transactions × 17 features (added `is_high_risk`)
- **Output Files**:
  - `data/processed/data_with_risk_target.csv` - Full dataset with target variable
  - `reports/figures/rfm_clusters.png` - Cluster visualization

## Technical Implementation

### Code Structure
- **Main Module**: `src/target_engineering.py`
- **Classes**:
  - `RFMAnalyzer`: Calculates RFM metrics
  - `CustomerSegmentation`: Performs clustering and creates risk labels
- **Main Function**: `create_proxy_target()` - Complete pipeline

### Key Features
1. ✅ **Snapshot Date**: Defined consistently for recency calculation
2. ✅ **Feature Scaling**: StandardScaler applied before clustering
3. ✅ **Reproducibility**: `random_state=42` set in K-Means
4. ✅ **Cluster Analysis**: Automatic identification of high-risk cluster
5. ✅ **Visualization**: Automatic generation of cluster plots
6. ✅ **Logging**: Comprehensive logging throughout the pipeline

## Validation Results

### Clustering Quality
- **Silhouette Score**: 0.5732
  - Indicates good cluster separation
  - Values range from -1 (poor) to 1 (excellent)
  - Score > 0.5 suggests meaningful clusters

### Cluster Balance
- **High-Risk Cluster (0)**: 38.11% - Reasonable proportion for modeling
- **Low-Risk Cluster (1)**: 61.89% - Majority of customers
- **VIP Cluster (2)**: 0.11% - Small elite group

### High-Risk Cluster Characteristics
- **Recency**: 60.88 days (inactive - haven't transacted recently)
- **Frequency**: 7.72 transactions (low engagement)
- **Monetary**: 89,738 (low total value)
- **Interpretation**: These customers are disengaged and represent higher credit risk

## Files and Outputs

### Source Code
- `src/target_engineering.py` - Complete implementation (578 lines)

### Documentation
- `docs/TASK4_TARGET_ENGINEERING.md` - Comprehensive documentation

### Generated Files
- `data/processed/data_with_risk_target.csv` - Dataset with target variable
- `reports/figures/rfm_clusters.png` - Cluster visualization

### Visualization
The cluster visualization includes:
1. Recency vs Frequency scatter plot
2. Recency vs Monetary scatter plot
3. Frequency vs Monetary scatter plot
4. Cluster size distribution (high-risk cluster highlighted in red)

## Usage Example

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

# Access results
print(f"High-Risk Customers: {metadata['high_risk_count']} ({metadata['high_risk_percentage']:.2f}%)")
print(f"High-Risk Cluster ID: {metadata['high_risk_cluster']}")
```

## Requirements Verification

All task requirements have been met:

1. ✅ **Calculate RFM Metrics**: Implemented with snapshot date
2. ✅ **Cluster Customers**: K-Means with 3 clusters, StandardScaler preprocessing
3. ✅ **Set random_state**: random_state=42 for reproducibility
4. ✅ **Define High-Risk Label**: Cluster 0 identified as high-risk based on engagement patterns
5. ✅ **Create is_high_risk column**: Binary target (0 or 1) created
6. ✅ **Integrate Target Variable**: Merged back to main dataset

## Rationale

### Why RFM Analysis?
- **Recency**: Recent customers are more engaged and likely to repay
- **Frequency**: More transactions indicate loyalty and engagement
- **Monetary**: Higher value customers are more valuable and reliable

### Why K-Means with 3 Clusters?
- Separates customers into distinct engagement levels
- **Cluster 0**: Disengaged (high-risk) - 38.11%
- **Cluster 1**: Standard engagement (low-risk) - 61.89%
- **Cluster 2**: Power users/VIPs (low-risk) - 0.11%

### High-Risk Definition
Customers are labeled high-risk if they exhibit:
- **High Recency**: Haven't transacted recently (disengaged)
- **Low Frequency**: Few total transactions
- **Low Monetary**: Low total value

These characteristics indicate higher likelihood of default on BNPL payments.

## Next Steps

With the `is_high_risk` target variable created, you can now:

1. ✅ **Task 4 Complete**: Proxy target variable created
2. **Task 5**: Train models using `is_high_risk` as the target
3. **Model Evaluation**: Evaluate performance on predicting credit risk
4. **Comparison**: Compare with existing `FraudResult` field
5. **Refinement**: Refine clustering if needed based on model performance

## Dependencies

Required packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

All dependencies are included in `requirements.txt`.

---

**Completion Date**: Current Session  
**Implementation Location**: `src/target_engineering.py`  
**Documentation**: `docs/TASK4_TARGET_ENGINEERING.md`  
**Status**: ✅ All deliverables completed and validated

