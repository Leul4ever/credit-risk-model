# Task 4 - Proxy Target Variable Engineering

## Overview

This module creates a credit risk proxy target variable using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to identify high-risk customers - those who are disengaged and likely to default.

## Implementation

### Step 1: Calculate RFM Metrics ✅

**Metrics Calculated:**
- **Recency**: Days since last transaction (from snapshot date)
- **Frequency**: Total number of transactions per customer
- **Monetary**: Total transaction value per customer

**Results:**
```
RFM Statistics (3,742 unique customers):
├── Recency (days): Mean=30.46, Median=24, Range=[0-90]
├── Frequency (transactions): Mean=25.56, Median=7, Range=[1-4,091]
└── Monetary (value): Mean=253,103, Median=32,000, Range=[50-104.9M]
```

**Snapshot Date:** 2019-02-13 10:01:28 (max transaction date in dataset)

### Step 2: Cluster Customers ✅

**K-Means Configuration:**
- **Algorithm**: K-Means clustering
- **Number of Clusters**: 3
- **Random State**: 42 (for reproducibility)
- **Pre-processing**: StandardScaler applied to RFM features
- **Silhouette Score**: 0.5732 (good clustering quality)

**Cluster Profiles:**

| Cluster | Recency (days) | Frequency | Monetary | Customers | Interpretation |
|---------|----------------|-----------|----------|-----------|----------------|
| **0** 🔴 | 60.88 | 7.72 | 89,738 | 1,426 (38%) | **High-Risk**: Inactive, low engagement |
| 1 | 11.72 | 34.70 | 224,757 | 2,312 (62%) | Low-Risk: Active, moderate engagement |
| 2 | 22.25 | 1,104.50 | 74.9M | 4 (0.1%) | VIP: Very high engagement |

### Step 3: Define and Assign High-Risk Label ✅

**Risk Score Calculation:**
```python
Risk_Score = (Recency_Score + Frequency_Score + Monetary_Score) / 3

Cluster 0: 0.9973 ⚠️ HIGH RISK
Cluster 1: 0.7193
Cluster 2: 0.1218
```

**High-Risk Cluster Identified:** Cluster 0
- High Recency (60.88 days - inactive)
- Low Frequency (7.72 transactions)
- Low Monetary Value (89,738)

**Binary Target Created:** `is_high_risk`
- **1**: Customer in Cluster 0 (high-risk)
- **0**: Customer in other clusters (low-risk)

**Distribution:**
- High Risk (1): 1,426 customers (38.11%)
- Low Risk (0): 2,316 customers (61.89%)

### Step 4: Integrate Target Variable ✅

**Merge Strategy:**
- Original dataset: 95,662 transactions × 16 features
- Merged dataset: 95,662 transactions × 17 features
- New column: `is_high_risk` (binary: 0 or 1)

**Output Files:**
1. `data/processed/data_with_risk_target.csv` - Full dataset with target
2. `reports/figures/rfm_clusters.png` - Cluster visualization

## Usage

### Basic Usage

```python
from target_engineering import create_proxy_target

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

print(f"High-Risk: {metadata['high_risk_percentage']:.2f}%")
```

### Using Individual Components

```python
from target_engineering import RFMAnalyzer, CustomerSegmentation

# Step 1: Calculate RFM
rfm_analyzer = RFMAnalyzer(snapshot_date='2019-02-13')
rfm_data = rfm_analyzer.calculate_rfm(df)

# Step 2: Cluster
segmentation = CustomerSegmentation(n_clusters=3, random_state=42)
rfm_clustered = segmentation.fit_kmeans(rfm_data)

# Step 3: Create labels
risk_labels = segmentation.create_risk_labels(rfm_clustered)

# Step 4: Merge
df_with_target = df.merge(risk_labels[['CustomerId', 'is_high_risk']], on='CustomerId')
```

## Visualization

The module automatically generates a comprehensive cluster visualization showing:
1. **Recency vs Frequency**: Scatter plot colored by cluster
2. **Recency vs Monetary**: Scatter plot colored by cluster
3. **Frequency vs Monetary**: Scatter plot colored by cluster
4. **Cluster Size Distribution**: Bar chart (high-risk cluster in red)

![RFM Clusters](../../reports/figures/rfm_clusters.png)

## Classes

### `RFMAnalyzer`
Calculates RFM metrics for customer segmentation.

**Methods:**
- `calculate_rfm(df, customer_col, date_col, value_col)` - Calculate RFM metrics
- `get_rfm_summary()` - Get summary statistics

### `CustomerSegmentation`
Performs K-Means clustering and identifies high-risk segments.

**Methods:**
- `fit_kmeans(rfm_data, customer_col)` - Fit K-Means clustering
- `create_risk_labels(rfm_clustered, customer_col)` - Create binary risk labels
- `visualize_clusters(rfm_clustered, save_path)` - Generate visualizations

## Rationale

### Why RFM?
1. **Recency**: Recent customers are more engaged
2. **Frequency**: More transactions indicate loyalty
3. **Monetary**: Higher value customers are valuable

### Why K-Means with 3 Clusters?
- Separates customers into distinct engagement levels
- **Cluster 0**: Disengaged (high-risk)
- **Cluster 1**: Standard engagement
- **Cluster 2**: Power users/VIPs

### High-Risk Definition
Customers are labeled high-risk if they exhibit:
- **High Recency**: Haven't transacted recently (disengaged)
- **Low Frequency**: Few total transactions
- **Low Monetary**: Low total value

These characteristics indicate higher likelihood of default.

## Validation

**Silhouette Score: 0.5732**
- Indicates good cluster separation
- Values range from -1 (poor) to 1 (excellent)
- Score > 0.5 suggests meaningful clusters

**Cluster Balance:**
- Cluster 0 (High-Risk): 38.11% - Reasonable proportion
- Cluster 1 (Low-Risk): 61.89% - Majority
- Cluster 2 (VIP): 0.11% - Small elite group

## Next Steps

With the `is_high_risk` target variable created, you can now:
1. **Train models** using this as the target (Task 5)
2. **Evaluate performance** on predicting credit risk
3. **Compare** with existing FraudResult field
4. **Refine** clustering if needed based on model performance

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## References

- [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(market_research))
- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Silhouette Score](https://scikit-learn.org/stable/modules/clustering.html#silhouette-score)
