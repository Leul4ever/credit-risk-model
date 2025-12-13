# Task 3 - Feature Engineering Pipeline

## Overview

This module implements a comprehensive, automated, and reproducible feature engineering pipeline using `sklearn.pipeline.Pipeline` to transform raw financial transaction data into a model-ready format.

## Features Implemented

### 1. **Customer Aggregate Features** ✅
- **Total Transaction Amount**: Sum of all amounts per customer
- **Average Transaction Amount**: Mean amount per customer  
- **Transaction Count**: Number of transactions per customer
- **Standard Deviation**: Variability of amounts per customer

### 2. **Temporal Feature Extraction** ✅
- **Transaction Hour**: Hour of day (0-23)
- **Transaction Day**: Day of month (1-31)
- **Transaction Month**: Month (1-12)
- **Transaction Year**: Year
- **Day of Week**: Weekday identifier (0-6)
- **Is Weekend**: Binary flag for weekend transactions
- **Time of Day**: Categorical (night/morning/afternoon/evening)

### 3. **Categorical Encoding** ✅
- **One-Hot Encoding**: For low-cardinality features (<= 10 unique values)
- **Label Encoding**: For high-cardinality features (> 10 unique values)
- Automatic strategy selection based on cardinality

### 4. **Missing Value Handling** ✅
- **Simple Imputation**: Mean, median, or mode strategies
- **KNN Imputation**: K-Nearest Neighbors for complex patterns
- **Column Removal**: Drops columns with > 50% missing values
- **Categorical Handling**: Mode imputation or 'missing' label

### 5. **Feature Scaling** ✅
- **Standardization**: Z-score normalization (mean=0, std=1)
- **Min-Max Scaling**: Range normalization [0, 1]
- Configurable exclusion of specific columns

### 6. **Weight of Evidence (WoE) and Information Value (IV)** ✅
- **WoE Transformation**: Encodes categorical variables based on target relationship
- **IV Calculation**: Measures predictive power of each feature
- **Feature Selection**: Automatic filtering by IV threshold
- **IV Interpretation**:
  - < 0.02: Useless
  - 0.02-0.1: Weak
  - 0.1-0.3: Medium
  - 0.3-0.5: Strong
  - \> 0.5: Very Strong

## Pipeline Architecture

```python
Pipeline([
    ('missing_values', MissingValueHandler()),
    ('temporal_features', TemporalFeatureExtractor()),
    ('customer_aggregates', CustomerAggregateFeatures()),
    ('woe_encoding', WoEIVEncoder()),
    ('categorical_encoding', CategoricalEncoder()),
    ('feature_scaling', FeatureScaler())
])
```

## Usage

### Basic Usage

```python
from feature_engineering import create_feature_engineering_pipeline

# Create pipeline
pipeline = create_feature_engineering_pipeline(
    include_woe=True,
    scaling_method='standard',
    imputation_strategy='mean'
)

# Fit and transform
df_transformed = pipeline.fit_transform(df)
```

### Running the Demo

```bash
python scripts/feature_engineering_demo.py
```

This will:
1. Load raw data
2. Create and fit the pipeline
3. Transform data with all features
4. Display WoE/IV analysis
5. Save the pipeline and processed data

### Custom Pipeline Configuration

```python
# Create custom pipeline
from feature_engineering import (
    CustomerAggregateFeatures,
    TemporalFeatureExtractor,
    CategoricalEncoder,
    FeatureScaler
)
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('aggregates', CustomerAggregateFeatures()),
    ('temporal', TemporalFeatureExtractor()),
    ('encoder', CategoricalEncoder(one_hot_threshold=5)),
    ('scaler', FeatureScaler(method='minmax'))
])
```

## Custom Transformers

All transformers inherit from `BaseEstimator` and `TransformerMixin` and follow sklearn conventions:

- `fit(X, y=None)`: Learn parameters from data
- `transform(X)`: Apply transformation
- `fit_transform(X, y=None)`: Combined fit and transform

### CustomerAggregateFeatures

```python
transformer = CustomerAggregateFeatures(
    customer_id_col='CustomerId',
    amount_col='Amount'
)
```

### TemporalFeatureExtractor

```python
transformer = TemporalFeatureExtractor(
    timestamp_col='TransactionStartTime'
)
```

### WoEIVEncoder

```python
transformer = WoEIVEncoder(
    categorical_cols=['ProductCategory', 'ChannelId'],
    target_col='FraudResult',
    iv_threshold=0.02
)

# Get IV report
iv_report = transformer.get_iv_report()
```

## Output

### Original vs Engineered Features

| Metric | Count |
|--------|-------|
| Original Features | 16 |
| Engineered Features | 35+ |
| Aggregate Features | 4 |
| Temporal Features | 7 |
| WoE Features | 5-10 (based on IV) |
| Encoded Features | 10-20 |

### Sample WoE/IV Output

```
Information Value (IV) Summary:
==================================================
ProductCategory                 : 0.4523 (Strong)
ChannelId                       : 0.2145 (Medium)
ProviderId                      : 0.1876 (Medium)
PricingStrategy                 : 0.0567 (Weak)
```

## Saved Artifacts

1. **Feature Pipeline**: `models/feature_pipeline.pkl`
   - Fitted sklearn Pipeline object
   - Reusable for consistent transformations

2. **Processed Data**: `data/processed/featured_data.csv`
   - Fully transformed dataset
   - Ready for model training

3. **IV Report**: Accessible via `WoEIVEncoder.get_iv_report()`

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
category-encoders
joblib
```

## Error Handling

All transformers include comprehensive error handling:
- File I/O exceptions
- Missing column validation
- Data type compatibility
- Unseen category handling in encoding
- Division by zero protection in WoE

## Logging

All operations are logged with:
- `INFO`: Normal progress updates
- `WARNING`: Non-critical issues
- `ERROR`: Critical failures with stack traces

## Benefits

✅ **Reproducible**: Same pipeline for training and prediction  
✅ **Automated**: All transformations in single `.fit_transform()` call  
✅ **Modular**: Add/remove transformers easily  
✅ **Production-Ready**: Serializable sklearn Pipeline  
✅ **Feature Selection**: Built-in IV-based feature importance  
✅ **Comprehensive**: All task requirements implemented  

## Next Steps (Task 4)

The pipeline output is ready for:
- Model training with SMOTE for class imbalance
- Cross-validation
- Hyperparameter tuning
- Model evaluation

## References

- [sklearn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)
- [Weight of Evidence and Information Value](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- [Custom Transformers in sklearn](https://scikit-learn.org/stable/developers/develop.html)
