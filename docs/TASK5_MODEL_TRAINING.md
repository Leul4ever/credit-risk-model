# Task 5 - Model Training and Tracking

## Overview

This module implements a robust, reproducible model training pipeline with experiment tracking using **MLflow**. It systematically trains, tunes, and evaluates multiple machine learning algorithms to predict credit risk.

## Key Components

### 1. Data Preparation ✅
- **Splitting**: 80/20 Train/Test split (`random_state=42`)
- **Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) applied to training data to handle class imbalance.
- **Preprocessing**: Integration with feature engineering pipeline.

### 2. Model Selection ✅
Training and evaluating the following algorithms:
- **Logistic Regression**: Baseline linear model
- **Decision Tree**: interpretable non-linear model
- **Random Forest**: Ensemble bagging model (robust to overfitting)
- **Gradient Boosting**: Ensemble boosting model (high performance)

### 3. Hyperparameter Tuning ✅
- **Method**: Random Search (`RandomizedSearchCV`)
- **Cross-Validation**: 5-fold CV to ensure generalizability
- **Optimization Metric**: ROC-AUC (Area Under Receiver Operating Characteristic Curve)

### 4. Experiment Tracking (MLflow) ✅
All runs are logged to MLflow including:
- **Parameters**: Hyperparameters values (e.g., `n_estimators`, `max_depth`)
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Artifacts**: Trained model object, Classification report (CSV)
- **Tags**: Model type, dataset info

### 5. Model Registry ✅
- Automatic selection of the best model based on ROC-AUC.
- Registration of the best model in the MLflow Model Registry for version control.

## Usage

### Run Training

```bash
python src/train_mlflow.py
```

### View Results

Start the MLflow UI to compare runs:
```bash
mlflow ui
```
Then navigate to `http://localhost:5000` in your browser.

## Unit Tests

Automated tests in `tests/test_data_processing.py` verify:
- Data loading and error handling
- Data cleaning logic
- Feature engineering correctness (aggregations, encoding)

Run tests with:
```bash
pytest tests/test_data_processing.py -v
```

## Results Summary

*(Results will be populated after training completes)*

| Model | ROC-AUC | F1 Score | Accuracy |
|-------|---------|----------|----------|
| Logistic Regression | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD |

**Best Model**: TBD

## Requirements
- `mlflow`
- `scikit-learn==1.5.2`
- `imbalanced-learn==0.12.4`
- `pytest`
