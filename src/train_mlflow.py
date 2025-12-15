"""
Model Training with MLflow Tracking

This module implements comprehensive model training with:
- Multiple model types (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Hyperparameter tuning (Grid Search and Random Search)
- MLflow experiment tracking
- Comprehensive evaluation metrics
- Model registry integration
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# Optional: XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI (local by default)
mlflow.set_tracking_uri("file:./mlruns")


def load_and_prepare_data(
    data_path: str,
    target_column: str = 'is_high_risk',
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data and prepare train/test splits with reproducibility.
    
    Args:
        data_path: Path to the processed data CSV
        target_column: Name of target column
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        use_smote: Whether to apply SMOTE for class imbalance
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("="*70)
    logger.info("DATA PREPARATION")
    logger.info("="*70)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Check target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Separate features and target
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    
    # Drop ID and non-numeric columns
    id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                  'CustomerId', 'TransactionStartTime']
    X = X.drop(columns=[col for col in id_columns if col in X.columns], errors='ignore')
    
    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    # Apply SMOTE if requested
    if use_smote:
        try:
            logger.info("Applying SMOTE to balance training data...")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Train set: {X_train.shape[0]} samples")
            logger.info(f"After SMOTE - Target distribution: {pd.Series(y_train).value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"SMOTE failed: {str(e)}. Proceeding without balancing.")
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC (handle edge cases)
    try:
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = 0.0
            logger.warning("Only one class present, ROC-AUC set to 0.0")
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
        metrics['roc_auc'] = 0.0
    
    return metrics


def train_model_with_mlflow(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    hyperparameters: Optional[Dict] = None,
    experiment_name: str = "credit_risk_modeling"
) -> Dict[str, Any]:
    """
    Train a model and log everything to MLflow.
    
    Args:
        model_name: Name of the model
        model: Model instance
        X_train, X_test, y_train, y_test: Training and test data
        hyperparameters: Dictionary of hyperparameters to log
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with model, predictions, and metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING {model_name.upper()}")
    logger.info(f"{'='*70}")
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name):
        # Log hyperparameters
        if hyperparameters:
            mlflow.log_params(hyperparameters)
        else:
            # Extract hyperparameters from model
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
        
        # Train model
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Log additional info
        mlflow.log_params({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'model_type': model_name
        })
        
        # Log model artifact
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=infer_signature(X_train, y_train)
        )
        
        # Log classification report as artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = f"mlruns/{experiment_name}/{mlflow.active_run().info.run_id}/classification_report.csv"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)
        
        # Print results
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics,
            'run_id': mlflow.active_run().info.run_id
        }


def get_model_configs() -> Dict[str, Dict]:
    """
    Define model configurations for training.
    
    Returns:
        Dictionary of model configurations
    """
    configs = {
        'Logistic Regression': {
            'model': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'hyperparameters': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(
                random_state=42
            ),
            'hyperparameters': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            'hyperparameters': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(
                random_state=42
            ),
            'hyperparameters': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        configs['XGBoost'] = {
            'model': xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss'
            ),
            'hyperparameters': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        configs['LightGBM'] = {
            'model': lgb.LGBMClassifier(
                random_state=42,
                verbose=-1
            ),
            'hyperparameters': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    
    return configs


def hyperparameter_tuning(
    model: Any,
    param_grid: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'grid',
    cv: int = 5,
    scoring: str = 'roc_auc',
    n_iter: int = 20
) -> Any:
    """
    Perform hyperparameter tuning using Grid Search or Random Search.
    
    Args:
        model: Base model instance
        param_grid: Parameter grid for tuning
        X_train, y_train: Training data
        method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_iter: Number of iterations for Random Search
        
    Returns:
        Best model with tuned hyperparameters
    """
    logger.info(f"\nHyperparameter Tuning using {method.upper()} Search...")
    
    if method == 'grid':
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
    else:  # random
        search = RandomizedSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            n_iter=n_iter,
            random_state=42,
            verbose=1
        )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best {scoring} score: {search.best_score_:.4f}")
    
    return search.best_estimator_


def train_all_models(
    data_path: str,
    target_column: str = 'is_high_risk',
    use_hyperparameter_tuning: bool = True,
    tuning_method: str = 'random',
    experiment_name: str = "credit_risk_modeling"
) -> Dict[str, Any]:
    """
    Train multiple models and track with MLflow.
    
    Args:
        data_path: Path to processed data
        target_column: Name of target column
        use_hyperparameter_tuning: Whether to perform hyperparameter tuning
        tuning_method: 'grid' or 'random'
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary of all model results
    """
    logger.info("="*70)
    logger.info("COMPREHENSIVE MODEL TRAINING WITH MLFLOW")
    logger.info("="*70)
    
    # Prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        data_path,
        target_column=target_column,
        random_state=42
    )
    
    # Get model configurations
    model_configs = get_model_configs()
    
    # Train each model
    results = {}
    
    for model_name, config in model_configs.items():
        try:
            model = config['model']
            hyperparams = config.get('hyperparameters', {})
            
            # Hyperparameter tuning if requested
            if use_hyperparameter_tuning and hyperparams:
                model = hyperparameter_tuning(
                    model,
                    hyperparams,
                    X_train,
                    y_train,
                    method=tuning_method
                )
            
            # Train and log to MLflow
            result = train_model_with_mlflow(
                model_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                hyperparameters=model.get_params() if hasattr(model, 'get_params') else None,
                experiment_name=experiment_name
            )
            
            results[model_name] = result
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    return results


def register_best_model(
    experiment_name: str = "credit_risk_modeling",
    metric: str = 'roc_auc',
    model_name: str = "credit_risk_model"
) -> str:
    """
    Find and register the best model in MLflow Model Registry.
    
    Args:
        experiment_name: MLflow experiment name
        metric: Metric to use for selecting best model
        model_name: Name for registered model
        
    Returns:
        Model version URI
    """
    logger.info(f"\n{'='*70}")
    logger.info("REGISTERING BEST MODEL")
    logger.info(f"{'='*70}")
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        raise ValueError("No runs found in experiment")
    
    # Find best run based on metric
    best_run = runs.loc[runs[f'metrics.{metric}'].idxmax()]
    
    logger.info(f"Best model: {best_run['tags.mlflow.runName']}")
    logger.info(f"Best {metric}: {best_run[f'metrics.{metric}']:.4f}")
    logger.info(f"Run ID: {best_run['run_id']}")
    
    # Register model
    model_uri = f"runs:/{best_run['run_id']}/model"
    registered_model = mlflow.register_model(model_uri, model_name)
    
    # Transition to Production
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    logger.info(f"\n✓ Model registered successfully!")
    logger.info(f"  Model Name: {model_name}")
    logger.info(f"  Version: {registered_model.version}")
    logger.info(f"  Stage: Production")
    
    return f"models:/{model_name}/Production"


if __name__ == "__main__":
    """Main execution"""
    import sys
    
    # Default data path
    data_path = "data/processed/data_with_risk_target.csv"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    try:
        # Train all models
        results = train_all_models(
            data_path,
            target_column='is_high_risk',
            use_hyperparameter_tuning=True,
            tuning_method='random',  # Use 'grid' for Grid Search
            experiment_name="credit_risk_modeling"
        )
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING SUMMARY")
        logger.info("="*70)
        
        for model_name, result in results.items():
            if 'error' not in result:
                metrics = result['metrics']
                logger.info(f"\n{model_name}:")
                logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Register best model
        try:
            model_uri = register_best_model(
                experiment_name="credit_risk_modeling",
                metric='roc_auc',
                model_name="credit_risk_model"
            )
            logger.info(f"\nBest model URI: {model_uri}")
        except Exception as e:
            logger.warning(f"Could not register model: {str(e)}")
        
        logger.info("\n" + "="*70)
        logger.info("Training completed successfully!")
        logger.info("View results in MLflow UI: mlflow ui")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

