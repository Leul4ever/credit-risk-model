import pandas as pd
import numpy as np
import os
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(data_path: str, model_save_path: str = 'models/fraud_model.pkl'):
    """
    Train the credit risk model with error handling and validation.
    
    Args:
        data_path: Path to the training data CSV
        model_save_path: Path to save the trained model
        
    Returns:
        dict: Training results and metrics
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is invalid
        Exception: For other training errors
    """
    try:
        # Import data processing functions
        from data_processing import load_data, clean_data, feature_engineering, prepare_train_test_split
        
        logger.info("="*60)
        logger.info("Starting model training pipeline")
        logger.info("="*60)
        
        # Load and prepare data
        logger.info(f"Loading data from {data_path}")
        df = load_data(data_path)
        
        logger.info("Cleaning data...")
        df_clean = clean_data(df)
        
        logger.info("Engineering features...")
        df_features = feature_engineering(df_clean)
        
        # Prepare train/test split
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df_features, 
            target_column='FraudResult',
            test_size=0.2,
            random_state=42
        )
        
        # Handle class imbalance with SMOTE
        logger.info("Handling class imbalance with SMOTE...")
        try:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            logger.info(f"Original train set: {X_train.shape}")
            logger.info(f"Balanced train set: {X_train_balanced.shape}")
            logger.info(f"Fraud rate after SMOTE: {y_train_balanced.mean():.4f}")
        except Exception as e:
            logger.warning(f"SMOTE failed: {str(e)}. Proceeding without balancing.")
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train Random Forest model
        logger.info("Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit the model
        model.fit(X_train_balanced, y_train_balanced)
        logger.info("Model training completed")
        
        # Make predictions on test set
        logger.info("Evaluating model on test set...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            roc_auc = None
        
        f1 = f1_score(y_test, y_pred)
        
        # Print evaluation results
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"\nROC AUC Score: {roc_auc:.4f}" if roc_auc else "\nROC AUC: N/A")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:")
        logger.info("\n" + str(confusion_matrix(y_test, y_pred)))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        logger.info("\n" + str(feature_importance.head(10)))
        
        # Save the model
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            
            # Save model and feature names
            model_data = {
                'model': model,
                'feature_names': list(X_train.columns),
                'training_date': datetime.now().isoformat(),
                'metrics': {
                    'roc_auc': float(roc_auc) if roc_auc else None,
                    'f1_score': float(f1)
                }
            }
            
            joblib.dump(model_data, model_save_path)
            logger.info(f"\nModel saved successfully to {model_save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise IOError(f"Failed to save model: {str(e)}")
        
        # Return results
        results = {
            'success': True,
            'model_path': model_save_path,
            'metrics': {
                'roc_auc': float(roc_auc) if roc_auc else None,
                'f1_score': float(f1),
                'train_samples': len(X_train_balanced),
                'test_samples': len(X_test)
            },
            'feature_importance': feature_importance.to_dict('records')[:10]
        }
        
        logger.info("="*60)
        logger.info("Training pipeline completed successfully!")
        logger.info("="*60)
        
        return results
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Invalid data: {str(e)}")
        raise
    except MemoryError:
        logger.error("Not enough memory to train the model")
        raise MemoryError("Insufficient memory for model training")
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise Exception(f"Training failed: {str(e)}")


def train_baseline_models(data_path: str):
    """
    Train multiple baseline models for comparison with error handling.
    
    Args:
        data_path: Path to the training data
        
    Returns:
        dict: Results for each model
    """
    try:
        from data_processing import load_data, clean_data, feature_engineering, prepare_train_test_split
        
        logger.info("Training baseline models for comparison...")
        
        # Load and prepare data
        df = load_data(data_path)
        df_clean = clean_data(df)
        df_features = feature_engineering(df_clean)
        X_train, X_test, y_train, y_test = prepare_train_test_split(df_features)
        
        # Apply SMOTE
        try:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        except:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            try:
                logger.info(f"\nTraining {name}...")
                model.fit(X_train_balanced, y_train_balanced)
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                results[name] = {
                    'f1_score': float(f1_score(y_test, y_pred)),
                    'roc_auc': float(roc_auc_score(y_test, y_pred_proba)) if y_pred_proba is not None else None
                }
                
                logger.info(f"{name} - F1: {results[name]['f1_score']:.4f}, ROC AUC: {results[name]['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        logger.error(f"Error in baseline model training: {str(e)}")
        raise


if __name__ == "__main__":
    """Run training pipeline when script is executed directly."""
    try:
        import sys
        
        # Get data path from command line or use default
        if len(sys.argv) > 1:
            data_path = sys.argv[1]
        else:
            data_path = "data/raw/data.csv"
        
        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            logger.info("Usage: python train.py [path_to_data.csv]")
            sys.exit(1)
        
        # Train the model
        results = train_model(data_path)
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Success: {results['success']}")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Test ROC AUC: {results['metrics']['roc_auc']:.4f}" if results['metrics']['roc_auc'] else "ROC AUC: N/A")
        logger.info(f"Test F1 Score: {results['metrics']['f1_score']:.4f}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
