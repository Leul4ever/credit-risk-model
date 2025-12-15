import pandas as pd
import numpy as np
import os
import logging
import joblib
from typing import Union, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Dict:
    """
    Load a trained model from disk or MLflow with error handling.
    
    Args:
        model_path: Path to the saved model file or MLflow URI
        
    Returns:
        dict: Model data including model object and metadata
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is invalid
    """
    try:
        # Check if it's an MLflow URI
        if model_path.startswith("models:/") or model_path.startswith("runs:/"):
            logger.info(f"Loading model from MLflow URI: {model_path}")
            import mlflow.sklearn
            model = mlflow.sklearn.load_model(model_path)
            
            # For MLflow models, we might not have metadata dict wrapped
            # So we wrap it in our expected structure
            return {
                'model': model,
                'feature_names': getattr(model, 'feature_names_in_', None), # Try to get from sklearn model
                'training_date': 'MLflow Registry',
                'metrics': {}
            }
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        if not model_path.endswith('.pkl'):
            raise ValueError(f"Invalid model file format. Expected .pkl, got: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        
        # Validate model data structure
        if not isinstance(model_data, dict):
            # It might be just the model object
            logger.warning("Loaded object is not a dictionary. Assuming it is the model itself.")
            return {
                'model': model_data,
                'feature_names': getattr(model_data, 'feature_names_in_', None),
                'training_date': 'Unknown',
                'metrics': {}
            }
        
        if 'model' not in model_data:
            raise ValueError("Invalid model format: 'model' key not found")
        
        if 'feature_names' not in model_data:
            logger.warning("Feature names not found in model data")
            model_data['feature_names'] = None
        
        logger.info(f"Model loaded successfully. Training date: {model_data.get('training_date', 'Unknown')}")
        
        return model_data
        
    except EOFError:
        logger.error(f"Corrupted model file: {model_path}")
        raise ValueError(f"Model file appears to be corrupted: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def preprocess_input(data: Union[pd.DataFrame, Dict, List[Dict]], expected_features: List[str] = None) -> pd.DataFrame:
    """
    Preprocess input data for prediction with validation.
    
    Args:
        data: Input data (DataFrame, dict, or list of dicts)
        expected_features: List of expected feature names
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            if not data:
                raise ValueError("Input list is empty")
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
        
        if df.empty:
            raise ValueError("Input data is empty")
        
        logger.info(f"Preprocessing {len(df)} samples for prediction")
        
        # Apply the same feature engineering as training
        try:
            from data_processing import feature_engineering
            df_processed = feature_engineering(df)
        except Exception as e:
            logger.warning(f"Feature engineering failed: {str(e)}. Using raw data.")
            df_processed = df.copy()
        
        # If expected features provided, ensure alignment
        if expected_features is not None:
            # Get numeric columns only
            df_processed = df_processed.select_dtypes(include=[np.number])
            
            # Add missing features with 0
            missing_features = set(expected_features) - set(df_processed.columns)
            for feature in missing_features:
                df_processed[feature] = 0
                logger.debug(f"Added missing feature: {feature}")
            
            # Remove unexpected features
            extra_features = set(df_processed.columns) - set(expected_features)
            if extra_features:
                df_processed = df_processed.drop(columns=list(extra_features))
                logger.debug(f"Removed {len(extra_features)} unexpected features")
            
            # Ensure correct column order
            df_processed = df_processed[expected_features]
        
        # Handle any NaN values
        df_processed = df_processed.fillna(0)
        
        logger.info(f"Preprocessing completed. Shape: {df_processed.shape}")
        return df_processed
        
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise ValueError(f"Failed to preprocess input: {str(e)}")


def predict(model_path: str, data: Union[pd.DataFrame, Dict, List[Dict]], 
            return_proba: bool = True) -> Union[np.ndarray, Dict]:
    """
    Make predictions using a trained model with comprehensive error handling.
    
    Args:
        model_path: Path to the saved model file
        data: Input data for prediction
        return_proba: If True, return probabilities; if False, return class labels
        
    Returns:
        Predictions (array or dict with predictions and probabilities)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If input data is invalid
        Exception: For other prediction errors
    """
    try:
        # Load the model
        model_data = load_model(model_path)
        model = model_data['model']
        feature_names = model_data.get('feature_names')
        
        # Preprocess input data
        X = preprocess_input(data, expected_features=feature_names)
        
        # Make predictions
        logger.info("Making predictions...")
        try:
            predictions = model.predict(X)
            
            if return_proba and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                fraud_proba = probabilities[:, 1]  # Probability of fraud class (1)
                
                results = {
                    'predictions': predictions.tolist(),
                    'fraud_probabilities': fraud_proba.tolist(),
                    'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' 
                                   for p in fraud_proba]
                }
                
                logger.info(f"Predictions completed: {sum(predictions)} fraud cases detected")
                return results
            else:
                logger.info(f"Predictions completed: {sum(predictions)} fraud cases detected")
                return predictions
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise Exception(f"Prediction failed: {str(e)}")


def predict_single(model_path: str, transaction_data: Dict) -> Dict:
    """
    Make prediction for a single transaction with detailed output.
    
    Args:
        model_path: Path to the saved model
        transaction_data: Dictionary containing transaction features
        
    Returns:
        dict: Detailed prediction result
        
    Raises:
        ValueError: If transaction data is invalid
    """
    try:
        if not isinstance(transaction_data, dict):
            raise ValueError("Transaction data must be a dictionary")
        
        if not transaction_data:
            raise ValueError("Transaction data is empty")
        
        logger.info(f"Processing single transaction prediction")
        
        # Make prediction
        result = predict(model_path, transaction_data, return_proba=True)
        
        # Format detailed result
        detailed_result = {
            'is_fraud': bool(result['predictions'][0]),
            'fraud_probability': float(result['fraud_probabilities'][0]),
            'risk_level': result['risk_level'][0],
            'recommendation': 'BLOCK' if result['fraud_probabilities'][0] > 0.7 
                            else 'REVIEW' if result['fraud_probabilities'][0] > 0.3 
                            else 'APPROVE',
            'confidence': float(max(result['fraud_probabilities'][0], 
                                   1 - result['fraud_probabilities'][0]))
        }
        
        logger.info(f"Prediction: {detailed_result['recommendation']} "
                   f"(Fraud prob: {detailed_result['fraud_probability']:.2%})")
        
        return detailed_result
        
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise


def batch_predict(model_path: str, data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Make predictions on a batch of data from CSV file.
    
    Args:
        model_path: Path to the saved model
        data_path: Path to input CSV file
        output_path: Optional path to save predictions
        
    Returns:
        pd.DataFrame: Original data with predictions added
        
    Raises:
        FileNotFoundError: If files don't exist
        Exception: For other errors
    """
    try:
        # Load data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading batch data from {data_path}")
        df = pd.read_csv(data_path)
        
        if df.empty:
            raise ValueError("Input data file is empty")
        
        logger.info(f"Loaded {len(df)} records for batch prediction")
        
        # Make predictions
        results = predict(model_path, df, return_proba=True)
        
        # Add predictions to dataframe
        df['fraud_prediction'] = results['predictions']
        df['fraud_probability'] = results['fraud_probabilities']
        df['risk_level'] = results['risk_level']
        
        # Save if output path provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving predictions: {str(e)}")
                raise IOError(f"Failed to save predictions: {str(e)}")
        
        # Log summary
        fraud_count = sum(results['predictions'])
        logger.info(f"Batch prediction completed: {fraud_count}/{len(df)} flagged as fraud "
                   f"({fraud_count/len(df)*100:.2f}%)")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise


if __name__ == "__main__":
    """Run prediction when script is executed directly."""
    try:
        import sys
        
        # Check if model and data paths provided
        if len(sys.argv) < 3:
            logger.info("Usage: python predict.py <model_path> <data_path> [output_path]")
            logger.info("Example: python predict.py models/fraud_model.pkl data/raw/data.csv predictions.csv")
            sys.exit(1)
        
        model_path = sys.argv[1]
        data_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"
        
        # Run batch prediction
        logger.info("="*60)
        logger.info("BATCH PREDICTION PIPELINE")
        logger.info("="*60)
        
        results_df = batch_predict(model_path, data_path, output_path)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total transactions: {len(results_df)}")
        logger.info(f"Flagged as fraud: {results_df['fraud_prediction'].sum()}")
        logger.info(f"Average fraud probability: {results_df['fraud_probability'].mean():.2%}")
        logger.info(f"\nRisk Level Distribution:")
        logger.info(results_df['risk_level'].value_counts().to_string())
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\nPrediction interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)
