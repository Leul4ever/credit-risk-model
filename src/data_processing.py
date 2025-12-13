import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file with error handling and validation.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    try:
        # Validate file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at: {filepath}")
        
        # Validate file extension
        if not filepath.endswith('.csv'):
            raise ValueError(f"Invalid file format. Expected CSV file, got: {filepath}")
        
        # Load data
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate data is not empty
        if df.empty:
            raise ValueError("Loaded dataframe is empty")
        
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        raise ValueError("CSV file contains no data")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {str(e)}")
        raise ValueError(f"Invalid CSV format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the data with validation.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
        
    Raises:
        ValueError: If dataframe is invalid
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        logger.info("Starting data cleaning process")
        df_clean = df.copy()
        
        # Handle missing values
        initial_rows = len(df_clean)
        missing_counts = df_clean.isnull().sum()
        
        if missing_counts.any():
            logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
            logger.info(f"Dropped {initial_rows - len(df_clean)} rows with missing values")
        else:
            logger.info("No missing values found")
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if initial_rows > len(df_clean):
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Validate required columns
        required_columns = ['Amount', 'Value', 'FraudResult']
        missing_cols = [col for col in required_columns if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Data type conversions with error handling
        try:
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        except Exception as e:
            logger.warning(f"Error converting numeric columns: {str(e)}")
        
        # Parse datetime if present
        if 'TransactionStartTime' in df_clean.columns:
            try:
                df_clean['TransactionStartTime'] = pd.to_datetime(
                    df_clean['TransactionStartTime'], 
                    errors='coerce'
                )
            except Exception as e:
                logger.warning(f"Error parsing datetime: {str(e)}")
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        raise


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for the model with error handling.
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
        
    Raises:
        ValueError: If required columns are missing
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        logger.info("Starting feature engineering")
        df_features = df.copy()
        
        # Temporal features
        if 'TransactionStartTime' in df_features.columns:
            try:
                df_features['TransactionStartTime'] = pd.to_datetime(df_features['TransactionStartTime'])
                df_features['TransactionHour'] = df_features['TransactionStartTime'].dt.hour
                df_features['TransactionDay'] = df_features['TransactionStartTime'].dt.day
                df_features['TransactionMonth'] = df_features['TransactionStartTime'].dt.month
                df_features['TransactionDayOfWeek'] = df_features['TransactionStartTime'].dt.dayofweek
                logger.info("Created temporal features")
            except Exception as e:
                logger.warning(f"Error creating temporal features: {str(e)}")
        
        # Amount-based features
        try:
            if 'Amount' in df_features.columns and 'Value' in df_features.columns:
                # Amount to Value ratio
                df_features['Amount_Value_Ratio'] = np.where(
                    df_features['Value'] != 0,
                    df_features['Amount'] / df_features['Value'],
                    0
                )
                
                # Transaction direction (credit/debit)
                df_features['IsCredit'] = (df_features['Amount'] > 0).astype(int)
                
                # Amount bins
                df_features['Amount_Bin'] = pd.cut(
                    df_features['Amount'].abs(), 
                    bins=[-np.inf, 100, 1000, 10000, np.inf],
                    labels=['very_small', 'small', 'medium', 'large']
                )
                
                logger.info("Created amount-based features")
        except Exception as e:
            logger.warning(f"Error creating amount features: {str(e)}")
        
        # Customer aggregation features
        try:
            if 'CustomerId' in df_features.columns:
                # Customer transaction count
                customer_counts = df_features.groupby('CustomerId').size().reset_index(name='Customer_Transaction_Count')
                df_features = df_features.merge(customer_counts, on='CustomerId', how='left')
                
                # Customer total value
                customer_total = df_features.groupby('CustomerId')['Value'].sum().reset_index(name='Customer_Total_Value')
                df_features = df_features.merge(customer_total, on='CustomerId', how='left')
                
                # Customer average amount
                customer_avg = df_features.groupby('CustomerId')['Amount'].mean().reset_index(name='Customer_Avg_Amount')
                df_features = df_features.merge(customer_avg, on='CustomerId', how='left')
                
                logger.info("Created customer aggregation features")
        except Exception as e:
            logger.warning(f"Error creating customer features: {str(e)}")
        
        # One-hot encoding for categorical variables
        try:
            categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId']
            available_cats = [col for col in categorical_cols if col in df_features.columns]
            
            if available_cats:
                df_features = pd.get_dummies(df_features, columns=available_cats, drop_first=True)
                logger.info(f"One-hot encoded {len(available_cats)} categorical columns")
        except Exception as e:
            logger.warning(f"Error in one-hot encoding: {str(e)}")
        
        logger.info(f"Feature engineering completed. Final shape: {df_features.shape}")
        return df_features
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise


def prepare_train_test_split(
    df: pd.DataFrame, 
    target_column: str = 'FraudResult',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare features and target, then split into train/test sets.
    
    Args:
        df: Dataframe with features
        target_column: Name of target column
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        ValueError: If target column missing or invalid parameters
    """
    try:
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        logger.info(f"Preparing train/test split with test_size={test_size}")
        
        # Separate features and target
        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()
        
        # Drop non-numeric or ID columns
        id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                     'CustomerId', 'TransactionStartTime']
        X = X.drop(columns=[col for col in id_columns if col in X.columns])
        
        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Train fraud rate: {y_train.mean():.4f}, Test fraud rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error in train/test split: {str(e)}")
        raise


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataframe to CSV with error handling.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the file
        
    Raises:
        IOError: If save fails
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
    except PermissionError:
        logger.error(f"Permission denied: Cannot write to {output_path}")
        raise IOError(f"Permission denied: {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise IOError(f"Failed to save data: {str(e)}")
