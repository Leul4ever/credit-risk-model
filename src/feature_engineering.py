"""
Feature Engineering Module for Credit Risk Model

This module provides a comprehensive, automated, and reproducible feature engineering
pipeline using sklearn.pipeline.Pipeline to transform raw data into model-ready format.

Features:
- Aggregate features per customer
- Temporal feature extraction
- Categorical encoding
- Missing value handling
- Feature scaling/normalization
- Weight of Evidence (WoE) and Information Value (IV) transformations
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from datetime import datetime

# Sklearn imports
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

# For WoE and IV
try:
    from category_encoders import WOEEncoder
    WOE_AVAILABLE = True
except ImportError:
    WOE_AVAILABLE = False
    logging.warning("category_encoders not available. WoE encoding will use custom implementation.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Create aggregate features per customer:
    - Total Transaction Amount
    - Average Transaction Amount  
    - Transaction Count
    - Standard Deviation of Transaction Amounts
    """
    
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.aggregate_stats = None
        
    def fit(self, X, y=None):
        """Calculate aggregate statistics per customer."""
        try:
            logger.info("Calculating customer aggregate features...")
            
            # Calculate aggregates
            self.aggregate_stats = X.groupby(self.customer_id_col)[self.amount_col].agg([
                ('total_amount', 'sum'),
                ('avg_amount', 'mean'),
                ('transaction_count', 'count'),
                ('std_amount', 'std')
            ]).reset_index()
            
            # Fill NaN in std (for customers with single transaction)
            self.aggregate_stats['std_amount'] = self.aggregate_stats['std_amount'].fillna(0)
            
            logger.info(f"Aggregate features calculated for {len(self.aggregate_stats)} unique customers")
            return self
            
        except Exception as e:
            logger.error(f"Error in CustomerAggregateFeatures.fit: {str(e)}")
            raise
    
    def transform(self, X):
        """Add aggregate features to dataset."""
        try:
            if self.aggregate_stats is None:
                raise ValueError("Transformer not fitted. Call fit() first.")
            
            X_transformed = X.copy()
            
            # Merge aggregate features
            X_transformed = X_transformed.merge(
                self.aggregate_stats,
                on=self.customer_id_col,
                how='left'
            )
            
            # Fill any missing values with 0 (for new customers not in training)
            agg_columns = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount']
            for col in agg_columns:
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].fillna(0)
            
            logger.info(f"Added {len(agg_columns)} aggregate features")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in CustomerAggregateFeatures.transform: {str(e)}")
            raise


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from transaction timestamp:
    - Transaction Hour
    - Transaction Day
    - Transaction Month
    - Transaction Year
    - Day of Week
    - Is Weekend
    """
    
    def __init__(self, timestamp_col='TransactionStartTime'):
        self.timestamp_col = timestamp_col
        
    def fit(self, X, y=None):
        """No fitting required for temporal extraction."""
        return self
    
    def transform(self, X):
        """Extract temporal features."""
        try:
            X_transformed = X.copy()
            
            if self.timestamp_col not in X_transformed.columns:
                logger.warning(f"{self.timestamp_col} not found. Skipping temporal features.")
                return X_transformed
            
            logger.info("Extracting temporal features...")
            
            # Convert to datetime if not already
            X_transformed[self.timestamp_col] = pd.to_datetime(
                X_transformed[self.timestamp_col], 
                errors='coerce'
            )
            
            # Extract features
            X_transformed['transaction_hour'] = X_transformed[self.timestamp_col].dt.hour
            X_transformed['transaction_day'] = X_transformed[self.timestamp_col].dt.day
            X_transformed['transaction_month'] = X_transformed[self.timestamp_col].dt.month
            X_transformed['transaction_year'] = X_transformed[self.timestamp_col].dt.year
            X_transformed['transaction_dayofweek'] = X_transformed[self.timestamp_col].dt.dayofweek
            X_transformed['is_weekend'] = (X_transformed['transaction_dayofweek'] >= 5).astype(int)
            
            # Add time-based bins
            X_transformed['time_of_day'] = pd.cut(
                X_transformed['transaction_hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
            
            logger.info("Added 7 temporal features")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in TemporalFeatureExtractor.transform: {str(e)}")
            raise


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values using:
    - Simple imputation (mean, median, mode)
    - KNN imputation
    - Removal (optional)
    """
    
    def __init__(self, strategy='mean', use_knn=False, n_neighbors=5, threshold=0.5):
        """
        Args:
            strategy: 'mean', 'median', 'most_frequent', or 'constant'
            use_knn: Whether to use KNN imputation
            n_neighbors: Number of neighbors for KNN
            threshold: Drop columns with missing ratio above this threshold
        """
        self.strategy = strategy
        self.use_knn = use_knn
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.imputer = None
        self.columns_to_drop = []
        
    def fit(self, X, y=None):
        """Fit imputer on data."""
        try:
            logger.info("Fitting missing value handler...")
            
            # Identify columns with too many missing values
            missing_ratios = X.isnull().sum() / len(X)
            self.columns_to_drop = missing_ratios[missing_ratios > self.threshold].index.tolist()
            
            if self.columns_to_drop:
                logger.warning(f"Dropping {len(self.columns_to_drop)} columns with >{self.threshold*100}% missing")
            
            # Select numeric columns for imputation
            X_numeric = X.select_dtypes(include=[np.number])
            
            # Initialize imputer
            if self.use_knn:
                self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            else:
                self.imputer = SimpleImputer(strategy=self.strategy)
            
            # Fit imputer
            if not X_numeric.empty:
                self.imputer.fit(X_numeric)
            
            return self
            
        except Exception as e:
            logger.error(f"Error in MissingValueHandler.fit: {str(e)}")
            raise
    
    def transform(self, X):
        """Impute missing values."""
        try:
            X_transformed = X.copy()
            
            # Drop columns with too many missing values
            if self.columns_to_drop:
                X_transformed = X_transformed.drop(columns=self.columns_to_drop, errors='ignore')
            
            # Impute numeric columns
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0 and self.imputer is not None:
                X_transformed[numeric_cols] = self.imputer.transform(X_transformed[numeric_cols])
            
            # Fill remaining categorical missing values with mode or 'missing'
            categorical_cols = X_transformed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if X_transformed[col].isnull().any():
                    mode_value = X_transformed[col].mode()
                    if len(mode_value) > 0:
                        X_transformed[col] = X_transformed[col].fillna(mode_value[0])
                    else:
                        X_transformed[col] = X_transformed[col].fillna('missing')
            
            logger.info("Missing values handled")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in MissingValueHandler.transform: {str(e)}")
            raise


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables using:
    - One-Hot Encoding for low-cardinality features
    - Label Encoding for high-cardinality features
    """
    
    def __init__(self, one_hot_threshold=10, drop_first=True):
        """
        Args:
            one_hot_threshold: Use one-hot if unique values <= threshold, else label encoding
            drop_first: Drop first category in one-hot to avoid multicollinearity
        """
        self.one_hot_threshold = one_hot_threshold
        self.drop_first = drop_first
        self.one_hot_cols = []
        self.label_encode_cols = []
        self.label_encoders = {}
        self.one_hot_encoder = None
        
    def fit(self, X, y=None):
        """Determine encoding strategy and fit encoders."""
        try:
            logger.info("Fitting categorical encoders...")
            
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                n_unique = X[col].nunique()
                
                if n_unique <= self.one_hot_threshold:
                    self.one_hot_cols.append(col)
                else:
                    self.label_encode_cols.append(col)
                    # Fit label encoder
                    le = LabelEncoder()
                    le.fit(X[col].astype(str))
                    self.label_encoders[col] = le
            
            logger.info(f"One-hot encoding: {len(self.one_hot_cols)} columns")
            logger.info(f"Label encoding: {len(self.label_encode_cols)} columns")
            
            return self
            
        except Exception as e:
            logger.error(f"Error in CategoricalEncoder.fit: {str(e)}")
            raise
    
    def transform(self, X):
        """Apply encoding transformations."""
        try:
            X_transformed = X.copy()
            
            # Label encoding
            for col in self.label_encode_cols:
                if col in X_transformed.columns:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    X_transformed[col] = X_transformed[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
            
            # One-hot encoding
            if self.one_hot_cols:
                X_transformed = pd.get_dummies(
                    X_transformed,
                    columns=self.one_hot_cols,
                    drop_first=self.drop_first,
                    prefix=self.one_hot_cols
                )
            
            logger.info("Categorical encoding completed")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in CategoricalEncoder.transform: {str(e)}")
            raise


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Normalize or standardize numerical features:
    - StandardScaler: mean=0, std=1
    - MinMaxScaler: range [0, 1]
    """
    
    def __init__(self, method='standard', exclude_cols=None):
        """
        Args:
            method: 'standard' or 'minmax'
            exclude_cols: List of columns to exclude from scaling
        """
        self.method = method
        self.exclude_cols = exclude_cols or []
        self.scaler = None
        self.scale_cols = []
        
    def fit(self, X, y=None):
        """Fit scaler on numerical columns."""
        try:
            logger.info(f"Fitting {self.method} scaler...")
            
            # Select numerical columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            self.scale_cols = [col for col in numeric_cols if col not in self.exclude_cols]
            
            if not self.scale_cols or self.method in [None, 'none']:
                logger.warning("Scaling disabled or no numerical columns")
                return self
            
            # Initialize scaler
            if self.method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            
            # Fit scaler
            self.scaler.fit(X[self.scale_cols])
            
            logger.info(f"Fitted scaler on {len(self.scale_cols)} columns")
            return self
            
        except Exception as e:
            logger.error(f"Error in FeatureScaler.fit: {str(e)}")
            raise
    
    def transform(self, X):
        """Scale numerical features."""
        try:
            if not self.scale_cols or self.scaler is None:
                return X.copy()
            
            X_transformed = X.copy()
            X_transformed[self.scale_cols] = self.scaler.transform(X[self.scale_cols])
            
            logger.info(f"Scaled {len(self.scale_cols)} numerical features")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in FeatureScaler.transform: {str(e)}")
            raise


class WoEIVEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) and Information Value (IV) transformation.
    
    WoE measures the strength of a category in separating good vs bad outcomes.
    IV measures the predictive power of a feature.
    
    WoE = ln(% of goods / % of bads)
    IV = Σ (% of goods - % of bads) * WoE
    """
    
    def __init__(self, categorical_cols=None, target_col='FraudResult', iv_threshold=0.02):
        """
        Args:
            categorical_cols: List of categorical columns to encode
            target_col: Binary target variable
            iv_threshold: Minimum IV to keep feature (0.02 = weak, 0.3 = strong)
        """
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.iv_threshold = iv_threshold
        self.woe_dict = {}
        self.iv_dict = {}
        self.selected_features = []
        
    def _calculate_woe_iv(self, X, feature):
        """Calculate WoE and IV for a feature."""
        try:
            df = pd.DataFrame({
                'feature': X[feature],
                'target': X[self.target_col]
            })
            
            # Group by feature value
            grouped = df.groupby('feature')['target'].agg(['count', 'sum'])
            grouped.columns = ['total', 'bad']
            grouped['good'] = grouped['total'] - grouped['bad']
            
            # Calculate distributions
            grouped['dist_good'] = grouped['good'] / grouped['good'].sum()
            grouped['dist_bad'] = grouped['bad'] / grouped['bad'].sum()
            
            # Avoid division by zero
            grouped['dist_good'] = grouped['dist_good'].replace(0, 0.0001)
            grouped['dist_bad'] = grouped['dist_bad'].replace(0, 0.0001)
            
            # Calculate WoE
            grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
            
            # Calculate IV
            grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']
            
            total_iv = grouped['iv'].sum()
            
            return grouped['woe'].to_dict(), total_iv
            
        except Exception as e:
            logger.warning(f"Error calculating WoE/IV for {feature}: {str(e)}")
            return {}, 0.0
    
    def fit(self, X, y=None):
        """Calculate WoE and IV for all categorical features."""
        try:
            if self.target_col not in X.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in data")
            
            logger.info("Calculating WoE and IV...")
            
            # Determine categorical columns if not specified
            if self.categorical_cols is None:
                self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Remove target from categorical columns if present
            if self.target_col in self.categorical_cols:
                self.categorical_cols.remove(self.target_col)
            
            # Calculate WoE and IV for each feature
            for col in self.categorical_cols:
                if col in X.columns:
                    woe, iv = self._calculate_woe_iv(X, col)
                    self.woe_dict[col] = woe
                    self.iv_dict[col] = iv
                    
                    if iv >= self.iv_threshold:
                        self.selected_features.append(col)
            
            # Log IV values
            logger.info("\nInformation Value (IV) Summary:")
            logger.info("=" * 50)
            for col, iv in sorted(self.iv_dict.items(), key=lambda x: x[1], reverse=True):
                strength = 'Strong' if iv > 0.3 else 'Medium' if iv > 0.1 else 'Weak'
                logger.info(f"{col:30s}: {iv:.4f} ({strength})")
            
            logger.info(f"\nSelected {len(self.selected_features)} features with IV >= {self.iv_threshold}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error in WoEIVEncoder.fit: {str(e)}")
            raise
    
    def transform(self, X):
        """Transform categorical features using WoE."""
        try:
            X_transformed = X.copy()
            
            for col in self.selected_features:
                if col in X_transformed.columns:
                    woe_map = self.woe_dict[col]
                    # Map values to WoE, explicitly convert to float, use 0 for unseen categories
                    X_transformed[f'{col}_woe'] = X_transformed[col].map(woe_map).astype(float).fillna(0)
                    # Optionally drop original column
                    # X_transformed = X_transformed.drop(columns=[col])
            
            logger.info(f"WoE transformation applied to {len(self.selected_features)} features")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in WoEIVEncoder.transform: {str(e)}")
            raise
    
    def get_iv_report(self) -> pd.DataFrame:
        """Get Information Value report."""
        iv_df = pd.DataFrame({
            'Feature': list(self.iv_dict.keys()),
            'IV': list(self.iv_dict.values())
        }).sort_values('IV', ascending=False)
        
        iv_df['Predictive_Power'] = iv_df['IV'].apply(
            lambda x: 'Useless' if x < 0.02 else
                     'Weak' if x < 0.1 else
                     'Medium' if x < 0.3 else
                     'Strong' if x < 0.5 else
                     'Very Strong'
        )
        
        return iv_df


def create_feature_engineering_pipeline(
    include_woe=True,
    scaling_method='standard',
    imputation_strategy='mean'
) -> Pipeline:
    """
    Create a complete feature engineering pipeline.
    
    Args:
        include_woe: Whether to include WoE/IV transformation
        scaling_method: 'standard' or 'minmax'
        imputation_strategy: 'mean', 'median', or 'most_frequent'
    
    Returns:
        sklearn Pipeline object
    """
    steps = [
        ('missing_values', MissingValueHandler(strategy=imputation_strategy)),
        ('temporal_features', TemporalFeatureExtractor()),
        ('customer_aggregates', CustomerAggregateFeatures()),
    ]
    
    if include_woe:
        steps.append(('woe_encoding', WoEIVEncoder()))
    
    steps.extend([
        ('categorical_encoding', CategoricalEncoder()),
        ('feature_scaling', FeatureScaler(method=scaling_method))
    ])
    
    pipeline = Pipeline(steps)
    
    logger.info("Feature engineering pipeline created with steps:")
    for i, (name, _) in enumerate(pipeline.steps, 1):
        logger.info(f"  {i}. {name}")
    
    return pipeline


def save_pipeline(pipeline, filepath='models/feature_pipeline.pkl'):
    """Save fitted pipeline to disk."""
    import joblib
    import os
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(pipeline, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pipeline: {str(e)}")
        raise


def load_pipeline(filepath='models/feature_pipeline.pkl'):
    """Load fitted pipeline from disk."""
    import joblib
    
    try:
        pipeline = joblib.load(filepath)
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    """Test the feature engineering pipeline."""
    import sys
    
    try:
        # Load data
        from data_processing import load_data
        
        data_path = "data/raw/data.csv"
        df = load_data(data_path)
        
        logger.info("="*60)
        logger.info("TESTING FEATURE ENGINEERING PIPELINE")
        logger.info("="*60)
        
        # Create and fit pipeline
        pipeline = create_feature_engineering_pipeline(
            include_woe=True,
            scaling_method='standard',
            imputation_strategy='mean'
        )
        
        # Fit and transform
        logger.info("\nFitting pipeline...")
        df_transformed = pipeline.fit_transform(df)
        
        logger.info(f"\nOriginal shape: {df.shape}")
        logger.info(f"Transformed shape: {df_transformed.shape}")
        logger.info(f"New features added: {df_transformed.shape[1] - df.shape[1]}")
        
        # Save pipeline
        save_pipeline(pipeline)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE TEST COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        sys.exit(1)
