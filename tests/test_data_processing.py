"""
Unit tests for data processing and feature engineering modules.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import clean_data, load_data
from feature_engineering import create_feature_engineering_pipeline

@pytest.fixture
def sample_data():
    """Create a sample dataframe for testing."""
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'TransactionStartTime': [
            '2019-02-01 00:00:00', '2019-02-01 12:00:00',
            '2019-02-02 00:00:00', '2019-02-02 12:00:00',
            '2019-02-03 00:00:00'
        ],
        'Amount': [100.0, 200.0, 50.0, 150.0, 300.0],
        'Value': [1000.0, 2000.0, 500.0, 1500.0, 3000.0],
        'FraudResult': [0, 0, 0, 0, 1],
        'ProductCategory': ['Financial', 'Airtime', 'Financial', 'Data', 'Financial'],
        'ChannelId': ['Channel1', 'Channel2', 'Channel1', 'Channel1', 'Channel2']
    }
    return pd.DataFrame(data)

def test_clean_data(sample_data):
    """Test data cleaning function."""
    # Introduce messy data
    messy_data = sample_data.copy()
    
    # Add duplicate row
    messy_data = pd.concat([messy_data, messy_data.iloc[[0]]], ignore_index=True)
    assert len(messy_data) == 6
    
    # Test cleaning
    cleaned_df = clean_data(messy_data)
    
    # Should remove duplicate
    assert len(cleaned_df) == 5
    # Should convert timestamp
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['TransactionStartTime'])

def test_feature_engineering_pipeline(sample_data):
    """Test the complete feature engineering pipeline."""
    # Prepare data
    df = sample_data.copy()
    
    # Create pipeline
    pipeline = create_feature_engineering_pipeline(
        include_woe=False,  # Skip WoE for simple test as it requires target
        imputation_strategy='mean',
        scaling_method=None  # Disable scaling for value verification
    )
    
    # Transform
    df_transformed = pipeline.fit_transform(df)
    
    # Verify features created
    columns = df_transformed.columns
    
    # 1. Aggregate features
    assert 'total_amount' in columns
    assert 'avg_amount' in columns
    assert 'transaction_count' in columns
    
    # 2. Temporal features
    assert 'transaction_hour' in columns
    assert 'transaction_day' in columns
    
    # 3. Encoding (One-Hot)
    # ChannelId has >1 unique values, so it should be one-hot encoded
    # Check for encoded columns (prefix + value)
    encoded_cols = [col for col in columns if 'ChannelId_' in col]
    assert len(encoded_cols) > 0
    
    # 4. Check row count preserved
    assert len(df_transformed) == len(df)
    
    # 5. Check customer values are correct
    # C1 has amounts 100, 200. Total should be 300.
    # Note: CustomerId gets one-hot encoded because it has low cardinality in this sample
    if 'CustomerId' in df_transformed.columns:
        c1_row = df_transformed[df_transformed['CustomerId'] == 'C1']
    elif 'CustomerId_C1' in df_transformed.columns:
        c1_row = df_transformed[df_transformed['CustomerId_C1'] == 1]
    else:
        # Fallback if label encoded or other
        c1_row = df_transformed.iloc[[0]] # Just pick first row which we know is C1
        
    c1_total = c1_row['total_amount'].iloc[0]
    assert c1_total == 300.0

def test_missing_value_handling():
    """Test correct handling of missing values."""
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C2', 'C3'],
        'Amount': [100.0, np.nan, 300.0],
        'TransactionStartTime': ['2019-01-01', '2019-01-02', '2019-01-03']
    })
    
    pipeline = create_feature_engineering_pipeline(include_woe=False)
    df_transformed = pipeline.fit_transform(df)
    
    # Missing amount should be filled (mean = 200)
    filled_amount = df_transformed['Amount'].iloc[1]
    assert not np.isnan(filled_amount)
    # Scaled value won't be exactly 200, but it shouldn't be NaN
