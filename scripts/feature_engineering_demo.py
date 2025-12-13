"""
Demo script to showcase the Feature Engineering Pipeline for Task 3.

This script demonstrates:
1. Loading raw data
2. Creating the automated pipeline
3. Fitting and transforming data
4. Analyzing feature importance with WoE/IV
5. Saving the pipeline for reuse
"""

import pandas as pd
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import (
    create_feature_engineering_pipeline,
    save_pipeline,
    load_pipeline
)
from data_processing import load_data, clean_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive feature engineering pipeline demo."""
    
    try:
        print("=" * 80)
        print("TASK 3 - FEATURE ENGINEERING PIPELINE DEMONSTRATION")
        print("=" * 80)
        print()
        
        # Step 1: Load Data
        print("Step 1: Loading Raw Data")
        print("-" * 80)
        data_path = "data/raw/data.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = load_data(data_path)
        print(f"✓ Loaded {len(df)} transactions with {len(df.columns)} features")
        print(f"  Original columns: {list(df.columns[:5])}... (showing first 5)")
        print()
        
        # Step 2: Basic Cleaning
        print("Step 2: Basic Data Cleaning")
        print("-" * 80)
        df_clean = clean_data(df)
        print(f"✓ Cleaned data shape: {df_clean.shape}")
        print(f"  Missing values: {df_clean.isnull().sum().sum()}")
        print()
        
        # Step 3: Create Pipeline
        print("Step 3: Creating Feature Engineering Pipeline")
        print("-" * 80)
        print("Pipeline Components:")
        print("  1. Missing Value Handler - Imputation")
        print("  2. Temporal Feature Extractor - Hour, Day, Month, Year")
        print("  3. Customer Aggregate Features - Total, Avg, Count, Std")
        print("  4. WoE/IV Encoder - Weight of Evidence transformation")
        print("  5. Categorical Encoder - One-hot + Label encoding")
        print("  6. Feature Scaler - Standardization")
        print()
        
        pipeline = create_feature_engineering_pipeline(
            include_woe=True,
            scaling_method='standard',
            imputation_strategy='mean'
        )
        print("✓ Pipeline created successfully")
        print()
        
        # Step 4: Fit and Transform
        print("Step 4: Fitting Pipeline on Training Data")
        print("-" * 80)
        df_transformed = pipeline.fit_transform(df_clean)
        print(f"✓ Pipeline fitted and data transformed")
        print()
        
        # Step 5: Analyze Results
        print("Step 5: Transformation Results")
        print("-" * 80)
        print(f"Original Features: {df_clean.shape[1]}")
        print(f"Engineered Features: {df_transformed.shape[1]}")
        print(f"New Features Added: {df_transformed.shape[1] - df_clean.shape[1]}")
        print()
        
        print("Feature Categories:")
        print(f"  • Aggregate Features: 4 (total, avg, count, std per customer)")
        print(f"  • Temporal Features: 7 (hour, day, month, year, dayofweek, weekend, time_of_day)")
        print(f"  • WoE Encoded Features: {sum(1 for col in df_transformed.columns if '_woe' in col)}")
        print(f"  • One-Hot Encoded: {sum(1 for col in df_transformed.columns if '_' in col and col not in df_clean.columns)}")
        print()
        
        # Step 6: WoE/IV Analysis
        print("Step 6: Weight of Evidence (WoE) and Information Value (IV) Analysis")
        print("-" * 80)
        woe_transformer = pipeline.named_steps['woe_encoding']
        iv_report = woe_transformer.get_iv_report()
        
        print("\nTop 10 Features by Information Value:")
        print(iv_report.head(10).to_string(index=False))
        print()
        
        print("IV Interpretation:")
        print("  < 0.02: Useless for prediction")
        print("  0.02 - 0.1: Weak predictive power")
        print("  0.1 - 0.3: Medium predictive power")
        print("  0.3 - 0.5: Strong predictive power")
        print("  > 0.5: Very strong (but check for overfitting)")
        print()
        
        # Step 7: Sample Transformed Data
        print("Step 7: Sample of Transformed Data")
        print("-" * 80)
        # Show numeric columns only for clarity
        numeric_cols = df_transformed.select_dtypes(include=['number']).columns[:10]
        print(df_transformed[numeric_cols].head().to_string())
        print("\n(Showing first 10 numeric columns)")
        print()
        
        # Step 8: Save Pipeline
        print("Step 8: Saving Pipeline for Production Use")
        print("-" * 80)
        pipeline_path = "models/feature_pipeline.pkl"
        save_pipeline(pipeline, pipeline_path)
        print(f"✓ Pipeline saved to: {pipeline_path}")
        print()
        
        # Step 9: Test Loading Pipeline
        print("Step 9: Testing Pipeline Loading")
        print("-" * 80)
        loaded_pipeline = load_pipeline(pipeline_path)
        print("✓ Pipeline loaded successfully")
        
        # Test transform with loaded pipeline
        test_sample = df_clean.head(10)
        test_transformed = loaded_pipeline.transform(test_sample)
        print(f"✓ Test transformation successful: {test_transformed.shape}")
        print()
        
        # Step 10: Save Processed Data
        print("Step 10: Saving Processed Data")
        print("-" * 80)
        output_path = "data/processed/featured_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_transformed.to_csv(output_path, index=False)
        print(f"✓ Processed data saved to: {output_path}")
        print(f"  Rows: {len(df_transformed)}")
        print(f"  Columns: {len(df_transformed.columns)}")
        print()
        
        # Summary
        print("=" * 80)
        print("TASK 3 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n✅ Feature Engineering Pipeline Summary:")
        print(f"   • Automated preprocessing with sklearn Pipeline")
        print(f"   • {df_transformed.shape[1] - df_clean.shape[1]} new features engineered")
        print(f"   • WoE/IV analysis for feature selection")
        print(f"   • Pipeline saved for reproducible transformations")
        print(f"   • Ready for model training (Task 4)")
        print("\n" + "=" * 80)
        
        return df_transformed, pipeline, iv_report
        
    except Exception as e:
        logger.error(f"Pipeline demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    df_transformed, pipeline, iv_report = main()
