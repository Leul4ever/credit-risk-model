"""
Proxy Target Variable Engineering Module

This module creates a credit risk target variable using RFM (Recency, Frequency, Monetary)
analysis and K-Means clustering to identify high-risk customers.

High-risk customers are defined as those who are disengaged, with low frequency
and low monetary value, indicating higher likelihood of default.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional

# Sklearn imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RFMAnalyzer:
    """
    RFM (Recency, Frequency, Monetary) Analysis for customer segmentation.
    
    Calculates:
    - Recency: Days since last transaction
    - Frequency: Total number of transactions
    - Monetary: Total transaction value
    """
    
    def __init__(self, snapshot_date: Optional[str] = None):
        """
        Initialize RFM Analyzer.
        
        Args:
            snapshot_date: Reference date for recency calculation (YYYY-MM-DD format)
                          If None, uses the maximum transaction date in data
        """
        self.snapshot_date = snapshot_date
        self.rfm_data = None
        
    def calculate_rfm(
        self,
        df: pd.DataFrame,
        customer_col: str = 'CustomerId',
        date_col: str = 'TransactionStartTime',
        value_col: str = 'Value'
    ) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        Args:
            df: Transaction dataframe
            customer_col: Column name for customer ID
            date_col: Column name for transaction date
            value_col: Column name for transaction value
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        try:
            logger.info("="*60)
            logger.info("CALCULATING RFM METRICS")
            logger.info("="*60)
            
            # Ensure date column is datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Define snapshot date
            if self.snapshot_date is None:
                self.snapshot_date = df[date_col].max()
                logger.info(f"Snapshot date set to max transaction date: {self.snapshot_date}")
            else:
                self.snapshot_date = pd.to_datetime(self.snapshot_date)
                logger.info(f"Using provided snapshot date: {self.snapshot_date}")
            
            # Calculate RFM per customer
            logger.info(f"\nCalculating RFM for {df[customer_col].nunique()} unique customers...")
            
            rfm = df.groupby(customer_col).agg({
                date_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency
                value_col: 'sum'  # Monetary (total value)
            })
            
            # Add frequency (count transactions per customer)
            rfm['Frequency'] = df.groupby(customer_col).size()
            
            # Reset index to make customer_col a regular column
            rfm = rfm.reset_index()
            
            # Rename columns properly
            rfm.columns = [customer_col, 'Recency', 'Monetary', 'Frequency']
            
            # Reorder columns for consistency
            rfm = rfm[[customer_col, 'Recency', 'Frequency', 'Monetary']]
            
            # Store results
            self.rfm_data = rfm
            
            # Log statistics
            logger.info("\nRFM Statistics:")
            logger.info("="*60)
            logger.info(f"\nRecency (days since last transaction):")
            logger.info(f"  Mean: {rfm['Recency'].mean():.2f}")
            logger.info(f"  Median: {rfm['Recency'].median():.2f}")
            logger.info(f"  Min: {rfm['Recency'].min()}")
            logger.info(f"  Max: {rfm['Recency'].max()}")
            
            logger.info(f"\nFrequency (number of transactions):")
            logger.info(f"  Mean: {rfm['Frequency'].mean():.2f}")
            logger.info(f"  Median: {rfm['Frequency'].median():.2f}")
            logger.info(f"  Min: {rfm['Frequency'].min()}")
            logger.info(f"  Max: {rfm['Frequency'].max()}")
            
            logger.info(f"\nMonetary (total transaction value):")
            logger.info(f"  Mean: {rfm['Monetary'].mean():.2f}")
            logger.info(f"  Median: {rfm['Monetary'].median():.2f}")
            logger.info(f"  Min: {rfm['Monetary'].min():.2f}")
            logger.info(f"  Max: {rfm['Monetary'].max():.2f}")
            
            logger.info(f"\nRFM calculation completed for {len(rfm)} customers")
            
            return rfm
            
        except Exception as e:
            logger.error(f"Error calculating RFM: {str(e)}")
            raise
    
    def get_rfm_summary(self) -> pd.DataFrame:
        """Get summary statistics of RFM data."""
        if self.rfm_data is None:
            raise ValueError("RFM data not calculated. Run calculate_rfm() first.")
        
        return self.rfm_data[['Recency', 'Frequency', 'Monetary']].describe()


class CustomerSegmentation:
    """
    Customer segmentation using K-Means clustering on RFM metrics.
    Identifies high-risk customer segments based on engagement patterns.
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize Customer Segmentation.
        
        Args:
            n_clusters: Number of clusters for K-Means
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_profiles = None
        self.high_risk_cluster = None
        
    def fit_kmeans(
        self,
        rfm_data: pd.DataFrame,
        customer_col: str = 'CustomerId'
    ) -> pd.DataFrame:
        """
        Fit K-Means clustering on RFM features.
        
        Args:
            rfm_data: DataFrame with RFM metrics
            customer_col: Column name for customer ID
            
        Returns:
            DataFrame with cluster assignments
        """
        try:
            logger.info("\n" + "="*60)
            logger.info("K-MEANS CLUSTERING")
            logger.info("="*60)
            
            # Extract RFM features
            rfm_features = rfm_data[['Recency', 'Frequency', 'Monetary']].copy()
            
            # Scale features
            logger.info(f"\nScaling RFM features using StandardScaler...")
            rfm_scaled = self.scaler.fit_transform(rfm_features)
            
            # Fit K-Means
            logger.info(f"Fitting K-Means with {self.n_clusters} clusters (random_state={self.random_state})...")
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            
            cluster_labels = self.kmeans.fit_predict(rfm_scaled)
            
            # Add cluster labels to data
            rfm_clustered = rfm_data.copy()
            rfm_clustered['Cluster'] = cluster_labels
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(rfm_scaled, cluster_labels)
            logger.info(f"\nSilhouette Score: {silhouette_avg:.4f}")
            logger.info("  (Score ranges from -1 to 1, higher is better)")
            
            # Calculate cluster profiles
            self._calculate_cluster_profiles(rfm_clustered)
            
            # Identify high-risk cluster
            self._identify_high_risk_cluster(rfm_clustered)
            
            logger.info("\nK-Means clustering completed successfully")
            
            return rfm_clustered
            
        except Exception as e:
            logger.error(f"Error in K-Means clustering: {str(e)}")
            raise
    
    def _calculate_cluster_profiles(self, rfm_clustered: pd.DataFrame):
        """Calculate mean RFM values for each cluster."""
        logger.info("\n" + "="*60)
        logger.info("CLUSTER PROFILES")
        logger.info("="*60)
        
        self.cluster_profiles = rfm_clustered.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Cluster': 'count'
        }).rename(columns={'Cluster': 'Customer_Count'})
        
        logger.info("\n" + str(self.cluster_profiles.round(2)))
        
        # Add interpretations
        logger.info("\nCluster Interpretations:")
        for cluster_id in range(self.n_clusters):
            profile = self.cluster_profiles.loc[cluster_id]
            
            # Interpret based on RFM values
            if profile['Recency'] > self.cluster_profiles['Recency'].median():
                recency_desc = "High Recency (inactive)"
            else:
                recency_desc = "Low Recency (active)"
            
            if profile['Frequency'] > self.cluster_profiles['Frequency'].median():
                frequency_desc = "High Frequency"
            else:
                frequency_desc = "Low Frequency"
            
            if profile['Monetary'] > self.cluster_profiles['Monetary'].median():
                monetary_desc = "High Value"
            else:
                monetary_desc = "Low Value"
            
            logger.info(f"  Cluster {cluster_id}: {recency_desc}, {frequency_desc}, {monetary_desc}")
            logger.info(f"    Customers: {int(profile['Customer_Count'])}")
    
    def _identify_high_risk_cluster(self, rfm_clustered: pd.DataFrame):
        """
        Identify the high-risk cluster based on engagement metrics.
        
        High-risk characteristics:
        - High Recency (haven't transacted recently)
        - Low Frequency (few transactions)
        - Low Monetary (low total value)
        """
        logger.info("\n" + "="*60)
        logger.info("IDENTIFYING HIGH-RISK CLUSTER")
        logger.info("="*60)
        
        # Normalize metrics for scoring (0-1 scale)
        profiles_normalized = self.cluster_profiles.copy()
        
        # For Recency: higher is worse (normalize and invert)
        profiles_normalized['Recency_Score'] = (
            profiles_normalized['Recency'] / profiles_normalized['Recency'].max()
        )
        
        # For Frequency: lower is worse (normalize and invert)
        profiles_normalized['Frequency_Score'] = 1 - (
            profiles_normalized['Frequency'] / profiles_normalized['Frequency'].max()
        )
        
        # For Monetary: lower is worse (normalize and invert)
        profiles_normalized['Monetary_Score'] = 1 - (
            profiles_normalized['Monetary'] / profiles_normalized['Monetary'].max()
        )
        
        # Calculate risk score (higher = more risky)
        # Equal weight to all three factors
        profiles_normalized['Risk_Score'] = (
            profiles_normalized['Recency_Score'] +
            profiles_normalized['Frequency_Score'] +
            profiles_normalized['Monetary_Score']
        ) / 3
        
        # Identify cluster with highest risk score
        self.high_risk_cluster = profiles_normalized['Risk_Score'].idxmax()
        
        logger.info("\nRisk Scores by Cluster:")
        for cluster_id in range(self.n_clusters):
            risk_score = profiles_normalized.loc[cluster_id, 'Risk_Score']
            is_high_risk = " ⚠️ HIGH RISK" if cluster_id == self.high_risk_cluster else ""
            logger.info(f"  Cluster {cluster_id}: {risk_score:.4f}{is_high_risk}")
        
        logger.info(f"\n✓ High-Risk Cluster Identified: Cluster {self.high_risk_cluster}")
        
        high_risk_profile = self.cluster_profiles.loc[self.high_risk_cluster]
        logger.info(f"\nHigh-Risk Cluster Profile:")
        logger.info(f"  Recency: {high_risk_profile['Recency']:.2f} days")
        logger.info(f"  Frequency: {high_risk_profile['Frequency']:.2f} transactions")
        logger.info(f"  Monetary: {high_risk_profile['Monetary']:.2f}")
        logger.info(f"  Customer Count: {int(high_risk_profile['Customer_Count'])}")
    
    def create_risk_labels(
        self,
        rfm_clustered: pd.DataFrame,
        customer_col: str = 'CustomerId'
    ) -> pd.DataFrame:
        """
        Create binary high-risk label based on cluster assignment.
        
        Args:
            rfm_clustered: DataFrame with cluster assignments
            customer_col: Column name for customer ID
            
        Returns:
            DataFrame with is_high_risk column
        """
        try:
            logger.info("\n" + "="*60)
            logger.info("CREATING RISK LABELS")
            logger.info("="*60)
            
            if self.high_risk_cluster is None:
                raise ValueError("High-risk cluster not identified. Run fit_kmeans() first.")
            
            # Create binary risk label
            risk_labels = rfm_clustered[[customer_col, 'Cluster']].copy()
            risk_labels['is_high_risk'] = (
                risk_labels['Cluster'] == self.high_risk_cluster
            ).astype(int)
            
            # Log distribution
            high_risk_count = risk_labels['is_high_risk'].sum()
            total_count = len(risk_labels)
            high_risk_pct = (high_risk_count / total_count) * 100
            
            logger.info(f"\nRisk Label Distribution:")
            logger.info(f"  High Risk (1): {high_risk_count} ({high_risk_pct:.2f}%)")
            logger.info(f"  Low Risk (0): {total_count - high_risk_count} ({100-high_risk_pct:.2f}%)")
            
            logger.info("\n✓ Risk labels created successfully")
            
            return risk_labels
            
        except Exception as e:
            logger.error(f"Error creating risk labels: {str(e)}")
            raise
    
    def visualize_clusters(
        self,
        rfm_clustered: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Create visualizations of customer clusters.
        
        Args:
            rfm_clustered: DataFrame with cluster assignments
            save_path: Optional path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 3D scatter would be ideal but we'll use 2D projections
            
            # Recency vs Frequency
            axes[0, 0].scatter(
                rfm_clustered['Recency'],
                rfm_clustered['Frequency'],
                c=rfm_clustered['Cluster'],
                cmap='viridis',
                alpha=0.6
            )
            axes[0, 0].set_xlabel('Recency (days)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Recency vs Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Recency vs Monetary
            axes[0, 1].scatter(
                rfm_clustered['Recency'],
                rfm_clustered['Monetary'],
                c=rfm_clustered['Cluster'],
                cmap='viridis',
                alpha=0.6
            )
            axes[0, 1].set_xlabel('Recency (days)')
            axes[0, 1].set_ylabel('Monetary Value')
            axes[0, 1].set_title('Recency vs Monetary')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Frequency vs Monetary
            scatter = axes[1, 0].scatter(
                rfm_clustered['Frequency'],
                rfm_clustered['Monetary'],
                c=rfm_clustered['Cluster'],
                cmap='viridis',
                alpha=0.6
            )
            axes[1, 0].set_xlabel('Frequency')
            axes[1, 0].set_ylabel('Monetary Value')
            axes[1, 0].set_title('Frequency vs Monetary')
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')
            
            # Cluster size distribution
            cluster_counts = rfm_clustered['Cluster'].value_counts().sort_index()
            colors = ['red' if i == self.high_risk_cluster else 'blue' for i in cluster_counts.index]
            axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.7)
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Number of Customers')
            axes[1, 1].set_title('Cluster Size Distribution\n(Red = High Risk)')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Cluster visualization saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")


def create_proxy_target(
    df: pd.DataFrame,
    customer_col: str = 'CustomerId',
    date_col: str = 'TransactionStartTime',
    value_col: str = 'Value',
    n_clusters: int = 3,
    random_state: int = 42,
    snapshot_date: Optional[str] = None,
    save_visualization: bool = True,
    viz_path: str = 'reports/figures/rfm_clusters.png'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete pipeline to create proxy target variable.
    
    Args:
        df: Transaction dataframe
        customer_col: Customer ID column
        date_col: Transaction date column
        value_col: Transaction value column
        n_clusters: Number of K-Means clusters
        random_state: Random seed
        snapshot_date: Reference date for recency
        save_visualization: Whether to save cluster plots
        viz_path: Path to save visualization
        
    Returns:
        Tuple of (DataFrame with merged target, metadata dict)
    """
    try:
        logger.info("\n" + "="*70)
        logger.info("PROXY TARGET VARIABLE ENGINEERING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Calculate RFM
        rfm_analyzer = RFMAnalyzer(snapshot_date=snapshot_date)
        rfm_data = rfm_analyzer.calculate_rfm(df, customer_col, date_col, value_col)
        
        # Step 2: Cluster customers
        segmentation = CustomerSegmentation(n_clusters=n_clusters, random_state=random_state)
        rfm_clustered = segmentation.fit_kmeans(rfm_data, customer_col)
        
        # Step 3: Create risk labels
        risk_labels = segmentation.create_risk_labels(rfm_clustered, customer_col)
        
        # Step 4: Visualize (optional)
        if save_visualization:
            import os
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            segmentation.visualize_clusters(rfm_clustered, save_path=viz_path)
        
        # Step 5: Merge back to original data
        logger.info("\n" + "="*60)
        logger.info("INTEGRATING TARGET VARIABLE")
        logger.info("="*60)
        
        df_with_target = df.merge(
            risk_labels[[customer_col, 'is_high_risk']],
            on=customer_col,
            how='left'
        )
        
        # Verify merge
        if df_with_target['is_high_risk'].isnull().any():
            logger.warning("Some records have null is_high_risk values")
        
        logger.info(f"\nOriginal shape: {df.shape}")
        logger.info(f"Final shape: {df_with_target.shape}")
        logger.info(f"\n✓ Target variable 'is_high_risk' successfully integrated")
        
        # Prepare metadata
        metadata = {
            'snapshot_date': str(rfm_analyzer.snapshot_date),
            'n_clusters': n_clusters,
            'random_state': random_state,
            'high_risk_cluster': int(segmentation.high_risk_cluster),
            'cluster_profiles': segmentation.cluster_profiles.to_dict(),
            'high_risk_count': int(risk_labels['is_high_risk'].sum()),
            'total_customers': len(risk_labels),
            'high_risk_percentage': float((risk_labels['is_high_risk'].sum() / len(risk_labels)) * 100)
        }
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        return df_with_target, metadata
        
    except Exception as e:
        logger.error(f"Error in proxy target pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    """Test the proxy target variable engineering pipeline."""
    import sys
    
    try:
        from data_processing import load_data
        
        logger.info("Testing Proxy Target Variable Engineering Pipeline")
        
        # Load data
        data_path = "data/raw/data.csv"
        df = load_data(data_path)
        
        # Create proxy target
        df_with_target, metadata = create_proxy_target(
            df,
            n_clusters=3,
            random_state=42,
            save_visualization=True
        )
        
        # Save results
        output_path = "data/processed/data_with_risk_target.csv"
        df_with_target.to_csv(output_path, index=False)
        logger.info(f"\nData with target saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"High-Risk Customers: {metadata['high_risk_count']} ({metadata['high_risk_percentage']:.2f}%)")
        print(f"Snapshot Date: {metadata['snapshot_date']}")
        print(f"High-Risk Cluster ID: {metadata['high_risk_cluster']}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
