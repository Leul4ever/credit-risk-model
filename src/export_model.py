
import mlflow
import joblib
import os
import shutil
import logging
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_best_model(experiment_name="credit_risk_modeling", output_dir="models", model_name="fraud_model.pkl"):
    """
    Export the best model from MLflow to a local pickle file.
    This avoids path issues when moving mlruns folder to Docker (Linux).
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, model_name)
        
        logger.info(f"Looking for best model in experiment: {experiment_name}")
        
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Search runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                  order_by=["metrics.roc_auc DESC"], 
                                  max_results=1)
        
        if runs.empty:
            raise ValueError("No runs found")
            
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        roc_auc = best_run['metrics.roc_auc']
        
        logger.info(f"Best run ID: {run_id} (ROC-AUC: {roc_auc:.4f})")
        
        # Construct model URI
        model_uri = f"runs:/{run_id}/model"
        
        # Load model
        logger.info(f"Loading model from {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get feature names (try multiple ways)
        feature_names = getattr(model, "feature_names_in_", None)
        
        # Prepare deployment package
        deployment_model = {
            'model': model,
            'feature_names': feature_names,
            'training_date': str(best_run['start_time']),
            'metrics': {
                'roc_auc': roc_auc,
                'run_id': run_id
            }
        }
        
        # Save to disk
        logger.info(f"Saving deployment model to {output_path}...")
        joblib.dump(deployment_model, output_path)
        
        logger.info("✅ Model exported successfully!")
        return output_path
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise

if __name__ == "__main__":
    export_best_model()
