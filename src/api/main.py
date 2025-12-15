from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .pydantic_models import Transaction, PredictionResponse, BatchPredictionRequest
import logging
import os
from typing import Dict, List
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Model API",
    description="API for fraud detection and credit risk assessment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for model path
# Default to Production stage in MLflow Registry
MODEL_PATH = os.getenv("MODEL_PATH", "models:/credit_risk_model/Production")

# In Docker, we might need adjustments, but we prioritize the Registry URI
if os.getenv("IS_DOCKER") and not (MODEL_PATH.startswith("models:/") or MODEL_PATH.startswith("runs:/")):
    logger.info("Docker environment detected with local path. Checking for registry preference...")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred", "error": str(exc)}
    )


@app.get("/")
async def read_root():
    """
    Root endpoint - health check.
    
    Returns:
        dict: API status and information
    """
    try:
        return {
            "message": "Credit Risk Model API is running",
            "version": "1.0.0",
            "status": "healthy",
            "endpoints": {
                "health": "/health",
                "predict_single": "/predict",
                "predict_batch": "/predict/batch",
                "model_info": "/model/info"
            }
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status including model availability
    """
    try:
        if MODEL_PATH.startswith("models:/") or MODEL_PATH.startswith("runs:/"):
            model_exists = True
        else:
            model_exists = os.path.exists(MODEL_PATH)
        
        health_status = {
            "status": "healthy" if model_exists else "degraded",
            "model_loaded": model_exists,
            "model_path": MODEL_PATH
        }
        
        if not model_exists:
            logger.warning(f"Model not found at {MODEL_PATH}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud for a single transaction with comprehensive error handling.
    
    Args:
        transaction: Transaction data
        
    Returns:
        PredictionResponse: Prediction results
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate model exists (skip check for MLflow URIs)
        if not MODEL_PATH.startswith("models:/") and not MODEL_PATH.startswith("runs:/"):
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model not found at {MODEL_PATH}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model not found. Please train a model first."
                )
        
        # Convert transaction to dict
        try:
            transaction_dict = transaction.dict()
        except Exception as e:
            logger.error(f"Error converting transaction to dict: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid transaction data format: {str(e)}"
            )
        
        # Make prediction
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from predict import predict_single
            
            logger.info(f"Processing prediction request for transaction")
            result = predict_single(MODEL_PATH, transaction_dict)
            
            # Create response
            response = PredictionResponse(
                is_fraud=result['is_fraud'],
                fraud_probability=result['fraud_probability'],
                risk_level=result['risk_level'],
                recommendation=result['recommendation'],
                confidence=result['confidence']
            )
            
            logger.info(f"Prediction completed: {response.recommendation}")
            return response
            
        except FileNotFoundError as e:
            logger.error(f"Model file error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not available: {str(e)}"
            )
        except ValueError as e:
            logger.error(f"Input validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid input data: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions with error handling.
    
    Args:
        request: Batch prediction request with list of transactions
        
    Returns:
        dict: Batch prediction results
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate model exists
        if not MODEL_PATH.startswith("models:/") and not MODEL_PATH.startswith("runs:/"):
            if not os.path.exists(MODEL_PATH):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model not found at {MODEL_PATH}"
                )
        
        # Validate batch size
        if len(request.transactions) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Transaction list is empty"
            )
        
        if len(request.transactions) > 1000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Batch size too large. Maximum 1000 transactions per request."
            )
        
        # Convert transactions to list of dicts
        try:
            transactions_data = [t.dict() for t in request.transactions]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid transaction data: {str(e)}"
            )
        
        # Make batch prediction
        try:
            import sys
            import pandas as pd
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from predict import predict
            
            logger.info(f"Processing batch prediction for {len(transactions_data)} transactions")
            
            results = predict(MODEL_PATH, transactions_data, return_proba=True)
            
            # Format response
            batch_results = {
                "total_transactions": len(transactions_data),
                "predictions": results['predictions'],
                "fraud_probabilities": results['fraud_probabilities'],
                "risk_levels": results['risk_level'],
                "summary": {
                    "total": len(results['predictions']),
                    "fraud_count": int(sum(results['predictions'])),
                    "fraud_percentage": float(sum(results['predictions']) / len(results['predictions']) * 100),
                    "high_risk_count": sum(1 for r in results['risk_level'] if r == 'High'),
                    "medium_risk_count": sum(1 for r in results['risk_level'] if r == 'Medium'),
                    "low_risk_count": sum(1 for r in results['risk_level'] if r == 'Low')
                }
            }
            
            logger.info(f"Batch prediction completed: {batch_results['summary']['fraud_count']} fraud cases")
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in batch predict: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model metadata and statistics
        
    Raises:
        HTTPException: If model not found or error loading info
    """
    try:
        if not MODEL_PATH.startswith("models:/") and not MODEL_PATH.startswith("runs:/"):
            if not os.path.exists(MODEL_PATH):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model not found at {MODEL_PATH}"
                )
        
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from predict import load_model
            
            model_data = load_model(MODEL_PATH)
            
            info = {
                "model_path": MODEL_PATH,
                "training_date": model_data.get('training_date', 'Unknown'),
                "metrics": model_data.get('metrics', {}),
                "feature_count": len(model_data.get('feature_names', [])) if model_data.get('feature_names') else 0,
                "model_type": type(model_data['model']).__name__
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error loading model info: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading model info: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    try:
        logger.info("Starting Credit Risk Model API...")
        logger.info(f"Model path: {MODEL_PATH}")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        raise
