from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class Transaction(BaseModel):
    """Schema for a single transaction with validation."""
    
    TransactionId: str = Field(..., description="Unique transaction identifier")
    BatchId: str = Field(..., description="Batch identifier")
    AccountId: str = Field(..., description="Account identifier")
    SubscriptionId: str = Field(..., description="Subscription identifier")
    CustomerId: str = Field(..., description="Customer identifier")
    CurrencyCode: str = Field(..., description="Currency code (e.g., UGX)")
    CountryCode: int = Field(..., ge=1, le=999, description="Country code")
    ProviderId: str = Field(..., description="Provider identifier")
    ProductId: str = Field(..., description="Product identifier")
    ProductCategory: str = Field(..., description="Product category")
    ChannelId: str = Field(..., description="Channel identifier")
    Amount: float = Field(..., description="Transaction amount (can be negative)")
    Value: float = Field(..., ge=0, description="Absolute transaction value")
    TransactionStartTime: str = Field(..., description="Transaction timestamp")
    PricingStrategy: int = Field(..., ge=0, le=4, description="Pricing strategy (0-4)")
    FraudResult: Optional[int] = Field(0, ge=0, le=1, description="Fraud result (0 or 1)")
    
    @validator('Amount')
    def validate_amount(cls, v):
        """Validate amount is a valid number."""
        if v is None:
            raise ValueError("Amount cannot be None")
        if abs(v) > 100000000:  # 100M limit
            raise ValueError("Amount exceeds maximum allowed value")
        return v
    
    @validator('Value')
    def validate_value(cls, v):
        """Validate value is positive."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        if v > 100000000:  # 100M limit
            raise ValueError("Value exceeds maximum allowed value")
        return v
    
    @validator('CurrencyCode')
    def validate_currency(cls, v):
        """Validate currency code."""
        if not v or len(v) != 3:
            raise ValueError("Currency code must be 3 characters")
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
                "TransactionId": "TransactionId_76871",
                "BatchId": "BatchId_36123",
                "AccountId": "AccountId_3957",
                "SubscriptionId": "SubscriptionId_887",
                "CustomerId": "CustomerId_4406",
                "CurrencyCode": "UGX",
                "CountryCode": 256,
                "ProviderId": "ProviderId_6",
                "ProductId": "ProductId_10",
                "ProductCategory": "airtime",
                "ChannelId": "ChannelId_3",
                "Amount": 1000.0,
                "Value": 1000,
                "TransactionStartTime": "2018-11-15T02:18:49Z",
                "PricingStrategy": 2,
                "FraudResult": 0
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    is_fraud: bool = Field(..., description="Whether transaction is classified as fraud")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    recommendation: str = Field(..., description="Action recommendation: APPROVE, REVIEW, or BLOCK")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in prediction (0-1)")
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        """Validate risk level is valid."""
        valid_levels = ['Low', 'Medium', 'High']
        if v not in valid_levels:
            raise ValueError(f"Risk level must be one of {valid_levels}")
        return v
    
    @validator('recommendation')
    def validate_recommendation(cls, v):
        """Validate recommendation is valid."""
        valid_recommendations = ['APPROVE', 'REVIEW', 'BLOCK']
        if v not in valid_recommendations:
            raise ValueError(f"Recommendation must be one of {valid_recommendations}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.15,
                "risk_level": "Low",
                "recommendation": "APPROVE",
                "confidence": 0.85
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    
    transactions: List[Transaction] = Field(..., min_items=1, max_items=1000, 
                                            description="List of transactions to predict")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        """Validate transactions list."""
        if not v:
            raise ValueError("Transactions list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 transactions")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "TransactionId": "TransactionId_76871",
                        "BatchId": "BatchId_36123",
                        "AccountId": "AccountId_3957",
                        "SubscriptionId": "SubscriptionId_887",
                        "CustomerId": "CustomerId_4406",
                        "CurrencyCode": "UGX",
                        "CountryCode": 256,
                        "ProviderId": "ProviderId_6",
                        "ProductId": "ProductId_10",
                        "ProductCategory": "airtime",
                        "ChannelId": "ChannelId_3",
                        "Amount": 1000.0,
                        "Value": 1000,
                        "TransactionStartTime": "2018-11-15T02:18:49Z",
                        "PricingStrategy": 2,
                        "FraudResult": 0
                    }
                ]
            }
        }


class ModelInfo(BaseModel):
    """Schema for model information response."""
    
    model_path: str = Field(..., description="Path to the model file")
    training_date: str = Field(..., description="When the model was trained")
    model_type: str = Field(..., description="Type of the trained model")
    feature_count: int = Field(..., ge=0, description="Number of features used")
    metrics: dict = Field(default={}, description="Model performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "model_path": "models/fraud_model.pkl",
                "training_date": "2024-12-13T12:00:00",
                "model_type": "RandomForestClassifier",
                "feature_count": 45,
                "metrics": {
                    "roc_auc": 0.95,
                    "f1_score": 0.89
                }
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Service status: healthy or degraded")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Path to the model file")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_path": "models/fraud_model.pkl"
            }
        }
