
from fastapi import FastAPI
from .pydantic_models import Transaction

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Credit Risk Model API is running"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    return {"risk_probability": 0.5}
