# Task 6: Model Deployment and CI/CD

## Overview
This document details the implementation of Task 6, which focuses on containerizing the trained credit risk model, deploying it as a REST API using FastAPI, creating a user-friendly dashboard with Streamlit, and establishing a CI/CD pipeline for automated testing and code quality checks.

## 1. Environment Setup

### Dependencies
The following packages were added to `requirements.txt`:
- `fastapi`: High-performance web framework for the API.
- `uvicorn`: ASGI server implementation to run the FastAPI app.
- `streamlit`: Framework for creating the interactive dashboard.
- `flake8`: Linter for enforcing code style and quality.
- `requests`: For making HTTP requests from the dashboard to the API.

## 2. API Development (FastAPI)

### Structure
- **File**: `src/api/main.py`
- **Models**: `src/api/pydantic_models.py`

### Features
- **Model Loading**: The API loads the trained model artifact. It supports loading from both the MLflow Model Registry (`models:/...`) and local storage (`models/fraud_model.pkl`) for robustness in Docker environments.
- **Endpoints**:
    - `GET /`: Root endpoint returning API status and version.
    - `GET /health`: Health check endpoint verifying if the model is loaded and the service is responsive.
    - `POST /predict`: Main inference endpoint. Accepts a single transaction and returns fraud probability, risk level, and recommendation.
    - `POST /predict/batch`: Handles multiple transactions in a single request.
    - `GET /model/info`: Returns metadata about the currently loaded model.
- **Validation**: Uses **Pydantic** models to strictly validate input data types and ranges before processing (e.g., ensuring `Amount` is a number, `CountryCode` matches expected format).

## 3. Dashboard Development (Streamlit)

### Structure
- **File**: `src/dashboard/app.py`

### Features
- **User Interface**: Provides a form to input all necessary transaction details (Amount, Provider, Channel, etc.).
- **Real-time Interaction**: Sends requests to the FastAPI backend and displays results immediately.
- **Visualizations**: 
    - Risk Gauge (Safe/High Risk)
    - Probability Progress Bar
    - Clear "Recommendation" status (APPROVE/REVIEW/BLOCK).
- **Docker Integration**: Automatically detects if it's running inside Docker to connect to the correct API hostname (`http://api:8000`).

## 4. Containerization (Docker)

To ensure consistency across environments, the application is fully containerized.

### API Container
- **Dockerfile**: `Dockerfile`
- **Base Image**: `python:3.10-slim`
- **Configuration**: Installs dependencies, copies the model and code, and exposes port **8000**.

### Dashboard Container
- **Dockerfile**: `Dockerfile.streamlit`
- **Base Image**: `python:3.10-slim`
- **Configuration**: Installs dependencies, copies dashboard code, and exposes port **8501**.

### Orchestration
- **File**: `docker-compose.yml`
- **Services**: Defines `api` and `dashboard` services.
- **Networking**: Creates a shared network so the Dashboard can talk to the API.
- **Volumes**: Mounts the source code for easier development and model updates.

## 5. CI/CD Pipeline (GitHub Actions)

### Workflow
- **File**: `.github/workflows/ci.yml`
- **Triggers**: Executed on every push to `main` and feature branches (`task-*`).

### Jobs
1. **Linting**: Runs `flake8` to check for syntax errors and coding style violations.
2. **Testing**: Runs `pytest` to execute the unit test suite (`tests/`).
3. **Build Integrity**: The pipeline fails if either linting or testing fails, preventing broken code from merging.

## 6. How to Run

### Using Docker (Recommended)
This will start both the API and Dashboard:
```bash
docker-compose up --build
```

- **Dashboard**: [http://localhost:8501](http://localhost:8501)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Local Development (Manual)
1. **Start API**:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```
2. **Start Dashboard**:
   ```bash
   streamlit run src/dashboard/app.py
   ```
