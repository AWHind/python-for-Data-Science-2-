from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
import mlflow.pyfunc
import mlflow.client
import pandas as pd
import os
import logging
from typing import List, Optional, Dict, Any
from enum import Enum

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# FastAPI App with Swagger
# =============================
app = FastAPI(
    title="Arrhythmia Prediction API",
    description="Advanced ML API for Arrhythmia prediction with multiple models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================
# Redirect root → Swagger
# =============================
@app.get("/", tags=["Root"])
def root():
    """Redirect to Swagger documentation"""
    return RedirectResponse(url="/docs")

# =============================
# CORS Configuration
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Enums
# =============================
class ModelType(str, Enum):
    """Available ML models"""
    RANDOM_FOREST = "RandomForest"
    SVM = "SVM"
    LOGISTIC_REGRESSION = "LogisticRegression"
    XGBOOST = "XGBoost"

# =============================
# Global Config & Models
# =============================
EXPECTED_FEATURES = 278
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Arrhythmia_Experiment")

# Model cache
loaded_models = {}
model_metrics = {}

def load_model(model_name: str) -> Any:
    """Load model from MLflow with caching"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    client = mlflow.client.MlflowClient()
    
    # Fetch run by model name
    experiment = client.get_experiment_by_name("Arrhythmia_Advanced_Experiment")
    if not experiment:
        raise RuntimeError("MLflow experiment not found")
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    target_run = None
    for run in runs:
        if run.data.params.get("model") == model_name:
            target_run = run
            break
    
    if not target_run:
        raise RuntimeError(f"No MLflow run found for model: {model_name}")
    
    model_uri = f"runs:/{target_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Cache metrics
    model_metrics[model_name] = {
        "accuracy": target_run.data.metrics.get("accuracy", 0),
        "f1_score": target_run.data.metrics.get("f1_score", 0),
        "run_id": target_run.info.run_id
    }
    
    loaded_models[model_name] = model
    logger.info(f"Model {model_name} loaded successfully")
    return model

# Try to load all models at startup
for model_type in ModelType:
    try:
        load_model(model_type.value)
    except Exception as e:
        logger.warning(f"Could not preload model {model_type.value}: {str(e)}")

# =============================
# Pydantic Models/Schemas
# =============================
class PatientData(BaseModel):
    """Patient ECG data for prediction"""
    features: List[float]
    model: Optional[ModelType] = ModelType.RANDOM_FOREST

class PredictionResponse(BaseModel):
    """Prediction result"""
    prediction: int
    model_used: str
    confidence: Optional[float] = None
    model_metrics: Optional[Dict] = None

class ModelInfo(BaseModel):
    """Information about available models"""
    name: str
    accuracy: float
    f1_score: float
    run_id: str

class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    loaded_models: List[str]
    total_features_expected: int

# =============================
# Routes
# =============================

@app.get("/health", response_model=HealthStatus, tags=["Health"])
def health():
    """Health check endpoint"""
    return {
        "status": "OK",
        "loaded_models": list(loaded_models.keys()),
        "total_features_expected": EXPECTED_FEATURES
    }


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
def get_models_info():
    """Get all available models and their metrics"""
    models_info = []
    for model_name in ModelType:
        if model_name.value in model_metrics:
            metrics = model_metrics[model_name.value]
            models_info.append(
                ModelInfo(
                    name=model_name.value,
                    accuracy=metrics["accuracy"],
                    f1_score=metrics["f1_score"],
                    run_id=metrics["run_id"]
                )
            )
    return models_info


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(data: PatientData):
    """
    Make a prediction using selected ML model
    
    - **features**: List of 278 ECG features
    - **model**: Model to use (default: RandomForest)
    """
    
    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(data.features)}"
        )

    model_name = data.model.value
    
    try:
        if model_name not in loaded_models:
            model = load_model(model_name)
        else:
            model = loaded_models[model_name]
        
        df = pd.DataFrame([data.features])
        prediction = model.predict(df)
        
        metrics = model_metrics.get(model_name, {})
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            model_used=model_name,
            confidence=metrics.get("accuracy", None),
            model_metrics=metrics
        )
    
    except Exception as e:
        logger.error(f"Prediction error for model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Predictions"])
def predict_batch(
    features_list: List[List[float]],
    model: ModelType = ModelType.RANDOM_FOREST
):
    """
    Make batch predictions for multiple patient records
    
    - **features_list**: List of feature arrays (each with 278 features)
    - **model**: Model to use
    """
    
    if not features_list or len(features_list) == 0:
        raise HTTPException(status_code=400, detail="Empty features list")
    
    results = []
    for features in features_list:
        if len(features) != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {EXPECTED_FEATURES} features per record"
            )
    
    model_name = model.value
    
    try:
        if model_name not in loaded_models:
            model_obj = load_model(model_name)
        else:
            model_obj = loaded_models[model_name]
        
        df = pd.DataFrame(features_list)
        predictions = model_obj.predict(df)
        
        results = [
            {
                "index": i,
                "prediction": int(pred),
                "model": model_name
            }
            for i, pred in enumerate(predictions)
        ]
        
        return {
            "total_records": len(results),
            "model_used": model_name,
            "predictions": results
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/metrics", tags=["Models"])
def get_model_metrics(model_name: str):
    """Get metrics for a specific model"""
    
    if model_name not in [m.value for m in ModelType]:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} not found. Available: {[m.value for m in ModelType]}"
        )
    
    if model_name not in model_metrics:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics for {model_name} not available"
        )
    
    return {
        "model": model_name,
        "metrics": model_metrics[model_name]
    }


@app.post("/report", tags=["Reports"])
def generate_report(data: PatientData):
    """
    Generate a PDF report of the prediction
    
    - **features**: List of 278 ECG features
    - **model**: Model to use
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    
    try:
        # Make prediction first
        if len(data.features) != EXPECTED_FEATURES:
            raise HTTPException(status_code=400, detail="Invalid feature count")
        
        model_name = data.model.value
        if model_name not in loaded_models:
            model = load_model(model_name)
        else:
            model = loaded_models[model_name]
        
        df = pd.DataFrame([data.features])
        prediction = model.predict(df)
        
        # Generate PDF
        file_path = "prediction_report.pdf"
        doc = SimpleDocTemplate(file_path)
        elements = []
        
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Rapport de Prédiction ECG - Arythmie", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))
        
        metrics = model_metrics.get(model_name, {})
        prediction_text = "Normal (Classe 0)" if prediction[0] == 0 else "Anomalie (Classe 1)"
        
        data_table = [
            ["Classe Prédite", prediction_text],
            ["Modèle Utilisé", model_name],
            ["Précision du Modèle", f"{metrics.get('accuracy', 'N/A')*100:.2f}%"],
            ["Score F1", f"{metrics.get('f1_score', 'N/A'):.4f}"],
            ["Nombre de Features", str(len(data.features))],
        ]
        
        table = Table(data_table)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename="rapport_prediction.pdf"
        )
    
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", tags=["Info"])
def get_api_info():
    """Get API information"""
    return {
        "title": "Arrhythmia Prediction API",
        "version": "1.0.0",
        "description": "ML models for ECG arrhythmia detection",
        "available_models": [m.value for m in ModelType],
        "expected_features": EXPECTED_FEATURES,
        "endpoints": {
            "/health": "Health check",
            "/models": "List all models",
            "/models/{model_name}/metrics": "Get model metrics",
            "/predict": "Make single prediction",
            "/predict/batch": "Make batch predictions",
            "/report": "Generate PDF report",
            "/docs": "Swagger UI"
        }
    }
