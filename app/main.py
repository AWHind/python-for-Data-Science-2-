from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import os
import sys
import logging
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

try:
    from model_cache import ModelCache
    HAS_MODEL_CACHE = True
except ImportError as e:
    HAS_MODEL_CACHE = False
    logger_init = logging.getLogger(__name__)
    logger_init.warning(f"Could not import model_cache: {e}")

# =============================
# Logging Setup
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# FastAPI App Configuration
# =============================
app = FastAPI(
    title="Arrhythmia Prediction API",
    description="Advanced ML API for Arrhythmia prediction with multiple models (RF, XGB, SVM, LR)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================
# Root Endpoint
# =============================
@app.get("/", tags=["Root"])
def root():
    """Redirect to Swagger documentation"""
    return RedirectResponse(url="/docs")

# =============================
# CORS Middleware
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
# Pydantic Models
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

class ModelMetricsResponse(BaseModel):
    """Model metrics response"""
    name: str
    accuracy: float
    f1_score: float
    run_id: str
    source: str

class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    api_version: str
    models_loaded: int
    cache_enabled: bool

# =============================
# Global Configuration
# =============================
EXPECTED_FEATURES = 278
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize model cache
model_cache = None
if HAS_MODEL_CACHE:
    cache_dir = os.path.join(BASE_DIR, "model_cache")
    model_cache = ModelCache(cache_dir=cache_dir)
    logger.info(f"✅ Model cache initialized at {cache_dir}")
else:
    logger.warning("⚠️ ModelCache not available - using dummy models as fallback")

# Preload all models
models_loaded = 0
if model_cache:
    for model_type in ModelType:
        try:
            model, info = model_cache.load_model(model_type.value)
            logger.info(f"✅ {model_type.value}: {info.accuracy:.2%} accuracy (source: {info.source})")
            models_loaded += 1
        except Exception as e:
            logger.error(f"❌ Failed to load {model_type.value}: {e}")

# =============================
# Health Check
# =============================
@app.get("/health", response_model=HealthStatus, tags=["Health"])
def health():
    """Health check endpoint"""
    return {
        "status": "OK",
        "api_version": "2.0.0",
        "models_loaded": models_loaded,
        "cache_enabled": HAS_MODEL_CACHE
    }

# =============================
# Models Endpoints
# =============================
@app.get("/models", response_model=List[ModelMetricsResponse], tags=["Models"])
def get_models_info():
    """Get all available models and their metrics"""
    if not model_cache:
        raise HTTPException(status_code=503, detail="Model cache not initialized")
    
    models_info = model_cache.get_all_models_info()
    return [
        ModelMetricsResponse(
            name=info["name"],
            accuracy=info["accuracy"],
            f1_score=info["f1_score"],
            run_id=info["run_id"],
            source=info["source"]
        )
        for info in models_info
    ]

@app.get("/models/{model_name}/metrics", tags=["Models"])
def get_model_metrics(model_name: str):
    """Get metrics for a specific model"""
    if not model_cache:
        raise HTTPException(status_code=503, detail="Model cache not initialized")
    
    valid_models = [m.value for m in ModelType]
    if model_name not in valid_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {valid_models}"
        )
    
    try:
        model, info = model_cache.load_model(model_name)
        return {
            "model": model_name,
            "accuracy": info.accuracy,
            "f1_score": info.f1_score,
            "run_id": info.run_id,
            "source": info.source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading {model_name}: {str(e)}")

# =============================
# Prediction Endpoints
# =============================
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(data: PatientData):
    """
    Make a prediction using selected ML model
    
    **Parameters:**
    - features: List of 278 ECG features (float values)
    - model: Model to use (RandomForest, XGBoost, SVM, or LogisticRegression)
    
    **Returns:**
    - prediction: 0 (Normal) or 1 (Arrhythmia)
    - model_used: Name of the model
    - confidence: Model accuracy
    - model_metrics: Detailed metrics
    """
    
    if not model_cache:
        raise HTTPException(status_code=503, detail="Model cache not initialized")
    
    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(data.features)}"
        )

    model_name = data.model.value
    
    try:
        # Load model
        model, model_info = model_cache.load_model(model_name)
        
        # Create DataFrame and predict
        df = pd.DataFrame([data.features])
        prediction = model.predict(df)
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            model_used=model_name,
            confidence=model_info.accuracy,
            model_metrics={
                "accuracy": model_info.accuracy,
                "f1_score": model_info.f1_score,
                "run_id": model_info.run_id,
                "source": model_info.source
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Prediction error for {model_name}: {str(e)}")
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
    
    **Parameters:**
    - features_list: List of feature arrays (each with 278 features)
    - model: Model to use
    
    **Returns:**
    - total_records: Number of predictions made
    - model_used: Name of the model
    - predictions: List of predictions with indices
    """
    
    if not model_cache:
        raise HTTPException(status_code=503, detail="Model cache not initialized")
    
    if not features_list or len(features_list) == 0:
        raise HTTPException(status_code=400, detail="Empty features list")
    
    # Validate all records
    for i, features in enumerate(features_list):
        if len(features) != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Record {i}: expected {EXPECTED_FEATURES} features, got {len(features)}"
            )
    
    model_name = model.value
    
    try:
        # Load model
        model_obj, _ = model_cache.load_model(model_name)
        
        # Create DataFrame and predict
        df = pd.DataFrame(features_list)
        predictions = model_obj.predict(df)
        
        results = [
            {
                "index": i,
                "prediction": int(pred),
                "class": "Normal" if int(pred) == 0 else "Arrhythmia"
            }
            for i, pred in enumerate(predictions)
        ]
        
        return {
            "total_records": len(results),
            "model_used": model_name,
            "predictions": results
        }
    
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# Report Generation
# =============================
@app.post("/report", tags=["Reports"])
def generate_report(data: PatientData):
    """
    Generate a PDF report of the prediction
    
    **Parameters:**
    - features: List of 278 ECG features
    - model: Model to use
    """
    
    if not model_cache:
        raise HTTPException(status_code=503, detail="Model cache not initialized")
    
    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(status_code=400, detail="Invalid feature count")
    
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        model_name = data.model.value
        model, model_info = model_cache.load_model(model_name)
        
        # Make prediction
        df = pd.DataFrame([data.features])
        prediction = model.predict(df)
        
        # Generate PDF
        file_path = "prediction_report.pdf"
        doc = SimpleDocTemplate(file_path)
        elements = []
        
        styles = getSampleStyleSheet()
        elements.append(Paragraph("CardioSense - Rapport de Prédiction ECG", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))
        
        prediction_text = "Normal (Classe 0)" if prediction[0] == 0 else "Anomalie (Classe 1)"
        
        data_table = [
            ["Classe Prédite", prediction_text],
            ["Modèle Utilisé", model_name],
            ["Précision", f"{model_info.accuracy*100:.2f}%"],
            ["F1 Score", f"{model_info.f1_score:.4f}"],
            ["Source", model_info.source],
            ["Nombre de Features", str(len(data.features))],
        ]
        
        table = Table(data_table)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
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
    
    except ImportError:
        raise HTTPException(status_code=501, detail="ReportLab not installed")
    except Exception as e:
        logger.error(f"❌ Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# API Info
# =============================
@app.post("/patient-report", tags=["Reports"])
def generate_patient_report(
    patient_name: str,
    patient_id: str,
    age: int,
    sex: str,
    prediction_result: int,
    model_used: str,
    confidence: float,
    data: PatientData
):
    """
    Generate a comprehensive PDF report for a patient including:
    - Patient information
    - Prediction result and confidence
    - Model used and its metrics
    - Key ECG parameters
    
    - **patient_name**: Patient's full name
    - **patient_id**: Unique patient identifier
    - **age**: Patient's age
    - **sex**: Patient's gender (M/F)
    - **prediction_result**: Prediction output (0 or 1)
    - **model_used**: Name of the model used
    - **confidence**: Confidence score of prediction
    - **data**: ECG data features
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from datetime import datetime
    
    try:
        file_path = f"patient_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(file_path, title=f"Rapport ECG - {patient_name}")
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=1
        )
        
        # Title
        elements.append(Paragraph("Rapport d'Analyse ECG", title_style))
        elements.append(Spacer(1, 0.3 * inch))
        
        # Patient Information Section
        elements.append(Paragraph("Informations du Patient", styles["Heading2"]))
        patient_data = [
            ["Nom Complet", patient_name],
            ["ID Patient", patient_id],
            ["Âge", f"{age} ans"],
            ["Genre", "Masculin" if sex == "M" else "Féminin"],
            ["Date du Rapport", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ]
        
        patient_table = Table(patient_data)
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Prediction Results Section
        elements.append(Paragraph("Résultats de la Prédiction", styles["Heading2"]))
        prediction_text = "Normal (Classe 0)" if prediction_result == 0 else "Anomalie Détectée (Classe 1)"
        prediction_color = colors.HexColor('#22c55e') if prediction_result == 0 else colors.HexColor('#ef4444')
        
        result_data = [
            ["Classe Prédite", prediction_text],
            ["Modèle Utilisé", model_used],
            ["Confiance", f"{confidence*100:.2f}%"],
            ["Nombre de Features", str(len(data.features))],
        ]
        
        result_table = Table(result_data)
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightyellow),
            ('TEXTCOLOR', (1, 0), (1, 0), prediction_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(result_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Model Metrics Section
        if model_used in model_cache.models_info if model_cache else {}:
            elements.append(Paragraph("Métriques du Modèle", styles["Heading2"]))
            model_info = model_cache.models_info.get(model_used, {}) if model_cache else {}
            
            metrics_data = [
                ["Métrique", "Valeur"],
                ["Accuracy", f"{model_info.get('accuracy', 0)*100:.2f}%"],
                ["F1 Score", f"{model_info.get('f1_score', 0):.4f}"],
                ["Source", model_info.get('source', 'Unknown')],
            ]
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 0.3 * inch))
        
        # Footer
        elements.append(Spacer(1, 0.5 * inch))
        footer_text = "Ce rapport est généré automatiquement par CardioSense. Consultez un professionnel de santé pour une interprétation complète."
        elements.append(Paragraph(footer_text, ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=1
        )))
        
        doc.build(elements)
        
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=f"rapport_{patient_id}.pdf"
        )
    
    except ImportError:
        raise HTTPException(status_code=501, detail="ReportLab not installed")
    except Exception as e:
        logger.error(f"❌ Patient report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.post("/upload-ecg", tags=["Files"])
async def upload_ecg_file(
    patient_id: str,
    patient_name: str,
    file_content: str
):
    """
    Upload ECG data file and validate format
    
    - **patient_id**: Unique patient identifier
    - **patient_name**: Patient's name
    - **file_content**: ECG file content (CSV, JSON, or raw data)
    """
    try:
        # Create upload directory
        upload_dir = os.path.join(BASE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(upload_dir, f"{patient_id}_{timestamp}_ecg.csv")
        
        with open(file_path, 'w') as f:
            f.write(file_content)
        
        return {
            "status": "success",
            "message": "ECG file uploaded successfully",
            "file_path": file_path,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "upload_time": timestamp
        }
    
    except Exception as e:
        logger.error(f"❌ ECG upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.get("/info", tags=["Info"])
def get_api_info():
    """Get comprehensive API information"""
    return {
        "title": "Arrhythmia Prediction API",
        "version": "2.0.0",
        "description": "Advanced ML models for ECG arrhythmia detection",
        "available_models": [m.value for m in ModelType],
        "features": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "model_metrics": "/models and /models/{model_name}/metrics",
            "pdf_report": "/report",
            "patient_report": "/patient-report",
            "ecg_upload": "/upload-ecg",
            "health_check": "/health"
        },
        "expected_features": EXPECTED_FEATURES,
        "docs": "https://localhost:8000/docs"
    }
