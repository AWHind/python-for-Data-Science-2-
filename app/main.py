from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging

# =============================
# Logging Configuration
# =============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# App Initialization
# =============================

app = FastAPI(title="Arrhythmia Prediction API")

# =============================
# Load Model from MLflow
# =============================

MODEL_URI = "models:/Arrhythmia_Model/1"  # بدل version إذا لزم

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise RuntimeError("Model could not be loaded")

# =============================
# Constants
# =============================

EXPECTED_FEATURES = 279  # حسب dataset متاعك

# =============================
# Schemas
# =============================

class PatientData(BaseModel):
    features: list[float]

class BatchPatientData(BaseModel):
    records: list[list[float]]

# =============================
# Health Check
# =============================

@app.get("/health")
def health():
    return {"status": "OK"}

# =============================
# Single Prediction
# =============================

@app.post("/predict")
def predict(data: PatientData):
    try:
        if len(data.features) != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {EXPECTED_FEATURES} features"
            )

        df = pd.DataFrame([data.features])
        prediction = model.predict(df)

        logger.info("Single prediction executed")

        return {
            "prediction": int(prediction[0])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# =============================
# Batch Prediction
# =============================

@app.post("/predict-batch")
def predict_batch(data: BatchPatientData):
    try:
        for record in data.records:
            if len(record) != EXPECTED_FEATURES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Each record must have {EXPECTED_FEATURES} features"
                )

        df = pd.DataFrame(data.records)
        predictions = model.predict(df)

        logger.info("Batch prediction executed")

        return {
            "predictions": predictions.tolist()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
