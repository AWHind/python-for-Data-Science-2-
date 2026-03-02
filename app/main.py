from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
import logging

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# FastAPI App
# =============================
app = FastAPI(
    title="Arrhythmia Prediction API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================
# Redirect root → Swagger
# =============================
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# =============================
# CORS (مهم للـ Frontend)
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Model Config
# =============================
EXPECTED_FEATURES = 278  # ✅ مهم: مش 279

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "mlruns",
    "923508638265794278",
    "f25aa9eda1cf4681a3e8ec760fb434e4",
    "artifacts",
    "model"
)

print("MODEL PATH:", MODEL_DIR)

if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Model path not found: {MODEL_DIR}")

model = mlflow.pyfunc.load_model(MODEL_DIR)
logger.info("Model loaded successfully")

# =============================
# Schema
# =============================
class PatientData(BaseModel):
    features: list[float]

# =============================
# Routes
# =============================
@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict")
def predict(data: PatientData):

    print("Received features:", len(data.features))

    if len(data.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, got {len(data.features)}"
        )

    try:
        df = pd.DataFrame([data.features])
        prediction = model.predict(df)

        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed"
        )
        from fastapi.responses import FileResponse
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import Table
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import TableStyle

        @app.get("/report")
        def generate_report():

            file_path = "prediction_report.pdf"

            doc = SimpleDocTemplate(file_path)
            elements = []

            styles = getSampleStyleSheet()
            elements.append(Paragraph("Rapport de Prediction ECG", styles["Heading1"]))
            elements.append(Spacer(1, 0.5 * inch))

            data = [
                ["Classe Predite", "Classe 1 (Normal)"],
                ["Modele", "Random Forest"],
                ["Confiance", "90%"],
            ]

            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))

            elements.append(table)
            doc.build(elements)

            return FileResponse(
                file_path,
                media_type="application/pdf",
                filename="rapport_prediction.pdf"
            )