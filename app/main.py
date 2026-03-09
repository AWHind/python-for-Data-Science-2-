from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
import logging

from app.rag import ask_ai
from app.voice import speech_to_text, text_to_speech

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

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
# CORS (للـ Frontend)
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
EXPECTED_FEATURES = 278

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
# Schemas
# =============================
class PatientData(BaseModel):
    features: list[float]


class ChatRequest(BaseModel):
    message: str

# =============================
# Health Check
# =============================
@app.get("/health")
def health():
    return {"status": "OK"}

# =============================
# Prediction Route
# =============================
@app.post("/predict")
def predict(data: PatientData):

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

# =============================
# PDF Report
# =============================
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

# =============================
# Chatbot
# =============================
@app.post("/chat")
def chat(req: ChatRequest):

    reply = ask_ai(req.message)

    return {
        "reply": reply
    }

# =============================
# Voice Chat
# =============================
@app.post("/voice-chat")
async def voice_chat(audio: bytes):

    text = speech_to_text(audio)

    answer = ask_ai(text)

    audio_reply = text_to_speech(answer)

    return {
        "text": answer,
        "audio": audio_reply
    }