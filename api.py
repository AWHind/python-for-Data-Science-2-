from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# =====================================
# App
# =====================================

app = FastAPI(
    title="Arrhythmia AI API",
    description="Arrhythmia detection + medical assistant",
    version="1.0"
)

# =====================================
# CORS
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# Models
# =====================================

class Question(BaseModel):
    message: str


class PredictInput(BaseModel):
    features: list


# =====================================
# Routes
# =====================================

@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "API working"}


@app.post("/chat")
def chat(q: Question):

    if "symptom" in q.message.lower():
        return {
            "response": "Common symptoms include palpitations, dizziness, chest pain and fatigue."
        }

    if "arrhythmia" in q.message.lower():
        return {
            "response": "Arrhythmia is an irregular heartbeat caused by abnormal electrical signals in the heart."
        }

    return {
        "response": "I can help you understand arrhythmia and heart symptoms."
    }


@app.post("/predict")
def predict(data: PredictInput):

    # لاحقاً سنربط MLflow model

    return {
        "prediction": "Arrhythmia detected",
        "confidence": 0.87
    }


@app.get("/model-info")
def model_info():

    return {
        "model": "RandomForest",
        "dataset": "UCI Arrhythmia",
        "features": 279
    }