# Arrhythmia Prediction API - Complete Documentation

## Overview
Complete ML API for ECG arrhythmia detection with 4 machine learning models integrated.

**API URL**: `http://127.0.0.1:8000`  
**Swagger Documentation**: `http://127.0.0.1:8000/docs`

---

## Available Models

| Model | Accuracy | F1 Score | Type |
|-------|----------|----------|------|
| RandomForest | ~0.95+ | ~0.92+ | Ensemble |
| SVM | ~0.90+ | ~0.88+ | Kernel Method |
| LogisticRegression | ~0.88+ | ~0.85+ | Linear |
| XGBoost | ~0.96+ | ~0.93+ | Gradient Boosting |

---

## Endpoints

### 1. Health Check
**GET** `/health`

Check API status and loaded models.

```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "OK",
  "loaded_models": ["RandomForest", "SVM", "LogisticRegression", "XGBoost"],
  "total_features_expected": 278
}
```

---

### 2. Get All Models
**GET** `/models`

Get information about all available models with their metrics.

```bash
curl http://127.0.0.1:8000/models
```

**Response:**
```json
[
  {
    "name": "RandomForest",
    "accuracy": 0.95,
    "f1_score": 0.92,
    "run_id": "abc123xyz"
  },
  {
    "name": "SVM",
    "accuracy": 0.90,
    "f1_score": 0.88,
    "run_id": "def456uvw"
  }
]
```

---

### 3. Get Model Metrics
**GET** `/models/{model_name}/metrics`

Get detailed metrics for a specific model.

```bash
curl http://127.0.0.1:8000/models/RandomForest/metrics
```

**Response:**
```json
{
  "model": "RandomForest",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.92,
    "run_id": "abc123xyz"
  }
}
```

**Available model names:**
- `RandomForest`
- `SVM`
- `LogisticRegression`
- `XGBoost`

---

### 4. Single Prediction
**POST** `/predict`

Make a single prediction with selected model.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [54, 1, 172, 78, 80, ...], // 278 values
    "model": "RandomForest"
  }'
```

**Request Body:**
```json
{
  "features": [
    54.0, 1.0, 172.0, 78.0, 80.0, 160.0, 370.0, 180.0, 100.0, 72.0, 6.0,
    // ... continues to 278 features total
  ],
  "model": "RandomForest"  // optional, defaults to RandomForest
}
```

**Response:**
```json
{
  "prediction": 0,
  "model_used": "RandomForest",
  "confidence": 0.95,
  "model_metrics": {
    "accuracy": 0.95,
    "f1_score": 0.92,
    "run_id": "abc123xyz"
  }
}
```

**Prediction Classes:**
- `0` = Normal ECG
- `1` = Arrhythmia Detected

**Available Models:**
- `RandomForest`
- `SVM`
- `LogisticRegression`
- `XGBoost`

---

### 5. Batch Predictions
**POST** `/predict/batch`

Make predictions for multiple patient records at once.

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "features_list": [
      [54, 1, 172, ...],
      [60, 2, 175, ...],
      [58, 1, 170, ...]
    ],
    "model": "XGBoost"
  }'
```

**Request Body:**
```json
{
  "features_list": [
    [54.0, 1.0, 172.0, ...],  // First patient - 278 features
    [60.0, 2.0, 175.0, ...],  // Second patient - 278 features
    [58.0, 1.0, 170.0, ...]   // Third patient - 278 features
  ],
  "model": "XGBoost"
}
```

**Response:**
```json
{
  "total_records": 3,
  "model_used": "XGBoost",
  "predictions": [
    {
      "index": 0,
      "prediction": 0,
      "model": "XGBoost"
    },
    {
      "index": 1,
      "prediction": 1,
      "model": "XGBoost"
    },
    {
      "index": 2,
      "prediction": 0,
      "model": "XGBoost"
    }
  ]
}
```

---

### 6. Generate PDF Report
**POST** `/report`

Generate a detailed PDF report of the prediction.

```bash
curl -X POST http://127.0.0.1:8000/report \
  -H "Content-Type: application/json" \
  -d '{
    "features": [54, 1, 172, ...],
    "model": "RandomForest"
  }' \
  --output rapport_prediction.pdf
```

**Request Body:**
```json
{
  "features": [54.0, 1.0, 172.0, ...],  // 278 features
  "model": "RandomForest"  // optional
}
```

**Response:** PDF file download

---

### 7. API Information
**GET** `/info`

Get comprehensive API information.

```bash
curl http://127.0.0.1:8000/info
```

**Response:**
```json
{
  "title": "Arrhythmia Prediction API",
  "version": "1.0.0",
  "description": "ML models for ECG arrhythmia detection",
  "available_models": ["RandomForest", "SVM", "LogisticRegression", "XGBoost"],
  "expected_features": 278,
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
```

---

## Error Handling

### Common Error Responses

**400 Bad Request** - Invalid feature count:
```json
{
  "detail": "Expected 278 features, got 100"
}
```

**404 Not Found** - Model not found:
```json
{
  "detail": "Model XYZ not found. Available: [RandomForest, SVM, LogisticRegression, XGBoost]"
}
```

**500 Internal Server Error** - Prediction failed:
```json
{
  "detail": "Prediction failed: [error message]"
}
```

---

## Feature Requirements

- **Total Features**: 278 numerical features
- **Feature Type**: Floating-point numbers
- **Feature Range**: Any valid numeric values (will be scaled internally)
- **Missing Values**: Not allowed (must provide all 278 features)

### Example Feature Generation

```python
import numpy as np

# Generate 278 features with random ECG-like values
features = np.concatenate([
    np.random.normal(100, 20, 50),    # Heart rate related
    np.random.uniform(0, 200, 100),   # Voltage measurements
    np.random.normal(50, 15, 128)     # Other ECG characteristics
])

assert len(features) == 278
```

---

## Integration with Frontend

### React/TypeScript Example

```typescript
const API_URL = "http://127.0.0.1:8000"

// Load available models
async function loadModels() {
  const response = await fetch(`${API_URL}/models`)
  return await response.json()
}

// Make prediction
async function predict(features: number[], model: string) {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features, model })
  })
  return await response.json()
}

// Batch predictions
async function batchPredict(featuresList: number[][], model: string) {
  const response = await fetch(`${API_URL}/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features_list: featuresList, model })
  })
  return await response.json()
}
```

---

## Running the Backend

### Prerequisites
```bash
pip install fastapi uvicorn mlflow pandas scikit-learn xgboost imblearn reportlab
```

### Start MLflow Server
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### Start API Server
```bash
cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at: `http://127.0.0.1:8000`  
Swagger documentation: `http://127.0.0.1:8000/docs`

---

## CORS Configuration

The API accepts requests from:
- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:3001`
- `http://127.0.0.1:3001`

To add more origins, edit `main.py` and modify the `allow_origins` list in the CORS middleware.

---

## Testing with Swagger

1. Go to `http://127.0.0.1:8000/docs`
2. Click on any endpoint
3. Click "Try it out"
4. Modify the request if needed
5. Click "Execute"

---

## Troubleshooting

### Models Not Loading
- Check MLflow server is running: `mlflow server --host 127.0.0.1 --port 5000`
- Check model paths in MLflow UI: `http://127.0.0.1:5000`

### Feature Count Mismatch
- Always provide exactly 278 features
- Check array length before sending

### CORS Errors
- Ensure Frontend URL is in `allow_origins` list
- Check browser console for specific origin

### Connection Refused
- Ensure API is running on `127.0.0.1:8000`
- Check no other service uses port 8000

---

## Performance Notes

- Single prediction: ~100-500ms depending on model
- Batch prediction (100 records): ~1-2 seconds
- All models are cached in memory after first use
- Metrics are loaded from MLflow on startup

---

## Future Enhancements

- [ ] Add authentication (JWT tokens)
- [ ] Add input validation/sanitization
- [ ] Add request logging
- [ ] Add performance monitoring
- [ ] Add model versioning
- [ ] Add prediction confidence intervals
- [ ] Add explainability features (SHAP)
- [ ] Add model retraining endpoint

---

## Version History

**v1.0.0** (Current)
- 4 ML models integrated
- Single & batch predictions
- PDF report generation
- MLflow tracking
- Complete Swagger documentation
