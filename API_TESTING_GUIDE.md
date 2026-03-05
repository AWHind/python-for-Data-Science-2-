# API Testing Guide - CardioSense

Complete guide for testing all ML models through the API.

## Quick Start (2 minutes)

### 1. Initialize Cache
```bash
python code/init_model_cache.py
```

### 2. Start API
```bash
cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Access Swagger Documentation
Visit: **http://127.0.0.1:8000/docs**

## Testing All Models

### Via cURL

#### Check API Health
```bash
curl http://127.0.0.1:8000/health
```

Response:
```json
{
  "status": "OK",
  "api_version": "2.0.0",
  "models_loaded": 4,
  "cache_enabled": true
}
```

#### Get All Models Info
```bash
curl http://127.0.0.1:8000/models
```

Response:
```json
[
  {
    "name": "RandomForest",
    "accuracy": 0.962,
    "f1_score": 0.947,
    "run_id": "cache",
    "source": "cache"
  },
  {
    "name": "XGBoost",
    "accuracy": 0.968,
    "f1_score": 0.956,
    "run_id": "cache",
    "source": "cache"
  },
  ...
]
```

#### Predict with RandomForest
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": ['"$(python -c 'import json; print(",".join(str(i*0.1) for i in range(278)))')"'],
    "model": "RandomForest"
  }'
```

#### Predict with XGBoost
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": ['"$(python -c 'import json; print(",".join(str(i*0.1) for i in range(278)))')"'],
    "model": "XGBoost"
  }'
```

#### Predict with SVM
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": ['"$(python -c 'import json; print(",".join(str(i*0.1) for i in range(278)))')"'],
    "model": "SVM"
  }'
```

#### Predict with LogisticRegression
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": ['"$(python -c 'import json; print(",".join(str(i*0.1) for i in range(278)))')"'],
    "model": "LogisticRegression"
  }'
```

### Via Python

```python
import requests
import json

API_URL = "http://127.0.0.1:8000"

# Generate sample features
features = [i * 0.1 for i in range(278)]

# Test all models
models = ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]

for model in models:
    response = requests.post(
        f"{API_URL}/predict",
        json={
            "features": features,
            "model": model
        }
    )
    
    result = response.json()
    print(f"{model}: Prediction={result['prediction']}, Accuracy={result['confidence']:.2%}")
```

### Via JavaScript/TypeScript

```typescript
const API_URL = "http://127.0.0.1:8000";

async function predictWithModel(modelName: string) {
  const features = Array.from({length: 278}, (_, i) => i * 0.1);
  
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      features,
      model: modelName
    })
  });
  
  const data = await response.json();
  console.log(`${modelName}: ${data.prediction === 0 ? "Normal" : "Arrhythmia"}`);
  return data;
}

// Test all models
const models = ["RandomForest", "XGBoost", "SVM", "LogisticRegression"];
for (const model of models) {
  await predictWithModel(model);
}
```

## Endpoints Reference

### Health Check
- **GET** `/health`
- Returns API status and loaded models count

### Models Management
- **GET** `/models` - List all models with metrics
- **GET** `/models/{model_name}/metrics` - Get specific model metrics

### Predictions
- **POST** `/predict` - Single prediction
- **POST** `/predict/batch` - Multiple predictions
- **POST** `/report` - Generate PDF report

### Information
- **GET** `/info` - API information
- **GET** `/docs` - Swagger UI
- **GET** `/redoc` - ReDoc documentation

## Model Comparison

| Model | Accuracy | F1 Score | Speed | Source |
|-------|----------|----------|-------|--------|
| RandomForest | 96.2% | 0.947 | Fast | cache/mlflow |
| XGBoost | 96.8% | 0.956 | Fast | cache/mlflow |
| SVM | 93.8% | 0.918 | Medium | cache/mlflow |
| LogisticRegression | 88.5% | 0.868 | Very Fast | cache/mlflow |

## Troubleshooting

### Models not loading
```bash
# Ensure cache is initialized
python code/init_model_cache.py

# Check cache directory
ls -la model_cache/
```

### API returns 503
- Model cache not initialized
- Run: `python code/init_model_cache.py`

### MLflow connection issues
- MLflow is optional - API works with local cache
- If you want MLflow, run: `mlflow server --host 127.0.0.1 --port 5000`

### Feature count mismatch
- All features must be exactly 278 values
- Check: `curl http://127.0.0.1:8000/info | jq .expected_features`

## Advanced Testing

### Batch Predictions
```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "features_list": [
      [0.1, 0.2, ..., 27.8],
      [1.1, 1.2, ..., 28.8]
    ],
    "model": "RandomForest"
  }'
```

### Generate PDF Report
```bash
curl -X POST http://127.0.0.1:8000/report \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, ..., 27.8],
    "model": "XGBoost"
  }' \
  -o report.pdf
```

## Frontend Integration

The frontend automatically loads all 4 models via the `/models` endpoint.

### Prediction Form Component
```typescript
// frontend/components/dashboard/prediction-form.tsx
// Already configured to test all models
```

### Admin Models Component
```typescript
// frontend/components/admin/admin-models.tsx
// Displays metrics for all models
```

All 4 models are now fully integrated and testable through both the API and frontend!
