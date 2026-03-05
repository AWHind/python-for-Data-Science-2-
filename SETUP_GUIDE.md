# Complete Setup Guide - Arrhythmia ML Application

This guide provides step-by-step instructions to set up and run the complete application with all ML models integrated.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Backend Setup](#backend-setup)
3. [Frontend Setup](#frontend-setup)
4. [Training Models](#training-models)
5. [Running the Application](#running-the-application)
6. [Verification Checklist](#verification-checklist)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.8+
- Node.js 16+ (for frontend)
- 4GB RAM minimum
- Port 5000, 8000, 3000 available

### Required Software
```bash
# Python packages (see requirements.txt)
fastapi, uvicorn, mlflow, pandas, scikit-learn, xgboost, 
imblearn, reportlab, pydantic

# Node packages (frontend)
react, next.js, typescript, tailwindcss
```

---

## Backend Setup

### Step 1: Install Python Dependencies

```bash
# Navigate to project root
cd /path/to/python-for-Data-Science-2-

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn mlflow pandas scikit-learn xgboost imbalanced-learn reportlab pydantic
```

### Step 2: Verify Dataset

Ensure the arrhythmia dataset is in the correct location:

```bash
# Expected path:
/path/to/project/data/arrhythmia.data

# Verify:
ls -la data/arrhythmia.data
```

### Step 3: Start MLflow Server

MLflow is required to track and manage ML models.

```bash
# In a new terminal, navigate to project root
cd /path/to/python-for-Data-Science-2-

# Start MLflow tracking server
mlflow server --host 127.0.0.1 --port 5000
```

**Expected output:**
```
[YYYY-MM-DD HH:MM:SS] [INFO] Starting MLflow server...
[YYYY-MM-DD HH:MM:SS] [INFO] Listening on http://127.0.0.1:5000
```

Access MLflow UI: `http://127.0.0.1:5000`

---

## Training Models

### Step 1: Train All Models

```bash
# In a terminal, navigate to project root
cd /path/to/python-for-Data-Science-2-

# Run the training script
python code/train_all_models.py
```

**Expected output:**
```
======================================================================
ARRHYTHMIA ML PIPELINE - TRAINING ALL MODELS
======================================================================

📊 Loading dataset...
   Loaded: XXXX records, XXX features

🧹 Cleaning data...
   ...

✂️ Splitting dataset...
   Train: XXXX records
   Test: XXX records

🚀 Training RandomForest...
✅ RandomForest Training Complete!
   Accuracy:  0.95XX
   F1 Score:  0.92XX
   ...

🚀 Training SVM...
✅ SVM Training Complete!
   ...

[Continues for LogisticRegression and XGBoost]

======================================================================
TRAINING COMPLETE - RESULTS SUMMARY
======================================================================

                 accuracy  f1_score  precision    recall   roc_auc
RandomForest       0.XXXX   0.XXXX    0.XXXX   0.XXXX   0.XXXX
SVM                0.XXXX   0.XXXX    0.XXXX   0.XXXX   0.XXXX
LogisticRegression 0.XXXX   0.XXXX    0.XXXX   0.XXXX   0.XXXX
XGBoost            0.XXXX   0.XXXX    0.XXXX   0.XXXX   0.XXXX

🏆 Best Model: XGBoost (Accuracy: 0.XXXX)

📁 Results saved to: training_results.json
📊 View results in MLflow: http://127.0.0.1:5000
✨ Training pipeline complete!
```

### Step 2: Verify Models in MLflow

1. Open MLflow UI: `http://127.0.0.1:5000`
2. You should see experiment: `Arrhythmia_Advanced_Experiment`
3. It should contain 4 runs (one for each model)
4. Each run should have:
   - Parameters (model name, hyperparameters)
   - Metrics (accuracy, F1, precision, etc.)
   - Artifacts (trained model files)

---

## Backend API Setup

### Step 1: Start FastAPI Server

```bash
# In a new terminal, navigate to project root
cd /path/to/python-for-Data-Science-2-

# Start the API server
cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Step 2: Verify API

```bash
# Health check
curl http://127.0.0.1:8000/health

# Expected response:
# {"status":"OK","loaded_models":["RandomForest","SVM","LogisticRegression","XGBoost"],"total_features_expected":278}
```

### Step 3: Access Swagger Documentation

Open in browser: `http://127.0.0.1:8000/docs`

You should see all available endpoints with interactive testing.

---

## Frontend Setup

### Step 1: Install Dependencies

```bash
# Navigate to frontend directory
cd front-end

# Install npm dependencies
npm install
```

### Step 2: Start Development Server

```bash
# From front-end directory
npm run dev
```

**Expected output:**
```
  ▲ Next.js XX.X.X
  - Local:        http://localhost:3000
  - Environments: .env.local

✓ Ready in XXXms
```

### Step 3: Verify Frontend

Open in browser: `http://localhost:3000`

The application should load. Check the browser console for any errors.

---

## Running the Application

### Complete Startup Sequence

**Terminal 1 - MLflow Server:**
```bash
cd /path/to/project
mlflow server --host 127.0.0.1 --port 5000
```

**Terminal 2 - Backend API:**
```bash
cd /path/to/project/app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 3 - Frontend:**
```bash
cd /path/to/project/front-end
npm run dev
```

Once all three are running:
- Frontend: `http://localhost:3000`
- Backend Swagger: `http://127.0.0.1:8000/docs`
- MLflow: `http://127.0.0.1:5000`

---

## Verification Checklist

### Backend Verification

- [ ] MLflow server running on `http://127.0.0.1:5000`
- [ ] All 4 models visible in MLflow experiment
- [ ] Each model has metrics (accuracy, F1, etc.)
- [ ] FastAPI running on `http://127.0.0.1:8000`
- [ ] Health check working: `curl http://127.0.0.1:8000/health`
- [ ] Swagger docs accessible: `http://127.0.0.1:8000/docs`

```bash
# Quick verification script
echo "Testing Backend..."

echo "1. Health check:"
curl -s http://127.0.0.1:8000/health | python -m json.tool

echo -e "\n2. Available models:"
curl -s http://127.0.0.1:8000/models | python -m json.tool

echo -e "\n3. API info:"
curl -s http://127.0.0.1:8000/info | python -m json.tool

echo -e "\n✅ Backend verification complete!"
```

### Frontend Verification

- [ ] Frontend loads at `http://localhost:3000`
- [ ] No console errors
- [ ] Prediction form visible
- [ ] Model selection dropdown shows 4 models
- [ ] API status shows "Connected"
- [ ] Can click "Predict" button (loads form)

### End-to-End Test

1. Open `http://localhost:3000`
2. Go to prediction form
3. Ensure API status shows "Connected"
4. Select a model from dropdown
5. Click "Predict" button
6. Verify prediction result appears
7. Models list shows all 4 models with metrics
8. Try different models - all should work

---

## API Endpoints Summary

### Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Redirect to Swagger |
| `/health` | GET | Check API status |
| `/models` | GET | List all models with metrics |
| `/models/{model_name}/metrics` | GET | Get specific model metrics |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/report` | POST | Generate PDF report |
| `/info` | GET | API information |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

### Quick Test Examples

```bash
# Test each endpoint

# 1. Health
curl http://127.0.0.1:8000/health

# 2. Get all models
curl http://127.0.0.1:8000/models

# 3. Get specific model metrics
curl http://127.0.0.1:8000/models/RandomForest/metrics

# 4. Make a prediction (with RandomForest)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [54, 1, 172, ...278 values...], "model": "RandomForest"}'

# 5. API info
curl http://127.0.0.1:8000/info
```

---

## Troubleshooting

### Issue: "Cannot connect to MLflow"

**Solution:**
```bash
# Verify MLflow is running
curl http://127.0.0.1:5000/

# If not running, start it:
mlflow server --host 127.0.0.1 --port 5000
```

### Issue: "Model not found" or "No MLflow run found"

**Solution:**
1. Ensure you've run `python code/train_all_models.py`
2. Check MLflow contains 4 runs
3. Restart the FastAPI server

```bash
# Restart FastAPI
cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Issue: "Expected 278 features, got X"

**Solution:**
The API requires exactly 278 features. The frontend automatically generates them, but if you're testing manually:

```python
import json
features = [54.0] * 278  # Generate 278 values

payload = {
    "features": features,
    "model": "RandomForest"
}

print(json.dumps(payload))
```

### Issue: "CORS Error" in Browser

**Solution:**
Check that frontend origin is in CORS whitelist in `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Add your origin here
        "http://127.0.0.1:3000",
        # ...
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Port Already in Use

**Solution:**
```bash
# Find and kill process on port
# On Mac/Linux:
lsof -ti:8000 | xargs kill -9  # Port 8000
lsof -ti:5000 | xargs kill -9  # Port 5000
lsof -ti:3000 | xargs kill -9  # Port 3000

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: Models Taking Long to Load

This is normal behavior. On first request, models are loaded from MLflow into memory. Subsequent requests are fast.

**Expected times:**
- First request: 2-5 seconds
- Subsequent requests: 100-500ms

---

## Monitoring and Logging

### MLflow Tracking

View all training runs and metrics:
```
http://127.0.0.1:5000
```

### FastAPI Logging

The API logs to console. Check for errors:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
[v0] Model RandomForest loaded successfully
[v0] Prediction result: ...
```

### Next.js Frontend

Check browser console for errors:
```
Chrome DevTools → Console tab
```

---

## Next Steps

1. **Improve Frontend UI** - Customize dashboard and prediction interface
2. **Add Authentication** - Secure API with JWT tokens
3. **Add Caching** - Cache predictions for common feature sets
4. **Model Retraining** - Add periodic retraining endpoint
5. **Advanced Analytics** - Add feature importance, SHAP explanations
6. **Performance Monitoring** - Track API response times, accuracy drift

---

## Contact & Support

For issues or questions:
1. Check logs in relevant terminal
2. Review error messages in browser console
3. Verify all services are running
4. Check API documentation at `http://127.0.0.1:8000/docs`

---

## Files Overview

```
/path/to/project/
├── app/
│   └── main.py              # FastAPI backend with all models
├── code/
│   ├── train_all_models.py  # Training script for all 4 models
│   ├── modeling.py          # Original training code
│   └── mlflow_tracking.py   # MLflow tracking examples
├── front-end/
│   ├── components/
│   │   └── dashboard/
│   │       └── prediction-form.tsx  # Updated prediction form
│   ├── app/
│   │   └── page.tsx         # Main page
│   └── package.json         # Frontend dependencies
├── data/
│   └── arrhythmia.data      # Dataset (must be present)
├── API_DOCUMENTATION.md     # Complete API documentation
├── SETUP_GUIDE.md          # This file
└── training_results.json   # Results from training (auto-generated)
```

---

## Configuration Files

### Backend Configuration (app/main.py)
- `EXPECTED_FEATURES = 278` - Number of input features
- `MODEL_DIR` - Path to MLflow models
- CORS origins - Whitelist of allowed frontend origins

### Frontend Configuration (front-end/.env.local)
- `NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"` - Backend URL

---

Good luck! 🚀
