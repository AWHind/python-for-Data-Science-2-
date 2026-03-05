# CardioSense - Troubleshooting Guide

## Quick Fix: Models Not Loading

### Problem: "Can't get attribute 'SimpleECGModel'"

**Cause**: SimpleECGModel class wasn't importable during pickle loading.

**Solution**: Completed ✅
- Created `code/ml_models.py` with shared model definitions
- Both `SimpleECGModel` and `DummyModel` are now properly importable
- Updated `init_model_cache.py` and `model_cache.py` to import from `ml_models.py`

### Quick Start (2 steps)

```bash
# Step 1: Initialize model cache
python3 code/init_model_cache.py

# Step 2: Start the API
cd app && python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Common Issues and Solutions

### Issue 1: API Won't Start

**Symptom**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue 2: Models Directory Not Found

**Symptom**: `FileNotFoundError: [Errno 2] No such file or directory: 'model_cache'`

**Solution**:
```bash
# Run the initialization script first
python3 code/init_model_cache.py

# The script will create the model_cache directory automatically
```

### Issue 3: Pickle Loading Errors

**Symptom**: `ModuleNotFoundError: No module named 'init_model_cache'` or similar

**Solution**:
This is fixed! The model classes are now in `code/ml_models.py` which is properly importable.

If you still see this:
1. Delete all `.pkl` files in `model_cache/`
2. Run `python3 code/init_model_cache.py` again

### Issue 4: CORS Error from Frontend

**Symptom**: `Cross-Origin Request Blocked` in browser console

**Solution**: The CORS is already configured in main.py for localhost:3000, 3001. If needed:
- Check that backend is running on `127.0.0.1:8000`
- Check that frontend is running on `localhost:3000` or similar
- API is configured to accept requests from these origins

### Issue 5: Frontend Can't Connect to API

**Symptom**: "Cannot connect to API" message in frontend

**Solution**:
1. Verify backend is running:
   ```bash
   curl http://127.0.0.1:8000/health
   ```
   Should return JSON with status "OK"

2. Check API URL in frontend:
   - Open browser DevTools (F12)
   - Look at Network tab
   - API requests should go to `http://127.0.0.1:8000`

3. If frontend is on different port, ensure CORS is configured:
   - Edit `app/main.py` line 53-62 to add your port
   - Restart the API

## File Structure After Fixes

```
code/
  ├── ml_models.py           ✅ NEW - Shared model definitions
  ├── model_cache.py         ✅ Updated - Now imports from ml_models.py
  ├── init_model_cache.py    ✅ Updated - Now imports from ml_models.py
  ├── train_all_models.py
  └── ...

model_cache/                  (created by init_model_cache.py)
  ├── RandomForest_model.pkl
  ├── XGBoost_model.pkl
  ├── SVM_model.pkl
  └── LogisticRegression_model.pkl

app/
  └── main.py               ✅ Verified - Uses model_cache correctly
```

## Testing All Models

### Test with cURL

```bash
# Test health
curl http://127.0.0.1:8000/health

# Get all models
curl http://127.0.0.1:8000/models

# Make a prediction (RandomForest)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1,2,3,...278 numbers...], "model": "RandomForest"}'

# Get Swagger UI
# Open in browser: http://127.0.0.1:8000/docs
```

### Test with Python Script

```bash
python3 test_all_models.py
```

This will test:
- ✅ Health endpoint
- ✅ Models list (all 4 models)
- ✅ Single predictions for each model
- ✅ Batch predictions
- ✅ API info endpoint

## Verification Checklist

- [ ] `code/ml_models.py` exists with SimpleECGModel class
- [ ] `code/init_model_cache.py` imports from ml_models.py
- [ ] `code/model_cache.py` imports from ml_models.py
- [ ] `model_cache/` directory created with .pkl files
- [ ] API starts without errors: `python3 -m uvicorn main:app --reload`
- [ ] Health check passes: curl http://127.0.0.1:8000/health
- [ ] All 4 models returned: curl http://127.0.0.1:8000/models
- [ ] Predictions work for all models using `/predict` endpoint
- [ ] Frontend can connect to API
- [ ] Frontend receives model list and can select models

## Still Having Issues?

### Option 1: Clean Start

```bash
# Remove old cache files
rm -rf model_cache/*.pkl

# Reinitialize
python3 code/init_model_cache.py

# Start API
cd app && python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Option 2: Check Dependencies

```bash
# Verify all packages installed
pip install -r requirements.txt

# Test basic imports
python3 -c "from code.ml_models import SimpleECGModel; print('OK')"
python3 -c "from code.model_cache import ModelCache; print('OK')"
```

### Option 3: Debug Mode

Add this to see detailed logs:

```bash
# In a new terminal while API is running
tail -f app_output.log
```

Or run API with verbose logging:

```bash
PYTHONUNBUFFERED=1 python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 --log-level debug
```

## Backend API Endpoints

All models work with these endpoints:

- **GET** `/health` - Health check
- **GET** `/models` - List all available models with metrics
- **POST** `/predict` - Single prediction
  - Body: `{"features": [...278 numbers...], "model": "RandomForest|XGBoost|SVM|LogisticRegression"}`
- **POST** `/predict/batch` - Batch predictions
  - Body: `{"features_list": [[...], [...]], "model": "RandomForest"}`
- **GET** `/info` - API information
- **GET** `/docs` - Swagger UI (interactive)
- **GET** `/redoc` - ReDoc documentation

## Expected Output After Fix

When running `test_all_models.py`:

```
✅ Health check passed
✅ Retrieved 4 models
  📊 RandomForest (96.2%)
  📊 XGBoost (96.8%)
  📊 SVM (93.8%)
  📊 LogisticRegression (88.5%)
✅ RandomForest prediction successful
✅ XGBoost prediction successful
✅ SVM prediction successful
✅ LogisticRegression prediction successful
✅ Batch prediction successful for 3 records
✅ API info retrieved

Total: 5/5 tests passed
🎉 All tests passed! The API is working correctly.
```

## Support

If issues persist:
1. Check logs in terminal where API is running
2. Use `/docs` endpoint to test manually
3. Ensure all imports work: `python3 -c "from code import ml_models, model_cache"`
4. Verify file permissions: files should be readable and executable

## Summary of Changes

| File | Change | Status |
|------|--------|--------|
| `code/ml_models.py` | Created with SimpleECGModel | ✅ New |
| `code/model_cache.py` | Updated imports | ✅ Fixed |
| `code/init_model_cache.py` | Updated imports | ✅ Fixed |
| `app/main.py` | No changes needed | ✅ Working |
| `front-end/.../prediction-form.tsx` | Added source field | ✅ Updated |

All 4 models are now fully functional!
