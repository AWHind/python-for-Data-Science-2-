# Debugging Path - Model Loading Issue

## Understanding the Problem

### Original Error
```
Can't get attribute 'SimpleECGModel'
```

### Why This Happened

When `pickle.load(file)` is called on a `.pkl` file containing a SimpleECGModel instance:

```
1. Python reads the pickle file
2. Pickle finds: "Class SimpleECGModel from module init_model_cache"
3. Python tries: from init_model_cache import SimpleECGModel
4. Error: ❌ Module init_model_cache is not in the import path!
5. Result: "Can't get attribute 'SimpleECGModel'"
```

### The Root Cause

SimpleECGModel was defined in `init_model_cache.py` but:
- `init_model_cache.py` is a script, not an importable module
- Pickle records the definition location
- When loading, pickle can't find the class anymore
- Fallback to dummy model (bad!)

---

## The Solution

### What We Fixed

1. **Created `code/ml_models.py`**
   - Moved SimpleECGModel here
   - Now in an importable module
   - Pickle can find it: `from code.ml_models import SimpleECGModel`

2. **Updated `code/model_cache.py`**
   - Changed: `from ml_models import SimpleECGModel, DummyModel`
   - Removed local DummyModel definition
   - Now uses importable classes

3. **Updated `code/init_model_cache.py`**
   - Changed: `from ml_models import SimpleECGModel`
   - Removed local SimpleECGModel definition
   - Creates instances using importable class

### How It Works Now

```
1. init_model_cache.py runs:
   model = SimpleECGModel("RandomForest")  (from ml_models.py)
   pickle.dump(model, file)
   → File recorded: "Class SimpleECGModel from code.ml_models"

2. model_cache.py loads the file:
   model = pickle.load(file)
   → Python: "Find SimpleECGModel in code.ml_models"
   → Success! ✅
   → model.predict() works!
```

---

## Debugging Checklist

If models still aren't loading, check this in order:

### 1. Check File Exists
```bash
ls -la code/ml_models.py
```
Should show the file exists (126 lines)

### 2. Check Import Works
```bash
python3 -c "from code.ml_models import SimpleECGModel; print('✅ OK')"
```
Should print: `✅ OK`

### 3. Check Pickle Works
```bash
python3 << 'EOF'
import pickle
import sys
sys.path.insert(0, 'code')
from ml_models import SimpleECGModel

# Create and pickle
model = SimpleECGModel("test")
with open('test.pkl', 'wb') as f:
    pickle.dump(model, f)

# Unpickle
with open('test.pkl', 'rb') as f:
    loaded = pickle.load(f)

print(f"✅ Pickle works: {loaded}")
EOF
```
Should print: `✅ Pickle works: SimpleECGModel(...)`

### 4. Check Cache Initialization
```bash
rm -rf model_cache/
python3 code/init_model_cache.py
ls -la model_cache/
```
Should create 4 .pkl files

### 5. Check Model Loading
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'code')
from model_cache import ModelCache

cache = ModelCache()
for model_name in ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]:
    model, info = cache.load_model(model_name)
    print(f"✅ {model_name}: {info.source}")
EOF
```
Should show 4 models loading (source: cache, mlflow, or dummy)

### 6. Check API Startup
```bash
cd app
python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Should show:
```
INFO:     Application startup complete
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 7. Check API Health
```bash
curl http://127.0.0.1:8000/health
```
Should return JSON with status "OK"

### 8. Check Models List
```bash
curl http://127.0.0.1:8000/models | python3 -m json.tool
```
Should list 4 models with metrics

### 9. Check Predictions
```bash
python3 test_all_models.py
```
Should show all tests passing

---

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError: No module named 'ml_models'

**Cause**: Import path incorrect

**Solution**:
```bash
# Check file exists
ls code/ml_models.py

# Check import works
python3 -c "import sys; sys.path.insert(0, 'code'); from ml_models import SimpleECGModel"

# If still fails, check Python path
python3 -c "import sys; print(sys.path)"
```

### Issue 2: Can't get attribute 'SimpleECGModel'

**Cause**: Old pickle files with wrong class path

**Solution**:
```bash
# Delete old pickle files
rm -rf model_cache/*.pkl

# Reinitialize
python3 code/init_model_cache.py

# Verify
ls model_cache/
# Should list 4 fresh .pkl files
```

### Issue 3: API starts but /models returns []

**Cause**: Model cache initialization failed silently

**Solution**:
```bash
# Check cache was created
ls -la model_cache/

# Check cache contains files
ls -la model_cache/*.pkl

# Try loading manually
python3 << 'EOF'
import sys
sys.path.insert(0, 'code')
from model_cache import ModelCache
cache = ModelCache()
models = cache.get_all_models_info()
print(f"Models: {len(models)}")
for m in models:
    print(f"  - {m['name']}: {m['source']}")
EOF
```

### Issue 4: Pickle errors about missing modules

**Cause**: Python path issues

**Solution**:
```bash
# Ensure you're in project root
pwd
# Should show: .../project

# Try re-creating cache
rm -rf model_cache/
python3 code/init_model_cache.py

# Check the generated files
python3 << 'EOF'
import pickle
import os

for f in os.listdir('model_cache'):
    if f.endswith('.pkl'):
        print(f"Testing {f}...")
        try:
            with open(f'model_cache/{f}', 'rb') as pf:
                model = pickle.load(pf)
            print(f"  ✅ {f} loads OK")
        except Exception as e:
            print(f"  ❌ {f} error: {e}")
EOF
```

---

## Step-by-Step Debugging

### Case 1: Models not loading in API

**Check 1**: API starting?
```bash
curl http://127.0.0.1:8000/health
```
If fails → Start API first

**Check 2**: Models endpoint returns data?
```bash
curl http://127.0.0.1:8000/models
```
If empty → Cache initialization failed

**Check 3**: Cache files exist?
```bash
ls -la model_cache/
```
If empty → Run `python3 code/init_model_cache.py`

**Check 4**: Can load directly?
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'code')
from model_cache import ModelCache
cache = ModelCache()
model, info = cache.load_model("RandomForest")
print(f"Loaded: {model}")
EOF
```
If fails → Check ml_models.py exists and imports work

### Case 2: Pickle errors when loading

**Check 1**: ml_models.py exists?
```bash
ls code/ml_models.py
```
If no → File wasn't created

**Check 2**: Can import?
```bash
python3 -c "from code.ml_models import SimpleECGModel; print('OK')"
```
If fails → Check file contents

**Check 3**: Check module path in pickle
```bash
python3 << 'EOF'
import pickle
import pickletools

with open('model_cache/RandomForest_model.pkl', 'rb') as f:
    pickletools.dis(f)  # Shows pickle contents
EOF
```
Look for: `GLOBAL 'code.ml_models' 'SimpleECGModel'` or similar

**Check 4**: Try recreating cache
```bash
rm -rf model_cache/
python3 code/init_model_cache.py
```

### Case 3: Frontend can't access models

**Check 1**: Is API running?
```bash
curl http://127.0.0.1:8000/health
```

**Check 2**: Can access models endpoint?
```bash
curl http://127.0.0.1:8000/models
```

**Check 3**: Check browser console for errors
- Open browser DevTools (F12)
- Network tab
- Look for failed requests to `/models`

**Check 4**: Check CORS
- API logs should show CORS headers
- Look for: `Access-Control-Allow-Origin`

**Check 5**: Check frontend URL
- Open: http://127.0.0.1:8000/docs
- Test endpoints manually in Swagger UI

---

## Verification Test

Run this complete test:

```bash
#!/bin/bash

echo "========== CardioSense Model Loading Test =========="
echo ""

# Test 1: File exists
echo -n "1. Checking code/ml_models.py... "
if [ -f "code/ml_models.py" ]; then
    echo "✅"
else
    echo "❌"
    exit 1
fi

# Test 2: Import works
echo -n "2. Testing import... "
if python3 -c "from code.ml_models import SimpleECGModel" 2>/dev/null; then
    echo "✅"
else
    echo "❌"
    exit 1
fi

# Test 3: Cache creates files
echo -n "3. Initializing cache... "
rm -rf model_cache/
if python3 code/init_model_cache.py > /dev/null 2>&1; then
    echo "✅"
else
    echo "❌"
    exit 1
fi

# Test 4: Pickle files created
echo -n "4. Checking pickle files... "
count=$(ls model_cache/*.pkl 2>/dev/null | wc -l)
if [ "$count" -eq "4" ]; then
    echo "✅ ($count files)"
else
    echo "❌ (found $count files, expected 4)"
    exit 1
fi

# Test 5: Load models
echo -n "5. Loading models... "
if python3 << 'EOF' 2>/dev/null
import sys
sys.path.insert(0, 'code')
from model_cache import ModelCache
cache = ModelCache()
for name in ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]:
    m, i = cache.load_model(name)
    assert m is not None
print("OK")
EOF
then
    echo "✅"
else
    echo "❌"
    exit 1
fi

# Test 6: API startup
echo -n "6. Starting API... "
cd app
timeout 5 python3 -m uvicorn main:app --reload > /dev/null 2>&1 &
sleep 2
cd ..
if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "✅"
else
    echo "❌"
    exit 1
fi

# Test 7: Models endpoint
echo -n "7. Testing /models endpoint... "
count=$(curl -s http://127.0.0.1:8000/models | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
if [ "$count" -eq "4" ]; then
    echo "✅ (4 models)"
else
    echo "❌ (found $count models)"
    exit 1
fi

echo ""
echo "========== All Tests Passed! =========="
```

---

## Next Steps If Still Failing

1. **Save this file content**
   ```bash
   cat debug.log  # Check for error messages
   ```

2. **Check directory structure**
   ```bash
   find . -name "*.py" | grep -E "(ml_models|model_cache|init)" | head -20
   ```

3. **Verify Python version**
   ```bash
   python3 --version  # Should be 3.7+
   ```

4. **Check dependencies**
   ```bash
   python3 -c "import numpy, pandas, sklearn, pickle; print('✅ All OK')"
   ```

5. **Read full solution docs**
   - MODEL_LOADING_FIX.md (technical)
   - TROUBLESHOOTING.md (solutions)
   - DETAILED_CHANGES.md (code changes)

---

## Key Takeaway

The fix is simple:
1. Classes need to be in an **importable module** (not a script)
2. Pickle records the module path when saving
3. Pickle uses that path when loading
4. If path doesn't work → Error!

We moved SimpleECGModel from `init_model_cache.py` (script) to `code/ml_models.py` (module) so pickle can always find it.

**Result**: All 4 models work! ✅
