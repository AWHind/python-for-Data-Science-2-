#!/usr/bin/env python
"""
Quick Test Script - Verify all models are working

Run this to verify the API and models are functioning correctly.
"""

import requests
import json
import time
import sys
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
EXPECTED_FEATURES = 278

def generate_sample_features():
    """Generate 278 sample ECG features"""
    return [float(i) * 0.1 for i in range(EXPECTED_FEATURES)]

def test_health():
    """Test API health check"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy")
            print(f"   - Status: {data['status']}")
            print(f"   - Version: {data['api_version']}")
            print(f"   - Models loaded: {data['models_loaded']}")
            print(f"   - Cache enabled: {data['cache_enabled']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        print(f"   Make sure backend is running: bash start-api.sh")
        return False

def test_models_endpoint():
    """Test models listing endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Models Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Found {len(models)} models")
            
            for model in models:
                print(f"\n   {model['name']}:")
                print(f"     - Accuracy: {model['accuracy']:.2%}")
                print(f"     - F1 Score: {model['f1_score']:.4f}")
                print(f"     - Source: {model['source']}")
            
            return len(models) == 4
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predictions():
    """Test predictions with all models"""
    print("\n" + "="*60)
    print("TEST 3: Predictions with All Models")
    print("="*60)
    
    models = ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]
    features = generate_sample_features()
    all_passed = True
    
    for model_name in models:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features, "model": model_name}
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = "Normal" if data['prediction'] == 0 else "Arrhythmia"
                print(f"✅ {model_name}: {prediction} (accuracy: {data['confidence']:.2%})")
            else:
                print(f"❌ {model_name}: {response.status_code} - {response.text}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")
            all_passed = False
    
    return all_passed

def test_batch_prediction():
    """Test batch predictions"""
    print("\n" + "="*60)
    print("TEST 4: Batch Predictions")
    print("="*60)
    
    features_list = [generate_sample_features() for _ in range(3)]
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"features_list": features_list, "model": "RandomForest"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful")
            print(f"   - Total records: {data['total_records']}")
            print(f"   - Model used: {data['model_used']}")
            print(f"   - Predictions: {[p['prediction'] for p in data['predictions']]}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_metrics():
    """Test individual model metrics"""
    print("\n" + "="*60)
    print("TEST 5: Individual Model Metrics")
    print("="*60)
    
    models = ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]
    all_passed = True
    
    for model_name in models:
        try:
            response = requests.get(f"{API_URL}/models/{model_name}/metrics")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {model_name}: Accuracy={data['accuracy']:.2%}, F1={data['f1_score']:.4f}")
            else:
                print(f"❌ {model_name}: {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CARDIOSENSE - QUICK TEST")
    print("="*60)
    print(f"Testing API at: {API_URL}")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    
    if not results[0][1]:
        print("\n❌ API is not running!")
        print("Start it with: bash start-api.sh")
        sys.exit(1)
    
    time.sleep(0.5)
    results.append(("Models Endpoint", test_models_endpoint()))
    time.sleep(0.5)
    results.append(("Predictions", test_predictions()))
    time.sleep(0.5)
    results.append(("Batch Predictions", test_batch_prediction()))
    time.sleep(0.5)
    results.append(("Model Metrics", test_model_metrics()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ All 4 models are working correctly!")
        print("\nYou can now:")
        print("  1. Open Swagger UI: http://127.0.0.1:8000/docs")
        print("  2. Test frontend: http://localhost:3000")
        print("  3. View docs: MODELS_INTEGRATION_COMPLETE.md")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        print("Check the output above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
