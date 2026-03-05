#!/usr/bin/env python3
"""
Test script to verify all models are working correctly
Tests the complete backend API with all 4 models
"""

import requests
import json
import sys
import time
from typing import List

API_URL = "http://127.0.0.1:8000"
EXPECTED_FEATURES = 278

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def generate_sample_features() -> List[float]:
    """Generate 278 sample ECG features"""
    import random
    features = [
        54, 1, 172, 78, 80, 160, 370, 180, 100, 72, 6,
        *[random.random() * 100 for _ in range(20)],
        *[random.random() * 500 for _ in range(50)]
    ]
    while len(features) < EXPECTED_FEATURES:
        features.append(random.random() * 50)
    return features[:EXPECTED_FEATURES]

def test_health():
    """Test health endpoint"""
    print_header("1. Testing Health Endpoint")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed")
            print(f"  Status: {data.get('status')}")
            print(f"  API Version: {data.get('api_version')}")
            print(f"  Loaded Models: {len(data.get('loaded_models', []))} models")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        return False

def test_models_list():
    """Test models list endpoint"""
    print_header("2. Testing Models Endpoint")
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print_success(f"Retrieved {len(models)} models")
            
            expected_models = ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]
            found_models = [m['name'] for m in models]
            
            for model in models:
                print(f"\n  📊 {model['name']}")
                print(f"     Accuracy: {model['accuracy']*100:.2f}%")
                print(f"     F1 Score: {model['f1_score']:.4f}")
                print(f"     Source: {model.get('source', 'unknown')}")
            
            missing = set(expected_models) - set(found_models)
            if missing:
                print_warning(f"Missing models: {missing}")
                return False
            else:
                print_success("All 4 models found!")
                return True
        else:
            print_error(f"Failed to get models: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Models endpoint error: {str(e)}")
        return False

def test_single_prediction(model_name: str) -> bool:
    """Test prediction with a specific model"""
    try:
        features = generate_sample_features()
        
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "features": features,
                "model": model_name
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction')
            confidence = data.get('confidence')
            
            class_name = "Normal (0)" if prediction == 0 else "Arrhythmia (1)"
            print(f"  Prediction: {class_name}")
            if confidence:
                print(f"  Confidence: {confidence*100:.2f}%")
            
            print_success(f"{model_name} prediction successful")
            return True
        else:
            print_error(f"{model_name} prediction failed: {response.status_code}")
            if response.text:
                print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print_error(f"{model_name} prediction error: {str(e)}")
        return False

def test_all_predictions():
    """Test predictions with all models"""
    print_header("3. Testing Predictions for All Models")
    
    models = ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]
    results = {}
    
    for model in models:
        print(f"\n  Testing {model}...")
        results[model] = test_single_prediction(model)
    
    return all(results.values())

def test_batch_prediction():
    """Test batch prediction"""
    print_header("4. Testing Batch Prediction")
    
    try:
        batch_features = [generate_sample_features() for _ in range(3)]
        
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={
                "features_list": batch_features,
                "model": "RandomForest"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print_success(f"Batch prediction successful for {len(predictions)} records")
            
            for pred in predictions:
                class_name = "Normal (0)" if pred['prediction'] == 0 else "Arrhythmia (1)"
                print(f"  Record {pred['index']}: {class_name}")
            
            return True
        else:
            print_error(f"Batch prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Batch prediction error: {str(e)}")
        return False

def test_api_info():
    """Test API info endpoint"""
    print_header("5. Testing API Info Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success("API info retrieved")
            
            print(f"\n  Title: {data.get('title')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Available Models: {len(data.get('available_models', []))}")
            print(f"  Expected Features: {data.get('expected_features')}")
            
            return True
        else:
            print_error(f"Failed to get API info: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"API info error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════╗")
    print("║   CardioSense Backend - Model Testing Suite       ║")
    print("║        Testing All 4 ML Models                    ║")
    print("╚════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    print_info(f"Testing API at: {API_URL}")
    time.sleep(1)
    
    # Run all tests
    results = {}
    
    results['health'] = test_health()
    time.sleep(0.5)
    
    results['models_list'] = test_models_list()
    time.sleep(0.5)
    
    results['predictions'] = test_all_predictions()
    time.sleep(0.5)
    
    results['batch'] = test_batch_prediction()
    time.sleep(0.5)
    
    results['info'] = test_api_info()
    
    # Print summary
    print_header("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20} {status}")
    
    print(f"\n  Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! The API is working correctly.{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some tests failed. Check the errors above.{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
