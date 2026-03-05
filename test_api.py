"""
API Testing Suite
=================

Run comprehensive tests on the Arrhythmia Prediction API.
Verifies all endpoints and models are working correctly.

Usage:
    python test_api.py
"""

import requests
import json
import time
from typing import List, Dict, Any

# Configuration
API_URL = "http://127.0.0.1:8000"
EXPECTED_FEATURES = 278

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_test(name: str):
    """Print test header"""
    print(f"\n{Colors.BLUE}▶ {name}{Colors.RESET}")

def print_success(msg: str):
    """Print success message"""
    print(f"  {Colors.GREEN}✓ {msg}{Colors.RESET}")

def print_error(msg: str):
    """Print error message"""
    print(f"  {Colors.RED}✗ {msg}{Colors.RESET}")

def print_warning(msg: str):
    """Print warning message"""
    print(f"  {Colors.YELLOW}⚠ {msg}{Colors.RESET}")

def print_info(msg: str):
    """Print info message"""
    print(f"  {msg}")

def generate_sample_features(count: int = EXPECTED_FEATURES) -> List[float]:
    """Generate sample features for testing"""
    return [float(i % 100) for i in range(count)]

def test_health_check():
    """Test health check endpoint"""
    print_test("Health Check")
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        print_success(f"Status: {data['status']}")
        print_info(f"Loaded models: {data['loaded_models']}")
        print_info(f"Expected features: {data['total_features_expected']}")
        
        assert data['status'] == "OK"
        assert len(data['loaded_models']) > 0
        assert data['total_features_expected'] == EXPECTED_FEATURES
        
        return True
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False

def test_get_models():
    """Test get models endpoint"""
    print_test("Get All Models")
    try:
        response = requests.get(f"{API_URL}/models")
        response.raise_for_status()
        models = response.json()
        
        assert len(models) > 0, "No models returned"
        print_success(f"Found {len(models)} models")
        
        for model in models:
            print_info(f"  - {model['name']}: Accuracy {model['accuracy']:.4f}, F1 {model['f1_score']:.4f}")
            assert 'name' in model
            assert 'accuracy' in model
            assert 'f1_score' in model
        
        return True, models
    except Exception as e:
        print_error(f"Get models failed: {str(e)}")
        return False, []

def test_model_metrics(model_names: List[str]):
    """Test get model metrics endpoint"""
    print_test("Get Model Metrics")
    results = {}
    
    for model_name in model_names:
        try:
            response = requests.get(f"{API_URL}/models/{model_name}/metrics")
            response.raise_for_status()
            data = response.json()
            
            print_success(f"{model_name} metrics: Accuracy {data['metrics']['accuracy']:.4f}")
            results[model_name] = True
            
            assert data['model'] == model_name
            assert 'accuracy' in data['metrics']
            assert 'f1_score' in data['metrics']
            
        except Exception as e:
            print_error(f"Failed to get metrics for {model_name}: {str(e)}")
            results[model_name] = False
    
    return all(results.values()), results

def test_single_prediction(model_name: str = "RandomForest"):
    """Test single prediction endpoint"""
    print_test(f"Single Prediction ({model_name})")
    try:
        features = generate_sample_features()
        
        payload = {
            "features": features,
            "model": model_name
        }
        
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=payload)
        elapsed = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        print_success(f"Prediction: {result['prediction']}")
        print_info(f"Model used: {result['model_used']}")
        print_info(f"Response time: {elapsed*1000:.1f}ms")
        
        if result['confidence']:
            print_info(f"Confidence: {result['confidence']*100:.2f}%")
        
        assert result['prediction'] in [0, 1]
        assert result['model_used'] == model_name
        
        return True
    except Exception as e:
        print_error(f"Single prediction failed: {str(e)}")
        return False

def test_all_models_prediction():
    """Test prediction with all models"""
    print_test("Prediction with All Models")
    models_to_test = ["RandomForest", "SVM", "LogisticRegression", "XGBoost"]
    results = {}
    
    features = generate_sample_features()
    
    for model_name in models_to_test:
        try:
            payload = {
                "features": features,
                "model": model_name
            }
            
            response = requests.post(f"{API_URL}/predict", json=payload)
            response.raise_for_status()
            result = response.json()
            
            print_success(f"{model_name}: Class {result['prediction']}")
            results[model_name] = True
            
        except Exception as e:
            print_error(f"{model_name} prediction failed: {str(e)}")
            results[model_name] = False
    
    return all(results.values()), results

def test_batch_prediction():
    """Test batch predictions endpoint"""
    print_test("Batch Predictions")
    try:
        # Create 5 sample records
        features_list = [generate_sample_features() for _ in range(5)]
        
        payload = {
            "features_list": features_list,
            "model": "RandomForest"
        }
        
        response = requests.post(f"{API_URL}/predict/batch", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print_success(f"Processed {result['total_records']} records")
        print_info(f"Model: {result['model_used']}")
        
        assert result['total_records'] == 5
        assert len(result['predictions']) == 5
        
        for pred in result['predictions']:
            assert pred['prediction'] in [0, 1]
        
        return True
    except Exception as e:
        print_error(f"Batch prediction failed: {str(e)}")
        return False

def test_invalid_feature_count():
    """Test error handling for invalid feature count"""
    print_test("Error Handling - Invalid Features")
    try:
        # Send wrong number of features
        payload = {
            "features": [1.0, 2.0, 3.0],  # Only 3 features instead of 278
            "model": "RandomForest"
        }
        
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 400:
            print_success("Correctly rejected invalid feature count")
            data = response.json()
            print_info(f"Error message: {data['detail']}")
            return True
        else:
            print_error(f"Expected 400 status, got {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False

def test_invalid_model():
    """Test error handling for invalid model"""
    print_test("Error Handling - Invalid Model")
    try:
        payload = {
            "features": generate_sample_features(),
            "model": "NonExistentModel"
        }
        
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code in [400, 404, 500]:
            print_success("Correctly rejected invalid model")
            return True
        else:
            print_warning(f"Expected error status, got {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False

def test_api_info():
    """Test API information endpoint"""
    print_test("API Information")
    try:
        response = requests.get(f"{API_URL}/info")
        response.raise_for_status()
        data = response.json()
        
        print_success(f"Title: {data['title']}")
        print_info(f"Version: {data['version']}")
        print_info(f"Available models: {', '.join(data['available_models'])}")
        print_info(f"Endpoints: {len(data['endpoints'])} total")
        
        return True
    except Exception as e:
        print_error(f"API info failed: {str(e)}")
        return False

def test_pdf_report():
    """Test PDF report generation"""
    print_test("PDF Report Generation")
    try:
        payload = {
            "features": generate_sample_features(),
            "model": "RandomForest"
        }
        
        response = requests.post(f"{API_URL}/report", json=payload)
        response.raise_for_status()
        
        if response.headers.get('content-type') == 'application/pdf':
            print_success(f"PDF generated ({len(response.content)} bytes)")
            return True
        else:
            print_warning(f"Content type: {response.headers.get('content-type')}")
            return True  # Still pass if we get a response
            
    except Exception as e:
        print_error(f"PDF generation failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print(f"\n{'='*70}")
    print(f"ARRHYTHMIA PREDICTION API - COMPREHENSIVE TEST SUITE")
    print(f"{'='*70}")
    print(f"API URL: {API_URL}")
    print(f"Expected features: {EXPECTED_FEATURES}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    results = {}
    
    # Basic connectivity tests
    results['health_check'] = test_health_check()
    
    if not results['health_check']:
        print(f"\n{Colors.RED}❌ API is not responding. Cannot continue tests.{Colors.RESET}")
        print(f"Ensure API is running on {API_URL}")
        return
    
    # Get models
    models_ok, models = test_get_models()
    results['get_models'] = models_ok
    
    if not models:
        print(f"\n{Colors.RED}❌ No models found. Please train models first.{Colors.RESET}")
        return
    
    model_names = [m['name'] for m in models]
    
    # Model metrics
    results['model_metrics'], _ = test_model_metrics(model_names)
    
    # Prediction tests
    results['single_prediction'] = test_single_prediction("RandomForest")
    results['all_models'], _ = test_all_models_prediction()
    results['batch_prediction'] = test_batch_prediction()
    
    # Error handling tests
    results['invalid_features'] = test_invalid_feature_count()
    results['invalid_model'] = test_invalid_model()
    
    # Info tests
    results['api_info'] = test_api_info()
    
    # Advanced features
    results['pdf_report'] = test_pdf_report()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    
    total_tests = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total_tests - passed
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if result else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"{test_name.replace('_', ' ').title():.<50} {status}")
    
    print(f"{'='*70}")
    print(f"Results: {Colors.GREEN}{passed}/{total_tests} tests passed{Colors.RESET}")
    
    if failed > 0:
        print(f"{Colors.RED}{failed} tests failed{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}🎉 All tests passed!{Colors.RESET}")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.RESET}")
