#!/usr/bin/env python3
"""
Setup Verification Script
=========================

Verifies that all components are properly installed and configured.

Usage:
    python verify_setup.py
"""

import os
import sys
import subprocess
import requests
import json
from pathlib import Path
from typing import Tuple, List

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_header(msg: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}{Colors.RESET}\n")

def check_success(name: str, details: str = ""):
    """Print success"""
    msg = f"{Colors.GREEN}✓{Colors.RESET} {name}"
    if details:
        msg += f" ({details})"
    print(msg)

def check_error(name: str, details: str = ""):
    """Print error"""
    msg = f"{Colors.RED}✗{Colors.RESET} {name}"
    if details:
        msg += f" - {details}"
    print(msg)

def check_warning(name: str, details: str = ""):
    """Print warning"""
    msg = f"{Colors.YELLOW}⚠{Colors.RESET} {name}"
    if details:
        msg += f" - {details}"
    print(msg)

def check_file_exists(path: str, description: str = "") -> bool:
    """Check if file exists"""
    exists = os.path.exists(path)
    if exists:
        check_success(f"File exists: {path}", description)
    else:
        check_error(f"File missing: {path}", description)
    return exists

def check_python_package(package: str, import_name: str = None) -> bool:
    """Check if Python package is installed"""
    if import_name is None:
        import_name = package.replace("-", "_")
    
    try:
        __import__(import_name)
        check_success(f"Python package: {package}")
        return True
    except ImportError:
        check_error(f"Python package missing: {package}")
        return False

def check_port_open(host: str, port: int, name: str) -> bool:
    """Check if port is open (service running)"""
    try:
        result = requests.get(f"http://{host}:{port}/", timeout=2)
        check_success(f"{name} running on {host}:{port}")
        return True
    except:
        check_error(f"{name} not running on {host}:{port}")
        return False

def check_api_endpoint(endpoint: str, name: str) -> bool:
    """Check if API endpoint is working"""
    try:
        response = requests.get(f"http://127.0.0.1:8000{endpoint}", timeout=2)
        if response.status_code == 200:
            check_success(f"API endpoint: {endpoint} ({name})")
            return True
        else:
            check_error(f"API endpoint: {endpoint} returned {response.status_code}")
            return False
    except Exception as e:
        check_error(f"API endpoint: {endpoint} - {str(e)}")
        return False

def check_dataset() -> bool:
    """Check if dataset exists"""
    dataset_path = os.path.join("data", "arrhythmia.data")
    if os.path.exists(dataset_path):
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        check_success(f"Dataset found", f"{size_mb:.1f} MB")
        return True
    else:
        check_error(f"Dataset not found at: {dataset_path}")
        return False

def check_directories() -> bool:
    """Check if required directories exist"""
    required_dirs = [
        ("app", "Backend"),
        ("code", "Code"),
        ("data", "Data"),
        ("front-end", "Frontend"),
    ]
    
    all_exist = True
    for dir_name, description in required_dirs:
        if os.path.isdir(dir_name):
            check_success(f"Directory: {dir_name}", description)
        else:
            check_error(f"Directory missing: {dir_name}", description)
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks"""
    
    print_header("PROJECT SETUP VERIFICATION")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    results = {
        "directories": False,
        "dataset": False,
        "python_packages": False,
        "backend": False,
        "mlflow": False,
        "frontend": False,
        "overall": False
    }
    
    # 1. Check directories
    print_header("1. Directory Structure")
    results["directories"] = check_directories()
    
    # 2. Check dataset
    print_header("2. Dataset")
    results["dataset"] = check_dataset()
    
    # 3. Check Python packages
    print_header("3. Python Packages")
    required_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("mlflow", "mlflow"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("imbalanced-learn", "imblearn"),
        ("reportlab", "reportlab"),
        ("pydantic", "pydantic"),
        ("requests", "requests"),
    ]
    
    all_packages = True
    for package, import_name in required_packages:
        if not check_python_package(package, import_name):
            all_packages = False
    
    results["python_packages"] = all_packages
    
    # 4. Check backend files
    print_header("4. Backend Files")
    backend_files = [
        ("app/main.py", "FastAPI application"),
        ("code/train_all_models.py", "Model training script"),
        ("code/modeling.py", "Original modeling code"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    all_backend_files = True
    for file_path, description in backend_files:
        if not check_file_exists(file_path, description):
            all_backend_files = False
    
    results["backend"] = all_backend_files
    
    # 5. Check documentation files
    print_header("5. Documentation")
    doc_files = [
        ("API_DOCUMENTATION.md", "API reference"),
        ("SETUP_GUIDE.md", "Setup instructions"),
        ("README_COMPLETE.md", "Complete guide"),
        ("QUICK_START.md", "Quick start guide"),
    ]
    
    for file_path, description in doc_files:
        check_file_exists(file_path, description)
    
    # 6. Check frontend
    print_header("6. Frontend")
    frontend_files = [
        ("front-end/package.json", "NPM configuration"),
        ("front-end/components/dashboard/prediction-form.tsx", "Prediction form"),
        ("front-end/components/admin/admin-models.tsx", "Admin dashboard"),
    ]
    
    all_frontend_files = True
    for file_path, description in frontend_files:
        if not check_file_exists(file_path, description):
            all_frontend_files = False
    
    results["frontend"] = all_frontend_files
    
    # 7. Check running services
    print_header("7. Running Services")
    print("Checking if services are running...\n")
    
    mlflow_running = check_port_open("127.0.0.1", 5000, "MLflow Server")
    results["mlflow"] = mlflow_running
    
    api_running = check_port_open("127.0.0.1", 8000, "FastAPI Backend")
    if api_running:
        # Check specific endpoints
        check_api_endpoint("/health", "Health check")
        check_api_endpoint("/models", "Get models")
        check_api_endpoint("/info", "API info")
        check_api_endpoint("/docs", "Swagger UI")
    
    frontend_running = check_port_open("localhost", 3000, "Frontend Server")
    
    if not api_running:
        print(f"\n{Colors.YELLOW}Note:{Colors.RESET} Backend not running. Start it with:")
        print(f"  cd app && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000")
    
    if not mlflow_running:
        print(f"\n{Colors.YELLOW}Note:{Colors.RESET} MLflow not running. Start it with:")
        print(f"  mlflow server --host 127.0.0.1 --port 5000")
    
    if not frontend_running:
        print(f"\n{Colors.YELLOW}Note:{Colors.RESET} Frontend not running. Start it with:")
        print(f"  cd front-end && npm run dev")
    
    # Summary
    print_header("Verification Summary")
    
    checks = [
        ("✓ Directories", results["directories"]),
        ("✓ Dataset", results["dataset"]),
        ("✓ Python packages", results["python_packages"]),
        ("✓ Backend files", results["backend"]),
        ("✓ Frontend files", results["frontend"]),
        ("✓ MLflow service", results["mlflow"]),
        ("✓ API service", results.get("backend", False)),
        ("✓ Frontend service", frontend_running),
    ]
    
    for check_name, passed in checks:
        if passed:
            print(f"{Colors.GREEN}✓{Colors.RESET} {check_name.replace('✓ ', '')}")
        else:
            print(f"{Colors.RED}✗{Colors.RESET} {check_name.replace('✓ ', '')}")
    
    # Final verdict
    print(f"\n{'='*70}")
    
    essential_checks = [
        results["directories"],
        results["dataset"],
        results["python_packages"],
        results["backend"],
        results["frontend"],
    ]
    
    if all(essential_checks):
        print(f"{Colors.GREEN}✓ SETUP VERIFICATION PASSED{Colors.RESET}")
        print("\nEssential components are installed and configured.")
        print("\nNext steps:")
        print("1. Start MLflow: mlflow server --host 127.0.0.1 --port 5000")
        print("2. Start Backend: cd app && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        print("3. Start Frontend: cd front-end && npm run dev")
        print("4. Train models (if not done): python code/train_all_models.py")
        print("5. Test API: python test_api.py")
    else:
        print(f"{Colors.RED}✗ SETUP VERIFICATION FAILED{Colors.RESET}")
        print("\nPlease install missing components:")
        if not results["directories"]:
            print("  - Missing required directories")
        if not results["dataset"]:
            print("  - Add dataset to data/arrhythmia.data")
        if not results["python_packages"]:
            print("  - Run: pip install -r requirements.txt")
        if not results["backend"]:
            print("  - Check backend files")
        if not results["frontend"]:
            print("  - Check frontend files")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Verification interrupted{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Error during verification: {str(e)}{Colors.RESET}")
