#!/bin/bash

# CardioSense API Startup Script
# This script initializes the model cache and starts the FastAPI server

set -e

echo "================================================"
echo "CardioSense API - Startup Script"
echo "================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "Step 1: Initializing model cache..."
python code/init_model_cache.py "$SCRIPT_DIR/model_cache"

echo ""
echo "Step 2: Starting FastAPI server..."
echo "Server will be available at: http://127.0.0.1:8000"
echo "API Documentation: http://127.0.0.1:8000/docs"
echo ""

cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
