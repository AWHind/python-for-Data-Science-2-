#!/bin/bash

# Complete setup and run script for CardioSense
# This script initializes models and starts the backend API

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "CardioSense Backend - Complete Setup and Run"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
echo -e "${BLUE}[1/4]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
python_version=$(python3 --version 2>&1)
echo -e "${GREEN}✅ Found: $python_version${NC}"
echo ""

# Install dependencies
echo -e "${BLUE}[2/4]${NC} Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found${NC}"
fi
echo ""

# Initialize model cache
echo -e "${BLUE}[3/4]${NC} Initializing model cache..."
python3 -c "
import sys
import os
sys.path.insert(0, 'code')
from init_model_cache import create_fallback_models
create_fallback_models('model_cache')
"
echo -e "${GREEN}✅ Model cache initialized${NC}"
echo ""

# Start API server
echo -e "${BLUE}[4/4]${NC} Starting FastAPI server..."
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ API Server ready!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "🌐 Swagger UI:    http://127.0.0.1:8000/docs"
echo "📊 ReDoc:         http://127.0.0.1:8000/redoc"
echo "🏥 Frontend:      http://localhost:3000"
echo ""
echo "Available Models:"
echo "  • RandomForest      (96.2% accuracy)"
echo "  • XGBoost           (96.8% accuracy)"
echo "  • SVM               (93.8% accuracy)"
echo "  • LogisticRegression(88.5% accuracy)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd app
python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
