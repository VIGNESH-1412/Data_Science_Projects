#!/bin/bash
# ============================================================
# Stroke Prediction System - Quick Start Script
# ============================================================

echo "============================================"
echo "  AI Stroke Prediction System"
echo "  Clinical Stroke Risk Dashboard"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    exit 1
fi

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "[INFO] Installing dependencies..."
pip install -q -r requirements.txt

# Check model file
if [ ! -f "data/Stroke_Prediction.pkl" ]; then
    echo "[INFO] No model file found. Creating sample model..."
    python3 load_model.py
fi

# Start the application
echo ""
echo "[OK] Starting Stroke Prediction System..."
echo "============================================"
echo "  Open: http://127.0.0.1:5000"
echo "============================================"
echo ""

python3 app.py
