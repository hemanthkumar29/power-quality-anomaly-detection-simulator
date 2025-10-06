#!/bin/bash

# Power Quality Anomaly Detection - Setup Script
# This script sets up the complete environment

echo "=========================================="
echo "Power Quality Anomaly Detection Setup"
echo "=========================================="

# Check Python version
echo ""
echo "[1/6] Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found"

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip -q

echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "[5/6] Installing dependencies..."
echo "This may take a few minutes..."

pip install -q numpy pandas scipy scikit-learn matplotlib seaborn
pip install -q xgboost lightgbm plotly streamlit
pip install -q tensorflow  # Optional, for neural networks

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "[6/6] Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p plots

echo "✓ Directories created"

# Summary
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Quick test: python quickstart.py"
echo "3. Train models: python train.py"
echo "4. Launch web app: streamlit run app.py"
echo "5. View tutorial: jupyter notebook tutorial.ipynb"
echo ""
echo "For more information, see README.md"
echo "=========================================="
