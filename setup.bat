@echo off
REM Power Quality Anomaly Detection - Setup Script for Windows
REM This script sets up the complete environment

echo ==========================================
echo Power Quality Anomaly Detection Setup
echo ==========================================

REM Check Python version
echo.
echo [1/6] Checking Python version...
python --version

if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

echo OK: Python found

REM Create virtual environment
echo.
echo [2/6] Creating virtual environment...
python -m venv venv

if errorlevel 1 (
    echo Error: Failed to create virtual environment
    exit /b 1
)

echo OK: Virtual environment created

REM Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat

echo OK: Virtual environment activated

REM Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip -q

echo OK: pip upgraded

REM Install dependencies
echo.
echo [5/6] Installing dependencies...
echo This may take a few minutes...

pip install -q numpy pandas scipy scikit-learn matplotlib seaborn
pip install -q xgboost lightgbm plotly streamlit
pip install -q tensorflow

if errorlevel 1 (
    echo Error: Failed to install dependencies
    exit /b 1
)

echo OK: Dependencies installed

REM Create necessary directories
echo.
echo [6/6] Creating project directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "plots" mkdir plots

echo OK: Directories created

REM Summary
echo.
echo ==========================================
echo OK: Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate environment: venv\Scripts\activate
echo 2. Quick test: python quickstart.py
echo 3. Train models: python train.py
echo 4. Launch web app: streamlit run app.py
echo 5. View tutorial: jupyter notebook tutorial.ipynb
echo.
echo For more information, see README.md
echo ==========================================

pause
