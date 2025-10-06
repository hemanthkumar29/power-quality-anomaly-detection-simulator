# 🔧 Troubleshooting Guide - Power Quality Anomaly Detection

This guide helps you resolve common issues when setting up and running the project.

## 📋 Table of Contents

1. [Installation Issues](#installation-issues)
2. [XGBoost/LightGBM Issues](#xgboostlightgbm-issues)
3. [Import Errors](#import-errors)
4. [Training Issues](#training-issues)
5. [Web App Issues](#web-app-issues)
6. [Performance Issues](#performance-issues)
7. [macOS Specific Issues](#macos-specific-issues)

---

## 🚨 Installation Issues

### Issue: Python not found
**Error**: `python: command not found` or `python3: command not found`

**Solution**:
```bash
# Check if Python is installed
which python3
python3 --version

# If not installed, download from python.org or use homebrew
brew install python3
```

### Issue: pip not found
**Error**: `pip: command not found`

**Solution**:
```bash
# Use python3 -m pip instead
python3 -m pip install -r requirements.txt

# Or install pip
python3 -m ensurepip --upgrade
```

### Issue: Permission denied when installing packages
**Error**: `Permission denied` or `Access denied`

**Solution**:
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Or use --user flag
pip install --user -r requirements.txt
```

---

## ⚡ XGBoost/LightGBM Issues

### Issue: XGBoost library not loaded (macOS)
**Error**: 
```
XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
Library not loaded: @rpath/libomp.dylib
```

**Solution**:
```bash
# Install OpenMP runtime using Homebrew
brew install libomp

# If you don't have Homebrew, install it first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install libomp
brew install libomp
```

**Alternative Solution** (if above doesn't work):
```bash
# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost

# Or try installing from conda
conda install -c conda-forge xgboost
```

**Workaround** (if still not working):
The code is now designed to work without XGBoost/LightGBM. It will:
- Skip these models automatically
- Use Random Forest and SVM instead
- Log warnings but continue execution

### Issue: LightGBM installation fails
**Error**: `Building wheel for lightgbm failed`

**Solution**:
```bash
# On macOS, install OpenMP first
brew install libomp cmake

# Install LightGBM
pip install lightgbm

# Or try pre-built wheel
pip install lightgbm --install-option=--precompiled
```

---

## 📦 Import Errors

### Issue: Module not found
**Error**: `ModuleNotFoundError: No module named 'numpy'` (or other modules)

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt

# Check installed packages
pip list
```

### Issue: ImportError with TensorFlow
**Error**: `ImportError: cannot import name 'something' from tensorflow`

**Solution**:
```bash
# TensorFlow is optional - you can skip neural networks
# Comment out tensorflow in requirements.txt

# Or install compatible version
pip install tensorflow==2.15.0

# On macOS M1/M2, use tensorflow-macos
pip install tensorflow-macos tensorflow-metal
```

---

## 🎓 Training Issues

### Issue: Models not found in web app
**Error**: `Models not loaded` or `No models found`

**Solution**:
```bash
# Train models first
python train.py

# Verify models are saved
ls models/
# Should show: random_forest.pkl, svm.pkl, etc.

# Then launch web app
streamlit run app.py
```

### Issue: Out of memory during training
**Error**: `MemoryError` or system becomes slow

**Solution**:
```bash
# Reduce dataset size
python train.py --n-samples 500

# Or reduce feature dimensions
# Edit src/feature_extraction.py to remove some features

# Close other applications
# Monitor memory: Activity Monitor (Mac) or Task Manager (Windows)
```

### Issue: Training is too slow
**Solution**:
```bash
# Use faster models
python train.py --random-forest --lightgbm

# Reduce samples
python train.py --n-samples 500

# Use fewer cross-validation folds
python train.py --no-cross-validate
```

---

## 🌐 Web App Issues

### Issue: Streamlit command not found
**Error**: `streamlit: command not found`

**Solution**:
```bash
# Activate virtual environment
source venv/bin/activate

# Install streamlit
pip install streamlit

# Or run with python -m
python -m streamlit run app.py
```

### Issue: Port already in use
**Error**: `Address already in use` or `Port 8501 is already in use`

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing streamlit process
# macOS/Linux:
pkill -f streamlit

# Windows:
taskkill /F /IM streamlit.exe
```

### Issue: Browser doesn't open
**Solution**:
```bash
# Manually open URL
# After running streamlit, open browser to:
http://localhost:8501

# Or disable auto-open
streamlit run app.py --server.headless true
```

### Issue: Web app shows errors
**Solution**:
```bash
# Check if models are trained
ls models/

# Train if missing
python train.py

# Clear streamlit cache
streamlit cache clear

# Restart streamlit
# Press Ctrl+C, then run again
streamlit run app.py
```

---

## 🐢 Performance Issues

### Issue: Feature extraction is slow
**Solution**:
- Expected: ~100 waveforms/second
- Use smaller dataset for testing
- Feature extraction is optimized; if still slow, check CPU usage

### Issue: Model training takes too long
**Solution**:
```bash
# Use faster models (LightGBM > XGBoost > Random Forest > SVM)
python train.py --lightgbm

# Reduce samples
python train.py --n-samples 500

# Skip visualization
python train.py --no-visualize
```

### Issue: Prediction is slow in web app
**Solution**:
- Expected: <10ms per prediction
- Use Random Forest or LightGBM (fastest)
- Close other browser tabs
- Check system resources

---

## 🍎 macOS Specific Issues

### Issue: brew command not found
**Solution**:
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (follow instructions after installation)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Issue: Permission denied on setup.sh
**Solution**:
```bash
# Make script executable
chmod +x setup.sh

# Then run
./setup.sh
```

### Issue: Apple Silicon (M1/M2) compatibility
**Solution**:
```bash
# Use Apple Silicon compatible Python
# Install via homebrew
brew install python@3.11

# For TensorFlow on Apple Silicon
pip install tensorflow-macos tensorflow-metal

# XGBoost should work after installing libomp
brew install libomp
```

---

## 🔍 Diagnostic Commands

### Check Python environment
```bash
python --version
which python
pip list
```

### Check installed packages
```bash
pip list | grep -E "numpy|pandas|scikit|xgboost|streamlit"
```

### Check project structure
```bash
ls -la
ls src/
ls models/
ls data/
```

### Check if models are trained
```bash
ls -lh models/
# Should show .pkl files
```

### Test imports
```bash
python -c "import numpy, pandas, sklearn; print('Core packages OK')"
python -c "import xgboost, lightgbm; print('Gradient boosting OK')"
python -c "import streamlit; print('Streamlit OK')"
```

---

## 🆘 Still Having Issues?

### Quick Fixes

1. **Restart everything**:
   ```bash
   # Deactivate and reactivate environment
   deactivate
   source venv/bin/activate
   ```

2. **Fresh install**:
   ```bash
   # Remove virtual environment
   rm -rf venv
   
   # Create new one
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Use minimal requirements**:
   ```bash
   # Install only core packages
   pip install numpy pandas scipy scikit-learn matplotlib seaborn streamlit
   
   # Test with Random Forest only (no XGBoost/LightGBM)
   python quickstart.py
   ```

### Check Error Logs

```bash
# Run with verbose output
python train.py 2>&1 | tee training.log

# Check for specific errors
cat training.log | grep -i error
```

### System Requirements

**Minimum**:
- Python 3.8+
- 4GB RAM
- 1GB free disk space

**Recommended**:
- Python 3.10+
- 8GB RAM
- 2GB free disk space
- macOS 11+ (Big Sur) or equivalent

---

## 📞 Get Help

If issues persist:

1. Check error messages carefully
2. Review documentation:
   - README.md
   - GETTING_STARTED.md
   - This file
3. Check Python version compatibility
4. Try minimal installation (core packages only)
5. Create issue on GitHub (if applicable)

---

## ✅ Verification Checklist

After troubleshooting, verify everything works:

```bash
# 1. Check Python
python --version  # Should be 3.8+

# 2. Check virtual environment
which python  # Should point to venv

# 3. Test imports
python -c "import numpy, sklearn; print('OK')"

# 4. Run quick test
python quickstart.py

# 5. Train models
python train.py --n-samples 500

# 6. Launch web app
streamlit run app.py
```

---

**Last Updated**: October 2025  
**Status**: ✅ Project fully functional with error handling for optional dependencies

*For more help, see README.md and GETTING_STARTED.md*
