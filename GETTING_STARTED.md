# 🚀 Getting Started with Power Quality Anomaly Detection

Welcome! This guide will help you get up and running in 5 minutes.

## ⚡ Prerequisites

- **Python 3.8+** installed on your system
- **Terminal/Command Prompt** access
- **5-10 minutes** of time

## 📋 Step-by-Step Guide

### Step 1: Verify Python Installation

Open terminal and run:

```bash
python --version
# or
python3 --version
```

You should see Python 3.8 or higher.

### Step 2: Navigate to Project Directory

```bash
cd /Users/hemanthkumar/power-quality
```

### Step 3: Run Setup Script

**On macOS/Linux:**
```bash
./setup.sh
```

**On Windows:**
```bash
setup.bat
```

This will:
- Create virtual environment
- Install all dependencies
- Create necessary directories

⏱️ **Time**: 3-5 minutes

### Step 4: Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You'll see `(venv)` in your terminal prompt.

### Step 5: Test the System

Run the quick test:

```bash
python quickstart.py
```

✅ This will:
- Generate sample data
- Extract features
- Train a model
- Make a prediction
- Create a visualization

⏱️ **Time**: 30 seconds

### Step 6: Train Full Models (Optional)

```bash
python train.py --n-samples 500
```

✅ This will:
- Generate larger dataset
- Train all 4 ML models
- Evaluate performance
- Save models and plots

⏱️ **Time**: 2-3 minutes

### Step 7: Launch Web Application

```bash
streamlit run app.py
```

✅ Your browser will open to `http://localhost:8501`

⏱️ **Instant**

## 🎮 Using the Web Application

1. **Select Data Source**:
   - "Generate Synthetic Waveform" (easiest)
   - "Upload CSV File" (your data)
   - "Use Sample Dataset" (pre-trained)

2. **Choose Anomaly Type**:
   - Normal
   - Sag
   - Swell
   - Harmonic
   - Outage

3. **Click "Generate Waveform"**

4. **View Visualizations**:
   - Time Domain tab
   - Frequency Domain tab
   - Combined tab

5. **Click "Classify Anomaly"**

6. **See Results**:
   - Predicted class
   - Confidence scores
   - Feature values

## 📊 Example Workflow

### Quick Demo (1 minute)

```bash
# 1. Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Quick test
python quickstart.py

# 3. Launch app
streamlit run app.py
```

### Full Training (5 minutes)

```bash
# 1. Train models with visualization
python train.py --n-samples 1000 --visualize

# 2. Check results
ls plots/
ls models/

# 3. Launch app
streamlit run app.py
```

### Interactive Learning (15 minutes)

```bash
# 1. Start Jupyter
jupyter notebook tutorial.ipynb

# 2. Run cells step by step
# 3. Experiment with parameters
```

## 💻 Command Cheat Sheet

### Training Commands
```bash
# Basic training
python train.py

# Quick training (fewer samples)
python train.py --n-samples 500

# Train specific models
python train.py --xgboost --lightgbm

# With cross-validation
python train.py --cross-validate

# Without visualization
python train.py --no-visualize
```

### Web App Commands
```bash
# Standard launch
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# Headless (no browser)
streamlit run app.py --server.headless true
```

### Testing Commands
```bash
# Quick system test
python quickstart.py

# Dataset information
python DATASETS.py

# Check what's installed
pip list | grep -E "numpy|pandas|scikit|xgboost|streamlit"
```

## 🔍 Verify Installation

Run these to verify everything is working:

```bash
# Check Python modules
python -c "import numpy, pandas, sklearn, xgboost, streamlit; print('✓ All imports OK')"

# Check project structure
ls src/
# Should show: data_loader.py, feature_extraction.py, etc.

# Check if models exist (after training)
ls models/
# Should show: .pkl files
```

## 🐛 Troubleshooting

### Issue: "Python not found"
**Solution**: Install Python 3.8+ from python.org

### Issue: "Permission denied: ./setup.sh"
**Solution**: 
```bash
chmod +x setup.sh
./setup.sh
```

### Issue: "Module not found"
**Solution**:
```bash
source venv/bin/activate  # Activate environment first
pip install -r requirements.txt
```

### Issue: "Models not found in web app"
**Solution**:
```bash
python train.py  # Train models first
```

### Issue: "Streamlit command not found"
**Solution**:
```bash
pip install streamlit
# or
python -m streamlit run app.py
```

### Issue: "Port already in use"
**Solution**:
```bash
streamlit run app.py --server.port 8502
```

## 📚 Next Steps

### Beginner Path
1. ✅ Run `quickstart.py`
2. ✅ Launch web app
3. ✅ Generate different anomalies
4. ✅ Try classifying them
5. ✅ Read README.md

### Intermediate Path
1. ✅ Train models with `train.py`
2. ✅ Review generated plots
3. ✅ Check confusion matrices
4. ✅ Try different sample sizes
5. ✅ Open tutorial.ipynb

### Advanced Path
1. ✅ Modify feature extraction
2. ✅ Add custom features
3. ✅ Tune hyperparameters
4. ✅ Test on real data
5. ✅ Deploy to production

## 🎯 What to Expect

### After quickstart.py
- ✅ Sample waveform PNG
- ✅ Trained Random Forest model
- ✅ Console output with metrics

### After train.py
- ✅ 4 trained models in `models/`
- ✅ Multiple plots in `plots/`
- ✅ Dataset saved in `data/`
- ✅ Training logs in console

### After streamlit run
- ✅ Web browser opens
- ✅ Interactive interface
- ✅ Real-time classification
- ✅ Beautiful visualizations

## 📖 Documentation Guide

- **README.md**: Comprehensive project documentation
- **QUICKREF.md**: Quick command reference
- **PROJECT_SUMMARY.md**: What was built
- **tutorial.ipynb**: Step-by-step interactive tutorial
- **DATASETS.py**: Dataset sources and information

## 🎨 Web App Features

### Data Sources
- 🎲 Generate synthetic waveforms
- 📤 Upload your CSV files
- 📊 Use pre-saved samples

### Visualizations
- 📈 Time-domain waveform
- 📊 Frequency spectrum (FFT)
- 🔄 Combined analysis
- 🎨 Interactive plots (zoom, pan, hover)

### Classification
- 🤖 4 ML models to choose from
- 📊 Confidence scores
- 📈 Feature values
- 🎯 Prediction results

### Features Displayed
- ⚡ RMS Voltage
- 📊 Peak Voltage
- 📉 Crest Factor
- 🌊 THD
- 📉 Dip Percentage
- 📈 Swell Percentage

## 💡 Pro Tips

1. **Start Small**: Use `--n-samples 500` for quick training
2. **Visualize**: Always use `--visualize` to see plots
3. **Save Models**: Models are automatically saved after training
4. **Use Web App**: Easiest way to test the system
5. **Check Plots**: Review `plots/` directory after training
6. **Try Tutorial**: Best way to understand the system
7. **Read Code**: All code is well-commented

## 🔗 Quick Links

| Task | Command |
|------|---------|
| Quick Test | `python quickstart.py` |
| Train Models | `python train.py` |
| Web App | `streamlit run app.py` |
| Tutorial | `jupyter notebook tutorial.ipynb` |
| Dataset Info | `python DATASETS.py` |

## ✅ Success Checklist

After following this guide, you should have:

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Quick test completed successfully
- [ ] Models trained (optional)
- [ ] Web app running
- [ ] Generated a waveform
- [ ] Classified an anomaly
- [ ] Viewed visualizations

## 🎉 You're Ready!

You now have a fully functional Power Quality Anomaly Detection system!

### Quick Access
```bash
# Every time you work on the project:
cd /Users/hemanthkumar/power-quality
source venv/bin/activate  # Activate environment
streamlit run app.py      # Launch app
```

### Need Help?
1. Check README.md
2. Review QUICKREF.md
3. Run tutorial.ipynb
4. Examine error messages
5. Check troubleshooting section above

---

**🎊 Happy Anomaly Detecting! ⚡**

*For detailed information, see README.md*
