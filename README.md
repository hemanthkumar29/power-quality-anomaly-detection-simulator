# ⚡ Power Quality Anomaly Detection Simulator

A comprehensive machine learning system for detecting and classifying power quality anomalies in electrical waveforms using real-time analysis and multiple ML models.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Overview

This simulator provides a software-based solution to detect and classify power quality (PQ) disturbances in electrical systems. It uses machine learning models to identify five types of anomalies:

- **Normal**: Stable sinusoidal waveform
- **Voltage Sag (Dip)**: Sudden drop in voltage (10-90% reduction)
- **Voltage Swell**: Sudden rise in voltage (10-80% increase)
- **Harmonic Distortion**: Non-sinusoidal waveform with harmonics
- **Outage (Interruption)**: Complete voltage loss

## ✨ Features

### Data Processing
- ✅ Synthetic waveform generation based on IEEE PQ standards
- ✅ Support for real PQ dataset loading (CSV format)
- ✅ Configurable sampling rates and durations
- ✅ Data augmentation and preprocessing

### Feature Extraction
- **Time-domain features**: RMS voltage, Peak voltage, Crest factor, Form factor, Energy
- **Frequency-domain features**: Total Harmonic Distortion (THD), Frequency deviation, Harmonic magnitudes
- **Anomaly-specific features**: Dip/Swell percentage, Zero-crossing rate

### Machine Learning Models
- 🌲 **Random Forest**: Baseline ensemble classifier
- 🔷 **Support Vector Machine (SVM)**: Kernel-based classifier for small datasets
- 🚀 **XGBoost**: Gradient boosting for high performance
- ⚡ **LightGBM**: Efficient gradient boosting
- 🧠 **Neural Networks (Optional)**: 1D CNN and LSTM architectures

### Evaluation & Visualization
- Cross-validation with stratified k-fold
- Comprehensive metrics (accuracy, F1-score, precision, recall)
- Confusion matrices
- Feature importance analysis
- Interactive waveform visualization (time & frequency domain)

### Web Application
- 🌐 **Streamlit web interface** for interactive analysis
- Upload custom waveforms or generate synthetic ones
- Real-time classification with confidence scores
- Interactive plots and visualizations

## 📋 Requirements

```
Python 3.8+
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
scikit-learn >= 1.3.0
xgboost >= 2.0.0
lightgbm >= 4.0.0
tensorflow >= 2.15.0 (optional, for neural networks)
matplotlib >= 3.7.0
seaborn >= 0.12.0
plotly >= 5.17.0
streamlit >= 1.28.0
```

## 🚀 Installation

1. **Clone the repository** (or create the project directory):
```bash
cd power-quality
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Train Models

Train all machine learning models on synthetic PQ dataset:

```bash
python train.py --n-samples 1000 --train-all --visualize
```

**Options**:
- `--n-samples`: Number of samples per class (default: 1000)
- `--test-size`: Test set proportion (default: 0.2)
- `--train-all`: Train all available models
- `--random-forest`: Train only Random Forest
- `--svm`: Train only SVM
- `--xgboost`: Train only XGBoost
- `--lightgbm`: Train only LightGBM
- `--cross-validate`: Perform k-fold cross-validation
- `--visualize`: Generate visualization plots
- `--use-saved`: Use previously saved dataset

**Example - Quick training**:
```bash
python train.py --n-samples 500 --train-all
```

**Example - Train specific models with cross-validation**:
```bash
python train.py --xgboost --lightgbm --cross-validate
```

### 2. Launch Web Application

Start the interactive Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

#### Web App Features:
- **Generate synthetic waveforms** for any anomaly type
- **Upload CSV files** with custom waveform data
- **Load sample data** from trained dataset
- **Visualize waveforms** in time and frequency domains
- **Classify anomalies** using trained ML models
- **View confidence scores** and feature values
- **Interactive plots** with zoom and pan

### 3. Use as Python Library

```python
from src.data_loader import PQDataLoader
from src.feature_extraction import FeatureExtractor
from src.model_training import PQModelTrainer

# Generate synthetic dataset
data_loader = PQDataLoader()
waveforms, labels = data_loader.generate_synthetic_dataset(n_samples=100)

# Extract features
feature_extractor = FeatureExtractor()
features = feature_extractor.extract_features_batch(waveforms)

# Load trained models and predict
trainer = PQModelTrainer()
trainer.load_models()
predictions, probabilities = trainer.predict(features, model_name='xgboost')

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

## 📁 Project Structure

```
power-quality/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading and synthetic generation
│   ├── feature_extraction.py   # Signal processing and feature extraction
│   ├── model_training.py       # ML model training and evaluation
│   ├── neural_network.py       # Neural network models (CNN, LSTM)
│   └── visualization.py        # Plotting and visualization utilities
├── data/
│   └── pq_dataset.npz          # Generated/saved dataset
├── models/
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── plots/
│   ├── sample_waveforms.png
│   ├── confusion_matrix_*.png
│   └── feature_importance_*.png
├── app.py                      # Streamlit web application
├── train.py                    # Training script
├── requirements.txt
└── README.md
```

## 🔬 Technical Details

### Synthetic Waveform Generation

The simulator generates realistic PQ waveforms based on power system standards:

- **Sampling Rate**: 6400 Hz (configurable)
- **Fundamental Frequency**: 60 Hz (50 Hz also supported)
- **Nominal Voltage**: 230V RMS (325V peak)
- **Duration**: 0.2 seconds per sample (configurable)

**Anomaly Characteristics**:
- **Sag**: 10-50% voltage reduction in middle 40% of waveform
- **Swell**: 10-40% voltage increase in middle 40% of waveform
- **Harmonic**: 3rd (20%), 5th (15%), 7th (10%) harmonic components
- **Outage**: Zero voltage in middle 40% of waveform

### Feature Extraction

**Time-Domain Features** (7):
- RMS Voltage
- Peak Voltage
- Crest Factor
- Form Factor
- Mean Absolute Voltage
- Standard Deviation
- Voltage Range

**Frequency-Domain Features** (9):
- Total Harmonic Distortion (THD)
- Frequency Deviation
- Individual harmonic magnitudes (1st-7th)

**Anomaly-Specific Features** (3):
- Dip Percentage
- Swell Percentage
- Zero Crossing Rate
- Signal Energy

**Total**: 20+ features extracted per waveform

### Model Performance

Expected performance on synthetic dataset (1000 samples/class):

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Random Forest | ~98% | ~0.98 | Fast |
| SVM (RBF) | ~96% | ~0.96 | Medium |
| XGBoost | ~99% | ~0.99 | Fast |
| LightGBM | ~99% | ~0.99 | Very Fast |
| 1D CNN | ~97% | ~0.97 | Slow |

*Note: Performance may vary with dataset size and quality*

## 📊 Dataset Information

### Built-in Synthetic Dataset
The system generates synthetic PQ waveforms based on IEEE standards. This ensures:
- Reproducible results
- Balanced classes
- Realistic waveform characteristics
- Configurable parameters

### Real Dataset Support

The system can load real PQ datasets in CSV format. To use your own data:

1. **Prepare CSV file** with format:
   ```
   sample_1, sample_2, ..., sample_n, label
   325.0, 320.1, ..., -310.5, Sag
   ```

2. **Place in `data/` directory** as `pq_dataset.csv`

3. **Load using data loader**:
   ```python
   data_loader = PQDataLoader()
   waveforms, labels = data_loader.load_real_dataset(source='local')
   ```

### Recommended Public Datasets

- **UCI Machine Learning Repository**: Power Quality Dataset
  - URL: https://archive.ics.uci.edu/
  
- **IEEE DataPort**: Power Quality Monitoring Datasets
  - URL: https://ieee-dataport.org/
  - Search: "power quality", "voltage sag", "harmonics"

- **Kaggle**: Power Quality Datasets
  - URL: https://www.kaggle.com/datasets
  - Search: "power quality", "electrical waveforms"

**Note**: Always check dataset licenses before use in production systems.

## 🎨 Visualization Examples

The system generates various visualizations:

1. **Sample Waveforms**: One example from each class
2. **Time-Frequency Analysis**: Combined time and frequency domain plots
3. **Confusion Matrices**: Model performance visualization
4. **Feature Importance**: Most important features for classification
5. **Interactive Plots**: Zoom, pan, and hover for details

## 🔧 Configuration

### Modify Sampling Parameters

Edit in `src/data_loader.py`:
```python
sampling_rate = 6400  # Hz
duration = 0.2        # seconds
frequency = 60        # Hz (50 for Europe)
```

### Adjust Model Hyperparameters

Edit in `train.py` or `src/model_training.py`:
```python
# Random Forest
n_estimators = 100
max_depth = 20

# XGBoost
learning_rate = 0.1
n_estimators = 100
```

### Change Feature Extraction

Edit in `src/feature_extraction.py` to add custom features:
```python
def extract_custom_feature(self, waveform):
    # Your custom feature logic
    return feature_value
```

## 🐛 Troubleshooting

### Issue: Models not loading in web app
**Solution**: Run `python train.py` first to train and save models.

### Issue: Import errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: Low model accuracy
**Solution**: Increase dataset size with `--n-samples 2000` or more.

### Issue: Streamlit connection error
**Solution**: Check if port 8501 is available or specify different port:
```bash
streamlit run app.py --server.port 8502
```

## 📝 License

This project uses:
- **MIT License** for original code
- **Synthetic Data**: Generated data can be used freely
- **Third-party Libraries**: Check individual library licenses

Real datasets may have their own licenses - please verify before commercial use.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Integration with real-time data acquisition systems
- Additional anomaly types (transients, flicker, etc.)
- More ML models (ensemble methods, deep learning)
- Mobile/web deployment
- Real-time streaming analysis

## 📧 Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check documentation in code files
- Review examples in `train.py` and `app.py`

## 🙏 Acknowledgments

- IEEE Power Quality Standards
- Scikit-learn, XGBoost, LightGBM teams
- Streamlit for web framework
- Open-source Python community

## 🔮 Future Enhancements

- [ ] Real-time waveform monitoring
- [ ] IoT sensor integration
- [ ] Cloud deployment (AWS, Azure)
- [ ] Mobile application
- [ ] Multi-phase analysis
- [ ] Power quality index calculation
- [ ] Automated report generation
- [ ] Alert system for critical anomalies

---

**Built with ❤️ for Electrical Engineers and Data Scientists**

*Power Quality Anomaly Detection - Making electrical systems smarter and safer*
