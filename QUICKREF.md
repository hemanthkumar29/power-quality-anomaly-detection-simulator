# Power Quality Anomaly Detection - Quick Reference Guide

## 🚀 Quick Start Commands

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Basic training (1000 samples/class)
python train.py

# Hybrid training with realistic data (RECOMMENDED)
python train.py --use-combined --n-samples 500

# Quick training (500 samples)
python train.py --n-samples 500

# Train specific models
python train.py --xgboost --lightgbm

# With cross-validation
python train.py --cross-validate

# Quick test
python quickstart.py
```

### Web Application
```bash
# Launch Streamlit app
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502
```

### Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook tutorial.ipynb
```

## 📊 API Quick Reference

### Data Loading
```python
from src.data_loader import PQDataLoader

# Generate synthetic data
loader = PQDataLoader()
waveforms, labels = loader.generate_synthetic_dataset(n_samples=100)

# Load combined (synthetic + realistic) dataset
from src.real_data_loader import load_combined_dataset
waveforms, labels = load_combined_dataset(n_synthetic=500, n_real=500)

# Save/load dataset
loader.save_dataset(waveforms, labels)
waveforms, labels = loader.load_saved_dataset()
```

### Feature Extraction
```python
from src.feature_extraction import FeatureExtractor

# Extract features
extractor = FeatureExtractor(sampling_rate=6400)
features = extractor.extract_features_batch(waveforms)

# Get single waveform features
features_dict = extractor.extract_all_features(waveform)
print(f"RMS: {features_dict['rms_voltage']}")
print(f"THD: {features_dict['thd']}")
```

### Model Training
```python
from src.model_training import PQModelTrainer

# Initialize trainer
trainer = PQModelTrainer()

# Prepare data
X_train, X_test, y_train, y_test = trainer.prepare_data(features, labels)

# Train models
trainer.train_random_forest(X_train, y_train)
trainer.train_xgboost(X_train, y_train)
trainer.train_lightgbm(X_train, y_train)

# Evaluate
results = trainer.evaluate_all_models(X_test, y_test)

# Save models
trainer.save_models()
```

### Prediction
```python
# Load saved models
trainer = PQModelTrainer()
trainer.load_models()

# Predict
predictions, probabilities = trainer.predict(features, model_name='xgboost')
print(f"Predicted: {predictions[0]}")
print(f"Confidence: {probabilities[0].max():.2%}")
```

### Visualization
```python
from src.visualization import PQVisualizer

visualizer = PQVisualizer()

# Plot waveform
fig = visualizer.plot_waveform(waveform, title="Voltage Sag")

# Plot FFT
fig = visualizer.plot_fft(waveform, title="Frequency Spectrum")

# Combined plot
fig = visualizer.plot_waveform_with_fft(waveform)

# Confusion matrix
fig = visualizer.plot_confusion_matrix(cm, class_names)

# Feature importance
fig = visualizer.plot_feature_importance(feature_names, importance)
```

## 🔧 Configuration

### Sampling Parameters
```python
# In data_loader.py or when generating
sampling_rate = 6400  # Hz
duration = 0.2        # seconds
frequency = 60        # Hz (50 for Europe)
```

### Model Hyperparameters
```python
# Random Forest
n_estimators = 100
max_depth = 20

# XGBoost
n_estimators = 100
learning_rate = 0.1
max_depth = 6

# LightGBM
n_estimators = 100
learning_rate = 0.1
max_depth = -1  # No limit
```

## 📈 Feature List

### Time-Domain (10 features)
- `rms_voltage`: Root mean square voltage
- `peak_voltage`: Maximum voltage
- `crest_factor`: Peak/RMS ratio
- `form_factor`: RMS/Mean ratio
- `mean_voltage`: Average absolute voltage
- `std_voltage`: Standard deviation
- `voltage_range`: Max - Min
- `zero_crossing_rate`: Normalized crossings
- `energy`: Signal energy
- `dip_percentage`: Voltage dip %
- `swell_percentage`: Voltage swell %

### Frequency-Domain (10+ features)
- `thd`: Total Harmonic Distortion
- `freq_deviation`: Frequency deviation
- `harmonic_1` to `harmonic_7`: Individual harmonics

## 🎯 Anomaly Classes

| Class | Description | Characteristics |
|-------|-------------|----------------|
| Normal | Stable sine wave | RMS ≈ 230V, THD < 5% |
| Sag | Voltage drop | 10-50% reduction |
| Swell | Voltage rise | 10-40% increase |
| Harmonic | Distortion | THD > 10%, harmonics present |
| Outage | Interruption | Voltage ≈ 0V for period |

## 📁 File Structure

```
power-quality/
├── src/               # Source code modules
├── data/              # Datasets (generated)
├── models/            # Trained models
├── plots/             # Generated plots
├── train.py           # Training script
├── app.py             # Web application
├── quickstart.py      # Quick test script
├── tutorial.ipynb     # Jupyter tutorial
└── requirements.txt   # Dependencies
```

## 🔍 Troubleshooting

### Common Issues

**"Models not found"**
```bash
python train.py  # Train models first
```

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"Streamlit won't start"**
```bash
# Check port availability
streamlit run app.py --server.port 8502
```

**"Low accuracy"**
```bash
# Increase dataset size
python train.py --n-samples 2000
```

## 📊 Performance Tips

1. **Faster Training**: Use LightGBM
2. **Better Accuracy**: Increase n_samples
3. **Quick Testing**: Use Random Forest
4. **Production**: Use XGBoost

## 🌐 Web App Usage

1. **Select data source**: Synthetic / Upload / Sample
2. **Generate/Load waveform**
3. **View visualizations**: Time, Frequency, Combined
4. **Extract features**: Automatic
5. **Classify**: Select model and predict
6. **View results**: Prediction + confidence scores

## 📝 Dataset Format

### CSV Upload Format
```csv
sample_1,sample_2,...,sample_n,label
325.0,320.1,...,-310.5,Sag
328.0,325.5,...,320.0,Normal
```

### NPZ Format (saved datasets)
```python
data = np.load('data/pq_dataset.npz')
waveforms = data['waveforms']  # (n_samples, n_points)
labels = data['labels']         # (n_samples,)
```

## 🔬 Advanced Usage

### Custom Feature Extraction
```python
# Add to feature_extraction.py
def extract_custom_feature(self, waveform):
    # Your calculation
    return custom_value
```

### Neural Network Training
```python
from src.neural_network import NeuralNetworkClassifier

nn_classifier = NeuralNetworkClassifier()
history = nn_classifier.train(
    X_train, y_train, 
    model_type='cnn',
    epochs=50
)
```

### Cross-Validation
```python
cv_results = trainer.cross_validate_model(
    model, X_full, y_full, cv=5
)
```

## 📞 Support

- Check README.md for detailed documentation
- Review tutorial.ipynb for examples
- Inspect source code in src/
- Run DATASETS.py for dataset information

## ⚡ Quick Tips

1. Start with `python quickstart.py`
2. Use `--visualize` flag for plots
3. Save datasets with `save_dataset()`
4. Check feature importance for insights
5. Use web app for interactive analysis
6. Increase samples for better accuracy
7. Try different models for comparison
8. Use cross-validation for robustness

---
*Power Quality Anomaly Detection v1.0*
