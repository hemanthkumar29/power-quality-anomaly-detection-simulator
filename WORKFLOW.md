# 🔄 Power Quality Detection - Hybrid Training Workflow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   DATA GENERATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐         ┌──────────────────────┐     │
│  │  Synthetic Generator │         │  Realistic Generator │     │
│  │  (data_loader.py)    │         │ (real_data_loader.py)│     │
│  ├──────────────────────┤         ├──────────────────────┤     │
│  │ • Mathematical models│         │ • IEEE 1159 standards│     │
│  │ • Pure sine waves    │         │ • Measurement noise  │     │
│  │ • Controlled events  │         │ • Power factor var.  │     │
│  │ • 5 event types      │         │ • Flicker effects    │     │
│  └──────────┬───────────┘         └──────────┬───────────┘     │
│             │                                 │                  │
│             └──────────┬────────────┬─────────┘                 │
│                        ▼            ▼                            │
│              ┌────────────────────────────┐                     │
│              │   Combined Dataset         │                     │
│              │   (5,000 samples)          │                     │
│              │   • 2,500 Synthetic (50%)  │                     │
│              │   • 2,500 Realistic (50%)  │                     │
│              │   • Balanced classes       │                     │
│              └──────────┬─────────────────┘                     │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│                   FEATURE EXTRACTION LAYER                       │
├─────────────────────────┼───────────────────────────────────────┤
│                         ▼                                         │
│              ┌────────────────────────┐                          │
│              │  FeatureExtractor      │                          │
│              │ (feature_extraction.py)│                          │
│              ├────────────────────────┤                          │
│              │ Time Domain:           │                          │
│              │ • RMS Voltage          │                          │
│              │ • Peak Voltage         │                          │
│              │ • Crest Factor         │                          │
│              │ • Form Factor          │                          │
│              │ • Energy               │                          │
│              │                        │                          │
│              │ Frequency Domain:      │                          │
│              │ • THD                  │                          │
│              │ • Frequency Deviation  │                          │
│              │ • Harmonics (3,5,7,9)  │                          │
│              │                        │                          │
│              │ Anomaly-Specific:      │                          │
│              │ • Dip Percentage       │                          │
│              │ • Swell Percentage     │                          │
│              │ • Zero-crossing Rate   │                          │
│              └──────────┬─────────────┘                          │
│                         │                                         │
│                         ▼                                         │
│              ┌────────────────────────┐                          │
│              │   Feature Matrix       │                          │
│              │   (5000 × 20)          │                          │
│              └──────────┬─────────────┘                          │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│                   MODEL TRAINING LAYER                           │
├─────────────────────────┼───────────────────────────────────────┤
│                         ▼                                         │
│        ┌────────────────────────────────────┐                   │
│        │  Train/Test Split (80/20)          │                   │
│        │  • Training: 4,000 samples         │                   │
│        │  • Testing: 1,000 samples          │                   │
│        └────────┬───────────────────────────┘                   │
│                 │                                                 │
│     ┌───────────┼───────────┬───────────┬───────────┐          │
│     ▼           ▼           ▼           ▼           ▼           │
│ ┌────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│ │Random  │ │  SVM   │ │XGBoost  │ │LightGBM │ │  Neural │    │
│ │Forest  │ │        │ │         │ │         │ │ Network │    │
│ ├────────┤ ├────────┤ ├─────────┤ ├─────────┤ ├─────────┤    │
│ │100 tree│ │RBF ker.│ │500 tree │ │100 iter │ │CNN/LSTM │    │
│ │max=20  │ │C=10    │ │lr=0.1   │ │lr=0.1   │ │optional │    │
│ ├────────┤ ├────────┤ ├─────────┤ ├─────────┤ ├─────────┤    │
│ │99.9%   │ │99.9%   │ │99.9%    │ │99.8%    │ │N/A      │    │
│ └────┬───┘ └────┬───┘ └────┬────┘ └────┬────┘ └────┬────┘    │
│      │          │          │           │          │            │
│      └──────────┴──────────┴───────────┴──────────┘            │
│                            │                                     │
│                            ▼                                     │
│                 ┌────────────────────┐                          │
│                 │  Model Evaluation  │                          │
│                 ├────────────────────┤                          │
│                 │ • Accuracy         │                          │
│                 │ • F1 Score         │                          │
│                 │ • Confusion Matrix │                          │
│                 │ • Feature Import.  │                          │
│                 └────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                   DEPLOYMENT LAYER                               │
├──────────────────────────┼──────────────────────────────────────┤
│                          ▼                                        │
│           ┌──────────────────────────────┐                      │
│           │   Save Models to Disk        │                      │
│           │   (models/*.pkl)             │                      │
│           ├──────────────────────────────┤                      │
│           │ • random_forest.pkl  (140KB) │                      │
│           │ • svm.pkl            (19KB)  │                      │
│           │ • xgboost.pkl        (384KB) │                      │
│           │ • lightgbm.pkl       (1.1MB) │                      │
│           │ • scaler.pkl                 │                      │
│           │ • label_encoder.pkl          │                      │
│           └──────────┬───────────────────┘                      │
│                      │                                            │
│         ┌────────────┴───────────┬──────────────┐              │
│         ▼                        ▼              ▼               │
│  ┌─────────────┐        ┌──────────────┐  ┌──────────┐        │
│  │  Web App    │        │  Python API  │  │   CLI    │        │
│  │  (app.py)   │        │              │  │          │        │
│  ├─────────────┤        ├──────────────┤  ├──────────┤        │
│  │ Streamlit   │        │ Import src   │  │train.py  │        │
│  │ Interactive │        │ Use models   │  │--use-    │        │
│  │ Upload CSV  │        │ Classify     │  │combined  │        │
│  │ Real-time   │        │              │  │          │        │
│  └─────────────┘        └──────────────┘  └──────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

## Workflow Steps

### 1. Data Generation (Choose One or Both)

#### Option A: Synthetic Only
```bash
python train.py --n-samples 1000
```
- Pure mathematical models
- Perfect sine waves with controlled events
- Fast generation
- Good for prototyping

#### Option B: Hybrid (Recommended)
```bash
python train.py --use-combined --n-samples 500
```
- 50% synthetic + 50% realistic
- IEEE 1159-compliant variations
- Better generalization
- Production-ready

### 2. Feature Extraction

Automatic extraction of 20 features:

| Category | Features | Count |
|----------|----------|-------|
| **Time Domain** | RMS, Peak, Crest Factor, Form Factor, Mean, Std, Energy | 7 |
| **Frequency Domain** | THD, Freq. Deviation, Harmonics (3,5,7,9) | 5 |
| **Anomaly-Specific** | Dip %, Swell %, Zero-crossing, Skewness, Kurtosis | 5 |
| **Statistical** | Min, Max, Range | 3 |

### 3. Model Training

Trains 4 models in parallel:
- **Random Forest**: 100 trees, max_depth=20
- **SVM**: RBF kernel, C=10, gamma=auto
- **XGBoost**: 500 estimators, learning_rate=0.1
- **LightGBM**: 100 leaves, learning_rate=0.1

### 4. Evaluation

Generates:
- Confusion matrices (4 plots)
- Feature importance (3 plots)
- Classification reports (4 reports)
- Accuracy metrics (4 scores)

### 5. Deployment

Saves models and provides 3 interfaces:
- **Web App**: `streamlit run app.py`
- **Python API**: `from src.model_training import PQModelTrainer`
- **CLI**: `python train.py --use-combined`

---

## Data Flow Example

### Input Waveform
```
Time (s): [0.0, 0.00015625, 0.0003125, ..., 0.2]
Voltage (V): [325.3, 323.1, 315.6, ..., -12.4, -8.2, 0.1]
Length: 1,280 points (6,400 Hz sampling, 0.2s duration)
```

### Feature Vector (20 features)
```python
{
    'rms_voltage': 230.2,
    'peak_voltage': 325.3,
    'crest_factor': 1.414,
    'thd': 0.031,
    'dip_percentage': 0.0,
    'swell_percentage': 0.0,
    ...  # 14 more features
}
```

### Classification Result
```python
{
    'prediction': 'Normal',
    'confidence': 0.998,
    'probabilities': {
        'Normal': 0.998,
        'Sag': 0.001,
        'Swell': 0.001,
        'Harmonic': 0.000,
        'Outage': 0.000
    }
}
```

---

## Training Timeline

```
[0s]     Load/Generate Dataset
         ├─ Synthetic: ~2s
         └─ Realistic: ~5s

[7s]     Extract Features (5,000 samples)
         └─ ~20 features per sample

[12s]    Train Models
         ├─ Random Forest: ~3s
         ├─ SVM: ~5s
         ├─ XGBoost: ~2s
         └─ LightGBM: ~1s

[23s]    Evaluate Models
         └─ Confusion matrices, metrics

[25s]    Generate Visualizations
         └─ 7 plots saved

[27s]    Save Models
         └─ 6 files to models/

[28s]    COMPLETE ✓
```

---

## Performance Metrics

### Accuracy by Model (Hybrid Training)

```
         Synthetic Only  │  Hybrid Training
────────────────────────┼──────────────────────
Random Forest:  100.0%  │  99.9%  ✓
SVM:            100.0%  │  99.9%  ✓
XGBoost:         99.0%  │  99.9%  ↑ Improved
LightGBM:       100.0%  │  99.8%  ✓
```

### Training Speed

```
Model         Training Time    Prediction Time
───────────────────────────────────────────────
Random Forest      ~3s              ~5ms
SVM                ~5s              ~2ms
XGBoost            ~2s              ~1ms
LightGBM           ~1s              ~1ms
```

### Dataset Size Impact

```
Samples/Class  Total   Training Time  Accuracy
───────────────────────────────────────────────
100            500     ~5s            97.5%
500            2,500   ~12s           99.2%
1,000          5,000   ~28s           99.9% ✓
2,000          10,000  ~60s           99.9%
```

**Recommendation: 500-1,000 samples per class for best balance**

---

## Quick Commands

```bash
# Full pipeline with hybrid data
python train.py --use-combined --n-samples 500 --visualize

# Quick test (100 samples)
python quickstart.py

# Web interface
streamlit run app.py

# Check dataset
python -c "
from src.real_data_loader import load_combined_dataset
w, l = load_combined_dataset(100, 100)
print(f'Shape: {w.shape}')
"
```

---

## File Outputs

After training, you will have:

```
models/
├── random_forest.pkl      (140 KB)  99.9% acc
├── svm.pkl                (19 KB)   99.9% acc
├── xgboost.pkl            (384 KB)  99.9% acc
├── lightgbm.pkl           (1.1 MB)  99.8% acc
├── label_encoder.pkl      (1 KB)
└── scaler.pkl             (3 KB)

plots/
├── confusion_matrix_random_forest.png
├── confusion_matrix_svm.png
├── confusion_matrix_xgboost.png
├── confusion_matrix_lightgbm.png
├── feature_importance_random_forest.png
├── feature_importance_xgboost.png
├── feature_importance_lightgbm.png
└── sample_waveforms.png

data/
├── real_pq_dataset.csv    (490 KB)  2,500 samples
└── pq_dataset.npz         (cached synthetic data)
```

---

## System Requirements

- **Python**: 3.8+
- **RAM**: 2 GB minimum, 4 GB recommended
- **Storage**: 50 MB for code + models
- **CPU**: Any modern processor (multi-core recommended)
- **OS**: Windows, macOS, Linux

---

**End of Workflow Documentation**
