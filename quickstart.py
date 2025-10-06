"""
Quick start script to test the Power Quality Anomaly Detection system
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from src.data_loader import PQDataLoader
from src.feature_extraction import FeatureExtractor
from src.model_training import PQModelTrainer
from src.visualization import PQVisualizer

def main():
    print("="*60)
    print("POWER QUALITY ANOMALY DETECTION - QUICK START")
    print("="*60)
    
    # 1. Generate a small dataset
    print("\n[1] Generating synthetic dataset...")
    data_loader = PQDataLoader()
    waveforms, labels = data_loader.generate_synthetic_dataset(n_samples=100)
    print(f"✓ Generated {len(waveforms)} waveforms")
    
    # 2. Extract features
    print("\n[2] Extracting features...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features_batch(waveforms)
    print(f"✓ Extracted {features.shape[1]} features")
    
    # 3. Train a quick model (Random Forest only)
    print("\n[3] Training Random Forest model...")
    trainer = PQModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(features, labels)
    trainer.train_random_forest(X_train, y_train)
    print("✓ Model trained")
    
    # 4. Evaluate
    print("\n[4] Evaluating model...")
    results = trainer.evaluate_model(
        trainer.models['random_forest'], 
        X_test, 
        y_test, 
        'Random Forest'
    )
    print(f"✓ Accuracy: {results['accuracy']:.4f}")
    print(f"✓ F1 Score: {results['f1_macro']:.4f}")
    
    # 5. Test prediction on one sample
    print("\n[5] Testing prediction on sample waveform...")
    sample_idx = 0
    sample_waveform = waveforms[sample_idx:sample_idx+1]
    sample_features = feature_extractor.extract_features_batch(sample_waveform)
    
    predictions, probabilities = trainer.predict(sample_features, 'random_forest')
    print(f"✓ Predicted: {predictions[0]}")
    print(f"✓ True label: {labels[sample_idx]}")
    
    # 6. Visualize
    print("\n[6] Creating visualization...")
    visualizer = PQVisualizer()
    fig = visualizer.plot_waveform(
        waveforms[sample_idx], 
        title=f"Sample Waveform - {labels[sample_idx]}",
        save_path="quickstart_waveform.png"
    )
    print("✓ Saved visualization to quickstart_waveform.png")
    
    print("\n" + "="*60)
    print("QUICK START COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run full training: python train.py")
    print("2. Launch web app: streamlit run app.py")
    print("="*60)

if __name__ == "__main__":
    main()
