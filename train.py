"""
Main Training Script for Power Quality Anomaly Detection
Orchestrates data loading, feature extraction, and model training
"""

import numpy as np
import logging
import argparse
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import PQDataLoader
from src.feature_extraction import FeatureExtractor
from src.model_training import PQModelTrainer
from src.visualization import PQVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main training pipeline"""
    
    logger.info("="*60)
    logger.info("POWER QUALITY ANOMALY DETECTION - TRAINING PIPELINE")
    logger.info("="*60)
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Step 1: Load/Generate Dataset
    logger.info("\n[STEP 1] Loading dataset...")
    data_loader = PQDataLoader(data_dir="data")
    
    if args.use_saved:
        try:
            waveforms, labels = data_loader.load_saved_dataset()
            logger.info("Loaded saved dataset")
        except FileNotFoundError:
            logger.info("No saved dataset found. Generating new dataset...")
            waveforms, labels = data_loader.generate_synthetic_dataset(
                n_samples=args.n_samples
            )
            data_loader.save_dataset(waveforms, labels)
    else:
        waveforms, labels = data_loader.generate_synthetic_dataset(
            n_samples=args.n_samples
        )
        data_loader.save_dataset(waveforms, labels)
    
    logger.info(f"Dataset shape: {waveforms.shape}")
    logger.info(f"Number of classes: {len(np.unique(labels))}")
    logger.info(f"Classes: {np.unique(labels)}")
    
    # Step 2: Visualize Sample Waveforms
    if args.visualize:
        logger.info("\n[STEP 2] Visualizing sample waveforms...")
        visualizer = PQVisualizer()
        
        # Plot one sample from each class
        unique_classes = np.unique(labels)
        sample_waveforms = []
        sample_labels = []
        
        for class_name in unique_classes:
            idx = np.where(labels == class_name)[0][0]
            sample_waveforms.append(waveforms[idx])
            sample_labels.append(class_name)
        
        visualizer.plot_multiple_waveforms(
            sample_waveforms,
            sample_labels,
            title="Sample Waveforms by Class",
            save_path="plots/sample_waveforms.png"
        )
        logger.info("Sample waveforms saved to plots/sample_waveforms.png")
    
    # Step 3: Extract Features
    logger.info("\n[STEP 3] Extracting features...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features_batch(waveforms)
    feature_names = feature_extractor.get_feature_names()
    
    logger.info(f"Extracted {features.shape[1]} features")
    logger.info(f"Feature names: {feature_names[:5]}... (showing first 5)")
    
    # Step 4: Train Models
    logger.info("\n[STEP 4] Training machine learning models...")
    trainer = PQModelTrainer(model_dir="models")
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features, labels, test_size=args.test_size
    )
    
    # Train all models
    if args.train_all:
        logger.info("Training all available models...")
        trainer.train_all_models(X_train, y_train)
    else:
        # Train individual models as specified
        if args.random_forest:
            trainer.train_random_forest(X_train, y_train)
        if args.svm:
            trainer.train_svm(X_train, y_train)
        if args.xgboost:
            result = trainer.train_xgboost(X_train, y_train)
            if result is None:
                logger.warning("XGBoost training skipped - library not available")
        if args.lightgbm:
            result = trainer.train_lightgbm(X_train, y_train)
            if result is None:
                logger.warning("LightGBM training skipped - library not available")
    
    # Check if any models were trained
    if not trainer.models:
        logger.error("No models were trained successfully!")
        return
    
    # Step 5: Evaluate Models
    logger.info("\n[STEP 5] Evaluating models...")
    results = trainer.evaluate_all_models(X_test, y_test)
    
    # Step 6: Visualize Results
    if args.visualize:
        logger.info("\n[STEP 6] Visualizing results...")
        visualizer = PQVisualizer()
        
        # Plot confusion matrices for each model
        for model_name, result in results.items():
            visualizer.plot_confusion_matrix(
                result['confusion_matrix'],
                trainer.class_names,
                title=f"Confusion Matrix - {model_name.upper()}",
                save_path=f"plots/confusion_matrix_{model_name}.png"
            )
            logger.info(f"Confusion matrix for {model_name} saved")
        
        # Plot feature importance for tree-based models
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            if model_name in trainer.models:
                importance = trainer.get_feature_importance(model_name)
                if importance is not None:
                    visualizer.plot_feature_importance(
                        feature_names,
                        importance,
                        title=f"Feature Importance - {model_name.upper()}",
                        save_path=f"plots/feature_importance_{model_name}.png"
                    )
                    logger.info(f"Feature importance for {model_name} saved")
    
    # Step 7: Cross-Validation (optional)
    if args.cross_validate:
        logger.info("\n[STEP 7] Performing cross-validation...")
        # Use full dataset for CV
        X_full = np.vstack([X_train, X_test])
        y_full_encoded = trainer.label_encoder.transform(
            np.concatenate([
                trainer.label_encoder.inverse_transform(y_train),
                trainer.label_encoder.inverse_transform(y_test)
            ])
        )
        
        for model_name, model in trainer.models.items():
            logger.info(f"\nCross-validating {model_name}...")
            cv_results = trainer.cross_validate_model(model, X_full, y_full_encoded)
    
    # Step 8: Save Models
    logger.info("\n[STEP 8] Saving models...")
    trainer.save_models()
    logger.info("All models saved successfully")
    
    # Print Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*60)
    logger.info(f"Dataset size: {len(waveforms)} samples")
    logger.info(f"Number of features: {features.shape[1]}")
    logger.info(f"Number of classes: {len(trainer.class_names)}")
    logger.info(f"Models trained: {list(trainer.models.keys())}")
    logger.info("\nBest model performance:")
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"  Model: {best_model[0]}")
    logger.info(f"  Accuracy: {best_model[1]['accuracy']:.4f}")
    logger.info(f"  F1 Score (macro): {best_model[1]['f1_macro']:.4f}")
    
    logger.info("\nModels saved in: models/")
    logger.info("Plots saved in: plots/")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Power Quality Anomaly Detection Models"
    )
    
    # Dataset arguments
    parser.add_argument(
        '--n-samples', 
        type=int, 
        default=1000,
        help='Number of samples per class to generate'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Proportion of test set'
    )
    parser.add_argument(
        '--use-saved', 
        action='store_true',
        help='Use previously saved dataset'
    )
    
    # Model training arguments
    parser.add_argument(
        '--train-all', 
        action='store_true',
        default=True,
        help='Train all available models'
    )
    parser.add_argument(
        '--random-forest', 
        action='store_true',
        help='Train Random Forest model'
    )
    parser.add_argument(
        '--svm', 
        action='store_true',
        help='Train SVM model'
    )
    parser.add_argument(
        '--xgboost', 
        action='store_true',
        help='Train XGBoost model'
    )
    parser.add_argument(
        '--lightgbm', 
        action='store_true',
        help='Train LightGBM model'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--cross-validate', 
        action='store_true',
        help='Perform cross-validation'
    )
    parser.add_argument(
        '--visualize', 
        action='store_true',
        default=True,
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    main(args)
