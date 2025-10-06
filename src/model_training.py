"""
Machine Learning Model Training and Evaluation Module
Supports Random Forest, SVM, XGBoost, LightGBM, and Neural Networks
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
import joblib
import os
import logging
from typing import Dict, Tuple, Any, List

logger = logging.getLogger(__name__)

# Try to import XGBoost and LightGBM, but continue without them if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. XGBoost models will be disabled.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. LightGBM models will be disabled.")


class PQModelTrainer:
    """Train and evaluate machine learning models for PQ anomaly classification"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize model trainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = None
    
    def prepare_data(
        self, 
        features: np.ndarray, 
        labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            features: Feature matrix
            labels: Labels array
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_
        logger.info(f"Classes: {self.class_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set size: {X_train_scaled.shape}")
        logger.info(f"Test set size: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 20,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest classifier...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        logger.info("Random Forest training complete")
        return model
    
    def train_svm(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        kernel: str = 'rbf',
        C: float = 1.0,
        **kwargs
    ) -> SVC:
        """
        Train Support Vector Machine classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            kernel: Kernel type
            C: Regularization parameter
            
        Returns:
            Trained model
        """
        logger.info("Training SVM classifier...")
        
        model = SVC(
            kernel=kernel,
            C=C,
            probability=True,  # Enable probability estimates
            random_state=42,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        self.models['svm'] = model
        
        logger.info("SVM training complete")
        return model
    
    def train_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ):
        """
        Train XGBoost classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            
        Returns:
            Trained model or None if XGBoost not available
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost is not available. Skipping XGBoost training.")
            return None
        
        logger.info("Training XGBoost classifier...")
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        logger.info("XGBoost training complete")
        return model
    
    def train_lightgbm(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        **kwargs
    ):
        """
        Train LightGBM classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            
        Returns:
            Trained model or None if LightGBM not available
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM is not available. Skipping LightGBM training.")
            return None
        
        logger.info("Training LightGBM classifier...")
        
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        
        logger.info("LightGBM training complete")
        return model
    
    def train_all_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training all available models...")
        
        # Train Random Forest
        self.train_random_forest(X_train, y_train)
        
        # Train SVM
        self.train_svm(X_train, y_train)
        
        # Train XGBoost if available
        if XGBOOST_AVAILABLE:
            self.train_xgboost(X_train, y_train)
        else:
            logger.warning("Skipping XGBoost (not available)")
        
        # Train LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm(X_train, y_train)
        else:
            logger.warning("Skipping LightGBM (not available)")
        
        logger.info(f"Trained {len(self.models)} models successfully")
        return self.models
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'f1_scores': f1,
            'support': support,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1 (macro): {f1_macro:.4f}")
        
        return results
    
    def evaluate_all_models(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results for each model
        """
        logger.info("Evaluating all models...")
        
        results = {}
        for model_name, model in self.models.items():
            results[model_name] = self.evaluate_model(
                model, X_test, y_test, model_name
            )
        
        # Print comparison
        self._print_model_comparison(results)
        
        return results
    
    def _print_model_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Print comparison of all models"""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        for model_name, result in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy:    {result['accuracy']:.4f}")
            logger.info(f"  F1 (macro):  {result['f1_macro']:.4f}")
            logger.info(f"  F1 (weighted): {result['f1_weighted']:.4f}")
        
        logger.info("\n" + "="*60)
    
    def cross_validate_model(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
        
        results = {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std()
        }
        
        logger.info(f"CV Accuracy: {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std']:.4f})")
        logger.info(f"CV F1 Score: {results['cv_f1_mean']:.4f} (+/- {results['cv_f1_std']:.4f})")
        
        return results
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> np.ndarray:
        """
        Get feature importance from tree-based models
        
        Args:
            model_name: Name of the model
            
        Returns:
            Array of feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            logger.warning(f"Model {model_name} does not have feature importances")
            return None
    
    def save_models(self):
        """Save all trained models"""
        logger.info("Saving models...")
        
        for model_name, model in self.models.items():
            filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
            joblib.dump(model, filepath)
            logger.info(f"Saved {model_name} to {filepath}")
        
        # Save label encoder and scaler
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, "label_encoder.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
        logger.info("Saved label encoder and scaler")
    
    def load_models(self):
        """Load saved models"""
        logger.info("Loading models...")
        
        model_files = {
            'random_forest': 'random_forest.pkl',
            'svm': 'svm.pkl',
            'xgboost': 'xgboost.pkl',
            'lightgbm': 'lightgbm.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                logger.info(f"Loaded {model_name}")
        
        # Load label encoder and scaler
        self.label_encoder = joblib.load(os.path.join(self.model_dir, "label_encoder.pkl"))
        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
        self.class_names = self.label_encoder.classes_
        logger.info("Loaded label encoder and scaler")
    
    def predict(
        self, 
        features: np.ndarray, 
        model_name: str = 'xgboost'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a trained model
        
        Args:
            features: Feature matrix
            model_name: Name of model to use
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions_encoded = model.predict(features_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)
        
        return predictions, probabilities
