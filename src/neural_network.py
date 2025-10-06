"""
Neural Network Models for Power Quality Classification
Implements 1D CNN and LSTM architectures
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import os
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class NeuralNetworkClassifier:
    """Neural network models for PQ anomaly classification"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize neural network classifier
        
        Args:
            model_dir: Directory to save models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
    
    def build_1d_cnn(
        self, 
        input_shape: Tuple[int],
        num_classes: int,
        filters: list = [64, 128, 256],
        kernel_size: int = 3
    ) -> models.Model:
        """
        Build 1D CNN model for waveform classification
        
        Args:
            input_shape: Shape of input waveforms (length,)
            num_classes: Number of output classes
            filters: List of filter sizes for each conv layer
            kernel_size: Size of convolution kernel
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building 1D CNN model...")
        
        model = models.Sequential([
            layers.Input(shape=(input_shape[0], 1)),
            
            # First convolutional block
            layers.Conv1D(filters[0], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Second convolutional block
            layers.Conv1D(filters[1], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Third convolutional block
            layers.Conv1D(filters[2], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model built with {model.count_params()} parameters")
        return model
    
    def build_lstm(
        self, 
        input_shape: Tuple[int],
        num_classes: int,
        lstm_units: list = [128, 64]
    ) -> models.Model:
        """
        Build LSTM model for waveform classification
        
        Args:
            input_shape: Shape of input waveforms (length,)
            num_classes: Number of output classes
            lstm_units: List of LSTM units for each layer
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model...")
        
        model = models.Sequential([
            layers.Input(shape=(input_shape[0], 1)),
            
            # LSTM layers
            layers.LSTM(lstm_units[0], return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(lstm_units[1]),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model built with {model.count_params()} parameters")
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        model_type: str = 'cnn',
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train neural network model
        
        Args:
            X_train: Training waveforms (raw, not features)
            y_train: Training labels
            X_val: Validation waveforms
            y_val: Validation labels
            model_type: 'cnn' or 'lstm'
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split if X_val not provided
            
        Returns:
            Training history
        """
        logger.info(f"Training {model_type.upper()} model...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        num_classes = len(self.label_encoder.classes_)
        
        # Reshape input for CNN/LSTM (add channel dimension)
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Build model
        if model_type.lower() == 'cnn':
            self.model = self.build_1d_cnn((X_train.shape[1],), num_classes)
        elif model_type.lower() == 'lstm':
            self.model = self.build_lstm((X_train.shape[1],), num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            validation_data = (X_val_reshaped, y_val_encoded)
            validation_split = None
        else:
            validation_data = None
        
        # Train model
        self.history = self.model.fit(
            X_train_reshaped,
            y_train_encoded,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training complete")
        return self.history.history
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test waveforms
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating model...")
        
        # Encode labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Reshape input
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test_reshaped, y_test_encoded, verbose=0)
        
        # Get predictions for detailed metrics
        y_pred_proba = self.model.predict(X_test_reshaped, verbose=0)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        
        from sklearn.metrics import f1_score, classification_report
        
        f1_macro = f1_score(y_test_encoded, y_pred_encoded, average='macro')
        f1_weighted = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
        
        results = {
            'loss': loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 (macro): {f1_macro:.4f}")
        
        return results
    
    def predict(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Input waveforms
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Reshape input
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Predict
        probabilities = self.model.predict(X_reshaped, verbose=0)
        predictions_encoded = np.argmax(probabilities, axis=1)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions, probabilities
    
    def save_model(self, model_name: str = 'neural_network'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save label encoder
        import joblib
        encoder_path = os.path.join(self.model_dir, f"{model_name}_encoder.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        logger.info(f"Label encoder saved to {encoder_path}")
    
    def load_model(self, model_name: str = 'neural_network'):
        """Load saved model"""
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load label encoder
        import joblib
        encoder_path = os.path.join(self.model_dir, f"{model_name}_encoder.pkl")
        self.label_encoder = joblib.load(encoder_path)
        logger.info(f"Label encoder loaded from {encoder_path}")
