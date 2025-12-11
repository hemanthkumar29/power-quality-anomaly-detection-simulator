"""
Data Loader Module for Power Quality Datasets
Handles loading real PQ datasets and generating synthetic waveforms
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PQDataLoader:
    """Load and manage Power Quality datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_real_dataset(self, source: str = "local") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load real PQ dataset from various sources
        
        Args:
            source: Dataset source ('local', 'uci', 'synthetic')
            
        Returns:
            Tuple of (waveforms, labels)
        """
        if source == "local":
            # Check for local CSV files
            local_file = os.path.join(self.data_dir, "pq_dataset.csv")
            if os.path.exists(local_file):
                logger.info(f"Loading local dataset from {local_file}")
                return self._load_csv_dataset(local_file)
        
        # If no real data available, generate synthetic dataset
        logger.warning("No real dataset found. Generating synthetic dataset...")
        return self.generate_synthetic_dataset(n_samples=1000)
    
    def _load_csv_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from CSV file"""
        df = pd.read_csv(filepath)
        
        # Assuming CSV has columns for waveform data and a 'label' column
        if 'label' in df.columns:
            labels = df['label'].values
            waveforms = df.drop('label', axis=1).values
        else:
            raise ValueError("Dataset must contain a 'label' column")
        
        return waveforms, labels
    
    def generate_synthetic_dataset(
        self, 
        n_samples: int = 1000,
        sampling_rate: int = 6400,
        duration: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic PQ waveform dataset using optimized vectorized operations
        
        Args:
            n_samples: Number of samples per class
            sampling_rate: Samples per second (Hz)
            duration: Duration of each waveform (seconds)
            
        Returns:
            Tuple of (waveforms, labels)
        """
        logger.info(f"Generating {n_samples} synthetic samples per class...")
        
        n_points = int(sampling_rate * duration)
        time = np.linspace(0, duration, n_points)
        frequency = 60  # 60 Hz power system
        
        # Pre-allocate arrays for better memory efficiency
        classes = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Outage']
        total_samples = n_samples * len(classes)
        waveforms = np.zeros((total_samples, n_points))
        labels = np.empty(total_samples, dtype=object)
        
        # Generate each class in batches
        idx = 0
        for class_name in classes:
            logger.info(f"Generating {class_name} waveforms...")
            
            if class_name == 'Normal':
                batch = self._generate_normal_batch(time, frequency, n_samples)
            elif class_name == 'Sag':
                batch = self._generate_sag_batch(time, frequency, n_samples)
            elif class_name == 'Swell':
                batch = self._generate_swell_batch(time, frequency, n_samples)
            elif class_name == 'Harmonic':
                batch = self._generate_harmonic_batch(time, frequency, n_samples)
            elif class_name == 'Outage':
                batch = self._generate_outage_batch(time, frequency, n_samples)
            
            waveforms[idx:idx+n_samples] = batch
            labels[idx:idx+n_samples] = class_name
            idx += n_samples
        
        # Shuffle the dataset
        indices = np.random.permutation(total_samples)
        waveforms = waveforms[indices]
        labels = labels[indices]
        
        logger.info(f"Generated dataset shape: {waveforms.shape}")
        return waveforms, labels
    
    def _generate_normal(self, time: np.ndarray, frequency: float) -> np.ndarray:
        """Generate normal sinusoidal waveform with small noise"""
        amplitude = 230 * np.sqrt(2)  # 230V RMS -> peak voltage
        waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        # Add small noise
        noise = np.random.normal(0, amplitude * 0.02, len(time))
        return waveform + noise
    
    def _generate_normal_batch(self, time: np.ndarray, frequency: float, n_samples: int) -> np.ndarray:
        """Generate multiple normal waveforms at once using vectorized operations"""
        amplitude = 230 * np.sqrt(2)
        n_points = len(time)
        
        # Generate all waveforms at once
        base_waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        waveforms = np.tile(base_waveform, (n_samples, 1))
        
        # Add noise to all waveforms
        noise = np.random.normal(0, amplitude * 0.02, (n_samples, n_points))
        return waveforms + noise
    
    def _generate_sag(self, time: np.ndarray, frequency: float) -> np.ndarray:
        """Generate voltage sag (dip) waveform"""
        amplitude = 230 * np.sqrt(2)
        waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        
        # Create sag in the middle portion
        sag_start = int(0.3 * len(time))
        sag_end = int(0.7 * len(time))
        sag_depth = np.random.uniform(0.1, 0.5)  # 10-50% voltage reduction
        waveform[sag_start:sag_end] *= (1 - sag_depth)
        
        noise = np.random.normal(0, amplitude * 0.02, len(time))
        return waveform + noise
    
    def _generate_sag_batch(self, time: np.ndarray, frequency: float, n_samples: int) -> np.ndarray:
        """Generate multiple sag waveforms at once using vectorized operations"""
        amplitude = 230 * np.sqrt(2)
        n_points = len(time)
        
        # Generate base waveforms
        base_waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        waveforms = np.tile(base_waveform, (n_samples, 1))
        
        # Apply sag to middle portion with random depths
        sag_start = int(0.3 * n_points)
        sag_end = int(0.7 * n_points)
        sag_depths = np.random.uniform(0.1, 0.5, (n_samples, 1))
        waveforms[:, sag_start:sag_end] *= (1 - sag_depths)
        
        # Add noise
        noise = np.random.normal(0, amplitude * 0.02, (n_samples, n_points))
        return waveforms + noise
    
    def _generate_swell(self, time: np.ndarray, frequency: float) -> np.ndarray:
        """Generate voltage swell waveform"""
        amplitude = 230 * np.sqrt(2)
        waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        
        # Create swell in the middle portion
        swell_start = int(0.3 * len(time))
        swell_end = int(0.7 * len(time))
        swell_magnitude = np.random.uniform(1.1, 1.4)  # 10-40% voltage increase
        waveform[swell_start:swell_end] *= swell_magnitude
        
        noise = np.random.normal(0, amplitude * 0.02, len(time))
        return waveform + noise
    
    def _generate_swell_batch(self, time: np.ndarray, frequency: float, n_samples: int) -> np.ndarray:
        """Generate multiple swell waveforms at once using vectorized operations"""
        amplitude = 230 * np.sqrt(2)
        n_points = len(time)
        
        # Generate base waveforms
        base_waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        waveforms = np.tile(base_waveform, (n_samples, 1))
        
        # Apply swell to middle portion with random magnitudes
        swell_start = int(0.3 * n_points)
        swell_end = int(0.7 * n_points)
        swell_magnitudes = np.random.uniform(1.1, 1.4, (n_samples, 1))
        waveforms[:, swell_start:swell_end] *= swell_magnitudes
        
        # Add noise
        noise = np.random.normal(0, amplitude * 0.02, (n_samples, n_points))
        return waveforms + noise
    
    def _generate_harmonic(self, time: np.ndarray, frequency: float) -> np.ndarray:
        """Generate waveform with harmonic distortion"""
        amplitude = 230 * np.sqrt(2)
        # Fundamental frequency
        waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        
        # Add harmonics (3rd, 5th, 7th)
        harmonics = [3, 5, 7]
        harmonic_magnitudes = [0.2, 0.15, 0.1]
        
        for harmonic, magnitude in zip(harmonics, harmonic_magnitudes):
            waveform += amplitude * magnitude * np.sin(2 * np.pi * frequency * harmonic * time)
        
        noise = np.random.normal(0, amplitude * 0.02, len(time))
        return waveform + noise
    
    def _generate_harmonic_batch(self, time: np.ndarray, frequency: float, n_samples: int) -> np.ndarray:
        """Generate multiple harmonic waveforms at once using vectorized operations"""
        amplitude = 230 * np.sqrt(2)
        n_points = len(time)
        
        # Pre-compute all harmonic components
        omega_t = 2 * np.pi * frequency * time
        fundamental = amplitude * np.sin(omega_t)
        harmonic_3 = amplitude * 0.2 * np.sin(3 * omega_t)
        harmonic_5 = amplitude * 0.15 * np.sin(5 * omega_t)
        harmonic_7 = amplitude * 0.1 * np.sin(7 * omega_t)
        
        # Sum all components
        base_waveform = fundamental + harmonic_3 + harmonic_5 + harmonic_7
        waveforms = np.tile(base_waveform, (n_samples, 1))
        
        # Add noise
        noise = np.random.normal(0, amplitude * 0.02, (n_samples, n_points))
        return waveforms + noise
    
    def _generate_outage(self, time: np.ndarray, frequency: float) -> np.ndarray:
        """Generate outage (interruption) waveform"""
        amplitude = 230 * np.sqrt(2)
        waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        
        # Create outage in the middle portion
        outage_start = int(0.4 * len(time))
        outage_end = int(0.8 * len(time))
        waveform[outage_start:outage_end] = 0
        
        noise = np.random.normal(0, amplitude * 0.02, len(time))
        waveform += noise
        waveform[outage_start:outage_end] = 0  # Ensure outage region stays zero
        
        return waveform
    
    def _generate_outage_batch(self, time: np.ndarray, frequency: float, n_samples: int) -> np.ndarray:
        """Generate multiple outage waveforms at once using vectorized operations"""
        amplitude = 230 * np.sqrt(2)
        n_points = len(time)
        
        # Generate base waveforms
        base_waveform = amplitude * np.sin(2 * np.pi * frequency * time)
        waveforms = np.tile(base_waveform, (n_samples, 1))
        
        # Add noise
        noise = np.random.normal(0, amplitude * 0.02, (n_samples, n_points))
        waveforms += noise
        
        # Apply outage to middle portion
        outage_start = int(0.4 * n_points)
        outage_end = int(0.8 * n_points)
        waveforms[:, outage_start:outage_end] = 0
        
        return waveforms
    
    def save_dataset(self, waveforms: np.ndarray, labels: np.ndarray, filename: str = "pq_dataset.npz"):
        """Save dataset to file"""
        filepath = os.path.join(self.data_dir, filename)
        np.savez(filepath, waveforms=waveforms, labels=labels)
        logger.info(f"Dataset saved to {filepath}")
    
    def load_saved_dataset(self, filename: str = "pq_dataset.npz") -> Tuple[np.ndarray, np.ndarray]:
        """Load previously saved dataset"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        data = np.load(filepath)
        return data['waveforms'], data['labels']


def get_dataset_info() -> Dict[str, str]:
    """
    Return information about available PQ datasets and their sources
    """
    datasets = {
        "Synthetic Generator": "Built-in synthetic waveform generator for all PQ anomaly types",
        "UCI ML Repository": "Power Quality Dataset - https://archive.ics.uci.edu/",
        "IEEE DataPort": "Various PQ monitoring datasets - https://ieee-dataport.org/",
        "Note": "This implementation uses synthetic data generation due to dataset availability. "
                "Real datasets can be integrated by placing CSV files in the data/ directory."
    }
    return datasets
