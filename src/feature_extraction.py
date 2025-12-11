"""
Feature Extraction Module for Power Quality Analysis
Extracts time-domain and frequency-domain features from waveforms
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from power quality waveforms"""
    
    def __init__(self, sampling_rate: int = 6400, fundamental_freq: float = 60.0):
        """
        Initialize feature extractor
        
        Args:
            sampling_rate: Sampling frequency in Hz
            fundamental_freq: Fundamental frequency of power system (50 or 60 Hz)
        """
        self.sampling_rate = sampling_rate
        self.fundamental_freq = fundamental_freq
    
    def extract_all_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """
        Extract all features from a single waveform
        
        Args:
            waveform: 1D array of voltage/current samples
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Time-domain features
        features['rms_voltage'] = self.calculate_rms(waveform)
        features['peak_voltage'] = self.calculate_peak(waveform)
        features['crest_factor'] = self.calculate_crest_factor(waveform)
        features['form_factor'] = self.calculate_form_factor(waveform)
        features['mean_voltage'] = np.mean(np.abs(waveform))
        features['std_voltage'] = np.std(waveform)
        features['voltage_range'] = np.ptp(waveform)
        
        # Frequency-domain features
        thd, harmonics = self.calculate_thd(waveform)
        features['thd'] = thd
        features['freq_deviation'] = self.calculate_frequency_deviation(waveform)
        
        # Add individual harmonic magnitudes (up to 7th harmonic)
        for i, harmonic_mag in enumerate(harmonics[:7], start=1):
            features[f'harmonic_{i}'] = harmonic_mag
        
        # Anomaly-specific features
        features['dip_percentage'] = self.calculate_dip_percentage(waveform)
        features['swell_percentage'] = self.calculate_swell_percentage(waveform)
        features['zero_crossing_rate'] = self.calculate_zero_crossing_rate(waveform)
        features['energy'] = self.calculate_energy(waveform)
        
        return features
    
    def extract_features_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """
        Extract features from multiple waveforms using vectorized operations
        
        Args:
            waveforms: 2D array of shape (n_samples, n_points)
            
        Returns:
            2D array of features shape (n_samples, n_features)
        """
        logger.info(f"Extracting features from {len(waveforms)} waveforms...")
        
        # Pre-allocate features array
        n_samples = len(waveforms)
        
        # Vectorized time-domain features
        rms_voltages = np.sqrt(np.mean(waveforms ** 2, axis=1))
        peak_voltages = np.max(np.abs(waveforms), axis=1)
        mean_voltages = np.mean(np.abs(waveforms), axis=1)
        std_voltages = np.std(waveforms, axis=1)
        voltage_ranges = np.ptp(waveforms, axis=1)
        
        # Crest and form factors (vectorized)
        crest_factors = np.where(rms_voltages > 0, peak_voltages / rms_voltages, 0)
        form_factors = np.where(mean_voltages > 0, rms_voltages / mean_voltages, 0)
        
        # Energy (vectorized)
        energies = np.sum(waveforms ** 2, axis=1)
        
        # Zero crossing rates (vectorized)
        sign_changes = np.diff(np.sign(waveforms), axis=1)
        zero_crossings = np.count_nonzero(sign_changes, axis=1)
        zero_crossing_rates = zero_crossings / waveforms.shape[1]
        
        # For frequency-domain and complex features, we still need per-sample processing
        # but we can optimize the critical path
        thd_values = np.zeros(n_samples)
        freq_deviations = np.zeros(n_samples)
        harmonic_features = np.zeros((n_samples, 7))
        dip_percentages = np.zeros(n_samples)
        swell_percentages = np.zeros(n_samples)
        
        # Process frequency-domain features with progress tracking
        for i, waveform in enumerate(waveforms):
            if i % 500 == 0 and i > 0:
                logger.info(f"Processed {i}/{n_samples} waveforms")
            
            # Calculate THD and harmonics
            thd, harmonics = self.calculate_thd(waveform)
            thd_values[i] = thd
            harmonic_features[i] = harmonics[:7]
            
            # Calculate frequency deviation
            freq_deviations[i] = self.calculate_frequency_deviation(waveform)
            
            # Calculate dip and swell percentages (optimized version)
            dip_percentages[i] = self._calculate_dip_percentage_fast(waveform)
            swell_percentages[i] = self._calculate_swell_percentage_fast(waveform)
        
        # Assemble all features in order
        features_array = np.column_stack([
            rms_voltages,
            peak_voltages,
            crest_factors,
            form_factors,
            mean_voltages,
            std_voltages,
            voltage_ranges,
            thd_values,
            freq_deviations,
            harmonic_features,
            dip_percentages,
            swell_percentages,
            zero_crossing_rates,
            energies
        ])
        
        logger.info(f"Feature extraction complete. Shape: {features_array.shape}")
        return features_array
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order"""
        # Create a dummy waveform to get feature names
        dummy_waveform = np.sin(2 * np.pi * self.fundamental_freq * np.linspace(0, 0.1, 640))
        features = self.extract_all_features(dummy_waveform)
        return list(features.keys())
    
    # Time-domain feature calculations
    
    def calculate_rms(self, waveform: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) voltage"""
        return np.sqrt(np.mean(waveform ** 2))
    
    def calculate_peak(self, waveform: np.ndarray) -> float:
        """Calculate peak voltage"""
        return np.max(np.abs(waveform))
    
    def calculate_crest_factor(self, waveform: np.ndarray) -> float:
        """
        Calculate crest factor (peak / RMS)
        Higher values indicate spikes or distortions
        """
        peak = self.calculate_peak(waveform)
        rms = self.calculate_rms(waveform)
        return peak / rms if rms > 0 else 0
    
    def calculate_form_factor(self, waveform: np.ndarray) -> float:
        """
        Calculate form factor (RMS / Mean absolute)
        """
        rms = self.calculate_rms(waveform)
        mean_abs = np.mean(np.abs(waveform))
        return rms / mean_abs if mean_abs > 0 else 0
    
    def calculate_energy(self, waveform: np.ndarray) -> float:
        """Calculate signal energy"""
        return np.sum(waveform ** 2)
    
    def calculate_zero_crossing_rate(self, waveform: np.ndarray) -> float:
        """Calculate zero crossing rate (normalized)"""
        zero_crossings = np.where(np.diff(np.sign(waveform)))[0]
        return len(zero_crossings) / len(waveform)
    
    # Frequency-domain feature calculations
    
    def calculate_thd(self, waveform: np.ndarray) -> tuple:
        """
        Calculate Total Harmonic Distortion (THD) and harmonic magnitudes
        
        Returns:
            Tuple of (THD value, list of harmonic magnitudes)
        """
        # Perform FFT
        n = len(waveform)
        fft_values = fft(waveform)
        fft_magnitude = np.abs(fft_values[:n//2])
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        
        # Find fundamental frequency component
        fund_idx = np.argmax(fft_magnitude)
        fundamental_mag = fft_magnitude[fund_idx]
        
        # Calculate harmonic magnitudes
        harmonics = []
        harmonic_sum_squared = 0
        
        for harmonic_num in range(1, 15):  # Check up to 14th harmonic
            # Find the harmonic frequency
            harmonic_freq = self.fundamental_freq * harmonic_num
            # Find closest frequency bin
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_mag = fft_magnitude[harmonic_idx]
            
            if harmonic_num == 1:
                # This is the fundamental
                fundamental_mag = harmonic_mag
            else:
                # This is a harmonic
                harmonic_sum_squared += harmonic_mag ** 2
            
            harmonics.append(harmonic_mag)
        
        # Calculate THD
        if fundamental_mag > 0:
            thd = np.sqrt(harmonic_sum_squared) / fundamental_mag
        else:
            thd = 0
        
        return thd, harmonics
    
    def calculate_frequency_deviation(self, waveform: np.ndarray) -> float:
        """
        Calculate frequency deviation from nominal frequency
        """
        # Use zero crossings to estimate frequency
        zero_crossings = np.where(np.diff(np.sign(waveform)))[0]
        
        if len(zero_crossings) < 2:
            return 0
        
        # Calculate average time between zero crossings
        avg_half_period = np.mean(np.diff(zero_crossings)) / self.sampling_rate
        
        if avg_half_period > 0:
            estimated_freq = 1 / (2 * avg_half_period)
            deviation = abs(estimated_freq - self.fundamental_freq)
        else:
            deviation = 0
        
        return deviation
    
    # Anomaly-specific feature calculations
    
    def calculate_dip_percentage(self, waveform: np.ndarray) -> float:
        """
        Calculate voltage dip (sag) percentage
        Maximum percentage drop from nominal voltage
        """
        return self._calculate_dip_percentage_fast(waveform)
    
    def _calculate_dip_percentage_fast(self, waveform: np.ndarray) -> float:
        """
        Optimized calculation of voltage dip percentage using vectorized operations
        """
        nominal_rms = 230  # Nominal voltage (RMS)
        
        # Calculate RMS in sliding windows using vectorized approach
        window_size = max(len(waveform) // 10, 10)
        step = window_size // 2
        
        # Pre-compute squared values
        waveform_sq = waveform ** 2
        
        # Use cumulative sum for efficient sliding window RMS
        cumsum = np.cumsum(np.insert(waveform_sq, 0, 0))
        
        # Calculate window RMS values efficiently
        window_starts = range(0, len(waveform) - window_size + 1, step)
        window_rms_values = []
        
        for start in window_starts:
            end = start + window_size
            window_sum = cumsum[end] - cumsum[start]
            window_rms_values.append(np.sqrt(window_sum / window_size))
        
        if window_rms_values:
            min_rms = min(window_rms_values)
        else:
            min_rms = np.sqrt(np.mean(waveform_sq))
        
        dip_percentage = max(0, (nominal_rms - min_rms) / nominal_rms * 100)
        return dip_percentage
    
    def calculate_swell_percentage(self, waveform: np.ndarray) -> float:
        """
        Calculate voltage swell percentage
        Maximum percentage increase from nominal voltage
        """
        return self._calculate_swell_percentage_fast(waveform)
    
    def _calculate_swell_percentage_fast(self, waveform: np.ndarray) -> float:
        """
        Optimized calculation of voltage swell percentage using vectorized operations
        """
        nominal_rms = 230  # Nominal voltage (RMS)
        
        # Calculate RMS in sliding windows using vectorized approach
        window_size = max(len(waveform) // 10, 10)
        step = window_size // 2
        
        # Pre-compute squared values
        waveform_sq = waveform ** 2
        
        # Use cumulative sum for efficient sliding window RMS
        cumsum = np.cumsum(np.insert(waveform_sq, 0, 0))
        
        # Calculate window RMS values efficiently
        window_starts = range(0, len(waveform) - window_size + 1, step)
        window_rms_values = []
        
        for start in window_starts:
            end = start + window_size
            window_sum = cumsum[end] - cumsum[start]
            window_rms_values.append(np.sqrt(window_sum / window_size))
        
        if window_rms_values:
            max_rms = max(window_rms_values)
        else:
            max_rms = np.sqrt(np.mean(waveform_sq))
        
        swell_percentage = max(0, (max_rms - nominal_rms) / nominal_rms * 100)
        return swell_percentage
    
    def apply_preprocessing(self, waveform: np.ndarray, 
                          normalize: bool = True,
                          denoise: bool = False) -> np.ndarray:
        """
        Apply preprocessing to waveform
        
        Args:
            waveform: Input waveform
            normalize: Whether to normalize the waveform
            denoise: Whether to apply denoising filter
            
        Returns:
            Preprocessed waveform
        """
        processed = waveform.copy()
        
        if denoise:
            # Apply low-pass filter to remove high-frequency noise
            nyquist = self.sampling_rate / 2
            cutoff = 1000  # Hz
            b, a = signal.butter(4, cutoff / nyquist, btype='low')
            processed = signal.filtfilt(b, a, processed)
        
        if normalize:
            # Normalize to zero mean and unit variance
            processed = (processed - np.mean(processed)) / (np.std(processed) + 1e-8)
        
        return processed
