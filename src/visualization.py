"""
Visualization Utilities for Power Quality Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class PQVisualizer:
    """Visualization tools for power quality waveforms and analysis"""
    
    def __init__(self, sampling_rate: int = 6400, fundamental_freq: float = 60.0):
        """
        Initialize visualizer
        
        Args:
            sampling_rate: Sampling frequency in Hz
            fundamental_freq: Fundamental frequency (50 or 60 Hz)
        """
        self.sampling_rate = sampling_rate
        self.fundamental_freq = fundamental_freq
    
    def plot_waveform(
        self, 
        waveform: np.ndarray, 
        title: str = "Power Quality Waveform",
        show_rms: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot time-domain waveform
        
        Args:
            waveform: Input waveform
            title: Plot title
            show_rms: Whether to show RMS line
            save_path: Path to save figure
        """
        time = np.arange(len(waveform)) / self.sampling_rate
        
        plt.figure(figsize=(12, 6))
        plt.plot(time, waveform, linewidth=1.5, label='Voltage')
        
        if show_rms:
            rms = np.sqrt(np.mean(waveform ** 2))
            plt.axhline(y=rms, color='r', linestyle='--', label=f'RMS = {rms:.2f}V')
            plt.axhline(y=-rms, color='r', linestyle='--')
        
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Voltage (V)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waveform plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_waveform_interactive(
        self, 
        waveform: np.ndarray, 
        title: str = "Power Quality Waveform"
    ) -> go.Figure:
        """
        Create interactive waveform plot using Plotly
        
        Args:
            waveform: Input waveform
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        time = np.arange(len(waveform)) / self.sampling_rate
        rms = np.sqrt(np.mean(waveform ** 2))
        
        fig = go.Figure()
        
        # Add waveform trace
        fig.add_trace(go.Scatter(
            x=time,
            y=waveform,
            mode='lines',
            name='Voltage',
            line=dict(color='blue', width=1.5)
        ))
        
        # Add RMS lines
        fig.add_hline(y=rms, line_dash="dash", line_color="red", 
                     annotation_text=f"RMS = {rms:.2f}V")
        fig.add_hline(y=-rms, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Voltage (V)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_fft(
        self, 
        waveform: np.ndarray, 
        title: str = "Frequency Spectrum",
        max_freq: float = 1000,
        save_path: Optional[str] = None
    ):
        """
        Plot frequency spectrum using FFT
        
        Args:
            waveform: Input waveform
            title: Plot title
            max_freq: Maximum frequency to display
            save_path: Path to save figure
        """
        n = len(waveform)
        fft_values = fft(waveform)
        fft_magnitude = np.abs(fft_values[:n//2])
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        
        # Limit to max_freq
        mask = freqs <= max_freq
        freqs = freqs[mask]
        fft_magnitude = fft_magnitude[mask]
        
        plt.figure(figsize=(12, 6))
        plt.plot(freqs, fft_magnitude, linewidth=1.5)
        
        # Mark fundamental frequency
        fund_idx = np.argmin(np.abs(freqs - self.fundamental_freq))
        plt.axvline(x=self.fundamental_freq, color='r', linestyle='--', 
                   label=f'Fundamental ({self.fundamental_freq} Hz)')
        
        # Mark harmonics
        for i in range(2, 8):
            harmonic_freq = i * self.fundamental_freq
            if harmonic_freq <= max_freq:
                plt.axvline(x=harmonic_freq, color='orange', linestyle=':', alpha=0.5,
                           label=f'{i}th Harmonic' if i == 2 else '')
        
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Magnitude', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"FFT plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_waveform_with_fft(
        self, 
        waveform: np.ndarray, 
        title: str = "Waveform Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot waveform and its FFT side by side
        
        Args:
            waveform: Input waveform
            title: Overall title
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Time domain
        time = np.arange(len(waveform)) / self.sampling_rate
        rms = np.sqrt(np.mean(waveform ** 2))
        
        ax1.plot(time, waveform, linewidth=1.5)
        ax1.axhline(y=rms, color='r', linestyle='--', alpha=0.7, label=f'RMS = {rms:.2f}V')
        ax1.axhline(y=-rms, color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Voltage (V)', fontsize=12)
        ax1.set_title('Time Domain', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain
        n = len(waveform)
        fft_values = fft(waveform)
        fft_magnitude = np.abs(fft_values[:n//2])
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        
        # Limit to 1000 Hz
        mask = freqs <= 1000
        freqs_limited = freqs[mask]
        fft_magnitude_limited = fft_magnitude[mask]
        
        ax2.plot(freqs_limited, fft_magnitude_limited, linewidth=1.5)
        ax2.axvline(x=self.fundamental_freq, color='r', linestyle='--', 
                   label=f'Fundamental ({self.fundamental_freq} Hz)')
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Magnitude', fontsize=12)
        ax2.set_title('Frequency Spectrum', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Combined plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray, 
        class_names: List[str],
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_feature_importance(
        self, 
        feature_names: List[str], 
        importance_values: np.ndarray,
        title: str = "Feature Importance",
        top_n: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importance_values: Feature importance values
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        # Sort by importance
        indices = np.argsort(importance_values)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance_values[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance, color='steelblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_multiple_waveforms(
        self, 
        waveforms: List[np.ndarray], 
        labels: List[str],
        title: str = "Sample Waveforms by Class",
        save_path: Optional[str] = None
    ):
        """
        Plot multiple waveforms in a grid
        
        Args:
            waveforms: List of waveforms
            labels: List of labels for each waveform
            title: Overall title
            save_path: Path to save figure
        """
        n_waveforms = len(waveforms)
        n_cols = 2
        n_rows = (n_waveforms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_waveforms > 1 else [axes]
        
        for i, (waveform, label) in enumerate(zip(waveforms, labels)):
            time = np.arange(len(waveform)) / self.sampling_rate
            axes[i].plot(time, waveform, linewidth=1)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Voltage (V)')
            axes[i].set_title(f'{label}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_waveforms, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Multi-waveform plot saved to {save_path}")
        
        return fig
