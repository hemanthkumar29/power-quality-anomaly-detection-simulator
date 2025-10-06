"""
Streamlit Web Application for Power Quality Anomaly Detection
Interactive interface for waveform visualization and classification
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import PQDataLoader
from src.feature_extraction import FeatureExtractor
from src.model_training import PQModelTrainer
from src.visualization import PQVisualizer

# Page configuration
st.set_page_config(
    page_title="Power Quality Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models (cached)"""
    try:
        trainer = PQModelTrainer(model_dir="models")
        trainer.load_models()
        return trainer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run train.py first to train the models.")
        return None


@st.cache_resource
def initialize_components():
    """Initialize data loader, feature extractor, and visualizer"""
    data_loader = PQDataLoader(data_dir="data")
    feature_extractor = FeatureExtractor()
    visualizer = PQVisualizer()
    return data_loader, feature_extractor, visualizer


def main():
    """Main application"""
    
    # Title and description
    st.title("⚡ Power Quality Anomaly Detection Simulator")
    st.markdown("""
    This application uses machine learning to detect and classify power quality anomalies 
    in electrical waveforms. Upload your own data or generate synthetic waveforms to test the system.
    """)
    
    # Initialize components
    data_loader, feature_extractor, visualizer = initialize_components()
    trainer = load_models()
    
    if trainer is None:
        st.warning("⚠️ Models not loaded. Please train models first by running: `python train.py`")
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Generate Synthetic Waveform", "Upload CSV File", "Use Sample Dataset"]
    )
    
    # Model selection
    available_models = list(trainer.models.keys())
    selected_model = st.sidebar.selectbox(
        "Select Classification Model:",
        available_models,
        index=available_models.index('xgboost') if 'xgboost' in available_models else 0
    )
    
    # Sampling parameters
    st.sidebar.subheader("Sampling Parameters")
    sampling_rate = st.sidebar.number_input(
        "Sampling Rate (Hz)", 
        min_value=1000, 
        max_value=10000, 
        value=6400, 
        step=100
    )
    duration = st.sidebar.slider(
        "Duration (seconds)", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.2, 
        step=0.1
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Waveform Input")
        
        waveform = None
        true_label = None
        
        # Handle different data sources
        if data_source == "Generate Synthetic Waveform":
            anomaly_type = st.selectbox(
                "Select Anomaly Type:",
                ["Normal", "Sag", "Swell", "Harmonic", "Outage"]
            )
            
            if st.button("🎲 Generate Waveform", type="primary"):
                # Generate waveform
                time = np.linspace(0, duration, int(sampling_rate * duration))
                frequency = 60  # 60 Hz
                
                # Use data loader methods
                if anomaly_type == "Normal":
                    waveform = data_loader._generate_normal(time, frequency)
                elif anomaly_type == "Sag":
                    waveform = data_loader._generate_sag(time, frequency)
                elif anomaly_type == "Swell":
                    waveform = data_loader._generate_swell(time, frequency)
                elif anomaly_type == "Harmonic":
                    waveform = data_loader._generate_harmonic(time, frequency)
                elif anomaly_type == "Outage":
                    waveform = data_loader._generate_outage(time, frequency)
                
                true_label = anomaly_type
                st.session_state.waveform = waveform
                st.session_state.true_label = true_label
                st.success(f"✅ Generated {anomaly_type} waveform")
        
        elif data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload CSV file with waveform data",
                type=['csv']
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    # Assume first column is the waveform
                    waveform = df.iloc[:, 0].values
                    st.session_state.waveform = waveform
                    st.success("✅ Waveform loaded from CSV")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        
        elif data_source == "Use Sample Dataset":
            if st.button("📥 Load Random Sample", type="primary"):
                try:
                    waveforms, labels = data_loader.load_saved_dataset()
                    idx = np.random.randint(0, len(waveforms))
                    waveform = waveforms[idx]
                    true_label = labels[idx]
                    st.session_state.waveform = waveform
                    st.session_state.true_label = true_label
                    st.success(f"✅ Loaded sample waveform (True label: {true_label})")
                except FileNotFoundError:
                    st.error("No saved dataset found. Please run train.py first.")
        
        # Display waveform if available
        if 'waveform' in st.session_state:
            waveform = st.session_state.waveform
            true_label = st.session_state.get('true_label', None)
            
            st.subheader("🌊 Waveform Visualization")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Domain", "Combined"])
            
            with tab1:
                fig = visualizer.plot_waveform_interactive(waveform, title="Voltage Waveform")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig_fft = visualizer.plot_fft(waveform, title="Frequency Spectrum")
                st.pyplot(fig_fft)
                plt.close()
            
            with tab3:
                fig_combined = visualizer.plot_waveform_with_fft(waveform)
                st.pyplot(fig_combined)
                plt.close()
    
    with col2:
        st.header("🔍 Analysis")
        
        if 'waveform' in st.session_state:
            waveform = st.session_state.waveform
            
            # Extract features
            with st.spinner("Extracting features..."):
                features_dict = feature_extractor.extract_all_features(waveform)
            
            # Display key features
            st.subheader("📈 Key Features")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("RMS Voltage", f"{features_dict['rms_voltage']:.2f} V")
                st.metric("Peak Voltage", f"{features_dict['peak_voltage']:.2f} V")
                st.metric("Crest Factor", f"{features_dict['crest_factor']:.3f}")
            
            with col_b:
                st.metric("THD", f"{features_dict['thd']:.3f}")
                st.metric("Dip %", f"{features_dict['dip_percentage']:.2f}%")
                st.metric("Swell %", f"{features_dict['swell_percentage']:.2f}%")
            
            # Classification
            st.subheader("🎯 Classification")
            
            if st.button("🚀 Classify Anomaly", type="primary"):
                with st.spinner(f"Classifying with {selected_model}..."):
                    # Prepare features
                    features_array = np.array([list(features_dict.values())])
                    
                    # Predict
                    predictions, probabilities = trainer.predict(
                        features_array, 
                        model_name=selected_model
                    )
                    
                    predicted_label = predictions[0]
                    
                    st.session_state.predicted_label = predicted_label
                    st.session_state.probabilities = probabilities
                
                # Display results
                predicted_label = st.session_state.predicted_label
                probabilities = st.session_state.probabilities
                
                # Show prediction with color coding
                if true_label:
                    if predicted_label == true_label:
                        st.success(f"### ✅ Predicted: **{predicted_label}**")
                        st.info(f"True Label: **{true_label}**")
                    else:
                        st.error(f"### ❌ Predicted: **{predicted_label}**")
                        st.info(f"True Label: **{true_label}**")
                else:
                    st.success(f"### Predicted: **{predicted_label}**")
                
                # Show confidence scores
                if probabilities is not None:
                    st.subheader("📊 Confidence Scores")
                    
                    prob_df = pd.DataFrame({
                        'Class': trainer.class_names,
                        'Probability': probabilities[0]
                    }).sort_values('Probability', ascending=False)
                    
                    st.dataframe(
                        prob_df.style.format({'Probability': '{:.2%}'}),
                        use_container_width=True
                    )
                    
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(prob_df['Class'], prob_df['Probability'], color='steelblue')
                    ax.set_xlabel('Probability')
                    ax.set_xlim(0, 1)
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            # Show all features in expander
            with st.expander("📋 View All Extracted Features"):
                features_df = pd.DataFrame(
                    list(features_dict.items()),
                    columns=['Feature', 'Value']
                )
                st.dataframe(features_df, use_container_width=True)
    
    # Model Information Section
    st.header("ℹ️ Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Classes", len(trainer.class_names))
        st.caption(f"Classes: {', '.join(trainer.class_names)}")
    
    with col2:
        st.metric("Available Models", len(trainer.models))
        st.caption(f"Models: {', '.join(available_models)}")
    
    with col3:
        st.metric("Features Extracted", len(feature_extractor.get_feature_names()))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 About Power Quality Anomalies
    
    - **Normal**: Stable sinusoidal waveform at nominal voltage
    - **Sag (Dip)**: Temporary decrease in voltage (10-90% reduction)
    - **Swell**: Temporary increase in voltage (10-80% increase)
    - **Harmonic**: Distortion caused by non-linear loads (multiples of fundamental frequency)
    - **Outage**: Complete loss of voltage (interruption)
    
    **Dataset Source**: Synthetic waveforms generated using IEEE power quality standards
    """)


if __name__ == "__main__":
    main()
