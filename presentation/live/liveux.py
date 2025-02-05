import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Load model and preprocessing components
def load_model():
    """Loads the trained XGBoost model, scaler, and feature names."""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    
    return model, scaler, feature_names

# Preprocess user input
def preprocess_input(user_input, scaler, feature_names):
    df = pd.DataFrame([user_input])
    df_scaled = scaler.transform(df)
    return df_scaled

# Streamlit UI Setup
def main():
    st.set_page_config(
        page_title="Accident Severity Prediction",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS Styling
    st.markdown("""
        <style>
            :root {
                --primary: #1e3a8a;
                --secondary: #bfdbfe;
            }
            .main { background-color: #f8fafc; }
            .sidebar .sidebar-content { background: var(--primary); color: white; }
            .stButton>button {
                background-color: var(--primary);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                width: 100%;
                transition: transform 0.2s;
            }
            .stButton>button:hover {
                background-color: #1d4ed8;
                transform: scale(1.05);
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚨 Accident Severity Predictor")
    st.markdown("""This predictive tool assesses road accident parameters to determine potential severity outcomes.""")

    # Load model components
    model, scaler, feature_names = load_model()

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Accident Parameters")
        user_input = {
            "day": st.number_input("Day of Month", min_value=1, max_value=31, value=15),
            "month": st.number_input("Month", min_value=1, max_value=12, value=6),
            "time": st.number_input("Time (HHMMSS)", min_value=0, max_value=235959, value=120000),
            "lat": st.number_input("Latitude", value=48.85),
            "long": st.number_input("Longitude", value=2.35),
            "maximum_speed": st.number_input("Speed Limit (km/h)", value=50),
            "age": st.number_input("Driver Age", min_value=18, max_value=100, value=30),
        }

    # Prediction section
    if st.button("🔍 Analyze Severity Risk"):
        with st.spinner("Evaluating risk factors..."):
            processed_input = preprocess_input(user_input, scaler, feature_names)
            prediction = model.predict(processed_input)[0]

        severity_mapping = {
            0: "Low Risk: Minor or No Injuries",
            1: "High Risk: Severe Injuries or Fatalities"
        }
        st.success(f"**Prediction Result:**  {severity_mapping[prediction]}")

    # Download section
    st.markdown("### 🗄️ Model Resources")
    for file, label in [("model.pkl", "Download Model"),
                        ("scaler.pkl", "Download Scaler"),
                        ("feature_names.pkl", "Download Features")]:
        with open(file, "rb") as f:
            st.download_button(label=f"📦 {label}", data=f, file_name=file, use_container_width=True)

if __name__ == "__main__":
    main()
