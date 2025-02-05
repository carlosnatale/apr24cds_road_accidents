import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Set Streamlit page configuration
st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚦", layout="wide")

# Load model, scaler, and feature names
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        with open("feature_names.pkl", "rb") as file:
            feature_names = pickle.load(file)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Preprocess user input
def preprocess_input(user_input, scaler, feature_names):
    df = pd.DataFrame([user_input])
    
    # Cyclical encoding for time variables
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['time_sin'] = np.sin(2 * np.pi * df['time'] / 86340000)
    df['time_cos'] = np.cos(2 * np.pi * df['time'] / 86340000)
    
    # Drop original time variables
    df.drop(columns=['day', 'month', 'time'], inplace=True)
    
    # Ensure numerical features exist in the DataFrame
    numerical_features = ['lat', 'long', 'upstream_terminal_number', 'distance_upstream_terminal', 'maximum_speed', 'age']
    for col in numerical_features:
        if col not in df.columns:
            df[col] = 0  # Set default value if missing
    
    # Standardize numerical features if scaler is available
    if scaler is not None:
        df[numerical_features] = scaler.transform(df[numerical_features])
    
    # One-hot encode categorical features
    categorical_features = ['lum', 'atm_condition', 'collision_type', 'route_category',
                            'traffic_regime', 'vehicle_category', 'user_category', 'gender']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Ensure all training features exist and are in correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Add missing categorical features
    df = df[feature_names]
    
    return df

# UI Layout
st.title("🚦 Accident Severity Prediction")
st.write("This tool helps assess accident severity based on various conditions.")

# Sidebar Inputs
st.sidebar.header("📌 Enter Accident Details")
with st.sidebar.expander("🕒 Time & Location", expanded=True):
    day = st.number_input("Day", min_value=1, max_value=31, value=15)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    time = st.number_input("Time (HHMMSS)", min_value=0, max_value=235959, value=120000)
    lat = st.number_input("Latitude", value=48.85)
    long = st.number_input("Longitude", value=2.35)

with st.sidebar.expander("🚗 Vehicle & Driver", expanded=False):
    max_speed = st.number_input("Max Speed (km/h)", value=50)
    age = st.number_input("Driver Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    upstream_terminal_number = st.number_input("Upstream Terminal Number", min_value=0, value=0)
    distance_upstream_terminal = st.number_input("Distance to Upstream Terminal (m)", min_value=0, value=0)

with st.sidebar.expander("🌦 Environmental Conditions", expanded=False):
    lum = st.selectbox("Lighting Condition", ["Daylight", "Night with public lighting"])
    atm_condition = st.selectbox("Weather Condition", ["Normal", "Light rain", "Heavy rain"])
    collision_type = st.selectbox("Collision Type", ["Frontal", "Rear-end", "Side impact"])
    route_category = st.selectbox("Route Category", ["Highway", "National road", "Municipal road"])

# Prediction Section
st.markdown("### 🔍 Prediction Result")
if st.button("🚀 Predict Severity"):
    with st.spinner("Analyzing accident details..."):
        model, scaler, feature_names = load_model()
        if model is not None:
            user_input = {
                "day": day, "month": month, "time": time,
                "lat": lat, "long": long,
                "maximum_speed": max_speed, "age": age,
                "upstream_terminal_number": upstream_terminal_number,
                "distance_upstream_terminal": distance_upstream_terminal
            }
            processed_input = preprocess_input(user_input, scaler, feature_names)
            if processed_input is not None:
                prediction = model.predict(processed_input)[0]
                severity_mapping = {0: "🟢 Minor Injury or No Injury", 1: "🔴 Severe Injury or Fatal"}
                st.success(f"**Prediction: {severity_mapping[prediction]}**")

# Download Section
st.markdown("### 📥 Download Model Files")
st.write("Download model files for further analysis:")
col1, col2, col3 = st.columns(3)
with col1:
    with open("model.pkl", "rb") as f:
        st.download_button("📦 Model", f, file_name="model.pkl")
with col2:
    with open("scaler.pkl", "rb") as f:
        st.download_button("📉 Scaler", f, file_name="scaler.pkl")
with col3:
    with open("feature_names.pkl", "rb") as f:
        st.download_button("📑 Feature Names", f, file_name="feature_names.pkl")
