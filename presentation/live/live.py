import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# Initialize Streamlit app
st.title("Injury Severity Prediction with XGBoost")
st.write("Enter accident details to predict severity")

# User Input Fields
col1, col2 = st.columns(2)

with col1:
    day = st.number_input("Day of Accident", min_value=1, max_value=31, step=1)
    month = st.number_input("Month of Accident", min_value=1, max_value=12, step=1)
    time = st.number_input("Time (Hour)", min_value=0, max_value=23, step=1)
    speed_limit = st.number_input("Speed Limit", min_value=10, max_value=130, step=10)
    weather = st.selectbox("Weather Conditions", ["Clear", "Rain", "Fog", "Snow", "Other"])
    road_type = st.selectbox("Road Type", ["Urban", "Rural", "Highway"])

with col2:
    light_conditions = st.selectbox("Lighting Conditions", ["Daylight", "Dark - Street lights", "Dark - No lights"])
    vehicle_type = st.selectbox("Vehicle Type", ["Car", "Motorcycle", "Bicycle", "Truck", "Bus", "Other"])
    driver_age = st.number_input("Driver Age", min_value=18, max_value=100, step=1)
    seatbelt_used = st.selectbox("Seatbelt Used", ["No", "Yes"])

# Encode categorical values
weather_dict = {"Clear": 0, "Rain": 1, "Fog": 2, "Snow": 3, "Other": 4}
road_dict = {"Urban": 0, "Rural": 1, "Highway": 2}
light_dict = {"Daylight": 0, "Dark - Street lights": 1, "Dark - No lights": 2}
vehicle_dict = {"Car": 0, "Motorcycle": 1, "Bicycle": 2, "Truck": 3, "Bus": 4, "Other": 5}
seatbelt_dict = {"No": 0, "Yes": 1}

# Preprocess Input Data
input_data = pd.DataFrame({
    "day": [day],
    "month": [month],
    "time": [time],
    "day_sin": [np.sin(2 * np.pi * day / 31)],
    "day_cos": [np.cos(2 * np.pi * day / 31)],
    "month_sin": [np.sin(2 * np.pi * month / 12)],
    "month_cos": [np.cos(2 * np.pi * month / 12)],
    "time_sin": [np.sin(2 * np.pi * time / 24)],
    "time_cos": [np.cos(2 * np.pi * time / 24)],
    "speed_limit": [speed_limit],
    "weather": [weather_dict[weather]],
    "road_type": [road_dict[road_type]],
    "light_conditions": [light_dict[light_conditions]],
    "vehicle_type": [vehicle_dict[vehicle_type]],
    "driver_age": [driver_age],
    "seatbelt_used": [seatbelt_dict[seatbelt_used]]
})

# Load model and scaler
model, scaler = load_model()

# Ensure input features match expected features
expected_features = scaler.feature_names_in_
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing features with default values

# Keep only expected columns, in correct order
input_data = input_data[expected_features]

# Normalize input
input_data_scaled = scaler.transform(input_data)

# Make Prediction
if st.button("Predict Injury Severity"):
    prediction = model.predict(input_data_scaled)
    severity = "Severely Injured" if prediction[0] == 1 else "Slightly Injured"
    st.success(f"Predicted Injury Severity: {severity}")