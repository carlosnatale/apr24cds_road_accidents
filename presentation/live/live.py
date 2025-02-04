import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
MODEL_PATH = "xgboost_model.json"  # Updated to JSON format
SCALER_PATH = "scaler.pkl"

def load_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)  # Load from JSON
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
    atm_condition = st.selectbox("Weather Conditions", [1, 2, 3, 4, 5])
    collision_type = st.selectbox("Collision Type", [1, 2, 3, 4, 5, 6])

with col2:
    lum = st.selectbox("Lighting Conditions", [1, 2, 3, 4, 5])
    user_category = st.selectbox("User Category", [1, 2, 3, 4])
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender", [1, 2])
    reason_travel = st.selectbox("Reason for Travel", [0, 1, 2, 3, 4, 5])
    safety_equipment1 = st.selectbox("Safety Equipment Used", [1, 2, 3, 4, 5, 6])

# Ensure input features match the scaler
expected_features = [
    "day", "month", "time", "day_sin", "day_cos", "month_sin", "month_cos", "time_sin", "time_cos", 
    "speed_limit", "atm_condition", "collision_type", "lum", "user_category", "age", "gender", 
    "reason_travel", "safety_equipment1"
]

# Preprocess Input Data
input_data = pd.DataFrame([{  
    "day": day,
    "month": month,
    "time": time,
    "day_sin": np.sin(2 * np.pi * day / 31),
    "day_cos": np.cos(2 * np.pi * day / 31),
    "month_sin": np.sin(2 * np.pi * month / 12),
    "month_cos": np.cos(2 * np.pi * month / 12),
    "time_sin": np.sin(2 * np.pi * time / 24),
    "time_cos": np.cos(2 * np.pi * time / 24),
    "speed_limit": speed_limit,
    "atm_condition": atm_condition,
    "collision_type": collision_type,
    "lum": lum,
    "user_category": user_category,
    "age": age,
    "gender": gender,
    "reason_travel": reason_travel,
    "safety_equipment1": safety_equipment1
}])

# Ensure input matches scaler's expected features
input_data = input_data[expected_features]

# Load model and scaler
model, scaler = load_model()

# Normalize input
input_data_scaled = scaler.transform(input_data)

# Make Prediction
if st.button("Predict Injury Severity"):
    dmatrix = xgb.DMatrix(input_data_scaled)  # Convert input for Booster
    prediction = model.predict(dmatrix)
    severity = "Severely Injured" if prediction[0] > 0.5 else "Slightly Injured"  # Adjust threshold if needed
    st.success(f"Predicted Injury Severity: {severity}")
