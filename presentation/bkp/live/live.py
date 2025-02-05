import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def load_model():
    """Load the trained XGBoost model, the standard scaler, and feature names."""
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    with open("feature_names.pkl", "rb") as file:
        feature_names = pickle.load(file)  # Ensure feature order matches training
    return model, scaler, feature_names

def preprocess_input(user_input, scaler, feature_names):
    """Preprocess user input to match the training format."""
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
    
    # List of numerical features expected by the scaler
    numerical_features = ['lat', 'long', 'upstream_terminal_number', 'distance_upstream_terminal', 'maximum_speed', 'age']
    
    # Ensure all expected features exist in the DataFrame
    for col in numerical_features:
        if col not in df.columns:
            df[col] = 0  # Default value for missing features
    
    # Standardize numerical features
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    # One-hot encode categorical features
    categorical_features = ['lum', 'atm_condition', 'collision_type', 'route_category',
                            'traffic_regime', 'vehicle_category', 'user_category', 'gender']
    df = pd.get_dummies(df, columns=categorical_features)
    
    # Ensure all training features exist and are in the correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Add missing categorical feature
    df = df[feature_names]  # Ensure correct column order
    
    return df

def main():
    st.title("Accident Severity Prediction")
    st.write("Enter accident details to predict severity.")
    
    # Load model, scaler, and feature names
    model, scaler, feature_names = load_model()
    
    # User input fields
    user_input = {
        "day": st.number_input("Day", min_value=1, max_value=31, value=15),
        "month": st.number_input("Month", min_value=1, max_value=12, value=6),
        "time": st.number_input("Time (HHMMSS)", min_value=0, max_value=235959, value=120000),
        "lat": st.number_input("Latitude", value=48.85),
        "long": st.number_input("Longitude", value=2.35),
        "maximum_speed": st.number_input("Max Speed (km/h)", value=50),
        "age": st.number_input("Driver Age", min_value=18, max_value=100, value=30),
        "lum": st.selectbox("Lighting Condition", options=[1, 2, 3, 4, 5]),
        "atm_condition": st.selectbox("Weather Condition", options=[1, 2, 3, 4, 5]),
        "collision_type": st.selectbox("Collision Type", options=[1, 2, 3, 4, 5, 6]),
        "route_category": st.selectbox("Route Category", options=[1, 2, 3, 4]),
        "traffic_regime": st.selectbox("Traffic Regime", options=[1, 2, 3, 4]),
        "vehicle_category": st.selectbox("Vehicle Category", options=[1, 2, 3, 4, 5]),
        "user_category": st.selectbox("User Category", options=[1, 2, 3, 4]),
        "gender": st.selectbox("Gender", options=[1, 2]),
    }
    
    # Prediction button
    if st.button("Predict Severity"):
        processed_input = preprocess_input(user_input, scaler, feature_names)
        prediction = model.predict(processed_input)[0]
        st.success(f"Predicted Severity: {prediction}")

if __name__ == "__main__":
    main()
