import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def load_model():
    """Load the trained XGBoost model and the standard scaler."""
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, scaler

def preprocess_input(user_input, scaler):
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
    
    # Standardize numerical features
    numerical_features = ['lat', 'long', 'upstream_terminal_number', 'distance_upstream_terminal', 'maximum_speed', 'age']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    return df

def main():
    st.title("Accident Severity Prediction")
    st.write("Enter accident details to predict severity.")
    
    # Load model and scaler
    model, scaler = load_model()
    
    # User input fields
    user_input = {
        "day": st.number_input("Day", min_value=1, max_value=31, value=15),
        "month": st.number_input("Month", min_value=1, max_value=12, value=6),
        "time": st.number_input("Time (HHMMSS)", min_value=0, max_value=235959, value=120000),
        "lat": st.number_input("Latitude", value=48.85),
        "long": st.number_input("Longitude", value=2.35),
        "maximum_speed": st.number_input("Max Speed (km/h)", value=50),
        "age": st.number_input("Driver Age", min_value=18, max_value=100, value=30),
    }
    
    # Prediction button
    if st.button("Predict Severity"):
        processed_input = preprocess_input(user_input, scaler)
        prediction = model.predict(processed_input)[0]
        st.success(f"Predicted Severity: {prediction}")

if __name__ == "__main__":
    main()
