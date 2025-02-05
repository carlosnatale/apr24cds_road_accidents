import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Define mappings for usability improvements
MAPPINGS = {
    "gender": {1: "Male", 2: "Female"},
    "lum": {
        1: "Daylight",
        2: "Dawn/Dusk",
        3: "Night without public lighting",
        4: "Night with public lighting off",
        5: "Night with public lighting on"
    },
    "atm_condition": {
        1: "Normal",
        2: "Light rain",
        3: "Heavy rain",
        4: "Snow/Hail",
        5: "Fog/Smoke",
        6: "Strong wind",
        7: "Dazzling weather",
        8: "Overcast",
        9: "Other"
    },
    "collision_type": {
        1: "Frontal collision",
        2: "Rear-end collision",
        3: "Side collision",
        4: "Chain collision",
        5: "Multiple collisions",
        6: "Other collision",
        7: "No collision"
    },
    "route_category": {
        1: "Highway",
        2: "National road",
        3: "Departmental road",
        4: "Municipal road"
    },
    "traffic_regime": {
        1: "One-way",
        2: "Two-way",
        3: "Separate lanes",
        4: "Variable lanes"
    },
    "vehicle_category": {
        1: "Bicycle",
        2: "Moped",
        3: "Motorcycle",
        4: "Car",
        5: "Truck",
        6: "Bus"
    }
}

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
    for col in categorical_features:
        if col not in df.columns:
            df[col] = 0  # Add missing categorical features with a default value
    df = pd.get_dummies(df, columns=categorical_features)
    
    # Ensure all training features exist and are in the correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Add missing categorical feature
    df = df[feature_names]  # Ensure correct column order
    
    return df

def main():
    st.set_page_config(page_title="Accident Severity Prediction", page_icon="🚦", layout="wide")

    st.markdown(
        """
        <style>
        .main {
            background-color: #f4f4f4;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stSelectbox label, .stNumberInput label {
            font-weight: bold;
        }
        .stSuccess {
            color: green;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🚦 Accident Severity Prediction")
    st.write("Enter accident details to predict severity. This tool helps identify risk levels for road safety.")

    # Load model, scaler, and feature names
    model, scaler, feature_names = load_model()

    # User input fields with mappings applied
    st.sidebar.header("Input Details")
    user_input = {
        "day": st.sidebar.number_input("Day", min_value=1, max_value=31, value=15),
        "month": st.sidebar.number_input("Month", min_value=1, max_value=12, value=6),
        "time": st.sidebar.number_input("Time (HHMMSS)", min_value=0, max_value=235959, value=120000),
        "lat": st.sidebar.number_input("Latitude", value=48.85),
        "long": st.sidebar.number_input("Longitude", value=2.35),
        "maximum_speed": st.sidebar.number_input("Max Speed (km/h)", value=50),
        "age": st.sidebar.number_input("Driver Age", min_value=18, max_value=100, value=30),
        "lum": st.sidebar.selectbox("Lighting Condition", options=list(MAPPINGS["lum"].keys()), format_func=lambda x: MAPPINGS["lum"][x]),
        "atm_condition": st.sidebar.selectbox("Weather Condition", options=list(MAPPINGS["atm_condition"].keys()), format_func=lambda x: MAPPINGS["atm_condition"][x]),
        "collision_type": st.sidebar.selectbox("Collision Type", options=list(MAPPINGS["collision_type"].keys()), format_func=lambda x: MAPPINGS["collision_type"][x]),
        "route_category": st.sidebar.selectbox("Route Category", options=list(MAPPINGS["route_category"].keys()), format_func=lambda x: MAPPINGS["route_category"][x]),
        "traffic_regime": st.sidebar.selectbox("Traffic Regime", options=list(MAPPINGS["traffic_regime"].keys()), format_func=lambda x: MAPPINGS["traffic_regime"][x]),
        "vehicle_category": st.sidebar.selectbox("Vehicle Category", options=list(MAPPINGS["vehicle_category"].keys()), format_func=lambda x: MAPPINGS["vehicle_category"][x]),
        "gender": st.sidebar.selectbox("Gender", options=list(MAPPINGS["gender"].keys()), format_func=lambda x: MAPPINGS["gender"][x]),
    }

    # Prediction button
    if st.button("Predict Severity"):
        processed_input = preprocess_input(user_input, scaler, feature_names)
        prediction = model.predict(processed_input)[0]
        severity_mapping = {
            0: "Not injured or Slightly injured",
            1: "Severely injured or Fatal Accident"
        }
        st.success(f"Predicted Severity: {severity_mapping[prediction]}")

    # Download section
    st.subheader("📥 Download Model Files")
    st.write("You can download the model files below for further use:")
    with open("model.pkl", "rb") as f:
        st.download_button("Download Model", f, file_name="model.pkl")
    with open("scaler.pkl", "rb") as f:
        st.download_button("Download Scaler", f, file_name="scaler.pkl")
    with open("feature_names.pkl", "rb") as f:
        st.download_button("Download Feature Names", f, file_name="feature_names.pkl")

if __name__ == "__main__":
    main()
