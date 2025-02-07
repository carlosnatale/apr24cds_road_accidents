import streamlit as st
import pickle
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
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

# UI Layout
st.title("🚦 Accident Severity Prediction")
st.write("This tool helps assess accident severity based on various conditions.")

# Sidebar Inputs
st.sidebar.header("📌 Enter Accident Details")
with st.sidebar.expander("🕒 Time & Location", expanded=True):
    day = st.number_input("Day", min_value=1, max_value=31, value=15)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    time = st.number_input("Time (HHMMSS)", min_value=0, max_value=235959, value=120000)

    # Default location in France (Paris)
    default_lat, default_long = 48.8566, 2.3522

    st.sidebar.write("### 🌍 Select Accident Location on Map")
    m = folium.Map(location=[default_lat, default_long], zoom_start=6)
    marker = folium.Marker([default_lat, default_long], draggable=True)
    marker.add_to(m)

    # Display the map and capture user selection
    map_data = st_folium(m, height=400, width=700)

    # Extract selected location
    if map_data["last_clicked"]:
        lat = map_data["last_clicked"]["lat"]
        long = map_data["last_clicked"]["lng"]
    else:
        lat, long = default_lat, default_long  # Default values

    st.sidebar.write(f"**Selected Latitude:** {lat}, **Longitude:** {long}")

with st.sidebar.expander("🚗 Vehicle & Driver", expanded=False):
    max_speed = st.number_input("Max Speed (km/h)", value=50)
    age = st.number_input("Driver Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    upstream_terminal_number = st.number_input("Upstream Terminal Number", min_value=0, value=0)
    distance_upstream_terminal = st.number_input("Distance to Upstream Terminal (m)", min_value=0, value=0)

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
            
            # Convert user input into DataFrame
            df = pd.DataFrame([user_input])
            
            # Standardize numerical features
            numerical_features = ["lat", "long", "maximum_speed", "age", "upstream_terminal_number", "distance_upstream_terminal"]
            if scaler is not None:
                df[numerical_features] = scaler.transform(df[numerical_features])

            # Make prediction
            prediction = model.predict(df)[0]
            severity_mapping = {0: "🟢 Minor Injury or No Injury", 1: "🔴 Severe Injury or Fatal"}
            st.success(f"**Prediction: {severity_mapping[prediction]}**")
