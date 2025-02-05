import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import shap
from lime import lime_tabular
import os
from PIL import Image
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gzip
import pickle

# Load the compressed dataset
file_path = "data.pkl.gz"

with gzip.open(file_path, "rb") as f:
    data_final = pickle.load(f)
data_final = data_final.drop_duplicates()

def main():
    st.set_page_config(page_title="Data Science Project Presentation", layout="wide")

    # Sidebar Navigation
    sections = [
        "Cover Page",
        "Part 1: Project Context and Initial Data Insights",
        "Part 2: Data Preprocessing and Feature Engineering",
        "Part 3: Modeling, Results, and Future Work",
        "Live Demonstration"
    ]
    choice = st.sidebar.radio("Go to:", sections)

    if choice == "Cover Page":
        part_0()
    elif choice == "Part 1: Project Context and Initial Data Insights":
        part_1()
    elif choice == "Part 2: Data Preprocessing and Feature Engineering":
        part_2()
    elif choice == "Part 3: Modeling, Results, and Future Work":
        part_3()
    elif choice == "Live Demonstration":
        live_demo()

def live_demo():
    st.title("🚦 Live Accident Severity Prediction")
    st.write("Enter accident details below to predict the severity of an accident.")
    
    # Load the trained model
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
    
    model, scaler, feature_names = load_model()
    
    # User input form
    st.sidebar.header("📌 Enter Accident Details")
    day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)
    month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
    time = st.sidebar.number_input("Time (HHMMSS)", min_value=0, max_value=235959, value=120000)
    lat = st.sidebar.number_input("Latitude", value=48.85)
    long = st.sidebar.number_input("Longitude", value=2.35)
    max_speed = st.sidebar.number_input("Max Speed (km/h)", value=50)
    age = st.sidebar.number_input("Driver Age", min_value=18, max_value=100, value=30)
    upstream_terminal_number = st.sidebar.number_input("Upstream Terminal Number", min_value=0, value=0)
    distance_upstream_terminal = st.sidebar.number_input("Distance to Upstream Terminal (m)", min_value=0, value=0)
    
    def preprocess_input(user_input, scaler, feature_names):
        df = pd.DataFrame([user_input])
        
        # Cyclical encoding for time variables
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['time_sin'] = np.sin(2 * np.pi * df['time'] / 86340000)
        df['time_cos'] = np.cos(2 * np.pi * df['time'] / 86340000)
        
        df.drop(columns=['day', 'month', 'time'], inplace=True)
        
        # Standardize numerical features
        numerical_features = ['lat', 'long', 'upstream_terminal_number', 'distance_upstream_terminal', 'maximum_speed', 'age']
        if scaler is not None:
            df[numerical_features] = scaler.transform(df[numerical_features])
        
        # Ensure all features exist and are in the correct order
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0  # Add missing categorical features
        df = df[feature_names]
        
        return df
    
    if st.button("🚀 Predict Severity"):
        with st.spinner("Analyzing accident details..."):
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

def part_0():
    st.title("Historic Road Accidents in France – A Study")
    # Cover Page Details

def part_1():
    st.header("Part 1: Project Context and Initial Data Insights")
    # Context and Initial Data Insights

def part_2():
    st.header("Part 2: Data Preprocessing and Feature Engineering")
    # Preprocessing Details

def part_3():
    st.header("Part 3: Modeling, Results, and Future Work")
    # Modeling and Results

if __name__ == "__main__":
    main()
