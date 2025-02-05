import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# ... [Keep the MAPPINGS dictionary and load_model/preprocess_input functions unchanged] ...

def main():
    st.set_page_config(
        page_title="Accident Severity Prediction",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS styling
    st.markdown("""
        <style>
            :root {
                --primary: #1e3a8a;
                --secondary: #bfdbfe;
            }
            
            .main {
                background-color: #f8fafc;
            }
            
            .sidebar .sidebar-content {
                background: var(--primary);
                color: white;
            }
            
            .st-bb {
                background-color: white;
            }
            
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
                color: white;
                transform: scale(1.05);
            }
            
            .stSuccess {
                background-color: #d1fae5 !important;
                color: #065f46 !important;
                border-radius: 8px;
                padding: 1rem;
            }
            
            .section-header {
                font-size: 1.2rem !important;
                color: var(--primary) !important;
                border-bottom: 2px solid var(--secondary);
                padding-bottom: 0.5rem;
                margin-top: 1.5rem !important;
            }
            
            .download-section {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    # Main content header
    st.title("🚨 Accident Severity Predictor")
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            This predictive analytics tool assesses road accident parameters to determine potential severity outcomes. 
            Input details below to receive instant risk evaluation.
        </div>
    """, unsafe_allow_html=True)

    # Load model components
    model, scaler, feature_names = load_model()

    # Sidebar organization
    with st.sidebar:
        st.header("Accident Parameters")
        st.markdown("---")
        
        with st.expander("📅 Date & Time", expanded=True):
            user_input = {
                "day": st.number_input("Day of Month", min_value=1, max_value=31, value=15),
                "month": st.number_input("Month", min_value=1, max_value=12, value=6),
                "time": st.number_input("Time (HHMMSS)", min_value=0, max_value=235959, value=120000,
                                      help="Enter time in 24h format, e.g.: 143000 = 2:30 PM"),
            }

        with st.expander("📍 Location Details", expanded=True):
            user_input.update({
                "lat": st.number_input("Latitude", value=48.85),
                "long": st.number_input("Longitude", value=2.35),
                "maximum_speed": st.number_input("Speed Limit (km/h)", value=50),
                "route_category": st.selectbox("Road Type", options=list(MAPPINGS["route_category"].keys()),
                                             format_func=lambda x: MAPPINGS["route_category"][x]),
            })

        with st.expander("👤 Driver Information", expanded=True):
            user_input.update({
                "age": st.number_input("Driver Age", min_value=18, max_value=100, value=30),
                "gender": st.selectbox("Driver Gender", options=list(MAPPINGS["gender"].keys()),
                                     format_func=lambda x: MAPPINGS["gender"][x]),
            })

        with st.expander("🚦 Accident Circumstances", expanded=True):
            user_input.update({
                "lum": st.selectbox("Lighting Conditions", options=list(MAPPINGS["lum"].keys()),
                                  format_func=lambda x: MAPPINGS["lum"][x]),
                "atm_condition": st.selectbox("Weather Conditions", options=list(MAPPINGS["atm_condition"].keys()),
                                            format_func=lambda x: MAPPINGS["atm_condition"][x]),
                "collision_type": st.selectbox("Collision Type", options=list(MAPPINGS["collision_type"].keys()),
                                             format_func=lambda x: MAPPINGS["collision_type"][x]),
                "traffic_regime": st.selectbox("Traffic Flow", options=list(MAPPINGS["traffic_regime"].keys()),
                                             format_func=lambda x: MAPPINGS["traffic_regime"][x]),
                "vehicle_category": st.selectbox("Vehicle Type", options=list(MAPPINGS["vehicle_category"].keys()),
                                               format_func=lambda x: MAPPINGS["vehicle_category"][x]),
            })

    # Prediction section
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔍 Analyze Severity Risk"):
            with st.spinner("Evaluating risk factors..."):
                processed_input = preprocess_input(user_input, scaler, feature_names)
                prediction = model.predict(processed_input)[0]
                
            severity_mapping = {
                0: "Low Risk: Minor or No Injuries",
                1: "High Risk: Severe Injuries or Fatalities"
            }
            st.success(f"""
                **Prediction Result:**  
                {severity_mapping[prediction]}
            """)
            st.markdown("---")
            st.markdown("""
                **Interpretation Guide:**  
                - *Low Risk:* Minor property damage or light injuries  
                - *High Risk:* Significant injuries or life-threatening situations
            """)

    # Download section
    st.markdown("---")
    st.markdown("### 🗄️ Model Resources")
    with st.container():
        cols = st.columns(3)
        download_data = [
            ("model.pkl", "Download Model", "📦"),
            ("scaler.pkl", "Download Scaler", "⚖️"),
            ("feature_names.pkl", "Download Features", "📋")
        ]
        
        for (file, label, icon), col in zip(download_data, cols):
            with col:
                with open(file, "rb") as f:
                    st.download_button(
                        label=f"{icon} {label}",
                        data=f,
                        file_name=file,
                        help=f"Download {file.split('.')[0].replace('_', ' ').title()}",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()