import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

# Constants
COLOR_PALETTE = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71']

# Configuration
st.set_page_config(page_title="Road Safety Analysis - France", layout="wide")

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, sep=';')

def styled_header(title):
    st.markdown(f"""
    <h2 style='color: #2c3e50; 
                border-bottom: 2px solid #e74c3c; 
                padding-bottom: 0.5rem;
                margin-bottom: 2rem;'>
        {title}
    </h2>
    """, unsafe_allow_html=True)

def sidebar_navigation():
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1rem; 
                    border-bottom: 1px solid #ddd; 
                    margin-bottom: 2rem;'>
            <h2 style='color: #2c3e50;'>Accident Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            st.image("road_accidents_France.png", 
                     use_column_width=True,
                     caption="Road Safety Visualization")
        except FileNotFoundError:
            st.warning("Presentation banner image not found")
        
        st.radio("Navigate Sections", [
            "Cover Page",
            "Part 1: Project Context and Initial Data Insights",
            "Part 2: Data Preprocessing and Feature Engineering",
            "Part 3: Modeling, Results, and Future Work"
        ], key='nav', label_visibility="collapsed")

def part_0():
    st.title("Historic Road Accidents in France – A Study")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Authors")
        st.markdown("""
        - [Carlos Natale](https://github.com/carlosnatale)
        - [Ehsan Jafari](https://github.com/Ehsanjafari1993)
        - [Stephen Waller](https://github.com/StephenWaller87)
        """)
    
    with col2:
        st.markdown("### Mentor")
        st.markdown("- [Manon Georget](https://github.com/manongeorget)")

def plot_gender_distribution():
    with st.spinner('Loading gender distribution data...'):
        df_usagers = load_data('usagers-2022.csv')
        gender_map = {1: "Male", 2: "Female"}
        
        fig = px.bar(df_usagers['sexe'].map(gender_map).value_counts(),
                     color_discrete_sequence=[COLOR_PALETTE[2], COLOR_PALETTE[1]],
                     labels={'value': 'Count', 'index': 'Gender'},
                     title="Accident Distribution by Gender")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                         xaxis_title="Gender",
                         yaxis_title="Number of Accidents")
        st.plotly_chart(fig, use_container_width=True)

def part_1():
    styled_header("Part 1: Project Context and Initial Data Insights")
    
    st.markdown("""
    ### 1.1 Context
    For this project, we are tasked with conducting a detailed analysis of the history of road accidents in France...
    """)
    
    with st.expander("Data Sources and Methodology"):
        st.markdown("""
        The data on this site contains details of road accidents from 2005 to 2022...
        The datasets themselves consisted of 4 .csv files per year 'usagers', 'vehicules', 'lieux' and 'carcteristiques'...
        """)
    
    st.markdown("### 1.2 Data Load and Analysis")
    
    with st.spinner('Loading vehicle data...'):
        df_vehicles = load_data("vehicules-2022.csv")
        st.dataframe(df_vehicles.head(10), use_container_width=True)
    
    st.markdown("### 1.3 Initial Findings")
    plot_gender_distribution()

def part_2():
    styled_header("Part 2: Data Preprocessing and Feature Engineering")
    
    st.markdown("""
    ### 2.1 Data Cleaning Pipeline
    Our preprocessing workflow included:
    - Missing value imputation
    - Outlier detection and handling
    - Categorical encoding
    - Temporal feature extraction
    """)
    
    with st.expander("Feature Engineering Details"):
        st.markdown("""
        Created new features including:
        - Time-based features (hour bins, weekend flags)
        - Geographic clusters
        - Vehicle age categories
        """)

def plot_model_metrics(metrics_df):
    fig = go.Figure()
    metrics = ['Slightly Injured F1', 'Severely Injured F1', 'Overall Accuracy']
    
    for metric, color in zip(metrics, COLOR_PALETTE):
        fig.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df[metric],
            name=metric,
            marker_color=color
        ))
    
    fig.update_layout(
        barmode='group',
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_shap_analysis():
    # Simulated SHAP values
    np.random.seed(42)
    shap_values = np.random.randn(1, 5)
    feature_names = ["Longitude", "Latitude", "Max Speed", "Safety Equipment", "Age"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=np.mean(shap_values),
        feature_names=feature_names
    ), max_display=7, show=False)
    
    plt.title("Feature Impact on Prediction", fontsize=14)
    plt.gcf().set_facecolor('white')
    st.pyplot(fig)

def part_3():
    styled_header("Part 3: Modeling, Results, and Future Work")
    
    # Model metrics
    metrics_data = {
        "Model": ["Random Forest", "XGBoost", "LightGBM"],
        "Slightly Injured F1": [0.97, 0.87, 0.90],
        "Severely Injured F1": [0.49, 0.59, 0.54],
        "Overall Accuracy": [0.85, 0.80, 0.83]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Feature importance
    features_data = {
        "Feature": ["Longitude", "Latitude", "Maximum Speed", "Safety Equipment", "Age"],
        "Importance": [0.25, 0.20, 0.15, 0.30, 0.10]
    }
    features_df = pd.DataFrame(features_data)
    
    # Layout
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_model_metrics(metrics_df)
    with col2:
        st.markdown("### Key Metrics")
        st.metric("Total Samples", "450,000+", help="Combined dataset records")
        st.metric("Top Feature", "Safety Equipment", "30% impact")
        st.metric("Best Model", "Random Forest", "85% Accuracy")
    
    st.markdown("---")
    plot_shap_analysis()
    
    st.markdown("""
    ### 3.4 Recommendations
    - **Safety Equipment:** Ensuring proper use can significantly reduce severe injuries
    - **Speed Management:** Implement targeted enforcement in high-risk areas
    - **Data Collection:** Expand weather and road condition monitoring
    """)

def main():
    sidebar_navigation()
    
    if st.session_state.nav == "Cover Page":
        part_0()
    elif st.session_state.nav == "Part 1: Project Context and Initial Data Insights":
        part_1()
    elif st.session_state.nav == "Part 2: Data Preprocessing and Feature Engineering":
        part_2()
    elif st.session_state.nav == "Part 3: Modeling, Results, and Future Work":
        part_3()

if __name__ == "__main__":
    main()