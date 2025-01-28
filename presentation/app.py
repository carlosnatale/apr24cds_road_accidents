import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import os
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Configuration and styling
st.set_page_config(page_title="Road Accident Analysis", layout="wide", page_icon="🚗")

# Custom CSS styling
custom_style = """
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(145deg, #2c3e50, #3498db);
        color: white;
    }
    .sidebar-title {
        font-size: 1.5em !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .footer {
        margin-top: 40px;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
        text-align: center;
    }
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# Cached data loading and model training
@st.cache_data
def load_sample_data():
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, random_state=42)
    feature_names = ["Longitude", "Latitude", "Max Speed", "Safety Equipment", "Driver Age"]
    return pd.DataFrame(X, columns=feature_names), y

@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def main():
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Accident Analysis Dashboard</p>', unsafe_allow_html=True)
        
        if os.path.exists("road_accidents_France.png"):
            st.image("road_accidents_France.png", use_column_width=True)
        else:
            st.warning("Preview image not found")
        
        navigation = st.radio("Navigate to:", [
            "Project Overview",
            "Data Analysis",
            "Feature Engineering",
            "Model Insights"
        ])

    # Load sample data and train model
    X, y = load_sample_data()
    model = train_model(X, y)

    # Page routing
    if navigation == "Project Overview":
        show_overview()
    elif navigation == "Data Analysis":
        show_data_analysis(X)
    elif navigation == "Feature Engineering":
        show_feature_engineering()
    elif navigation == "Model Insights":
        show_model_insights(X, model)

def show_overview():
    st.title("Road Accident Severity Analysis in France")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://via.placeholder.com/400x250.png?text=Accident+Analysis", use_column_width=True)
    
    with col2:
        st.markdown("""
        ### Project Overview
        Comprehensive analysis of road accidents in France to identify key factors influencing accident severity.
        - **Time Period**: 2015-2022
        - **Data Sources**: National Road Safety Database, Weather API
        - **Target**: Accident Severity Classification
        """)
    
    st.markdown("---")
    st.subheader("Project Team")
    cols = st.columns(3)
    team_members = [
        ("Carlos Natale", "Data Scientist", "https://github.com/carlosnatale"),
        ("Ehsan Jafari", "ML Engineer", "https://github.com/Ehsanjafari1993"),
        ("Stephen Waller", "Data Analyst", "https://github.com/StephenWaller87")
    ]
    
    for col, (name, role, link) in zip(cols, team_members):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <h4>{name}</h4>
                <p>{role}</p>
                <small>[GitHub]({link})</small>
            </div>
            """, unsafe_allow_html=True)

def show_data_analysis(X):
    st.title("Data Analysis")
    
    st.subheader("Data Distribution")
    selected_feature = st.selectbox("Select feature to visualize", X.columns)
    
    fig = px.histogram(X, x=selected_feature, nbins=30, 
                      title=f"Distribution of {selected_feature}",
                      color_discrete_sequence=["#3498db"])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Correlation Matrix")
    corr_matrix = X.corr()
    fig = px.imshow(corr_matrix, labels=dict(x="Features", y="Features"),
                   x=X.columns, y=X.columns,
                   color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

def show_feature_engineering():
    st.title("Feature Engineering Process")
    
    st.markdown("""
    ### Key Feature Transformations
    - **Temporal Features**: 
        - Hour of day extracted from timestamp
        - Weekend/holiday flags
    - **Geospatial Features**:
        - Distance to nearest hospital
        - Road type classification
    - **Driver Features**:
        - Age categories
        - Experience level
    """)
    
    st.image("https://via.placeholder.com/800x300.png?text=Feature+Engineering+Pipeline", use_column_width=True)

def show_model_insights(X, model):
    st.title("Model Insights and Interpretability")
    
    # Model performance metrics
    st.subheader("Model Performance Metrics")
    metrics = {
        "Accuracy": 0.87,
        "Precision (Severe)": 0.63,
        "Recall (Severe)": 0.58,
        "F1 Score (Severe)": 0.60
    }
    
    cols = st.columns(4)
    for col, (name, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <h4>{name}</h4>
                <h2>{value:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    fig = px.bar(x=X.columns, y=importance, 
                labels={'x': 'Features', 'y': 'Importance'},
                color=importance, color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    
    # SHAP explanations
    try:
        st.subheader("SHAP Interpretation")
        sample = X.iloc[[0]]  # Explain first sample
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        
        st.markdown("#### Global Feature Impact")
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        st.pyplot(fig1)
        
        st.markdown("#### Local Explanation (Waterfall Plot)")
        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0][0],
            base_values=explainer.expected_value[0],
            data=sample.values[0],
            feature_names=X.columns.tolist()
        ), show=False)
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"SHAP visualization error: {str(e)}")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Project Repository: <a href="https://github.com/your-repo">GitHub</a></p>
        <p>Contact: team@accident-analysis.fr | Follow us on Twitter: @AccidentAnalysts</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()