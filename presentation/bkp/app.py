import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Configuration
st.set_page_config(page_title="Road Accident Analysis", layout="wide", page_icon="🚗")

@st.cache_data
def load_sample_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_clusters_per_class=1,
        random_state=42
    )
    feature_names = ["Longitude", "Latitude", "Max Speed", "Safety Equipment", "Driver Age"]
    return pd.DataFrame(X, columns=feature_names), y

@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def main():
    # Load data and train model
    X, y = load_sample_data()
    model = train_model(X, y)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("road_accidents_France.png", use_column_width=True)
        nav = st.radio("Navigation", [
            "Data Overview", 
            "Feature Analysis",
            "Model Insights"
        ])

    # Page routing
    if nav == "Data Overview":
        st.header("Dataset Summary")
        st.dataframe(X.describe())
        
    elif nav == "Feature Analysis":
        st.header("Feature Distributions")
        selected_feature = st.selectbox("Choose feature", X.columns)
        fig = px.histogram(X, x=selected_feature, nbins=30)
        st.plotly_chart(fig)
        
    elif nav == "Model Insights":
        st.header("Model Interpretation")
        
        # Feature importance
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        fig = px.bar(importance, x="Feature", y="Importance")
        st.plotly_chart(fig)

        # SHAP explanations
        st.subheader("SHAP Values")
        explainer = shap.TreeExplainer(model)
        sample = X.iloc[:10]  # Explain first 10 samples
        shap_values = explainer.shap_values(sample)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, sample, plot_type="bar")
        st.pyplot(fig)

if __name__ == "__main__":
    main()