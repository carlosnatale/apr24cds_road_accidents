import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import os

def main():
    st.set_page_config(page_title="Data Science Project Presentation", layout="wide")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    sections = [
        "Cover Page",
        "Part 1: Project Context and Initial Data Insights",
        "Part 2: Data Preprocessing and Feature Engineering",
        "Part 3: Modeling, Results, and Future Work"
    ]
    choice = st.sidebar.radio("Go to:", sections)

    if choice == "Cover Page":
        display_cover_page()
    elif choice == "Part 1: Project Context and Initial Data Insights":
        display_project_context()
    elif choice == "Part 2: Data Preprocessing and Feature Engineering":
        display_data_preprocessing()
    elif choice == "Part 3: Modeling, Results, and Future Work":
        display_modeling_results()

def display_cover_page():
    st.title("Historic Road Accidents in France – A Study")

    st.header("Authors")
    st.markdown("- [Carlos Natale](https://github.com/carlosnatale)")
    st.markdown("- [Ehsan Jafari](https://github.com/Ehsanjafari1993)")
    st.markdown("- [Stephen Waller](https://github.com/StephenWaller87)")

    st.header("Mentor")
    st.markdown("- [Manon Georget](https://github.com/manongeorget)")

def display_project_context():
    st.header("Part 1: Project Context and Initial Data Insights")

    st.subheader("1.1 Context")
    st.write("""
    For this project, we conducted a detailed analysis of road accident data in France provided by the Ministry of the Interior. 
    The objective was to analyze, clean, and format the data to understand correlations and build predictive models for accident severity.
    """)

    st.subheader("1.2 Data Insights")
    st.write("The initial datasets consisted of four CSV files per year from 2019 to 2022, focusing on users, vehicles, locations, and accident characteristics.")

    df = load_csv("vehicules-2022.csv")
    if df is not None:
        st.write("Sample Data from the 2022 'Vehicles' CSV:")
        st.dataframe(df.head(10))
    
    st.subheader("1.3 Accident Analysis by Gender")
    df_usagers = load_csv('usagers-2022.csv')
    if df_usagers is not None:
        plot_gender_analysis(df_usagers)


def display_data_preprocessing():
    st.header("Part 2: Data Preprocessing and Feature Engineering")

    st.subheader("2.1 Data Cleaning")
    st.write("Details about data cleaning processes, handling missing values, and combining datasets.")

    st.subheader("2.2 Feature Engineering")
    st.write("Key features created for predictive modeling and their importance.")


def display_modeling_results():
    st.header("Part 3: Modeling, Results, and Future Work")

    st.subheader("3.1 Model Performance")
    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost", "LightGBM"],
        "Slightly Injured F1": [0.97, 0.87, 0.90],
        "Severely Injured F1": [0.49, 0.59, 0.54],
        "Overall Accuracy": [0.85, 0.80, 0.83]
    })
    plot_model_performance(metrics_df)

    st.subheader("3.2 Feature Importance")
    features_df = pd.DataFrame({
        "Feature": ["Longitude", "Latitude", "Maximum Speed", "Safety Equipment", "Age"],
        "Importance": [0.25, 0.20, 0.15, 0.30, 0.10]
    })
    plot_feature_importance(features_df)

    st.subheader("3.3 SHAP Analysis")
    st.write("SHAP values provide insights into model predictions.")
    plot_shap_analysis()

    st.subheader("3.4 Recommendations")
    st.write("""
    - **Safety Equipment:** Promote proper use to reduce injuries.
    - **Speed Management:** Enforce speed limits in high-risk areas.
    - **Targeted Interventions:** Focus on high-risk groups and locations.
    """)

# Utility Functions
def load_csv(file_path):
    """Loads a CSV file and returns a DataFrame."""
    try:
        return pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found.")
        return None

def plot_gender_analysis(df):
    """Plots gender distribution in accidents."""
    value_counts = df['sexe'].value_counts()
    fig = px.bar(value_counts, x=value_counts.index, y=value_counts.values,
                 labels={'x': 'Gender', 'y': 'Count'}, title="Accidents by Gender")
    st.plotly_chart(fig)

def plot_model_performance(metrics_df):
    """Plots model performance metrics."""
    fig = px.bar(metrics_df, x="Model", y=["Slightly Injured F1", "Severely Injured F1", "Overall Accuracy"],
                 barmode="group", title="Model Performance Metrics",
                 labels={"value": "Score", "Model": "Models"})
    st.plotly_chart(fig)

def plot_feature_importance(features_df):
    """Plots feature importance."""
    fig = px.bar(features_df, x="Importance", y="Feature", orientation='h',
                 title="Feature Importance", labels={"Importance": "Relative Importance"})
    st.plotly_chart(fig)

def plot_shap_analysis():
    """Plots a SHAP analysis example."""
    shap_values = np.random.rand(1, 5)  # Example SHAP values
    feature_names = ["Longitude", "Latitude", "Maximum Speed", "Safety Equipment", "Age"]
    example_instance = np.random.rand(1, 5)  # Example feature values
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=np.mean(shap_values),
        data=example_instance[0],
        feature_names=feature_names
    )
    fig, ax = plt.subplots()
    shap.waterfall_plot(explanation)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
