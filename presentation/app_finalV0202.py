import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import shap
from lime import lime_tabular
import os
from PIL import Image
from scipy import stats  # Added missing import

def main():
    st.set_page_config(page_title="Data Science Project Presentation", layout="wide")

    # Custom Sidebar Style
    sidebar_style = """
    <style>
        [data-testid="stSidebar"] {
            background-color: #f4f4f4;
            border-right: 1px solid #ddd;
        }
        [data-testid="stSidebar"] h2 {
            color: #333;
        }
        .sidebar-content {
            display: flex;
            align-items: center;
            flex-direction: column;
        }
    </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)

    # Display Image in Sidebar
    image_path = "road_accidents_France.png"
    with st.sidebar:
        if os.path.exists(image_path):
            st.image(image_path)
        else:
            st.error("Image file 'road_accidents_France.png' not found. Please ensure it is in the correct location.")

        st.title("Navigation")
        sections = [
            "Cover Page",
            "Part 1: Project Context and Initial Data Insights",
            "Part 2: Data Preprocessing and Feature Engineering",
            "Part 3: Modeling, Results, and Future Work"
        ]
        choice = st.radio("Go to:", sections)

    # Section Content
    if choice == "Cover Page":
        part_0()
    elif choice == "Part 1: Project Context and Initial Data Insights":
        part_1()
    elif choice == "Part 2: Data Preprocessing and Feature Engineering":
        part_2()
    elif choice == "Part 3: Modeling, Results, and Future Work":
        part_3()

def part_0():
    st.title("Historic Road Accidents in France – A Study")
    st.header("Authors")
    st.markdown("- [Carlos Natale](https://github.com/carlosnatale)")
    st.markdown("- [Ehsan Jafari](https://github.com/Ehsanjafari1993)")
    st.markdown("- [Stephen Waller](https://github.com/StephenWaller87)")
    st.header("Mentor")
    st.markdown("- [Manon Georget](https://github.com/manongeorget)")

def part_1():
    st.header("Part 1: Project Context and Initial Data Insights")
    
    st.subheader("1.1 Context")
    st.write("For this project, we are tasked with conducting a detailed analysis of the history of road accidents in France using data provided by the French government through the Ministry of the Interior and Overseas Territories.")

    st.write("The objective is to analyse the available data, clean and format it technically, and to understand any correlation between data items that could provide insight and statistical significance of the data objects. This can lead to improved predictions of the likelihood of road accidents and their severity. We will also utilize advanced data analytics techniques, such as machine learning, to build predictive models and derive actionable insights.")

    st.subheader("1.2 Data Load and Analysis")
    st.write("The initial objective of this project is to have the data analysed, processed, cleaned and combined as required, in order to have a singular dataset that can be used for complex statistical analysis and modelling in order to create predictive models to forecast accident severity.")

    st.subheader("1.2.1 Understanding and Manipulation of the Data")
    st.write("As mentioned above, the data that is being used for analysis is sourced from the Ministry of the Interior and Overseas Territories, which is freely published")

    try:
        df = pd.read_csv("vehicules-2022.csv", sep=';')
    except FileNotFoundError:
        st.error("File 'vehicules-2022.csv' not found.")
        return

    st.subheader("1.2.2 Data Example")
    st.dataframe(df.head(10))
    st.write("As you can see from this example, there are many different variable that can be used for analysis.")

    st.subheader("1.2.3 Combining The Data")
    try:
        df2 = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("Combined data file 'data.csv' not found.")
        return

    st.dataframe(df2.head(10))
    st.write(df2.shape)

    # Chi-square Analysis with error handling
    try:
        data = pd.read_csv('data.csv')
        if 'gravity' not in data.columns:
            st.error("Column 'gravity' missing in dataset.")
            return
            
        data = data.drop_duplicates()
        data = data.loc[:, data.isna().sum() / len(data) < 0.15]

        def fill_na_with_distribution(df, column):
            if df[column].isna().all():
                df[column].fillna("Unknown", inplace=True)
                return
            value_counts = df[column].value_counts(normalize=True)
            values = value_counts.index.tolist()
            probabilities = value_counts.values.tolist()
            nans_to_fill = df[column].isna().sum()
            fill_values = np.random.choice(values, size=nans_to_fill, p=probabilities)
            df.loc[df[column].isna(), column] = fill_values

        for column in data.columns:
            if data[column].isna().sum() > 0:
                fill_na_with_distribution(data, column)

        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'gravity' not in categorical_columns:
            st.error("'gravity' column not found in categorical columns.")
            return
        categorical_columns.remove('gravity')

        chi_square_results_all = {}
        for column in categorical_columns:
            contingency_table = pd.crosstab(data['gravity'], data[column])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            chi_square_results_all[column] = {
                'Chi-square statistic': chi2,
                'p-value': p,
                'Degrees of freedom': dof,
                'Expected frequencies': expected
            }

        filtered_variables = [var for var in chi_square_results_all.keys() if var not in {'AccID', 'vehicleID', 'num_veh'}]
        filtered_chi_square_stats = [chi_square_results_all[var]['Chi-square statistic'] for var in filtered_variables]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(filtered_variables, filtered_chi_square_stats, color='skyblue')
        ax.set_xlabel('Chi-square Statistic')
        ax.set_title('Chi-square Statistics for Variables with Gravity')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.gca().invert_yaxis()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Error during Chi-square analysis: {str(e)}")

def part_2():
    st.header("Part 2: Data Preprocessing and Feature Engineering")
    st.subheader("2.1 Data Cleaning")
    st.write("In this section, we cleaned the data by handling missing values, removing duplicates, and correcting inconsistencies.")

def part_3():
    st.header("Part 3: Modeling, Results, and Future Work")
    
    # SHAP Analysis with image checks
    st.title("SHAP Analysis Dashboard")
    tab_xgb, tab_lgb, tab_rf = st.tabs(["XGBoost", "LightGBM", "Random Forest"])

    with tab_xgb:
        st.header("XGBoost SHAP Analysis")
        xgb_image_path = "shap1 - xgboost.png"
        if os.path.exists(xgb_image_path):
            xgb_image = Image.open(xgb_image_path)
            st.image(xgb_image, use_column_width=True)
        else:
            st.warning("SHAP image for XGBoost not found.")

    with tab_lgb:
        st.header("LightGBM SHAP Analysis")
        lgb_image_path = "shap1 - lightgbm.png"
        if os.path.exists(lgb_image_path):
            lgb_image = Image.open(lgb_image_path)
            st.image(lgb_image, use_column_width=True)
        else:
            st.warning("SHAP image for LightGBM not found.")

    with tab_rf:
        st.header("Random Forest SHAP Analysis")
        rf_image_path = "shap1 - random forest.png"
        if os.path.exists(rf_image_path):
            rf_image = Image.open(rf_image_path)
            st.image(rf_image, use_column_width=True)
        else:
            st.warning("SHAP image for Random Forest not found.")

if __name__ == "__main__":
    main()