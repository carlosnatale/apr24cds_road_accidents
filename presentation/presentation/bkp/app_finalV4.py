import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import shap
from lime import lime_tabular
import os
from scipy import stats
import pickle

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

    st.write("In the long term, this analysis could bring economic benefits to both the population of France and the French government. The reduction of accidents or their severity would have far-reaching benefits due to the socio-economic impact of road accidents, such as road closures, injuries, or even death. Accident prevention will help reduce healthcare costs, alleviate the financial burden on victims and their families, and minimize the economic impact of traffic disruptions. Furthermore, enhanced road safety will lead to increased productivity and economic efficiency.")

    st.write("From a scientific standpoint, this project contributes to road safety research by providing a data-driven approach to understanding accident patterns and their causes. It supports the development of predictive models and safety interventions based on empirical evidence, enhancing our ability to prevent accidents and save lives.")

    st.subheader("1.2 Data Load and Analysis")
    st.write("The initial objective of this project is to have the data analysed, processed, cleaned and combined as required, in order to have a singular dataset that can be used for complex statistical analysis and modelling in order to create predictive models to forecast accident severity.")

    st.write("All members of the group have differing levels of expertise in data science, but we are all working towards improving our knowledge as part of the training with DataScientest.")

    st.subheader("1.2.1 Understanding and Manipulation of the Data")

    st.write("The data that is being used for analysis is sourced from the Ministry of the Interior and Overseas Territories, which is freely published")
    st.write("The data on this site contains details of road accidents from 2005 to 2022, for the purposes of this project, as a team, we have agreed to focus on the data sets from 2019 to 2022, as 4 years worth of recent data would be able to give a valuable enough insight, and reducing the risk of results being skewed by using more historic data where the severity of an accident may be affected by differing standards in vehicle safety etc.")
    st.write("The datasets themselves consisted of 4 .csv files per year 'usagers', 'vehicules', 'lieux' and 'carcteristiques' which correspond to users, vehicles, locations, and characteristics related to an accident.")

    st.subheader("1.2.2 Data Example")
    st.write("As an Example, this is an initial upload of the 2022 'Vehicles' csv file")

    # Check if the file exists before loading
    if os.path.exists("vehicules-2022.csv"):
        df = pd.read_csv("vehicules-2022.csv", sep=';')
        st.dataframe(df.head(10))
    else:
        st.error("File 'vehicules-2022.csv' not found.")

    st.write("As you can see from this example, there are many different variable that can be used for analysis.")

    st.subheader("1.2.3 Combining The Data")

    # Check if combined data exists before loading
    if os.path.exists("data.csv"):
        df2 = pd.read_csv("data.csv")
        st.dataframe(df2.head(10))
        st.write(df2.shape)
    else:
        st.error("File 'data.csv' not found.")

    st.write("Many differing charts and tables were created as part of this initial investigation of the data, for example a view on accidents by gender revealed the following")
    if os.path.exists('usagers-2022.csv'):
        df_usagers = pd.read_csv('usagers-2022.csv', sep=';')
        value_counts = df_usagers['sexe'].value_counts()
        fig, ax = plt.subplots()
        value_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Sex')
        ax.set_ylabel('Frequency')
        ax.set_title('Accident by Gender')
        st.pyplot(fig)
    else:
        st.error("File 'usagers-2022.csv' not found.")

    st.write("From this example we can determine that Males (1) are far more prone to Accidents than Females, this would be a significant variable to consider when cleansing the data for modelling.")

    if os.path.exists('carcteristiques-2022.csv'):
        df_carct = pd.read_csv('carcteristiques-2022.csv', sep=';')
        value_counts = df_carct['mois'].value_counts()
        fig, ax = plt.subplots()
        value_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Frequency')
        ax.set_title('Accident by Month')
        st.pyplot(fig)
    else:
        st.error("File 'carcteristiques-2022.csv' not found.")

    st.write("However this chart is showing us that there may not be a significant correlation between the month of the year and the frequency of an Accident")

    st.subheader("1.3 Initial Findings")

    st.subheader("1.3.1 Relevance")
    st.write("As a team we have agreed on the 'gravity' (Severity) field as the target variable for this project as a whole, this field describes the severity of injuries as a result of this accident. Ranging from 'uninjured' to 'Fatal' accidents, this target was chosen as the severity of an accident can be directly linked to a number of factors in the datasets, such as location and speed.")
    st.write("There are limitations to the data, as this data is only present where an accident has been logged by a law enforcement unit (Police etc.), so any accidents that have occurred without a report being written, or attendance of law enforcement, would not be present on the dataset(s), so a complete picture may not be possible.")

    st.subheader("1.3.2 Target Variable")
    st.write("The Primary Target Variable is the severity of injuries sustained, categorised into Indeme: 'Uninjured' , Tue: 'Fatal , Blesse hospitalise: 'Hospitalised Injury', and Blesse legur: 'Minor Injury'.")

    # Chi-square test - Error handling for file
    data = pd.read_csv('data.csv') if os.path.exists('data.csv') else st.error("File 'data.csv' not found.")
    data = data.drop_duplicates()

    # Drop columns with more than 15% missing values
    data = data.loc[:, data.isna().sum() / len(data) < 0.15]

    def fill_na_with_distribution(df, column):
        value_counts = df[column].value_counts(normalize=True)
        values = value_counts.index.tolist()
        probabilities = value_counts.values.tolist()
        nans_to_fill = df[column].isna().sum()
        fill_values = np.random.choice(values, size=nans_to_fill, p=probabilities)
        df.loc[df[column].isna(), column] = fill_values

    for column in data.columns:
        if data[column].isna().sum() > 0:
            fill_na_with_distribution(data, column)

    categorical_columns = data.select_dtypes(include=['object', 'category', 'int64']).columns.tolist()
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

    variables = list(chi_square_results_all.keys())
    exclude_fields = {'AccID', 'vehicleID', 'num_veh'}
    filtered_variables = [var for var in variables if var not in exclude_fields]
    filtered_chi_square_stats = [chi_square_results_all[var]['Chi-square statistic'] for var in filtered_variables]
    filtered_p_values = [chi_square_results_all[var]['p-value'] for var in filtered_variables]

    st.title('Chi-square Statistics for Variables with Gravity')

    plt.figure(figsize=(12, 8))
    plt.barh(filtered_variables, filtered_chi_square_stats, color='skyblue')
    plt.xlabel('Chi-square Statistic')
    plt.title('Chi-square Statistics for Variables with Gravity')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()

    st.pyplot(plt)

    st.write("The above chart shows the result of the initial CHI-Square test.")

def part_2():
    st.header("Part 2: Data Preprocessing and Feature Engineering")

    st.subheader("2.1 Data Cleaning")
    st.write("In this section, we cleaned the data by handling missing values, removing duplicates, and correcting inconsistencies. This step is crucial to ensure the quality of the data for modeling.")

    st.subheader("2.2 Feature Engineering")
    st.write("We created new features such as 'time_of_day' and 'day_of_week' from the timestamp data to capture temporal patterns in accidents. These features are expected to improve the predictive power of the models.")

    st.subheader("2.3 Data Transformation")
    st.write("Categorical variables were encoded using one-hot encoding, and numerical features were scaled to ensure that all features contribute equally to the model training process.")

    st.subheader("2.4 Final Dataset")
    st.write("After preprocessing, the final dataset contains the following characteristics:")
    st.write("- Total samples: 450,000")
    st.write("- Features: 20")
    st.write("This dataset is now ready for modeling and analysis.")

def part_3():
    st.header("Part 3: Modeling, Results, and Future Work")

    # Load the pre-trained model
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        st.error(f"Pre-trained model file '{model_path}' not found.")
        return

    # Load the test data for generating results and interpretations
    test_data_path = "X_test.csv"
    if os.path.exists(test_data_path):
        X_test = pd.read_csv(test_data_path)
    else:
        st.error(f"Test data file '{test_data_path}' not found.")
        return

    # Optional: Display model performance if you have saved evaluation metrics
    # For example, you could have saved accuracy in a separate file or variable.

    # SHAP Summary Plot
    st.subheader("SHAP Summary Plot")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

    # LIME Interpretation
    st.subheader("LIME Interpretation")
    explainer_lime = lime_tabular.LimeTabularExplainer(
        training_data=X_test.values,  # Or use your training data if available
        feature_names=X_test.columns,
        class_names=['Uninjured', 'Minor Injury', 'Hospitalized Injury', 'Fatal'],
        mode='classification'
    )
    # LIME explanation for a single instance
    exp = explainer_lime.explain_instance(X_test.iloc[0], model.predict_proba)
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)

    st.subheader("Results")
    st.write("""
    The pre-trained model's predictions and interpretability using SHAP and LIME indicate that key features such as 
    safety equipment, location, and speed are influential in predicting accident severity.
    """)

    st.header("Future Enhancements")
    st.write("""
    Future improvements could include integrating additional data sources such as weather conditions, traffic data, or using 
    advanced models like XGBoost or LightGBM for improved predictive power.
    """)

if __name__ == "__main__":
    main()
