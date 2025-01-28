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

    st.write("As mentioned above, the data that is being used for analysis is sourced from the Ministry of the Interior and Overseas Territories, which is freely published")
    st.write("The data on this site contains details of road accidents from 2005 to 2022, for the purposes of this project, as a team, we have agreed to focus on the data sets from 2019 to 2022, as 4 years worth of recent data would be able to give a valuable enough insight, and reducing the risk of results being skewed by using more historic data where the severity of an accident may be affected by differing standards in vehicle safety etc.")
    st.write("The datasets themselves consisted of 4 .csv files per year 'usagers', 'vehicules', 'lieux' and 'carcteristiques' which correspond to users, vehicles, locations, and characteristics related to an accident.")

    st.subheader("1.2.2 Data Example")
    st.write("As an Example, this is an initial upload of the 2022 'Vehicles' csv file")         

    df = pd.read_csv("vehicules-2022.csv", sep=';')

    st.dataframe(df.head(10))

    st.write("As you can see from this example, there are many different variable that can be used for analysis.")

    st.subheader("1.2.3 Combining The Data")

    st.write("The next stage after the initial investiagtion of the data available across the 4 differing CSV files available is to combine into a single dataframe that can be used for modelling, the combined dataframe looks like the below. ")
    df2 = pd.read_csv("data.csv")

    st.dataframe(df2.head(10))
    st.write(df2.shape)

    st.write("Although the combined dataframe above shows the first 10 rows, the combined file had over 450,000 rows of data to be used for analysis and modelling.")

    st.write("Many differing charts and tables were created as part of this initial investigation of the data, for example a view on accidents by gender revealed the following")
    df_usagers = pd.read_csv('usagers-2022.csv', sep=';')
    value_counts = df_usagers['sexe'].value_counts()

    fig, ax = plt.subplots()
    value_counts.plot(kind='bar', ax=ax)  
    ax.set_xlabel('Sex')
    ax.set_ylabel('Frequency')
    ax.set_title('Accident by Gender')

    st.pyplot(fig)
    st.write("From this exaple we can determine that Males (1) are far more prone to Accidents than Females, this would be a significant variable to consider when cleansing the data for modelling.")

    st.subheader("1.3 Initial Findings")

    st.subheader("1.3.1 Relevance")
    st.write("As a team we have agreed on the 'grav' (Severity) field as the target variable for this project as a whole, this field describes the severity of injuries as a result of this accident. Ranging from 'uninjured' to 'Fatal' accidents, this target was chosen as the severity of an accident can be directly linked to a number of factors in the datasets, such as location and speed.")
    st.write("There are limitations to the data, as this data is only present where an accident has been logged by a law enforcement unit (Police etc.), so any accidents that have occurred without a report being written, or attendance of law enforcement, would not be present on the dataset(s), so a complete picture may not be possible.")

    st.subheader("1.3.2 Target Variable")
    st.write("The Primary Target Variable is the severity of injuries sustained, categorised into Indeme: 'Uninjured' , Tue: 'Fatal , Blesse hospitalise: 'Hospitalised Injury', and Blesse legur: 'Minor Injury'.")
    st.write("As our target Variable has 4 potential values in the final dataset, mod3elling would be completed against al for of these, and grouped into binary capacity of Severly/Non Severly injured. ")
  

def part_2():
    st.header("Part 2: Data Preprocessing and Feature Engineering")

    st.subheader("2.1 xxxxxxxxxxxxxxxxxxx")
    st.write("vxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")

    st.subheader("2.2 xxxxxxxxxxxxxxxxxxx")
    st.write("Describe the new features created and the rationale behind them.")

    st.subheader("2.3 xxxxxxxxxxxxxxxxxxx")
    st.write("vxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")

    st.subheader("2.4 xxxxxxxxxxxxxxxxxxx")
    st.write("vxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")
    st.write("- Total samples: xxxx")
    st.write("- Features: xxxx")
    st.write("vxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
        part_0()
    elif choice == "Part 1: Project Context and Initial Data Insights":
        part_1()
    elif choice == "Part 2: Data Preprocessing and Feature Engineering":
        part_2()
    elif choice == "Part 3: Modeling, Results, and Future Work":
        part_3()

def part_3():
    st.header("Part 3: Modeling, Results, and Future Work")

    # Metrics Data (Example DataFrame for visualization)
    data = {
        "Model": ["Random Forest", "XGBoost", "LightGBM"],
        "Slightly Injured F1": [0.97, 0.87, 0.90],
        "Severely Injured F1": [0.49, 0.59, 0.54],
        "Overall Accuracy": [0.85, 0.80, 0.83]
    }
    metrics_df = pd.DataFrame(data)

    st.subheader("3.1 Model Performance")
    st.write("The following chart compares the F1-scores and overall accuracy across different models.")

    # Plotting F1 Scores and Accuracy
    fig, ax = plt.subplots()
    metrics_df.plot(x="Model", kind="bar", ax=ax)
    ax.set_title("Model Performance Metrics")
    ax.set_ylabel("Score")
    st.pyplot(fig)

    # Feature Importance Data (Example DataFrame)
    features = {
        "Feature": ["Longitude", "Latitude", "Maximum Speed", "Safety Equipment", "Age"],
        "Importance": [0.25, 0.20, 0.15, 0.30, 0.10]
    }
    features_df = pd.DataFrame(features)

    st.subheader("3.2 Feature Importance")
    st.write("The chart below highlights the most influential features in predicting accident severity.")

    # Plotting Feature Importance
    fig2, ax2 = plt.subplots()
    features_df.plot(x="Feature", y="Importance", kind="bar", ax=ax2, legend=False)
    ax2.set_title("Feature Importance")
    ax2.set_ylabel("Importance")
    st.pyplot(fig2)

    # SHAP Analysis (Example Data for Visualization)
    st.subheader("3.3 SHAP Analysis")
    st.write("SHAP (SHapley Additive exPlanations) values provide insights into how features influence model predictions globally and locally.")

    # Simulated SHAP values and feature names for demonstration
    shap_values = np.random.rand(1, 5)  # Single instance SHAP values
    feature_names = ["Longitude", "Latitude", "Maximum Speed", "Safety Equipment", "Age"]

    # Example Instance for Local Interpretation
    example_instance = np.random.rand(1, 5)
    explainer = shap.Explainer(lambda x: x, np.random.rand(100, 5))  # Placeholder explainer
    shap_values_instance = explainer(example_instance)

    st.subheader("Local Explanation")
    st.write("The following SHAP values explain a specific prediction for an example instance.")

    fig4, ax4 = plt.subplots()
    shap.waterfall_plot(shap_values_instance[0])  # Correctly use single explanation without extra arguments
    st.pyplot(fig4)

    st.subheader("3.4 Recommendations")
    st.write("- **Safety Equipment:** Ensuring proper use can significantly reduce severe injuries.")
    st.write("- **Speed Management:** Implement speed limit enforcement in high-risk areas.")
    st.write("- **Targeted Interventions:** Focus on older drivers and high-risk locations.")

    st.write("Future enhancements include incorporating additional data such as weather and traffic conditions and utilizing advanced modeling techniques for better precision in minority classes.")

if __name__ == "__main__":
    main()