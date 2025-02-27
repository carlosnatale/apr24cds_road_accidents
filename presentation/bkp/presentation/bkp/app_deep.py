import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

def part_3():
    st.header("Part 3: Modeling, Results, and Future Work")

	# Set up the page
	st.set_page_config(page_title="Accident Severity Prediction", layout="wide")

	# Title and Introduction
	st.title("Accident Severity Prediction Analysis")
	st.markdown("""
	This presentation provides a comprehensive analysis of machine learning models used for predicting accident severity. 
	The models evaluated include **Random Forest**, **XGBoost**, and **LightGBM**. We will explore their performance, 
	feature importance, and real-life recommendations based on SHAP and LIME interpretability.
	""")

	# Section 1: Model Performance Overview
	st.header("Model Performance Overview")

	# Random Forest
	st.subheader("Random Forest")
	st.markdown("""
	- **Overall Performance**: High F1-scores across most classes, especially for non-fatal cases.
	- **Strength in Binary Grouping**: Excelled in Non-Fatal vs. Fatal grouping with an F1-score of 0.97 for non-fatal cases.
	- **Balance of Precision and Recall**: Balanced approach helps capture different injury severities without overfitting.
	""")

	# XGBoost
	st.subheader("XGBoost")
	st.markdown("""
	- **Top Performer in Non-Fatal**: Achieved the highest F1-scores for non-fatal cases (0.98).
	- **Handling of Severe Cases**: Strong recall for severe injury classifications (F1-score of 0.59).
	- **Efficiency with Imbalanced Data**: Boosting technique effectively handles class imbalance.
	""")

	# LightGBM
	st.subheader("LightGBM")
	st.markdown("""
	- **Consistent High Scores**: Competitive F1 scores across different classes (0.94 for non-fatal cases).
	- **Good Recall for Minor Injury Cases**: F1-score of 0.60 for minor injuries.
	- **Efficiency and Speed**: Faster training and predictions due to gradient-based technique.
	""")

	# Section 2: Detailed Model Analysis
	st.header("Detailed Model Analysis")

	# Random Forest Detailed Analysis
	st.subheader("Random Forest Detailed Analysis")
	st.markdown("""
	- **Binary Classification**: Resampling into binary classes (Slightly Injured vs. Severely Injured) improved accuracy.
	- **LIME Interpretations**: Key features like location and time remain important after resampling.
	- **Conclusion**: Random Forest is a strong option for this problem, but it has high processing costs.
	""")

	# XGBoost Detailed Analysis
	st.subheader("XGBoost Detailed Analysis")
	st.markdown("""
	- **Model Performance**: High precision for Slightly Injured (0.95) but struggles with Severely Injured (precision 0.46).
	- **Feature Importance**: Location (latitude, longitude) and safety equipment are top features.
	- **SHAP and LIME**: Safety equipment and speed are critical factors in predicting severe injuries.
	""")

	# LightGBM Detailed Analysis
	st.subheader("LightGBM Detailed Analysis")
	st.markdown("""
	- **Model Performance**: High accuracy (0.83) but struggles with precision for Severely Injured (0.46).
	- **Feature Importance**: Age, location, and safety equipment are top features.
	- **SHAP and LIME**: Safety equipment and mobile obstacles are key predictors of severe injuries.
	""")

	# Section 3: Feature Importance and Interpretability
	st.header("Feature Importance and Interpretability")

	# SHAP Summary Plot
	st.subheader("SHAP Summary Plot")
	st.markdown("""
	The SHAP summary plot shows the most important features influencing the model's predictions. 
	Top features include **safety equipment**, **location**, and **maximum speed**.
	""")

	# Example SHAP plot (you need to load your SHAP values)
	# Assuming `shap_values` and `X_test` are available
	# shap.summary_plot(shap_values, X_test, show=False)
	# st.pyplot(plt.gcf())

	# LIME Interpretation
	st.subheader("LIME Interpretation")
	st.markdown("""
	LIME provides local interpretability for individual predictions. For example, the absence of safety equipment 
	and high speed are key factors in predicting severe injuries.
	""")

	# Example LIME plot (you need to load your LIME explainer)
	# Assuming `explainer` and `X_test` are available
	# exp = explainer.explain_instance(X_test.iloc[0], model.predict_proba)
	# exp.show_in_notebook()
	# st.pyplot(plt.gcf())

	# Section 4: Real-Life Recommendations
	st.header("Real-Life Recommendations")
	st.markdown("""
	Based on the SHAP and LIME insights, the following recommendations can help reduce accident severity:
	- **Speed Regulations**: Enforce speed limits in high-risk areas.
	- **Safety Equipment**: Promote the use of seat belts and airbags.
	- **Road Infrastructure**: Improve lighting and road conditions in accident-prone areas.
	- **Driver Training**: Offer refresher courses for older drivers.
	""")

	# Section 5: Future Enhancements
	st.header("Future Enhancements")
	st.markdown("""
	To further improve the models, consider integrating additional data sources:
	- **Weather Data**: Precipitation, wind speed, and temperature.
	- **Traffic Flow**: Traffic volume and congestion levels.
	- **Vehicle-Specific Data**: Maintenance records and safety ratings.
	- **Driver Behavior**: Speeding history and phone usage while driving.
	""")

	# Section 6: Conclusion
	st.header("Conclusion")
	st.markdown("""
	The combination of **SHAP-driven insights** and robust deployment strategies can significantly enhance road safety. 
	Real-time applications, improved infrastructure, and informed policy decisions based on these models have the potential 
	to reduce accident severity and save lives.
	""")

if __name__ == "__main__":
    main()