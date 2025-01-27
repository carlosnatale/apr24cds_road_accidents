import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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



st.write("For example:")
col1, col2 = st.columns(2)
with col1:
    st.write("xxxxxxxxxxxxxxxxxxx.")
with col2:
    st.write("xxxxxxxxxxxxxxxxxxx.")
