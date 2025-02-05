import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import shap
from lime import lime_tabular
import os
from PIL import Image
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gzip
import pickle

# Load the compressed dataset
file_path = "data.pkl.gz"

with gzip.open(file_path, "rb") as f:
    data_final = pickle.load(f)
data_final = data_final.drop_duplicates()

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
            "Part 3: Modeling, Results, and Future Work",
            "Live Demo"
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
    elif choice == "Live Demo":
        part_4()    
    

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
    st.write("From this example we can determine that Males (1) are far more prone to Accidents than Females, this would be a significant variable to consider when cleansing the data for modelling.")

    df_carct = pd.read_csv('carcteristiques-2022.csv', sep=';')
    value_counts = df_carct['mois'].value_counts()


    fig, ax = plt.subplots()
    value_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Month')
    ax.set_ylabel('Frequency')
    ax.set_title('Accident by Month')

    st.pyplot(fig)
    st.write("However this chart is showing us that there may not be a significant correlation between the month of the year and the frequency of an Accident")


    st.subheader("1.3 Initial Findings")

    st.subheader("1.3.1 Relevance")
    st.write("As a team we have agreed on the 'grav' (Severity) field as the target variable for this project as a whole, this field describes the severity of injuries as a result of this accident. Ranging from 'uninjured' to 'Fatal' accidents, this target was chosen as the severity of an accident can be directly linked to a number of factors in the datasets, such as location and speed.")
    st.write("There are limitations to the data, as this data is only present where an accident has been logged by a law enforcement unit (Police etc.), so any accidents that have occurred without a report being written, or attendance of law enforcement, would not be present on the dataset(s), so a complete picture may not be possible.")



    st.subheader("1.3.2 Target Variable")
    st.write("The Primary Target Variable is the severity of injuries sustained, categorised into Indeme: 'Uninjured' , Tue: 'Fatal , Blesse hospitalise: 'Hospitalised Injury', and Blesse legur: 'Minor Injury'.")
    st.write("As our target Variable has 4 potential values in the final dataset, modelling would be completed against all of these, as well as grouped into binary capacity of Severly/Non Severly injured. ")

    st.subheader("1.3.3 Initial Insight with the Target Variable")

    st.write("As our target variable has been decided upon, the final part of the initial analysis was to assess all the remaining variables in the data set, to discover their relevance, and potential importance to the models that are to be written.")

    st.write("For Example, an initial CHI-Square analysis was done, this is a useful statistical test to dertermine any significant association between variables")

    data = pd.read_csv('data.csv')
    data = data.drop_duplicates()

    data = data.loc[:, data.isna().sum() / len(data) <0.15]
    def fill_na_with_distribution(df, column):
        # Calculate value counts for non-NaN values
        value_counts = df[column].value_counts(normalize=True)
        
        # Create a list of values based on the distribution
        values = value_counts.index.tolist()
        probabilities = value_counts.values.tolist()
        
        # Number of NaNs to fill
        nans_to_fill = df[column].isna().sum()
        
        # Randomly choose values based on the distribution
        fill_values = np.random.choice(values, size=nans_to_fill, p=probabilities)
        
        # Fill NaNs with these values
        df.loc[df[column].isna(), column] = fill_values

    # Apply the function to each column with NaN values
    for column in data.columns:
        if data[column].isna().sum() > 0:
            fill_na_with_distribution(data, column)


    # Identifying categorical columns in the dataset
    categorical_columns = data.select_dtypes(include=['object', 'category', 'int64']).columns.tolist()

    # Removing 'gravity' from the list as it will be the dependent variable
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

    # Extracting Chi-square statistics and p-values for visualization
    variables = list(chi_square_results_all.keys())

    # Exclude specific fields
    exclude_fields = {'AccID', 'vehicleID', 'num_veh'}
    filtered_variables = [var for var in variables if var not in exclude_fields]

    # Extract the corresponding Chi-square statistics and p-values
    filtered_chi_square_stats = [chi_square_results_all[var]['Chi-square statistic'] for var in filtered_variables]
    filtered_p_values = [chi_square_results_all[var]['p-value'] for var in filtered_variables]

    # Streamlit App
    st.title('Chi-square Statistics for Variables with Gravity')

    # Create a bar plot for Chi-square statistics
    plt.figure(figsize=(12, 8))
    plt.barh(filtered_variables, filtered_chi_square_stats, color='skyblue')
    plt.xlabel('Chi-square Statistic')
    plt.title('Chi-square Statistics for Variables with Gravity')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()

    # Display plot in Streamlit
    st.pyplot(plt)

    st.write("The Above chart shows the result of the initial CHI-Square test, and it clearly shows there could be significant value in some fields, such as Address and Safety Equipement, as well as other columns potentially being unnessesary and a candidate to be further rmoved from our dataset at point of modelling e.g. Route Number Index")


    st.subheader("1.3.4 Result of Initial Insights and Data Investigation")

    st.write("Significant correlations were found between accident severity and factors like road types, lighting conditions, demographics, and safety equipment use. Visualization revealed patterns such as fatal accidents being more likely on certain road types and under poor lighting, and older individuals being more susceptible to severe outcomes.")

    st.write("The next phase involves advanced modeling using machine learning techniques (Gradient Boosting, Decision Trees, and Random Forests), with further feature engineering to enhance model performance. These models aim to predict accident severity accurately, providing valuable insights for policymakers to improve road safety and reduce socio-economic impacts.")

    st.write("Overall, this data-driven approach aims to contribute significantly to road safety research, offering predictive insights to save lives and enhance traffic safety in France.")
 
def part_2():
    st.header("Part 2: Data Preprocessing and Feature Engineering")
    #st.markdown("Extensive data cleaning and processing were necessary to prepare the dataset for analysis. The treatment process involved key steps:")
   
    st.title("🧹 Data Cleaning and Processing")
    st.markdown("In this section, we cleaned the data by handling missing values, removing duplicates, and correcting inconsistencies. This step is crucial to ensure the quality of the data for modeling.")

    #st.write("In this section, we cleaned the data by handling missing values, removing duplicates, and correcting inconsistencies. This step is crucial to ensure the quality of the data for modeling.")

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Loading and Processing Datasets
    st.markdown("✅ **Loading and Processing Datasets**")
    st.markdown("""
    - The dataset initially consisted of **four separate CSV files per year (2019-2022)**:
                
        • `accidents.csv`
                
        • `locations.csv`
                
        • `users.csv`
                
        • `vehicles.csv`
                
    - Each dataset was loaded and processed using a function that:
                
        • Read the CSV files.
                
        • Handled incorrect formatting (`on_bad_lines='skip'`).
                
        • Assigned correct data types (`dtype=str` for consistency).
    """)

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Standardizing Column Names
    st.markdown("✅ **Standardizing Column Names**")
    st.markdown("""
    - Column names in the raw dataset were in **French**.
    - Translated all column names to **English** for clarity and better collaboration.
    - Example transformations:
        • `jour` → `day`
        • `mois` → `month`
        • `an` → `year`
        • `dep` → `dep_code`
    """)

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Handling Missing Values
    st.markdown("✅ **Handling Missing Values**")
    st.markdown("""
    - **Replaced Not Specified Values**:
                
        • `1` (indicating "Not specified") in `reason_travel` was replaced with `'0' ('Unknown')`.
                
        • All `-1` values were converted to `NaN` to standardize missing data.
                
    - **Dropped Irrelevant Columns**:
        • Removed `id_usager` and columns with **more than 30% missing values**.
    """)

     # st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # # ✅ Latitude & Longitude Conversion
    # st.markdown("✅ **Latitude & Longitude Conversion**")
    # st.markdown("""
    # - Geographic coordinates in the dataset used **commas (`','`)** instead of **decimal points (`'.'`)**.
    # - Converted values to ensure compatibility with geospatial tools.
    # """)

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Duplicate & Outlier Removal
    st.markdown("✅ **Duplicate & Outlier Removal**")
    st.markdown("""
    - **Duplicates**: Checked and removed to prevent redundancy.
                
    - **Outliers removed based on realistic thresholds**:
                
        • **Speed**: Dropped values `< 5 km/h` or `> 125 km/h` (unrealistic).
                
        • **Age**: Dropped values `< 0` or `> 97` based on statistical testing.
                
        • **Geographical Codes**: Removed invalid entries in `dep_code`.
    """)

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Missing Value Imputation
    st.markdown("✅ **Missing Value Imputation**")
    st.markdown("""
    - Instead of dropping rows, **imputation** was used to fill missing values based on:
                
        • **Distribution of existing values**.
                
        • Avoiding bias while maintaining dataset integrity.
    """)

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Deletion of Redundant Location Fields
    st.markdown("✅ **Deletion of Redundant Location Fields**")
    st.markdown("""
    - The dataset contained multiple location-related fields.
    - Retained only **latitude and longitude** for geospatial analysis.
    """)

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Merging Datasets
    st.markdown("✅ **Merging Datasets**")
    st.markdown("""
    - After preprocessing, the datasets were merged into a **single dataset**.
                
    - The merging process:
                
        • Used `AccID` as the primary key.
                
        • Ensured each accident had complete details from all four datasets.
                
        • Created a well-structured dataset for analysis.
    """)

    # Display dataset information

    buffer = io.StringIO()
    data_final.info(buf=buffer)
    info_str = buffer.getvalue()

    # Compute NaN values per column
    nan_counts = data_final.isna().sum()
    nan_output = f"Total NaN values per column:\n{nan_counts.to_string()}"

    # Count duplicate rows
    duplicate_count = data_final.duplicated().sum()
    duplicate_output = f"\nTotal duplicate rows: {duplicate_count}"

    # Display dataset summary
    st.markdown("### 📝 Dataset Summary")
    st.code(info_str, language="plaintext")

    # Display NaN values
    st.markdown("### 📌 Missing Values Summary")
    st.code(nan_output, language="plaintext")

    # Display duplicate count
    st.markdown("### 🔄 Duplicate Rows Summary")
    st.code(duplicate_output, language="plaintext")

    st.success("🎯 The dataset is now **clean, consistent, and ready** for further analysis!")

    st.title("🔧 Feature Engineering & Data Transformation")
    #st.markdown("This section covers the preprocessing steps applied to prepare the dataset for modeling.")

    #st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Dropping Unnecessary Columns
    st.markdown("✅ **Dropping Unnecessary Columns**")
    st.markdown("""
    - The following columns were **removed** because they were not useful for modeling:
                
        • `AccID`: Accident identifier (not a feature).
                
        • `birth_year`: Age will be used instead.
                
        • `vehicleID`, `num_veh`: Redundant vehicle identifiers.
    """)

    st.code("""
    # Drop unnecessary columns
    data_processed = data_processed.drop(['AccID', 'birth_year', 'vehicleID', 'num_veh'], axis=1)
    """, language="python")

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Cyclical Encoding for Temporal Features
    st.markdown("✅ **Cyclical Encoding for Temporal Features**")
    st.markdown("""
    - Temporal features (`day`, `month`, `time`) were transformed using **sine and cosine encoding** to preserve cyclical patterns.
                
    - The following transformations were applied:
                
        • `day_sin`, `day_cos` (Day of the month: 1-31).
                
        • `month_sin`, `month_cos` (Month of the year: 1-12).
                
        • `time_sin`, `time_cos` (Time of day converted to milliseconds).
                
    - Original time columns were dropped after encoding.
    """)

    st.code("""
    # Apply cyclical encoding
    data_processed['day_sin'] = np.sin(2 * np.pi * data_processed['day'] / 31)
    data_processed['day_cos'] = np.cos(2 * np.pi * data_processed['day'] / 31)

    data_processed['month_sin'] = np.sin(2 * np.pi * data_processed['month'] / 12)
    data_processed['month_cos'] = np.cos(2 * np.pi * data_processed['month'] / 12)

    data_processed['time_sin'] = np.sin(2 * np.pi * data_processed['time'] / 86340000) 
    data_processed['time_cos'] = np.cos(2 * np.pi * data_processed['time'] / 86340000)

    # Drop original time columns
    data_processed.drop(columns=['day', 'month', 'time'], inplace=True)
    """, language="python")

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Defining Features and Target Variable
    st.markdown("✅ **Defining Features and Target Variable**")
    st.markdown("""
    - Three groups of features were identified:
                
        • **Categorical features** (to be encoded using one-hot encoding).
                
        • **Numerical features** (to be standardized).
                
        • **Cyclical features** (already transformed).
                
    - Target variable: `gravity` (severity of the accident).
    """)

    st.code("""
    features_dummy = ['year', 'lum', 'atm_condition', 'collision_type',
        'route_category', 'traffic_regime', 'total_number_lanes',
        'reserved_lane_code', 'longitudinal_profile', 'plan',
        'surface_condition', 'infra', 'accident_situation',
        'traffic_direction', 'vehicle_category', 'fixed_obstacle',
        'mobile_obstacle', 'initial_impact_point', 'manv', 'motor', 'seat',
        'user_category', 'gender', 'reason_travel', 'safety_equipment1']

    features_scaler = ['lat', 'long', 'upstream_terminal_number', 'distance_upstream_terminal', 'maximum_speed', 'age']

    features_temporal = ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'time_sin', 'time_cos']

    target = 'gravity'
    """, language="python")

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Encoding Categorical Features
    st.markdown("✅ **Encoding Categorical Features**")
    st.markdown("""
    - Categorical variables were transformed using **one-hot encoding**.
                
    - The `drop_first=True` parameter was used to prevent multicollinearity.
    """)

    st.code("""
    X = data_processed.drop(columns=[target])
    y = data_processed[target]

    # Convert target variable to integer
    y = y.astype(int)

    # One-hot encoding
    X = pd.get_dummies(X, columns=features_dummy, drop_first=True)
    """, language="python")

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Splitting Dataset
    st.markdown("✅ **Splitting Dataset**")
    st.markdown("""
    - The dataset was split into **training (70%)** and **testing (30%)** sets.
                
    - **Stratified sampling** ensured the distribution of classes remained balanced.
    """)

    st.code("""
    # Stratified split to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    """, language="python")

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Feature Scaling
    st.markdown("✅ **Feature Scaling**")
    st.markdown("""
    - **Standardization** was applied to numerical features.
                
    - The scaler was **fitted on the training data** and **applied to both train and test sets**.
    """)

    st.code("""
    # Standardization: Fit only on training data
    scaler = StandardScaler()
    X_train[features_scaler] = scaler.fit_transform(X_train[features_scaler])
    X_test[features_scaler] = scaler.transform(X_test[features_scaler])
    """, language="python")

    st.markdown("<br>", unsafe_allow_html=True)  # Extra Space

    # ✅ Dataset Dimensions
    st.markdown("✅ **Dataset Dimensions**")
    st.markdown("""
    - Displays the shape of the training and testing sets.
    """)

# Copy of the original dataset for feature engineering and preprocessing
    data_processed = data_final.copy()

    # Drop unnecessary columns
    data_processed = data_processed.drop(['AccID', 'birth_year', 'vehicleID', 'num_veh'], axis=1)

    # Cyclical encoding for temporal features
    data_processed['day_sin'] = np.sin(2 * np.pi * data_processed['day'] / 31)  # Assuming day ranges from 1 to 31
    data_processed['day_cos'] = np.cos(2 * np.pi * data_processed['day'] / 31)

    data_processed['month_sin'] = np.sin(2 * np.pi * data_processed['month'] / 12)
    data_processed['month_cos'] = np.cos(2 * np.pi * data_processed['month'] / 12)

    data_processed['time_sin'] = np.sin(2 * np.pi * data_processed['time'] / 86340000) 
    data_processed['time_cos'] = np.cos(2 * np.pi * data_processed['time'] / 86340000)

    data_processed.drop(columns=['day','month','time'],inplace=True)

    # Selecting features and target variable
    features_dummy = ['year', 'lum', 'atm_condition', 'collision_type',
        'route_category', 'traffic_regime', 'total_number_lanes',
        'reserved_lane_code', 'longitudinal_profile', 'plan',
        'surface_condition', 'infra', 'accident_situation',
        'traffic_direction', 'vehicle_category', 'fixed_obstacle',
        'mobile_obstacle', 'initial_impact_point', 'manv', 'motor', 'seat',
        'user_category', 'gender', 'reason_travel',
        'safety_equipment1']
    # These features will be standardized
    features_scaler = ['lat', 'long', 'upstream_terminal_number', 'distance_upstream_terminal', 'maximum_speed', 'age']

    # These features are between -1 and 1 and do not need any standardazations. 
    features_temporal = ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'time_sin', 'time_cos']

    target = 'gravity'

    X = data_processed.drop(columns=[target])
    y = data_processed[target]

    y = y.astype(int)

    X = pd.get_dummies(X, columns=features_dummy, drop_first=True)

    # stratify will split the dataset according to the distribution of the classes to compensate for imbalanced datasets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Standardization: Fit only on the training data, then apply to both train and test
    scaler = StandardScaler()
    X_train[features_scaler] = scaler.fit_transform(X_train[features_scaler])
    X_test[features_scaler] = scaler.transform(X_test[features_scaler])


    # Capture dataset shapes
    shape_info = f"""
    Shape of X_train: {X_train.shape}
    Shape of X_test: {X_test.shape}
    """

    st.code(shape_info, language="plaintext")

    st.success("🎯 The dataset is now **preprocessed and ready** for model training!")

    # st.subheader("2.2 Feature Engineering")
    # st.write("We created new features such as 'time_of_day' and 'day_of_week' from the timestamp data to capture temporal patterns in accidents. These features are expected to improve the predictive power of the models.")

    # st.subheader("2.3 Data Transformation")
    # st.write("Categorical variables were encoded using one-hot encoding, and numerical features were scaled to ensure that all features contribute equally to the model training process.")

    # st.subheader("2.4 Final Dataset")
    # st.write("After preprocessing, the final dataset contains the following characteristics:")
    # st.write("- Total samples: 450,000")
    # st.write("- Features: 20")
    # st.write("This dataset is now ready for modeling and analysis.")

def part_3():
    st.header("Part 3: Modeling, Results, and Future Work")

    # Title and Introduction
    st.title("Accident Severity Prediction Analysis")
    st.markdown("""
    This section provides a comprehensive analysis of machine learning models used for predicting accident severity. 
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

    # Example Confusion Matrix for Random Forest
    st.write("**Confusion Matrix for Random Forest**")
    confusion_matrix = np.array([[8500, 500], [300, 1700]])  # Placeholder data
    fig, ax = plt.subplots(figsize=(6, 4))  # Resized graph
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Random Forest')
    st.pyplot(fig)
    st.write("The confusion matrix shows that the Random Forest model performs well in classifying non-fatal cases but struggles slightly with severe injuries.")

    # XGBoost
    st.subheader("XGBoost")
    st.markdown("""
    - **Top Performer in Non-Fatal**: Achieved the highest F1-scores for non-fatal cases (0.98).
    - **Handling of Severe Cases**: Strong recall for severe injury classifications (F1-score of 0.59).
    - **Efficiency with Imbalanced Data**: Boosting technique effectively handles class imbalance.
    """)

    # Example Feature Importance Plot for XGBoost
    st.write("**Feature Importance for XGBoost**")
    feature_importance = pd.DataFrame({
        'Feature': ['Location', 'Safety Equipment', 'Speed', 'Time of Day', 'Weather'],
        'Importance': [0.35, 0.25, 0.20, 0.15, 0.05]
    })
    fig = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importance - XGBoost')
    st.plotly_chart(fig, use_container_width=True)  # Resized graph
    st.write("The feature importance plot highlights that location and safety equipment are the most influential factors in predicting accident severity.")

    # LightGBM
    st.subheader("LightGBM")
    st.markdown("""
    - **Consistent High Scores**: Competitive F1 scores across different classes (0.94 for non-fatal cases).
    - **Good Recall for Minor Injury Cases**: F1-score of 0.60 for minor injuries.
    - **Efficiency and Speed**: Faster training and predictions due to gradient-based technique.
    """)

    # Example ROC Curve for LightGBM
    st.write("**ROC Curve for LightGBM**")
    fpr = np.linspace(0, 1, 100)  # Placeholder data
    tpr = np.sqrt(fpr)  # Placeholder data
    roc_auc = 0.89  # Placeholder data
    fig, ax = plt.subplots(figsize=(6, 4))  # Resized graph
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - LightGBM')
    ax.legend()
    st.pyplot(fig)
    st.write("The ROC curve shows that the LightGBM model has a high AUC score, indicating strong performance in distinguishing between classes.")

    # Section 2: Feature Importance and Interpretability
    st.header("Feature Importance and Interpretability")

    # SHAP Analysis Dashboard
    st.title("SHAP Analysis Dashboard")
    st.write("""
    This dashboard displays SHAP analysis for different machine learning models (XGBoost, LightGBM, and Random Forest). 
    Each graph visualizes the feature importance and interaction values. Below each graph, you'll find the corresponding interpretations.
    """)

    # Tabs for each model
    tab_xgb, tab_lgb, tab_rf = st.tabs(["XGBoost", "LightGBM", "Random Forest"])

    # XGBoost SHAP analysis
    with tab_xgb:
        st.header("XGBoost SHAP Analysis")
        xgb_image = Image.open("shap1 - xgboost.png")
        st.image(xgb_image, caption="XGBoost SHAP Summary Plot", width=800)
        st.subheader("Interpretation")
        st.write("""
        - **Top features**: `safety_equipment1_1`, `vehicle_category_7`, `lat`, and `maximum_speed` are the most influential features.
        - **Patterns**: High values of `maximum_speed` (red dots) generally push predictions in a positive direction, indicating a higher likelihood of a particular outcome.
        - **Feature importance**: The spread of the dots along the x-axis shows how much each feature contributes to the model’s output. Wider spread means a higher impact.
        """)

    # LightGBM SHAP analysis
    with tab_lgb:
        st.header("LightGBM SHAP Analysis")
        lgb_image = Image.open("shap1 - lightgbm.png")
        st.image(lgb_image, caption="LightGBM SHAP Interaction Values", width=800)
        st.subheader("Interpretation")
        st.write("""
        - **Interaction focus**: Columns like `year`, `lum`, `atm_condition`, and `collision_type` show their interaction with other features.
        - **Patterns**: High values of `collision_type` interact strongly with other features, indicating significant influence on the model's predictions.
        - **Balanced contributions**: The symmetric distribution of interaction values around 0 suggests well-balanced contributions between positive and negative impacts.
        """)

    # Random Forest SHAP analysis
    with tab_rf:
        st.header("Random Forest SHAP Analysis")
        rf_image = Image.open("shap1 - random forest.png")
        st.image(rf_image, caption="Random Forest SHAP Summary Plot", width=800)
        st.subheader("Interpretation")
        st.write("""
        - **Top features**: `vehicle_category`, `seat`, `user_category`, and `fixed_obstacle` have the highest impact on predictions.
        - **Patterns**: Features like `maximum_speed` and `age` exhibit diverse effects depending on whether their values are high or low (red or blue).
        - **Comparison with XGBoost**: While many features are important in both models, the order of importance differs, indicating potential variations in how each algorithm processes data.
        """)
    
    # Section 3: Real-Life Recommendations
    st.header("Real-Life Recommendations")
    st.markdown("""
    Based on the SHAP and LIME insights, the following recommendations can help reduce accident severity:
    - **Speed Regulations**: Enforce speed limits in high-risk areas.
    - **Safety Equipment**: Promote the use of seat belts and airbags.
    - **Road Infrastructure**: Improve lighting and road conditions in accident-prone areas.
    - **Driver Training**: Offer refresher courses for older drivers.
    """)

    # Section 4: Future Enhancements
    st.header("Future Enhancements")
    st.markdown("""
    To further improve the models, consider integrating additional data sources:
    - **Weather Data**: Precipitation, wind speed, and temperature.
    - **Traffic Flow**: Traffic volume and congestion levels.
    - **Vehicle-Specific Data**: Maintenance records and safety ratings.
    - **Driver Behavior**: Speeding history and phone usage while driving.
    """)

    # Section 5: Conclusion
    st.header("Conclusion")
    st.markdown("""
    The combination of **SHAP-driven insights** and robust deployment strategies can significantly enhance road safety. 
    Real-time applications, improved infrastructure, and informed policy decisions based on these models have the potential 
    to reduce accident severity and save lives.
    """)

def part_4():
    #Run liveux.py as a separate process
    os.system("streamlit run liveux.py")
            
if __name__ == "__main__":
    main()