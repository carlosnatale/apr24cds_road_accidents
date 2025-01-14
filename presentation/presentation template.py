import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def main():
    st.set_page_config(page_title="Data Science Project Presentation", layout="wide")

    # Main Page Content
    st.title("Historic Road Accidents in France – A Study")
    st.markdown("### Authors:")
    st.markdown("- **Carlos Natale**")
    st.markdown("- **Stephen Waller**")
    st.markdown("- **Ehsan Jafari**")

    st.write("This project provides an in-depth analysis of road accidents in France, combining data preprocessing, modeling, and interpretative insights to improve road safety. Use the navigation panel to explore the project.")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    sections = [
        "Part 1: Project Context and Initial Data Insights",
        "Part 2: Data Preprocessing and Feature Engineering",
        "Part 3: Modeling, Results, and Future Work"
    ]
    choice = st.sidebar.radio("Go to:", sections)

    # Section Content
    if choice == "Part 1: Project Context and Initial Data Insights":
        part_1()
    elif choice == "Part 2: Data Preprocessing and Feature Engineering":
        part_2()
    elif choice == "Part 3: Modeling, Results, and Future Work":
        part_3()

def part_1():
    st.header("Part 1: Project Context and Initial Data Insights")

    st.subheader("1.1 Project Overview")
    st.write("Provide a brief description of the project's motivation, objectives, and significance.")

    st.subheader("1.2 Dataset Description")
    st.write("Describe the dataset used, including its source, size, and key characteristics.")

    st.subheader("1.3 Initial Data Insights")
    st.write("Present initial findings and insights from the dataset using visuals and summary statistics.")
    st.write("For example:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Add your visualization here.")
    with col2:
        st.write("Add your statistics or insights here.")

def part_2():
    st.header("Part 2: Data Preprocessing and Feature Engineering")

    st.subheader("2.1 Data Cleaning")
    st.write("Explain how missing values, duplicates, and outliers were handled.")

    st.subheader("2.2 Feature Engineering")
    st.write("Describe the new features created and the rationale behind them.")

    st.subheader("2.3 Transformation and Scaling")
    st.write("Discuss any transformations or scaling applied to the data.")

    st.subheader("2.4 Final Dataset")
    st.write("Summarize the prepared dataset, including the number of features and samples.")
    st.write("- Total samples: 447,670")
    st.write("- Features: 39")
    st.write("Include key features such as accident conditions, demographics, and road attributes.")

def part_3():
    st.header("Part 3: Modeling, Results, and Future Work")

    st.subheader("3.1 Modeling Approaches")
    st.write("Describe the algorithms used and their respective configurations:")
    st.write("- **Random Forest:** Efficient for high-dimensional data, optimized using hyperparameter tuning.")
    st.write("- **XGBoost:** Powerful gradient boosting technique, handled class imbalances effectively. Results show strong performance in recall for 'Severely Injured' cases.")
    st.write("- **AdaBoost:** Ensemble method focusing on misclassified instances.")
    st.write("- **KNN, Logistic Regression, and SVM:** Explored for specific scenarios with respective preprocessing strategies, but with varying success rates.")

    st.subheader("3.2 Evaluation Metrics")
    st.write("Key metrics used to evaluate the models:")
    st.write("- **Accuracy:** Overall correctness of predictions.")
    st.write("- **Precision, Recall, and F1-Score:** For imbalanced classes like 'Severely Injured' and 'Fatal'.")
    st.write("- **ROC-AUC:** To assess the classifier's performance in distinguishing classes.")

    st.subheader("3.3 Results")
    st.write("Summarize the outcomes of the modeling efforts:")
    st.write("- Random Forest achieved the highest F1-score of 0.91 for slightly injured cases.")
    st.write("- XGBoost excelled in recall for severely injured cases with an F1-score of 0.59 after threshold tuning.")
    st.write("- AdaBoost performed well in non-fatal classifications but struggled with precision for severe cases.")
    st.write("- KNN was effective for non-fatal cases but faced challenges with severely injured classifications.")

    # Interactive Graphs
    st.write("### Feature Importance - Random Forest")
    feature_importance = pd.DataFrame({
        "Feature": [f"Feature {i}" for i in range(1, 11)],
        "Importance": np.random.rand(10)
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(feature_importance, x="Importance", y="Feature", orientation='h', title="Feature Importance")
    st.plotly_chart(fig)

    st.write("### Confusion Matrix - XGBoost")
    confusion_matrix = pd.DataFrame({
        "Actual": ["Slightly Injured", "Slightly Injured", "Severely Injured", "Severely Injured"],
        "Predicted": ["Slightly Injured", "Severely Injured", "Slightly Injured", "Severely Injured"],
        "Count": [50, 10, 8, 32]
    })

    fig = px.imshow(
        confusion_matrix.pivot(index="Actual", columns="Predicted", values="Count"),
        title="Confusion Matrix",
        color_continuous_scale="Blues",
        labels={"color": "Count"}
    )
    st.plotly_chart(fig)

    st.subheader("3.4 Strengths and Weaknesses")
    st.write("Highlight the strengths and weaknesses of the chosen approaches:")
    st.write("- **Strengths:** Effective handling of large datasets, strong recall for minority classes, and interpretable feature importance in tree-based models.")
    st.write("- **Weaknesses:** Challenges with severe class imbalance, limited precision for fatal cases, and sensitivity to dataset noise in models like SVM.")

    st.subheader("3.5 Future Work")
    st.write("Outline potential improvements and next steps:")
    st.write("- Explore advanced ensemble methods and deep learning architectures.")
    st.write("- Apply synthetic data generation techniques to address class imbalance.")
    st.write("- Investigate the impact of temporal trends using ARIMA models.")
    st.write("- Further optimize hyperparameters to improve precision for minority classes.")

if __name__ == "__main__":
    main()
