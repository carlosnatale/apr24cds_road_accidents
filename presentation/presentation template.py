import streamlit as st

def main():
    st.set_page_config(page_title="Data Science Project Presentation", layout="wide")
    st.title("Final Data Science Project Presentation")

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

def part_3():
    st.header("Part 3: Modeling, Results, and Future Work")

    st.subheader("3.1 Model Selection and Training")
    st.write("Detail the models used and the training process.")

    st.subheader("3.2 Evaluation Metrics and Results")
    st.write("Present evaluation metrics and results with visuals such as confusion matrices or performance charts.")
    st.write("For example:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Add your chart or metric visualization here.")
    with col2:
        st.write("Add interpretation or key findings here.")

    st.subheader("3.3 Future Work")
    st.write("Discuss potential improvements, future directions, or additional questions to explore.")

if __name__ == "__main__":
    main()
