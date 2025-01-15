import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
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
            st.image(image_path, use_column_width=True)
        else:
            st.error("Image file 'road_accidents_France.png' not found. Please ensure it is in the correct location.")

        st.title("Navigation")
        sections = [
            "Part 0: Cover Page",
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

    st.subheader("1.1 xxxxxxxxxxxxxxxxxxx")
    st.write("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")

    st.subheader("1.2 xxxxxxxxxxxxxxxxxxx")
    st.write("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")

    st.subheader("1.3 xxxxxxxxxxxxxxxxxxx")
    st.write("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")
    st.write("For example:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("xxxxxxxxxxxxxxxxxxx.")
    with col2:
        st.write("xxxxxxxxxxxxxxxxxxx.")

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

    st.subheader("3.1 xxxxxxxxxxxxxxxxxxx")
    st.write("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")

    st.subheader("3.2 xxxxxxxxxxxxxxxxxxx")
    st.write("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")

    st.subheader("3.3 xxxxxxxxxxxxxxxxxxx")
    st.write("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")
    st.write("For example:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("xxxxxxxxxxxxxxxxxxx.")
    with col2:
        st.write("xxxxxxxxxxxxxxxxxxx.")

if __name__ == "__main__":
    main()
