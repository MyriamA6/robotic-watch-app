import pandas as pd
import streamlit as st
import os

def home() :
    general_directory = os.path.dirname(os.path.abspath(__file__))

    dataset_file=os.path.join(general_directory,"..","..","data/humanoids_data.csv")
    df = pd.read_csv(dataset_file, delimiter=";")

    st.title("Welcome to the humanoïd robot analysis board :robot_face:")
    st.write("Feel free to modify the following settings as you wish to update the analysis of the dashboard.")
    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (df[column] < lower_bound) | (df[column] > upper_bound)


    # Function to generalise the print of the special robots for a given feature
    # Allows users to keep or not the special robots' values for which they were detected
    def show_outliers(df, column, unit=""):
        st.markdown(f"### :red[Detected special robots in **{column}**:]")
        mask = detect_outliers_iqr(df, column)
        unique_robots = df[mask]
        tp = df.copy()
        if unique_robots.empty:
            st.info("None")
            return tp
        st.dataframe(unique_robots)

        tp[str(column) + "- spe"] = None
        tp.loc[mask, str(column) + "- spe"] = "Special"

        choice = st.radio(f"Keep or remove special robots in {column}?",
                          [":green[Keep]", ":red[Remove]"], index=None)
        if choice == ":red[Remove]":
            tp.loc[mask, column] = None
            st.success(f"Special robots removed from **{column}**.")
        elif choice == ":green[Keep]":
            st.info("Special robots kept unchanged.")
        else:
            st.warning("Please make a selection.")
        return tp

    with st.container(border=True):
        st.subheader("Managing Unique Robots :")
        st.write("### Who are they ? :bulb:")
        st.write("Data points that stand out significantly from the rest of the observations in the dataset. "
                 "**Robots** with exceptionally high, low or unusual given feature.")
        st.write("These unusual values can sometimes skew analysis or models, so it's important to compare analysis with and without them.")
        st.write(
            ":large_green_circle: For each characteristic where unique robots are detected, you'll have the option to either **keep** them "
            "in your analysis or **remove** them. Your choice will directly affect the calculations and visualizations "
            "that follow on this page, allowing you to explore the data with or without these extreme values."
        )
        # Detection of special robots for several features
        with st.expander("View **Cost** based unique robots :money_with_wings:"):
            work_df=show_outliers(df,"Cost(USD)","USD")

        with st.expander("View **Payload** based unique robots :package:"):
            work_df=show_outliers(work_df,"Two Hand Payload (kg)","kg")

        with st.expander("View **Autonomy** based unique robots :battery:"):
            work_df=show_outliers(work_df,"Autonomy (hour)","Hour")

        with st.expander("View **DOF** based unique robots :wave:"):
            work_df=show_outliers(work_df,"Total Degrees of Freedom (DOF)","kg")

        with st.expander("View **Speed** based unique robots :running_woman:"):
            work_df=show_outliers(work_df,"Speed (m/s)","kg")

    st.divider()

    # Allowing the user to do the analysis considering wheeled robots or not
    st.write(":blue_car: Some robots are wheeled but wheeled robots can be quite different from bipedal ones and impact the analysis.")
    include_wheeled = st.radio(
        "Do you want to remove wheeled robots ?",
        [":green[Keep wheeled robots]", ":red[Remove wheeled robots]"],
        index=None, key="with_without_wheeled"
    )

    if include_wheeled == ":red[Remove wheeled robots]":
        work_df=work_df[work_df["Mobility Type"]!="Wheeled"]
        st.success("Wheeled robots have been removed.")
    elif include_wheeled == ":green[Keep wheeled robots]":
        st.info("Wheeled robots have been kept.")
    else :
        st.warning("Please make a selection.")


    # Attention : each time we change page the file is created once again
    work_df.to_csv(os.path.join(general_directory,"../../data/updated_humanoid_data.csv"), index=False)
