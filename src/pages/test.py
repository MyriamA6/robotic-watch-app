import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
from streamlit.components.v1 import html
import os
import json
import re

# testing place

df = pd.read_csv("../../data/updated_humanoid_data.csv")
vision_sensors = list(df["Vision Sensors type"].dropna())
all_sensors = " , ".join(vision_sensors).replace("\n"," ").replace("\r"," ")
separated_sensors = all_sensors.split(",")
list_of_sensors=[]
for sensor in separated_sensors:
    list_of_sensors.append(sensor.strip())
result = list(set(list_of_sensors))
result.remove('')

"""
camera_rate = pd.read_csv(os.path.join(os.path.dirname(__file__),"..","..","data/vision_sensor_rating.csv"))
    def compute_camera_overall_rate(robot):
        cameras = camera_rate["Vision sensors type"]
        robot_vision_sensor = robot["Vision Sensors type"]
        res = 0
        cpt = 0
        for cam in cameras:
            if str(cam).lower() in str(robot_vision_sensor).lower():
                res += camera_rate[camera_rate["Vision sensors type"] == cam]["Rate out of 10"].iloc[0]
                cpt += 1
        if cpt==0:
            return 0
        return res / cpt

    def compute_quality_price_kpi(robot):

        weights = {
            "Speed (m/s)": 0.10,
            "Autonomy (hour)": 0.25,
            "Total Degrees of Freedom (DOF)": 0.2,
            "Two Hand Payload (kg)": 0.3,
            "Camera Score": 0.15
        }

        speed = min_max_scale(df,"Speed (m/s)",robot.get("Speed (m/s)", 0))
        autonomy = min_max_scale(df,"Autonomy (hour)",robot.get("Autonomy (hour)", 0))
        dof = min_max_scale(df,"Total Degrees of Freedom (DOF)",robot.get("Total Degrees of Freedom (DOF)", 0))
        payload = min_max_scale(df,"Two Hand Payload (kg)",robot.get("Two Hand Payload (kg)", 0))
        cost =robot.get("Cost(USD)", 1)  /1000

        # Je doute
        camera_global_rate = min_max_scale(camera_rate,"Rate out of 10",compute_camera_overall_rate(robot))

        quality_score = (
                weights["Speed (m/s)"] * speed +
                weights["Autonomy (hour)"] * autonomy +
                weights["Total Degrees of Freedom (DOF)"] * dof +
                weights["Two Hand Payload (kg)"] * payload +
                weights["Camera Score"] * camera_global_rate
        )

        kpi = quality_score / np.log(cost + 1)

        return kpi

    work_df['KPI_Quality_Price'] = work_df.apply(compute_quality_price_kpi, axis=1)

    fig_quality_price=px.scatter(
        work_df,
        x="Cost(USD)",
        y="KPI_Quality_Price",
        hover_name="Robot Name+A1:AB1"
    )

    st.plotly_chart(fig_quality_price, use_container_width=True)"""
