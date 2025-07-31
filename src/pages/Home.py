import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from streamlit.components.v1 import html
import json
import re
from collections import Counter
import base64
from io import BytesIO
import qrcode

def home() :
    general_directory = os.path.dirname(os.path.abspath(__file__))

    dataset_file=os.path.join(general_directory,"..","..","data/humanoid_data_cleaned2207.csv")
    df = pd.read_csv(dataset_file)

    df = df.replace(
        to_replace=r'(?i)^\s*(n/d|n\.d|//n|n/a|na|nan|none|null)\s*$',
        value=None,
        regex=True
    )

    numerical_columns = df[["Cost (USD)", "Weight (kg)", "Height(cm)", "Speed (m/s)", "Autonomy (hour)", "Total Degrees of Freedom (DOF)",
         "Body Degrees of Freedom (DOF)", "Hands Degrees of Freedom (DOF)", "Two Hand Payload (kg)"]]

    # Converting the right columns to numeric
    for col in numerical_columns.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(df.columns)
    df = df.dropna(subset="Year Unveiled")
    df["Year Unveiled"] = df["Year Unveiled"].astype(int)
    df["Country"]=df["Country"].str.strip()
    df["Region"]=df["Region"].str.strip()


    df.to_csv(os.path.join(general_directory,"../../data/updated_humanoid_data.csv"), index=False)

    # Retrieval of the path to the current directory
    general_directory = os.path.dirname(os.path.abspath(__file__))

    anchor_ids =["title","Disclaimer", "news","calvin","means","map-repartition","company-repartition","physical-prop","s2", "top-robots-1","humanoid-creators","camera-type","percep-info","ai-tech","safety","s3","img-hum-flow","primary-use-case","robots-produced"]

    # Retrieval of the dataset
    dataset_file=os.path.join(general_directory,"..","../data/updated_humanoid_data.csv")
    df = pd.read_csv(dataset_file)

    df = df.replace(
        to_replace=r'(?i)^\s*(n/d|n\.d|//n|n/a|na|none|null)\s*$',
        value=None,
        regex=True
    )

    # Description of the content of the webpage
    st.subheader("   ",anchor="title")
    st.title(":bar_chart: Humanoid Insights")
    st.divider()

    st.markdown("""
    <div style='border: 2px solid #90caf9; background-color: #e3f2fd; padding: 1.5em; border-radius: 10px; font-size: 1.1em;'>
        <p style="font-size:1.3em">
            This application allows you to explore and compare the current <b>humanoid robots</b> available on the market.<br><br>
            🏆 <b>Rankings</b> of top humanoid robots<br>
            📊 <b>Average specs</b> of humanoid robots<br>
            📈 Analysis of <b>companies</b> on the market <br><br>
            Dive in and discover the future of robotics!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    # Disclaimer Section

    st.subheader("   ",anchor="Disclaimer")
    st.subheader("   ")

    st.markdown("""
    <div style='border: 1px solid #ffa726; background-color: #fff3e0; padding: 1.5em; border-radius: 10px;'>
  <h2 style='color: #e65100; margin-top: 0;'>⚠️ Disclaimer</h2>
  
  <p style="font-size: 1.5em; line-height: 1.6; color: #333;">
    The data presented on this page were gathered from several public online sources, including:
  </p>
  
  <ul style="font-size: 1.3em; line-height: 1.6; padding-left: 1.5em; color: #444;">
    <li><i>Humanoid.guide</i></li>
    <li><i>www.aparobot.com</i></li>
    <li><i>humanoidroboticstechnology.com</i></li>
  </ul>
  
  <p style="font-size: 1.5em; line-height: 1.6; color: #333;">
    As such, the information may be incomplete, outdated, or inaccurate.<br>
    It is provided for informational purposes only and should not be considered definitive or fully reliable.<br>
    Use it solely to gain a general approximation of the current state of the humanoid robotics market.
  </p>
  
  <p style="font-size: 1.5em; font-weight: bold; color: #b71c1c;">
     Independent verification is strongly recommended before making any decisions.
  </p>
</div>

    """, unsafe_allow_html=True)


    st.subheader("   ", anchor="news")

    st.subheader(":newspaper: Latest in Humanoid Robotics   -   ***:grey[humanoidsdaily]***")
    from .website_retrieval import website_retrieval

    news = website_retrieval()
    html_blocks = ""
    for idx, (i, item) in enumerate(news.items()):
        html_blocks += f"""
    <div class="news-item" id="news-{i}"
         style="
           max-width: 1000px;
           max-height : 500px;
           margin: 1.5rem auto;
           padding: 1.5rem 1rem;
           background: #ffffffcc;
           border-radius: 16px;
           box-shadow: 0 8px 16px rgba(0,0,0,0.12);
           text-align: center;
           font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
           color: #333;
           overflow: visible;
           backdrop-filter: blur(8px);
         "
    >
      <h2 style="margin: 0 1.8rem 1rem; font-weight: 700; font-size: 1.6rem; color: #0078D7;">
        {item['title']}
      </h2>
      <p style="margin: 0 1.8rem 1rem; font-size: 1rem; line-height: 1.4;">
        {item['summary']}
      </p>
      <p style="color: grey; margin-bottom: 1rem; font-size: 0.9rem;">
        {item['date']}
      </p>
      <img src="{item['image']}"
           style="
           max-height : 300px;
       display: block;
       margin: 0 auto 1.25rem;
       border-radius: 16px;
       box-shadow: 0 4px 12px rgba(0,0,0,0.1);
     "
      />
    </div>
    """

    js_code_news=f"""
                <script>
                // === News slideshow logic ===
                let newsIndex = 0;
                const newsItems = document.querySelectorAll(".news-item");
                const newsDelay = 6000;

                function showNews(index) {{
                    newsItems.forEach((el, i) => {{
                        el.style.display = (i === index) ? "block" : "none";
                    }});
                }}

                function cycleNews() {{
                    if (newsItems.length === 0) return;
                    showNews(newsIndex);
                    newsIndex = (newsIndex + 1) % newsItems.length;
                    setTimeout(cycleNews, newsDelay);
                }}

                // Launch both carousels on page load
                window.addEventListener("load", () => {{
                    cycleNews();
                }});
            </script>
            """



    html(f"""<div>{html_blocks}</div>{js_code_news}
             """, height=700)

    st.subheader("   ", "calvin")
    st.video("https://www.youtube.com/watch?v=YddS-aI097Q&t=4s", autoplay=True, loop=True)


    # Mean of several data
    df_wheeled = df[df["Mobility Type"]=="wheeled"]
    weight_mean = round(df_wheeled["Weight (kg)"].mean(), 2)
    height_mean = round(df_wheeled["Height(cm)"].mean(), 2)
    payload_mean = round(df_wheeled["Two Hand Payload (kg)"].mean(), 2)
    dof_mean = round(df_wheeled["Total Degrees of Freedom (DOF)"].mean(), 2)
    speed_mean = round(df_wheeled["Speed (m/s)"].mean(), 2)
    autonomy_mean = round(df_wheeled["Autonomy (hour)"].mean(), 2)

    df_without_wheeled = df[df["Mobility Type"]!="wheeled"]
    weight_mean_without_wheeled = round(df_without_wheeled["Weight (kg)"].mean(), 2)
    height_mean_without_wheeled = round(df_without_wheeled["Height(cm)"].mean(), 2)
    payload_mean_without_wheeled = round(df_without_wheeled["Two Hand Payload (kg)"].mean(), 2)
    dof_mean_without_wheeled = round(df_without_wheeled["Total Degrees of Freedom (DOF)"].mean(), 2)
    speed_mean_without_wheeled = round(df_without_wheeled["Speed (m/s)"].mean(), 2)
    autonomy_mean_without_wheeled = round(df_without_wheeled["Autonomy (hour)"].mean(), 2)


    st.subheader("   ", anchor="means")
    st.subheader("   ")
    st.subheader("   ")
    st.subheader("   ")
    st.subheader("   ")
    # Titre principal
    st.markdown("<h1 style='font-size: 90px;'>📊 What is the current market overview ?</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    # Sous-titre
    st.markdown("<h2 style='font-size: 70px;'>Essential Figures to Remember</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # --- BIPEDAL HUMANOID ROBOTS ---
    col1.markdown("<p style='font-size: 60px; font-weight: bold;'>🤖 Bipedal humanoid robots</p>",
                  unsafe_allow_html=True)
    col1.markdown(
        f"<div style='font-size:50px;'>⚖️ <b>Average weight:</b><br><span style='font-size:50px; color:#1f77b4;'>{weight_mean_without_wheeled} kg</span></div>",
        unsafe_allow_html=True)
    col1.markdown(
        f"<div style='font-size:50px;'>📏 <b>Average height:</b><br><span style='font-size:50px; color:#1f77b4;'>{height_mean_without_wheeled} cm</span></div>",
        unsafe_allow_html=True)
    col1.markdown(
        f"<div style='font-size:50px;'>🦾 <b>Average payload:</b><br><span style='font-size:50px; color:#1f77b4;'>{payload_mean_without_wheeled} kg</span></div>",
        unsafe_allow_html=True)

    col2.markdown("<br><br>", unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size:50px;'>🧠 <b>Total DOF:</b><br><span style='font-size:50px; color:#1f77b4;'>{dof_mean_without_wheeled}</span></div>",
        unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size:50px;'>🏃 <b>Average speed:</b><br><span style='font-size:50px; color:#1f77b4;'>{speed_mean_without_wheeled} m/s</span></div>",
        unsafe_allow_html=True)
    col2.markdown(
        f"<div style='font-size:50px;'>🔋 <b>Autonomy:</b><br><span style='font-size:50px; color:#1f77b4;'>{autonomy_mean_without_wheeled} h</span></div>",
        unsafe_allow_html=True)

    # --- WHEELED HUMANOID ROBOTS ---
    col3.markdown("<p style='font-size: 60px; font-weight: bold;'>🛞 Wheeled humanoid robots</p>",
                  unsafe_allow_html=True)
    col3.markdown(
        f"<div style='font-size:50px;'>⚖️ <b>Average weight:</b><br><span style='font-size:50px; color:#e15759;'>{weight_mean} kg</span></div>",
        unsafe_allow_html=True)
    col3.markdown(
        f"<div style='font-size:50px;'>📏 <b>Average height:</b><br><span style='font-size:50px; color:#e15759;'>{height_mean} cm</span></div>",
        unsafe_allow_html=True)
    col3.markdown(
        f"<div style='font-size:50px;'>🦾 <b>Average payload:</b><br><span style='font-size:50px; color:#e15759;'>{payload_mean} kg</span></div>",
        unsafe_allow_html=True)

    col4.markdown("<br><br>", unsafe_allow_html=True)
    col4.markdown(
        f"<div style='font-size:50px;'>🧠 <b>Total DOF:</b><br><span style='font-size:50px; color:#e15759;'>{dof_mean}</span></div>",
        unsafe_allow_html=True)
    col4.markdown(
        f"<div style='font-size:50px;'>🏃 <b>Average speed:</b><br><span style='font-size:50px; color:#e15759;'>{speed_mean} m/s</span></div>",
        unsafe_allow_html=True)
    col4.markdown(
        f"<div style='font-size:50px;'>🔋 <b>Autonomy:</b><br><span style='font-size:50px; color:#e15759;'>{autonomy_mean} h</span></div>",
        unsafe_allow_html=True)

    # Marge avant séparation
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.subheader("   ")
    st.subheader("   ")

    st.subheader("   ")
    st.subheader("   ")

    st.subheader("   ")
    st.divider()


    # European country without russia
    europe = [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan",
        "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
        "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
        "Finland", "France", "Georgia", "Germany", "Greece", "Hungary",
        "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia",
        "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova",
        "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway",
        "Poland", "Portugal", "Romania", "San Marino", "Serbia",
        "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey",
        "Ukraine", "United Kingdom"
    ]

    country_dataset=os.path.join(general_directory,"..","../data/country_co.csv")
    country_data = pd.read_csv(country_dataset)

    country_data = country_data.merge(df, how='right', left_on='Country', right_on='Country')
    country_counts = country_data.groupby(["Country", "Latitude", "Longitude"]).size().reset_index(name="Robot Count")
    europe_counts = {}

    for (idx, row) in country_counts.iterrows():
        if row["Country"] in europe:
            europe_counts[row["Country"]] = row["Robot Count"]

    country_counts = country_counts[~country_counts["Country"].isin(europe)]

    country_counts.loc[len(country_counts)] = ["Europe", 49, 14, sum(europe_counts.values())]
    # Showing the repartition of creators of robots
    # The bigger the point, the more the corresponding country has recently invented robots
    st.subheader("   ",anchor="map-repartition")
    st.markdown("<h1 style='font-size:70px; font-weight:bold;'>Global Distribution Of Humanoid Robot Creators 🌎</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:50px; color:#2986FF'><b>The bigger the point, the more a country has created humanoid robots.</b></div>",
        unsafe_allow_html=True)
    with st.container(border=True):
        fig = px.scatter_map(country_counts,
                             lat="Latitude", lon="Longitude",
                             size="Robot Count",
                             hover_name="Country",
                             hover_data=["Robot Count"],
                             color="Country",
                             size_max=200,
                             zoom=3,
                             width=1000,
                             height=1500)

        fig.update_layout(mapbox_style="carto-positron")
        fig.update_layout(
            annotations=[
                dict(
                    x=0.5, y=-0.1, xref="paper", yref="paper",
                    text="Point size corresponds to number of humanoid robots created",
                    showarrow=False,
                    font=dict(size=12),
                    xanchor='center'
                )
            ]
        )

        st.plotly_chart(fig)

    # Grouping countries that have not created many robots
    low_count_country = country_counts[country_counts["Robot Count"] <= 1]["Robot Count"].sum()
    distribution_companies_country = country_counts[country_counts["Robot Count"] > 1]

    if low_count_country > 0:
        distribution_companies_country = pd.concat([
            distribution_companies_country,
            pd.DataFrame({"Country": ["Other countries"], "Robot Count": [low_count_country]})
            # Others corresponds to companies having invented only one robot
        ], ignore_index=True)

    # This chart shows how companies building humanoid robots are distributed across different countries, based on their count.
    st.subheader("   ", anchor="company-repartition")
    st.markdown("<h1 style='font-size:70px; font-weight:bold;'>🌎 Humanoid Robots Main Producers Around the World</h1>", unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1 :
        with st.container(border=True):
            fig_rep = px.pie(
                distribution_companies_country,
                values="Robot Count",
                names="Country",
                title="Worldwide robot creators distribution"
            )
            fig_rep.update_layout(autosize=True,
                width=500,
                title_font_size=50,
                height=1500, legend=dict(font=dict(size=40)))

            fig_rep.update_traces(
                textinfo="percent+label",  # Affiche le pourcentage ET le label
                textfont_size=40  # ← Taille du texte (ajuste selon tes besoins)
            )

            st.plotly_chart(fig_rep, use_container_width=True)

    with p2 :
        with st.container(border=True):
            df_europe = pd.DataFrame(list(europe_counts.items()), columns=["Country", "Robot Count"])
            fig_europe=px.pie(
                df_europe,
                values="Robot Count",
                names="Country",
                title ="European robot creators repartition"
            )
            fig_europe.update_layout(autosize=True,
                                     width=500,
                                     title_font_size=40,
                                     legend=dict(font=dict(size=40)),
                                     height=1500)
            fig_europe.update_traces(
                textinfo="percent+label",  # Affiche le pourcentage ET le label
                textfont_size=40  # ← Taille du texte (ajuste selon tes besoins)
            )
            st.plotly_chart(fig_europe, use_container_width=True)


    st.subheader("   ", anchor="physical-prop")
    st.markdown("<h1 style='font-size:70px; font-weight:bold;'>Physical properties based repartition of robots</h1>", unsafe_allow_html=True)

    # Scatter plot of the robots based on their weight and height
    with st.container(border=True):
        data_interactive=df.copy()
        data_interactive = data_interactive.dropna(subset=["Weight (kg)", "Height(cm)", "Two Hand Payload (kg)"])
        fig = px.scatter(
            data_interactive,
            x="Height(cm)",
            y="Weight (kg)",
            size="Two Hand Payload (kg)",
            color="Robot Name",
            hover_name="Robot Name",
            hover_data=["Robot Name","Company","Country","Region","Year Unveiled", "Two Hand Payload (kg)"])

        fig.update_layout(legend=dict(font=dict(size=40)))
        fig.update_layout(autosize=False, width=1000, height=1500)
        st.markdown("""
        <p style="font-size:30px; line-height:1.4;">
        This scatter plot shows the relationship between robot <b>height</b> and <b>weight</b>. 
        The <b>bubble size</b> indicates the <b>two-hand payload capacity</b> (in kg). 
        It helps identify trends between physical dimensions and lifting capabilities.
        </p>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig)

    # Function to show the n-best robots in a given category
    # The order of sort is also chosen by the dev
    def show_top(df, column, ascending=False, n=3, unit="", info_sup=None):
        top_df = df.copy()
        top_df[column] = pd.to_numeric(top_df[column], errors="coerce")
        top_df = top_df.dropna(subset=[column])
        top_df = top_df.sort_values(by=column, ascending=ascending).head(n)

        for _, row in top_df.iterrows():
            name = f"<b style='font-size:50px;'>{row['Robot Name']}</b>"
            company = f"<span style='font-size:40px;'>{row['Company']}</span>"
            value = f"<span style='font-size:40px; color:#1f77b4;'><b>{row[column]}</b> {unit}</span>"

            if info_sup is not None and isinstance(row[info_sup], str):
                extra = f"<br><i style='font-size:40px;'>NB: {row[info_sup]}</i>"
            else:
                extra = ""

            st.markdown(
                f"<div style='font-size:40px; margin-bottom: 20px; margin-top:20px; margin-left: 25px; margin-right: 25px;'>{name} - {company} : {value}{extra}</div>",
                unsafe_allow_html=True
            )

    st.subheader("   ", anchor="s2")
    st.markdown("<h1 style='font-size:90px; font-weight:bold;'>Who are the key players ?</h1>", unsafe_allow_html=True)
    st.subheader("   ", anchor="top-robots-1")
    st.markdown("<h1 style='font-size:70px; font-weight:bold;'>🏆 Top Robots by Category</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    # Board of the best robots in each selected category
    with c1:
        with st.container(border=True):
            st.markdown("<h3 style='font-size:40px;'>🔋 <b>Best Autonomy</b></h3>", unsafe_allow_html=True)
            show_top(df, "Autonomy (hour)", ascending=False, unit="hours")

    with c2:
        with st.container(border=True):
            st.markdown("<h3 style='font-size:40px;'>📦 <b>Best Payload</b></h3>", unsafe_allow_html=True)
            show_top(df, "Two Hand Payload (kg)", ascending=False, unit="kg")

    with c3:
        with st.container(border=True):
            st.markdown("<h3 style='font-size:40px;'>💸 <b>Cheapest Robots</b></h3>", unsafe_allow_html=True)
            show_top(df, "Cost (USD)", ascending=True, unit="USD")

    c4, c5, c6 = st.columns(3)

    with c4:
        with st.container(border=True):
            st.markdown("<h3 style='font-size:40px;'>🏃‍♀️ <b>Fastest Robots</b></h3>", unsafe_allow_html=True)
            show_top(df, "Speed (m/s)", ascending=False, unit="m/s", info_sup="Mobility Type")

    with c5:
        with st.container(border=True):
            st.markdown("<h3 style='font-size:40px;'>⚖️ <b>Lightest Robots</b></h3>", unsafe_allow_html=True)
            show_top(df, "Weight (kg)", ascending=True, unit="kg")

    with c6:
        with st.container(border=True):
            st.markdown("<h3 style='font-size:40px;'>👋 <b>Higher DOF</b></h3>", unsafe_allow_html=True)
            show_top(df, "Total Degrees of Freedom (DOF)", ascending=False)

        # Creation of a pie chart of the main companies in the humanoid robots market
        # On all companies, wheeled robots included

    distribution_companies = df.groupby("Company").size().reset_index(name="Count")

    distribution_companies = distribution_companies[distribution_companies["Count"] > 1].sort_values(by='Count',
                                                                                                     ascending=False)

    st.subheader("   ", anchor="humanoid-creators")
    st.subheader("Main humanoid robots creators :")
    names = distribution_companies["Company"].unique()
    cpt=0
    for name in names:
        cpt+=1
        with st.container(border=True):
            cb1, cb2 = st.columns(2)
            with cb1:
                country = df[df["Company"] == name]["Country"].iloc[0]
                st.markdown(
                    f"<p style='font-size:24px; font-weight:bold; color:rgb(46, 134, 193)'>{name} - Location : {country} </p>"
                    , unsafe_allow_html=True)
            with cb2:
                nb_robots = str(distribution_companies[distribution_companies["Company"] == name]["Count"].iloc[0])
                st.markdown(f"**Has recently created **:red[{nb_robots}]** robots.** ")
        if cpt==6 :
            break

    st.divider()

    st.subheader("    ", anchor="camera-type")
    st.subheader("Repartition of perception sensors type used by companies")
    df_camera = df.copy()
    df_camera["Vision Sensors type"].fillna("Not specified", inplace=True)

    df_sensor_map = pd.read_csv(os.path.join(general_directory, "../../data/vision_sensors.csv"), sep=";")

    sensor_groups = {}
    for _, row in df_sensor_map.iterrows():
        key = row["Vision_sensors"].strip().lower()
        keywords = [kw.strip().lower() for kw in row["other_names"].split(",")]
        sensor_groups[key] = keywords

    def classify_sensor(sensor_name):
        sensor_name = sensor_name.strip().lower()
        for group, keywords in sensor_groups.items():
            if any(kw in sensor_name for kw in keywords):
                return group
        return sensor_name

    vision_sensors = df_camera["Vision Sensors type"].tolist()
    all_sensors = ", ".join(vision_sensors).replace("\n", " ").replace("\r", " ")
    separated_sensors = [s.strip() for s in all_sensors.split(",") if s.strip()]

    classified_sensors = [classify_sensor(sensor) for sensor in separated_sensors]
    sensor_counts = Counter(classified_sensors)

    df_camera_count = pd.DataFrame(sensor_counts.items(), columns=["Camera Type", "Count"])
    fig_camera = px.pie(df_camera_count, values="Count", names="Camera Type",
                        title="Distribution of Vision Sensor Types")
    fig_camera.update_layout(autosize=False, width=1000, height=1500)
    st.plotly_chart(fig_camera, use_container_width=True)

    st.subheader("  ")
    st.subheader("  ")
    st.subheader("  ")

    st.subheader("  ", anchor="percep-info")
    st.markdown("""
    <h2 style='
        font-size:70px;
        font-weight:bold;
    '>
        🔎 Some information about perception sensors :
    </h2>
    """, unsafe_allow_html=True)
    st.subheader("   ")
    st.subheader("   ")
    percep_sensors = [
        {
            "title": "What is an RGB Camera?",
            "content": """
                    <p>
                      <strong>RGB</strong> stands for Red, Green, and Blue, i.e the primary colors of light.<br>
                      An RGB camera captures color images using visible light (400–700 nm).
                    </p>
                """,
            "source":"www.e-consystems.com",
            "img":"rgb.png"
        },
        {
            "title": "What is an RGBD Camera?",
            "content": """
                        <p>
                          An RGBD camera shows both the colors of a scene (RGB) and how far things are (D = Depth).<br>
                          It works like a normal camera but also adds 3D information to each image.
                        </p>
                    """,
            "source":"www.e-consystems.com",
            "img":"rgbd.png"
        },
        {
            "title": "What are LiDAR sensors ?",
            "content": """
                           <p>
                              LiDAR uses laser light to measure distances.<br>
                              It sends light to objects and measures how long it takes to bounce back.<br>
                              It helps create a 3D map of the surroundings.
                            </p>
                            """,
            "source": "www.synopsys.com",
            "img": "lidar.png"
        },
        {
            "title": "What are Ultrasonic sensors ?",
            "content": """
                       <p>
                          Ultrasonic sensors use sound waves (too high for humans to hear) to measure distance.<br>
                          They send out a sound, wait for it to bounce back, and calculate how far the object is.
                        </p>
                        """,
            "source":"www.sick.com",
            "img":"ultrasonic.png"
        },
        {
            "title": "What are RaDAR sensors ?",
            "content":"""
            <p>
              Radar uses radio waves (instead of sound or light) to detect objects and measure their distance, speed, and angle.<br>
              It sends out signals and listens for the echoes that bounce back<br>
              To be sure about what it detects, radar repeats measurements and compares the results (called tracking).<br>
              At least two antennas are needed to know the direction of an object.
            </p>
            """,
            "source":"www.sick.com",
            "img":"radar.png"

        },
        {
            "title": "What is a fisheye ?",
            "content": """
                <p>
                  A fisheye camera has a very wide lens that captures a much larger view than regular cameras, i.e. up to 180° or more.<br>
                  It creates curved, distorted images, which helps see more of a scene at once.<br>
                </p>
                """,
            "img": "fisheye_lens.png"

        },
        {
            "title": "What is an Eagle Eye camera ?",
            "content": """
                <p>
                  An Eagle Eye camera gives a top-down or full-scene view, like seeing from above.<br>
                  It combines images from multiple cameras to show a wide, 360° view from above.<br>
                </p>
                """,
            "img": "eagle_eye.png"

        }
    ]

    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode()
        return f"data:image/png;base64,{b64}"

    html_block_perception_info = ""
    for idx, c in enumerate(percep_sensors):
        img_show=os.path.join(general_directory,"..","..","data/images",c['img'])
        if not os.path.exists(img_show):
            print("here")
            img_show=os.path.join(general_directory,"..","..","data/images","not_found.png")
        html_block_perception_info += f"""
        <div class="percep-block" id="percep-{idx}"
    style="
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        background-color: #fff;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 4rem;
        color: #222;
        cursor: default;
        display: flex;
        gap: 2rem;
        align-items: center;
        justify-content: space-between;
    "
>
<!-- Image -->
    <div style="flex: 0 0 900px; max-width: 900px;">
        <img src="{img_to_base64(img_show)}"
            style="
                width: 100%;
                min-height :600px;
                border-radius: 40px;
                margin: 60px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                display: block;
                object-fit: cover;
            "
        />
    </div>
    <!-- Text Content -->
    <div style="flex: 1;">
        <h3 style="margin:100px;margin-bottom: 0.75rem; font-weight: 700; font-size: 4rem; display: flex; align-items: center; gap: 0.5rem;">
            {c['title']}
        </h3>
        <div style="margin:100px;color: #444; font-size: 3rem; line-height: 1.5;">
            {c['content']}
        </div>
    </div>
</div>

    """

    js_switch_percep = """
        <script>
        // === percep slideshow logic ===
        let percepIndex = 0;
        const percepItems = document.querySelectorAll(".percep-block");
        const percepDelay = 6000;

        function showpercep(index) {
        percepItems.forEach((el, i) => {
            // Use flex display to maintain layout when showing
            el.style.display = (i === index) ? "flex" : "none";
        });
        }

        function cycleperceps() {
            if (percepItems.length === 0) return;
            showpercep(percepIndex);
            percepIndex = (percepIndex + 1) % percepItems.length;
            setTimeout(cycleperceps, percepDelay);
        }

        // Démarrage au chargement de la page
        window.addEventListener("load", () => {
            cycleperceps();
        });
        </script>
        """

    html(f"""{html_block_perception_info}{js_switch_percep}
""",height=1500)



    st.subheader("  ", anchor="ai-tech")
    st.subheader(":computer: AI Analysis :")
    col3,col4=st.columns(2)
    # --- Colonne 3 : Répartition des technologies d'IA utilisées ---
    with col3:
        with st.container(border=True):
            st.subheader("Repartition of AI technologies used in the robots :")

            df_temp = df.copy()
            df_temp["AI Technology used"].fillna("Not specified", inplace=True)

            # Joindre toutes les valeurs et normaliser les séparateurs
            value_list = df_temp["AI Technology used"].tolist()
            all_values = "; ".join(value_list).replace("\n", " ").replace("\r", " ")

            # Normaliser les séparateurs multiples : and, &, /, ,
            for sep in [" and ", " & ", "/", ","]:
                all_values = all_values.replace(sep, ";")

            # Séparer, nettoyer
            cleaned_values = [s.strip() for s in all_values.split(";") if s.strip()]

            # Compter les occurrences
            value_counts = Counter(cleaned_values)
            df_counts = pd.DataFrame(value_counts.items(), columns=["AI Technology used", "Count"])

            # Calcul des pourcentages
            total = df_counts["Count"].sum()
            df_counts["Percentage"] = df_counts["Count"] / total * 100

            # Regrouper les < 3% en "Others"
            df_counts["AI Technology used"] = df_counts.apply(
                lambda row: row["AI Technology used"] if row["Percentage"] >= 3 else "Others",
                axis=1
            )

            # Regrouper à nouveau après remplacement
            df_counts = df_counts.groupby("AI Technology used", as_index=False)["Count"].sum()

            # Créer et afficher le graphique
            fig = px.pie(df_counts, values="Count", names="AI Technology used", title="Distribution of AI Technologies")
            fig.update_layout(autosize=False, width=500, height=1500)
            fig.update_traces(
                textinfo="percent+label",  # Affiche le pourcentage ET le label
                textfont_size=40  # ← Taille du texte (ajuste selon tes besoins)
            )
            st.plotly_chart(fig, use_container_width=True)
    # --- Colonne 4 : Capacité des robots à parler naturellement ---
    with col4:
        with st.container(border=True):
            st.subheader(
                "How many robots actually are able to converse naturally? :speaking_head_in_silhouette:")

            df["Can the robot converse naturally?"] = df["Can the robot converse naturally?"].fillna("no").replace("",
                                                                                                                   "no")
            df_talk = df.groupby("Can the robot converse naturally?").size().reset_index(name="Count")

            fig_talk = px.pie(df_talk, values="Count", names="Can the robot converse naturally?")
            fig_talk.update_layout(autosize=False, width=500, height=1500)
            fig_talk.update_traces(
                textinfo="percent+label",  # Affiche le pourcentage ET le label
                textfont_size=40  # ← Taille du texte (ajuste selon tes besoins)
            )
            st.plotly_chart(fig_talk, use_container_width=True)

    st.subheader("  ", anchor="safety")
    st.subheader("🛡️ Different Security Standards Implemented")
    with st.container(border=True):

        st.markdown(
            "<h2 style='font-size:48px;'>Safety features used in the robots</h2>",
            unsafe_allow_html=True
        )

        # 1. Copier et nettoyer les données
        df_temp = df.copy()
        df_temp["Safety Features"].fillna("not specified", inplace=True)

        # 2. Concaténer et normaliser les séparateurs
        value_list = df_temp["Safety Features"].tolist()
        all_values = "; ".join(value_list).replace("\n", " ").replace("\r", " ")
        for sep in [" and ", " & ", "/", ","]:
            all_values = all_values.replace(sep, ";")

        # 3. Séparer, nettoyer, mettre en minuscule
        cleaned_values = [s.strip().lower() for s in all_values.split(";") if s.strip()]

        # 4. Compter les occurrences
        value_counts = Counter(cleaned_values)
        df_counts = pd.DataFrame(value_counts.items(), columns=["safety feature", "count"])

        # 5. Calculer les pourcentages
        total = df_counts["count"].sum()
        df_counts["percentage"] = df_counts["count"] / total * 100

        # 6. Trier par pourcentage décroissant
        df_counts = df_counts.sort_values(by="percentage", ascending=False).reset_index(drop=True)

        # 8. Show 7 random safety features
        random_rows = df_counts.sample(n=min(7, len(df_counts)))  # handles if df_counts has < 7 rows


        for _, row in random_rows.iterrows():
            st.markdown(
                f"<p style='font-size:28px;'>• <b>{row['safety feature'].capitalize()}</b> "
                f"({row['count']} occurrences, {row['percentage']:.1f}%)</p>",
                unsafe_allow_html=True
            )

    st.subheader("   ", anchor="s3")
    st.markdown(
        "<h1 style='font-size:90px;'>What do the humanoid robots actually do?</h1>",
        unsafe_allow_html=True
    )

    st.subheader("  ")
    st.subheader("  ", anchor="img-hum-flow")

    st.markdown(
        "<h1 style='font-size:70px;'>Understanding humanoid robots - <i>Morgan Stanley</i></h1>",
        unsafe_allow_html=True
    )
    st.image(os.path.join(general_directory, "..", "..", "data/images/robot_representation.png"))

    # Histogram of Primary Use-case of robots
    st.subheader("   ", anchor="primary-use-case")
    st.markdown(
        "<h1 style='font-size:70px;'>Primary use case distribution for humanoid robots</h1>",
        unsafe_allow_html=True
    )

    with st.container(border=True):
        df["Primary Use-Case"] = df["Primary Use-Case"].fillna("Not specified").astype(str)
        frequency_primaryusecase = df["Primary Use-Case"].value_counts().reset_index(name="Count")

        fig = px.pie(
            frequency_primaryusecase,
            names="Primary Use-Case",
            values="Count",
        )

        fig.update_traces(
            textinfo="percent+label",  # Affiche pourcentages + labels
            textfont_size=40  # Agrandit la taille du texte
        )

        fig.update_layout(
            autosize=False,
            width=1000,
            height=1500,
            xaxis_title="Primary Use-Case",
            yaxis_title="Number of Robots"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("   ", anchor="robots-produced")
    st.subheader("   ")


    # Function to normalize a given value
    def min_max_scale(df,cat,val):
        col_min = df[cat].min()
        col_max = df[cat].max()

        if pd.isna(val) or pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            scaled_val = 0
        else:
            scaled_val = (val - col_min) / (col_max - col_min)

        return scaled_val

    # Retrieving robots currently produced

    st.subheader("   ")
    st.subheader("   ")
    st.subheader("   ")
    df_prod=df[df["Status"]=="In Production"]

    st.markdown(f"""
        <div style="
            background-color: #f1f3f6;
            border-left: 10px solid #4b8bbe;
            padding: 40px 45px;
            border-radius: 12px;
            margin-bottom: 30px;
            font-family: 'Segoe UI', sans-serif;
        ">

          <h2 style="margin-top: 0; color: #2c3e50; font-size: 70px;">🤖 Global Humanoid Robots in Production</h2>

          <p style="font-size: 48px; color: #333;">
            🔧 <strong>This section presents the robots that are currently produced around the world.</strong>
          </p>

          <p style="font-size: 48px; color: #333;">
            📊 Currently, there are 
            <span style="font-weight: bold; color: #e74c3c;">{len(df_prod)}</span> 
            humanoid robots listed as <strong>'In Production'</strong>.
          </p>

          <ul style="font-size: 38px; line-height: 1.8; color: #333; padding-left: 30px;">
            <li>🧠 Key specifications: mobility, autonomy, sensors…</li>
            <li>💵 Estimated cost (if available)</li>
            <li>🏭 Production capacity per year</li>
            <li>📈 Radar chart summarizing technical specs</li>
          </ul>

        </div>
    """, unsafe_allow_html=True)
    st.subheader("   ")
    st.subheader("   ")
    st.subheader("   ")
    st.subheader("   ")

    for idx, robot in df_prod.iterrows():
        st.markdown(
            f"<h2 style='font-size: 48px;'>🤖 <b>{robot['Robot Name']}</b> - <i style='color: grey;'>{robot['Company']}</i></h2>",
            unsafe_allow_html=True)

        # Compter les NaNs dans les catégories critiques
        categories = [
            "Height(cm)", "Weight (kg)", "Total Degrees of Freedom (DOF)",
            "Two Hand Payload (kg)", "Speed (m/s)", "Autonomy (hour)"
        ]

        missing_count = sum(
            robot[col] is None or pd.isna(robot[col]) for col in categories
        )

        missing_threshold = 4

        if missing_count > missing_threshold:
            st.markdown(
                f"<h3 style='font-size: 32px; color: gray;'>Not enough information available for this robot.</h3>",
                unsafe_allow_html=True)
            continue

        with st.container(border=True):
            anchor_ids.append("robot-info" + str(idx))
            subcol1, subcol2 = st.columns(2)

            with subcol1:
                image_path = os.path.join(general_directory, "..", "../data/images/" + robot['Robot Name'] + ".png")
                if os.path.exists(image_path):
                    st.image(image_path, width=300)
                else:
                    st.markdown("<p style='font-size: 28px;'>🖼️ No image found</p>", unsafe_allow_html=True)

                st.markdown("<h3 style='font-size: 36px; padding-top : 30px; padding-bottom : 30px;'>🔧 <b>SPECS INFO</b></h3>", unsafe_allow_html=True)
                for col in categories:
                    if robot[col] is not None and pd.notna(robot[col]):
                        st.markdown(f"<div style='font-size: 28px; padding-bottom : 30px;'>• <b>{col}:</b> {robot[col]}</div>",
                                    unsafe_allow_html=True)

            with subcol2:
                cost_robot = robot['Cost (USD)']
                if cost_robot is not None and pd.notna(cost_robot):
                    st.markdown(
                        f"<div style='font-size: 32px; padding-bottom: 30px;'>💰 <b>Cost:</b> {cost_robot}</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size: 32px; padding-bottom: 30px;'>💰 <b>Cost:</b> Not found</div>",
                                unsafe_allow_html=True)

                prod_cap = robot['Production Capacity (units per year)']
                if prod_cap is not None and pd.notna(prod_cap):
                    st.markdown(
                        f"<div style='font-size: 32px; padding-bottom: 30px;'>🏭 <b>Quantity:</b> {prod_cap} units/year</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='font-size: 32px; padding-bottom: 30px;'>🏭 <b>Quantity:</b> Not found</div>",
                        unsafe_allow_html=True)

                # Radar chart
                values = []
                for cat in categories:
                    try:
                        val = float(robot.get(cat))
                    except:
                        val = np.nan
                    values.append(min_max_scale(df, cat, val))

                values += values[:1]
                radar_categories = categories + [categories[0]]  # Don't overwrite original list

                fig = go.Figure(
                    data=[
                        go.Scatterpolar(
                            r=values,
                            theta=radar_categories,
                            fill='toself',
                            name=robot['Robot Name']
                        )
                    ],
                    layout=go.Layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1])
                        ),
                        showlegend=False,
                        title=dict(
                            text="Radar Chart Technical Specs 🔧",
                            font=dict(size=40)
                        ),
                        font=dict(size=30),
                        width=700,
                        height=700
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("   ")
        st.subheader("   ")

    st.divider()

    anchor_ids.extend(["s4","country-cost-comp","tot-fundraise","market-description","val-share","warnings","poll"])

    st.subheader("    ")
    st.subheader("    ", anchor="s4")
    st.markdown("<h1 style='font-size: 90px;'>What is the current economic situation?</h1>",
                unsafe_allow_html=True)
    st.subheader("    ")
    st.subheader("   ", anchor="country-cost-comp")
    st.markdown("<h1 style='font-size: 70px;'>Average Robot Cost by Country </h1>",
                unsafe_allow_html=True)

    country_robot_cost_europe = df[df["Country"].isin(europe)]["Cost (USD)"].mean()

    df_country_robot_cost_exp_europe = df[~df["Country"].isin(europe)].groupby("Country")["Cost (USD)"].agg(
        "mean").reset_index(name="Average Cost (USD)")
    df_cost_europe = pd.DataFrame([
        {"Country": "Europe",
         "Average Cost (USD)": country_robot_cost_europe}
    ])

    df_country_robot_cost_exp_europe = pd.concat([df_country_robot_cost_exp_europe, df_cost_europe],
                                                 ignore_index=True).dropna().sort_values("Average Cost (USD)",
                                                                                         ascending=False)

    fig_robot_cost = px.bar(
        df_country_robot_cost_exp_europe,
        x="Country",
        y="Average Cost (USD)",
        labels={"Country": "Country", "Average Cost (USD)": "Average Cost (USD)"},
        color="Country",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text=df_country_robot_cost_exp_europe["Average Cost (USD)"].round(2),
        height = 1500# Add rounded cost as text
    )

    fig_robot_cost.update_traces(textposition='outside',
                                textfont=dict(size=40)
                                 )  # Puts text above the bars
    fig_robot_cost.update_layout(xaxis_tickangle=-45,
                                xaxis = dict(
                                    title_font=dict(size=40),  # taille du label de l'axe X
                                    tickfont=dict(size=30)  # taille des ticks (valeurs) de l'axe X
                                ),
                                yaxis = dict(
                                    title_font=dict(size=40),  # taille du label de l'axe Y
                                    tickfont=dict(size=30)  # taille des ticks (valeurs) de l'axe Y
                                ),
                                 legend = dict(font=dict(size=40),
                                    title = dict(
                                        text="Country",
                                        font=dict(size=40)  # taille du titre de la légende
                                    ),
                                               )
                                 )
    st.plotly_chart(fig_robot_cost, use_container_width=True)
    st.divider()



    st.subheader("   ", anchor="tot-fundraise")
    st.markdown("<h1 style='font-size: 70px;'>📊 Total Fundraising in Humanoid Robotics (2018+)</h1>",
                unsafe_allow_html=True)
    df_humanoid_market = pd.read_csv(os.path.join(general_directory,"../../data/humanoid_company_market_analysis.csv"), delimiter=";", index_col="Company")
    df_humanoid_market = df_humanoid_market.T

    cols = list(df_humanoid_market.columns)
    fundraising_cols = [col for col in cols if col.startswith("Fundraising")]
    for col in fundraising_cols:
        df_humanoid_market[col] = df_humanoid_market[col].fillna(0)
        df_humanoid_market[col] = df_humanoid_market[col].astype(int)
    df_cum = df_humanoid_market[fundraising_cols].cumsum(axis=1)
    df_tp = df_cum.reset_index().rename(columns={'index': 'Company'})
    df_new = df_tp.melt(id_vars='Company', var_name='Year', value_name='Cumulative Fundraising')

    df_new['Year'] = df_new['Year'].str.replace('Fundraising ', '').astype(int)

    df_new_2 = df_new[df_new['Year'] >= 2018]

    # Somme des levées de fonds par année (pas cumulées)
    df_total = df_humanoid_market[fundraising_cols].sum().reset_index()
    df_total.columns = ["Year", "Total Fundraising"]
    df_total["Year"] = df_total["Year"].str.replace("Fundraising ", "").astype(int)
    df_total = df_total[df_total["Year"] >= 2018]  # filtrage à partir de 2018

    # Graphique
    fig = px.bar(
        df_total,
        x="Year",
        y="Total Fundraising",
        labels={"Total Fundraising": "Total Fundraising ($USD)"},
        text=df_total["Total Fundraising"].round(2),  # Add rounded cost as text
        height=1500
    )

    fig.update_traces(textfont=dict(size=40)
                                 )

    fig.update_layout(xaxis = dict(
                                    title_font=dict(size=40),  # taille du label de l'axe X
                                    tickfont=dict(size=30)  # taille des ticks (valeurs) de l'axe X
                                ),
                                yaxis = dict(
                                    title_font=dict(size=40),  # taille du label de l'axe Y
                                    tickfont=dict(size=30)  # taille des ticks (valeurs) de l'axe Y
                                ),
        annotations=[
            dict(
                x=2025,
                y=df_total.loc[df_total["Year"] == 2025, "Total Fundraising"].values[0],
                text="Data only up to mid-2025",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color="red", size=30)
            ),
            dict(
                x=2024,
                y=df_total.loc[df_total["Year"] == 2024, "Total Fundraising"].values[0]-100_000_000,
                ax=2023,
                ay=df_total.loc[df_total["Year"] == 2023, "Total Fundraising"].values[0]+100_000_000,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="AI included in robotics",
                showarrow=True,
                arrowhead=3,
                font=dict(color="red", size=30),
                arrowcolor="black"
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)



    st.subheader(" ", anchor="market-description")
    st.markdown("<h1 style='font-size: 70px;'>Companies' Market Valuations in Humanoid Robotics (2025)</h1>",
                unsafe_allow_html=True)

    side1, side2 =st.columns(2)
    with st.container(border=True):
        market = df_new_2[df_new_2['Year'] == 2025].sort_values(by='Cumulative Fundraising', ascending=False)

        fig_market = px.bar(
            market.iloc[:10],
            x='Company',
            y='Cumulative Fundraising',
            title="Fundraising Distribution Among Top 10 Humanoid Companies",
            text='Cumulative Fundraising'
        )

        fig_market.update_layout(
            xaxis=dict(
                title_font=dict(size=40),  # taille du label de l'axe X
                tickfont=dict(size=30)  # taille des ticks (valeurs) de l'axe X
            ),
            yaxis=dict(
                title_font=dict(size=40),  # taille du label de l'axe Y
                tickfont=dict(size=30)  # taille des ticks (valeurs) de l'axe Y
            ),
            height=1500,
            xaxis_title="Company",
            yaxis_title="Cumulative Fundraising (USD)",
            uniformtext_mode='hide'
        )

        fig_market.update_traces(textfont=dict(size=40)
                          )


        st.plotly_chart(fig_market, use_container_width=True)


    st.subheader(" ", anchor="val-share")
    st.subheader("2025 Valuation Share of Leading Humanoid Robotics Firms")
    with st.container(border=True):
        df_valuation = df_humanoid_market["Valuation 2025"].reset_index().rename(
            columns={'index': 'Company'}).sort_values(by='Valuation 2025', ascending=False)

        top5 = df_valuation.iloc[:5]

        top5['Valuation 2025'] = top5['Valuation 2025'].astype(float)

        total_val = top5['Valuation 2025'].sum()

        fig_valuation = px.pie(
            top5,
            values='Valuation 2025',
            names='Company',
            hole=0.5,
        )
        fig_valuation.update_layout(
            autosize=False,
            height=1500,
            annotations=[dict(
                text=f"""${total_val:,.0f}""",
                x=0.5,
                y=0.5,
                font_size=17,
                showarrow=False
            )]
        )
        st.plotly_chart(fig_valuation, use_container_width=True)
        st.markdown(
            "This chart illustrates the projected 2025 valuation share of the **top five humanoid robotics companies**. "
            "Each slice represents the relative market valuation of a company, with the total valuation shown at the center of the chart."
        )

    # Page title

    st.subheader("   ", anchor="warnings")
    st.title("🤖 Challenges of Building a Humanoid Robot")
    st.subheader("   ")


    challenges = [
        {
            "icon": "🦿",
            "title": "Mechanics & Realistic Mobility",
            "content": """
            Designing a body that can <strong>walk, run, jump, or bend</strong> is complex. It involves:
            <ul>
                <li>Lightweight but strong materials.</li>
                <li>Compact and powerful actuators.</li>
                <li>Dynamic balancing for bipedal locomotion.</li>
            </ul>
            """
        },
        {
            "icon": "👁️",
            "title": "Perception and Vision",
            "content": """
            A humanoid robot needs to <strong>understand its environment in real time</strong> using various sensors:
            <ul>
                <li><strong>RGB cameras, Fisheye, RGB-D, LiDAR, radar, ultrasonic</strong>, etc.</li>
                <li>Object and face recognition, obstacle detection.</li>
                <li>Real-time mapping and localization.</li>
            </ul>
            Processing this massive data stream requires high-performance computing and smart filtering.
            """
        },
        {
            "icon": "🧠",
            "title": "Intelligence and Decision-Making",
            "content": """
            Perception alone is not enough. The robot must also <strong>think, plan, and adapt</strong>:
            <ul>
                <li>Understand human behavior and intent.</li>
                <li>React to unexpected events and edge cases.</li>
                <li>Communicate naturally via speech or gestures (NLP, large language models).</li>
            </ul>
            This is where <strong>AI and reinforcement learning</strong> play a major role.
            """
        },
        {
            "icon": "🔋",
            "title": "Power Supply and Efficiency",
            "content": """
            Humanoid robots consume a lot of energy, especially with full-body movement and onboard processing.
            <ul>
                <li>Battery capacity is limited by size and weight.</li>
                <li>Energy optimization is essential for autonomy and safety.</li>
                <li>Need to choose an optimal battery, not overheating too much after an action from the robot</li>
            </ul>
            """
        },
        {
            "icon": "🧍",
            "title": "Human-Robot Interaction & Safety",
            "content": """
            Humanoids operate in spaces built for humans, often <strong>next to people</strong>:
            <ul>
                <li>Must detect and avoid collisions.</li>
                <li>Behaviors must be socially acceptable and predictable.</li>
                <li>Interfaces (voice, facial expressions, gestures) should be intuitive and non-threatening.</li>
            </ul>
            This is crucial for applications in healthcare, education, or homes.
            """
        },
        {
            "icon": "💸",
            "title": "Cost and Manufacturing",
            "content": """
            Current humanoid robots are expensive prototypes with limited production:
            <ul>
                <li>High R&D costs (hardware + software).</li>
                <li>Difficulty scaling from lab to factory.</li>
            </ul>
            Reducing cost while maintaining performance is a major issue toward mass adoption.
            """
        },
        {
            "icon": "🤝",
            "title": "Social Acceptance and Trust",
            "content": """
            Even if a humanoid robot is technically advanced, it won’t be effective unless <strong>people feel comfortable interacting with it</strong>.
            <ul>
                <li><strong>Appearance</strong>: Too human-like can scare people, while too robotic may feel unrelatable.</li>
                <li><strong>Behavior</strong>: Must be predictable, respectful, and non-invasive.</li>
                <li><strong>Trust</strong>: Especially important in healthcare, childcare, or elderly care.</li>
            </ul>
            <strong>Designing for empathy, transparency, and reliability is as important as engineering.</strong>
            """
        }
    ]


    # Render HTML blocks
    html_block_info_robots=""
    for idx, c in enumerate(challenges):
        html_block_info_robots += f"""
    <div class="challenge-block" id="challenge-{idx}"
        style="
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            background-color: #fff;
            padding: 1.5rem 2rem;
            margin: 1rem 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 1rem;
            color: #222;
            cursor: default;
        "
    >
        <h3 style="margin-bottom: 0.75rem; font-weight: 600; font-size: 1.3rem; display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.7rem;">{c['icon']}</span> {c['title']}
        </h3>
        <div style="color: #444; font-size: 1.5rem; line-height: 1.5;">
            {c['content']}
        </div>
    </div>
"""




    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSdrzUFnm5xeL1qE285aBtBawOI21HqpfjalVDAYe1L0EEGE1Q/viewform?usp=dialog"

    def qr_code_base64(url):
        img = qrcode.make(url)
        buf = BytesIO()
        img.save(buf)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{data}"

    qr_base64 = qr_code_base64(form_url)

    html_block_info_robots += f"""
<div class="challenge-block" id="challenge-qr"
     style="
       max-width: 320px;
       margin: 1.5rem auto;
       padding: 1.5rem 1rem;
       background: #ffffffcc;
       border-radius: 16px;
       box-shadow: 0 8px 16px rgba(0,0,0,0.12);
       text-align: center;
       font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       font-size: 1.1rem;
       color: #333;
       backdrop-filter: blur(8px);
     "
>
  <h3 style="font-weight: 700; font-size: 1.4rem; margin-bottom: 1rem; color: #0078D7;">
    📝 Participate in the poll
  </h3>
  <img src="{qr_base64}" width="200" alt="QR Code pour sondage"
       style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 1rem;">
</div>
"""



    js_switch_challenge="""
    <script>
    // === Challenge slideshow logic ===
    let challengeIndex = 0;
    const challengeItems = document.querySelectorAll(".challenge-block");
    const challengeDelay = 6000;

    function showChallenge(index) {
        challengeItems.forEach((el, i) => {
            el.style.display = (i === index) ? "block" : "none";
        });
    }

    function cycleChallenges() {
        if (challengeItems.length === 0) return;
        showChallenge(challengeIndex);
        challengeIndex = (challengeIndex + 1) % challengeItems.length;
        setTimeout(cycleChallenges, challengeDelay);
    }

    // Démarrage au chargement de la page
    window.addEventListener("load", () => {
        cycleChallenges();
    });
    </script>
    """
    html(f"""<div>{html_block_info_robots}</div>{js_switch_challenge}
         """, height=600)

    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_QhPDKKjQxkbVKkmwiu8jujseBms7anUs9Jd42Nj-0Wjnf2HnAZQVLa8WVcac5UrkB0jnap5KA0hk/pub?gid=867491542&single=true&output=csv"
    df_sond = pd.read_csv(url)

    st.subheader("   ", anchor = "poll")
    st.subheader("📊 Poll results ")

    counts = df_sond["What is your feeling about robots ?"].value_counts().reset_index()
    counts.columns = ["Feeling", "Count"]
    if "What is your feeling about robots ?" in df_sond.columns:
        fig_sond = px.bar(
            counts,
            x="Feeling",
            y="Count",
            color="Feeling",
            title="Feeling regarding robots",
            text="Count",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        st.plotly_chart(fig_sond, use_container_width=True)


    st.divider()

    # javascript to allow autoscroll
    display_time_seconds = 20  # Display time of each section

    js_code = f"""
            <script>
                // === Autoscroll ===
                const sectionIds = {json.dumps(anchor_ids)};
                let currentSectionIndex = 0;
                const displayTime = {display_time_seconds * 1000};
                const specialAnchor = "news";
                const specialAnchor2 = "warnings";
                const specialAnchor3 = "calvin";
                const specialAnchor4 = "percep-info";
                const specialDelay = 50000;

                function scrollToSection() {{
                    const targetId = sectionIds[currentSectionIndex];
                    const targetElement = window.parent.document.getElementById(targetId);
                    if (targetElement) {{
                        targetElement.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                    }}
                    const nextDelay = (targetId === specialAnchor || targetId === specialAnchor2 || targetId === specialAnchor3 || targetId === specialAnchor4) ? specialDelay : displayTime;
                    currentSectionIndex = (currentSectionIndex + 1) % sectionIds.length;
                    setTimeout(scrollToSection, nextDelay);
                }}
                setTimeout(scrollToSection, 500);</script>"""
    #html(f"""{js_code}""")