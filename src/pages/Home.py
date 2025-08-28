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

    """
        Main page of the Streamlit app showing insights on humanoid robots.

        This function:
        - Loads and cleans the humanoid robot dataset.
        - Displays key statistics and averages for humanoid robots.
        - Shows latest news retrieved from an external source.
        - Provides a disclaimer about the data sources and reliability.
        - Visualizes geographic distribution and other relevant metrics of humanoid robots.
        - Displays graphics of financial data of some of the most important companies in the database

        The UI is built with Streamlit components and uses Plotly for charts.
    """

    # Loading the general path to access the wanted files
    general_directory = os.path.dirname(os.path.abspath(__file__))

    dataset_file=os.path.join(general_directory,"..","..","data/humanoid_data_cleaned.csv")

    # Cleaning the database of robots before use
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
    df = df.dropna(subset="Year Unveiled")
    df["Year Unveiled"] = df["Year Unveiled"].astype(int)
    df["Country"]=df["Country"].str.strip()
    df["Region"]=df["Region"].str.strip()

    # Retrieval of the path to the current directory
    general_directory = os.path.dirname(os.path.abspath(__file__))

    #list containing the ids of each section shown in the app
    anchor_ids =["title","Disclaimer", "news","calvin","means","map-repartition","company-repartition","physical-prop","s2", "top-robots-1","humanoid-creators","camera-type","percep-info","ai-tech","safety","s3","img-hum-flow","primary-use-case","robots-produced"]

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


    # News section:
    # - Fetches latest humanoid robotics news via web scraping
    # - Adds JS slideshow to cycle through news items automatically
    st.subheader("   ")
    st.subheader("   ")
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

    # Allowing the use of html in th streamlit app
    html(f"""<div>{html_blocks}</div>{js_code_news}
             """, height=700)

    # Calvin display section
    st.subheader("   ", "calvin")
    st.video("https://www.youtube.com/watch?v=YddS-aI097Q&t=4s", autoplay=True, loop=True)


    # Calculate and display average specifications for humanoid robots
    # separated by mobility type: wheeled vs. bipedal.
    # For each group, compute means of weight, height, payload,
    # degrees of freedom, speed, and autonomy.
    df_wheeled = df[df["Mobility Type"]=="wheeled"]
    weight_mean = round(df_wheeled["Weight (kg)"].mean(), 2)
    height_mean = round(df_wheeled["Height(cm)"].mean(), 2)
    payload_mean = round(df_wheeled["Two Hand Payload (kg)"].mean(), 2)
    dof_mean = round(df_wheeled["Total Degrees of Freedom (DOF)"].mean(), 2)
    speed_mean = round(df_wheeled["Speed (m/s)"].mean(), 2)
    autonomy_mean = round(df_wheeled["Autonomy (hour)"].mean(), 2)

    df_without_wheeled = df[df["Mobility Type"]=="bipedal"]
    weight_mean_without_wheeled = round(df_without_wheeled["Weight (kg)"].mean(), 2)
    height_mean_without_wheeled = round(df_without_wheeled["Height(cm)"].mean(), 2)
    payload_mean_without_wheeled = round(df_without_wheeled["Two Hand Payload (kg)"].mean(), 2)
    dof_mean_without_wheeled = round(df_without_wheeled["Total Degrees of Freedom (DOF)"].mean(), 2)
    speed_mean_without_wheeled = round(df_without_wheeled["Speed (m/s)"].mean(), 2)
    autonomy_mean_without_wheeled = round(df_without_wheeled["Autonomy (hour)"].mean(), 2)


    st.subheader("   ", anchor="means")
    st.title("What is the current market overview ?")
    st.subheader("   ")
    st.subheader("Essential Figures to Remember")

    with st.container(border=True):
        col1, col2, col3,col4 = st.columns(4)
        col1.markdown("**Bipedal humanoid robots**")
        col1.metric(label="Average weight in kg :scales:", value=weight_mean_without_wheeled)
        col1.metric(label="Average height in cm :straight_ruler:", value=height_mean_without_wheeled)
        col1.metric(label="Average payload in kg :mechanical_arm:", value=payload_mean_without_wheeled)
        col2.markdown("   ")
        col2.markdown("   ")
        col2.write("   ")
        col2.metric(label="Average total degrees of freedom :feather:", value=dof_mean_without_wheeled)
        col2.metric(label="Average speed in m/s :running_man:", value=speed_mean_without_wheeled)
        col2.metric(label="Average autonomy in hours :battery:", value=autonomy_mean_without_wheeled)
        col3.markdown("**Wheeled humanoid robots**")
        col3.metric(label="Average weight in kg :scales:", value=weight_mean)
        col3.metric(label="Average height in cm :straight_ruler:", value=height_mean)
        col3.metric(label="Average payload in kg :mechanical_arm:", value=payload_mean)
        col4.markdown("   ")
        col4.markdown("   ")
        col4.write("   ")
        col4.metric(label="Average total degrees of freedom :feather:", value=dof_mean)
        col4.metric(label="Average speed in m/s :running_man:", value=speed_mean)
        col4.metric(label="Average autonomy in hours :battery:", value=autonomy_mean)

    st.subheader("   ")
    st.subheader("   ")

    st.subheader("   ")
    st.divider()

    # Section dedicated to showing the number of robots unveiled by country
    # European country are grouped without russia

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

    country_dataset=os.path.join(general_directory,"..","../data/countries_co.csv")
    country_data = pd.read_csv(country_dataset)

    country_data = country_data.merge(df, how='right', left_on='Country', right_on='Country')
    country_counts = country_data.groupby(["Country", "Latitude", "Longitude"]).size().reset_index(name="Robot Count")
    europe_counts = {}

    for (idx, row) in country_counts.iterrows():
        if row["Country"] in europe:
            europe_counts[row["Country"]] = row["Robot Count"]

    country_counts = country_counts[~country_counts["Country"].isin(europe)]

    country_counts.loc[len(country_counts)] = ["Europe", 49, 14, sum(europe_counts.values())]

    st.subheader("   ",anchor="map-repartition")
    st.subheader("Global Distribution of Humanoid Robots Revealed by Country :earth_americas:")
    st.markdown("**:blue[The bigger the point, the more a country has revealed humanoid robots.]**")
    with st.container(border=True):
        fig = px.scatter_map(country_counts,
                             lat="Latitude", lon="Longitude",
                             size="Robot Count",
                             hover_name="Country",
                             hover_data=["Robot Count"],
                             color="Country",
                             size_max=50,
                             zoom=1,
                             width=1000,
                             height=550)

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
    st.subheader("Humanoid Robots Main Producers Around the World :earth_americas:: ")

    p1, p2 = st.columns(2)
    with p1 :
        with st.container(border=True):
            fig_rep = px.pie(
                distribution_companies_country,
                values="Robot Count",
                names="Country",
                title="Worldwide robot creators distribution"
            )
            fig_rep.update_layout(autosize=False,
                width=500,
                height=500)
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
            fig_europe.update_layout(autosize=False,
                                     width=500,
                                     height=500)
            st.plotly_chart(fig_europe, use_container_width=True)


    st.subheader("   ", anchor="physical-prop")
    st.subheader("Physical properties based repartition of robots")

    # Scatter plot of the robots based on their weight and height
    # The bigger the point, the heavier a robot can lift
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

        fig.update_layout(legend=dict(font=dict(size=10)))
        fig.update_layout(autosize=False, width=1000, height=400)
        st.write(
            "This scatter plot shows the relationship between robot **height** and **weight**. "
            "The **bubble size** indicates the **two-hand payload capacity** (in kg). "
            "It helps identify trends between physical dimensions and lifting capabilities."
        )

        st.plotly_chart(fig)

    # Function to show the n-best robots in a given category
    # The order of sort is also chosen by the dev
    def show_top(df, column, ascending=False, n=3, unit="", info_sup=None):
        """
            Display the top n robots based on a specified category.
            Parameters:
            -----------
            df : pandas.DataFrame
                DataFrame containing the robot data.
            column : str
                The name of the column to sort by (e.g., 'Weight (kg)', 'Speed (m/s)', etc.).
            ascending : bool, optional (default=False)
                Sort order. False for descending (top values first), True for ascending.
            n : int, optional (default=3)
                Number of top robots to display.
            unit : str, optional (default="")
                Unit string to append after the value (e.g., 'kg', 'm/s').
            info_sup : str or None, optional (default=None)
                Name of an additional column whose content will be shown as a note after the metric, if present.

            Behavior:
            ---------
            - Converts the specified column to numeric, ignoring non-convertible values.
            - Sorts the DataFrame based on the column and order specified.
            - Displays the top n rows as a markdown list with robot name, company, value with unit, and optional extra info.

            Example usage:
            --------------
            show_top(df, column="Weight (kg)", ascending=True, n=5, unit="kg", info_sup="Notes")
        """

        top_df = df.copy()
        top_df[column] = pd.to_numeric(top_df[column], errors="coerce")
        top_df = top_df.dropna(subset=[column])
        top_df = top_df.sort_values(by=column, ascending=ascending).head(n)

        for _, row in top_df.iterrows():
            if info_sup is not None and isinstance(row[info_sup],str):
                st.markdown(
                    f"- **{row['Robot Name']}** ({row['Company']}): {row[column]} {unit}. NB: " + row[info_sup])
            else:
                st.markdown(f"- **{row['Robot Name']}** ({row['Company']}): {row[column]} {unit}")

    st.subheader("   ", anchor="s2")
    st.title("Who are the key players ?")
    st.subheader("   ", anchor="top-robots-1")
    st.subheader(":trophy: Top Robots by Category")
    c1, c2, c3 = st.columns(3)

    # Board of the best robots in each selected category
    with c1:
        with st.container(border=True):
            st.subheader(":battery: Best Autonomy", anchor="b-auto")
            show_top(df, "Autonomy (hour)", ascending=False, unit="hours")

    with c2:
        with st.container(border=True):
            st.subheader(":package: Best Payload", anchor="b-payload")
            show_top(df, "Two Hand Payload (kg)", ascending=False, unit="kg")

    with c3:
        with st.container(border=True):
            st.subheader(":money_with_wings: Cheapest Robots", anchor="b-money")
            show_top(df, "Cost (USD)", ascending=True, unit="USD")

    c4, c5, c6 = st.columns(3)
    with c4:
        with st.container(border=True):
            st.subheader(":woman-running: Fastest Robots", anchor="b-speed")
            show_top(df, "Speed (m/s)", ascending=False, unit="m/s", info_sup="Mobility Type")

    with c5:
        with st.container(border=True):
            st.subheader(":scales: Lightest Robots", anchor="b-scales")
            show_top(df, "Weight (kg)", ascending=True, unit="kg")

    with c6:
        with st.container(border=True):
            st.subheader(":wave: Higher DOF", anchor="b-dof")
            show_top(df, "Total Degrees of Freedom (DOF)", ascending=False)

    # This section displays the top 5 companies that unveiled the higher number of robots
    distribution_companies = df.groupby("Company").size().reset_index(name="Count")

    distribution_companies = distribution_companies[distribution_companies["Count"] > 1].sort_values(by='Count',
                                                                                                     ascending=False)

    st.subheader("   ", anchor="humanoid-creators")
    st.subheader("Main humanoid robots creators :")
    names = distribution_companies["Company"].unique()[:5]
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

    # This section analyzes the distribution of perception sensor types used by companies.
    # It processes the 'Vision Sensors type' column, replacing missing values with "Not specified".
    # Then, it loads a mapping CSV file containing sensor groupings and related keywords to standardize sensor types.

    st.subheader("    ", anchor="camera-type")
    st.subheader("Repartition of perception sensors type used by companies")
    df_camera = df.copy()
    df_camera["Vision Sensors type"].fillna("Not specified", inplace=True)

    vision_sensors = df_camera["Vision Sensors type"].tolist()
    all_sensors = "; ".join(vision_sensors).replace("\n", " ").replace("\r", " ")
    separated_sensors = [s.strip() for s in all_sensors.split(";") if s.strip()]

    classified_sensors = separated_sensors
    sensor_counts = Counter(classified_sensors)

    df_camera_count = pd.DataFrame(sensor_counts.items(), columns=["Camera Type", "Count"])
    fig_camera = px.pie(df_camera_count, values="Count", names="Camera Type",
                        title="Distribution of Vision Sensor Types")
    fig_camera.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig_camera, use_container_width=True)

    # Section that gives more information on the different perception sensors generally used in robots
    st.subheader("  ", anchor="percep-info")
    st.subheader("Some information about perception sensors :")
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
            "source": "unfound",
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
            "source": "unfound",
            "img": "eagle_eye.png"

        }
    ]

    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode()
        return f"data:image/png;base64,{b64}"

    # Formatting in html and then adding javascript for dynamism
    html_block_perception_info = ""
    for idx, c in enumerate(percep_sensors):
        img_show=os.path.join(general_directory,"..","..","data/images",c['img'])
        if not os.path.exists(img_show):
            img_show=os.path.join(general_directory,"..","..","data/images","not_found.png")
        html_block_perception_info += f"""
        <div class="percep-block" id="percep-{idx}"
            style="
                border-radius: 12px;
                max-height: 600px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                background-color: #fff;
                padding: 1.5rem 2rem;
                margin: 1rem 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 1.5rem;
                color: #222;
                cursor: default;
                display: flex;
                gap: 2rem;
                align-items: center;
                justify-content: space-between;
            "
            >
            <!-- Image -->
                <div style="flex: 0 0 300px; max-width: 300px;">
                    <img src="{img_to_base64(img_show)}"
                        style="
                            width: 100%;
                            max-height: 300px;
                            border-radius: 16px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                            display: block;
                            object-fit: cover;
                        "
                    />
                </div>
                <!-- Text Content -->
                <div style="flex: 1;">
                    <h3 style="margin-bottom: 0.75rem; font-weight: 600; font-size: 1.7rem; display: flex; align-items: center; gap: 0.5rem;">
                        {c['title']}
                    </h3>
                    <div style="color: #444; font-size: 1.5rem; line-height: 1.5;">
                        {c['content']}
                    </div>
                    <div style="color: grey; font-size: 1rem; line-height: 1.5;">
                        Source : {c['source']}
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

    html(f"""{html_block_perception_info}{js_switch_percep}""",height=600)


    # This section shows the distribution of AI technology used in humanoid robots (reinforcement learning, deep learning, ...
    st.subheader("  ", anchor="ai-tech")
    st.subheader(":computer: AI Analysis :")
    col3,col4=st.columns(2)
    with col3:
        with st.container(border=True):
            st.subheader("Repartition of AI technologies used in the robots :")

            # Cleaning the specific column
            df_temp = df.copy()
            df_temp["AI Technology used"].fillna("Not specified", inplace=True)

            value_list = df_temp["AI Technology used"].tolist()
            all_values = "; ".join(value_list).replace("\n", " ").replace("\r", " ")

            for sep in [" and ", " & ", "/", ","]:
                all_values = all_values.replace(sep, ";")

            cleaned_values = [s.strip() for s in all_values.split(";") if s.strip()]

            value_counts = Counter(cleaned_values)
            df_counts = pd.DataFrame(value_counts.items(), columns=["AI Technology used", "Count"])

            total = df_counts["Count"].sum()
            df_counts["Percentage"] = df_counts["Count"] / total * 100

            # We ignore AI technologies that are negligible
            df_counts["AI Technology used"] = df_counts.apply(
                lambda row: row["AI Technology used"] if row["Percentage"] >= 3 else "Others",
                axis=1
            )

            df_counts = df_counts.groupby("AI Technology used", as_index=False)["Count"].sum()

            fig = px.pie(df_counts, values="Count", names="AI Technology used", title="Distribution of AI Technologies")
            fig.update_layout(autosize=False, width=500, height=450)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Distribution of robots according to their ability to speak naturally or not
        with st.container(border=True):
            st.subheader(
                "How many robots actually are able to converse naturally? :speaking_head_in_silhouette:")

            df["Can the robot converse naturally?"] = df["Can the robot converse naturally?"].fillna("no").replace("",
                                                                                                                   "no")
            df_talk = df.groupby("Can the robot converse naturally?").size().reset_index(name="Count")

            fig_talk = px.pie(df_talk, values="Count", names="Can the robot converse naturally?")
            fig_talk.update_layout(autosize=False, width=500, height=450)
            st.plotly_chart(fig_talk, use_container_width=True)

    # Function that creates for each elem of "from_col" a new column with header "keyword"
    def derive_spe_columns(df, from_col, keyword):
        pattern = fr"\b{re.escape(keyword)}\b"
        df[keyword] = df[from_col].str.contains(pattern, case=False, na=False).astype(int)
        return df

    # Distribution of the safety features used in the humanoid robots
    st.subheader("  ", anchor="safety")
    st.subheader("🛡️ Different Security Standards Implemented")
    with st.container(border=True):

        st.subheader("Safety features used in the robots")

        df_temp = df.copy()
        df_temp["Safety Features"].fillna("not specified", inplace=True)

        value_list = df_temp["Safety Features"].tolist()
        all_values = "; ".join(value_list).replace("\n", " ").replace("\r", " ")
        for sep in [" and ", " & ", "/", ","]:
            all_values = all_values.replace(sep, ";")

        cleaned_values = [s.strip().lower() for s in all_values.split(";") if s.strip()]

        value_counts = Counter(cleaned_values)
        df_counts = pd.DataFrame(value_counts.items(), columns=["safety feature", "count"])

        total = df_counts["count"].sum()
        df_counts["percentage"] = df_counts["count"] / total * 100

        df_counts = df_counts.sort_values(by="percentage", ascending=False).reset_index(drop=True)

        st.dataframe(df_counts.style.format({"percentage": "{:.1f}%"}))

    st.subheader("   ", anchor="s3")

    st.title("What do the humanoid robots actually do?")
    st.subheader("  ")

    st.subheader("  ", anchor="img-hum-flow")
    st.subheader("Understanding humanoid robots - ***:grey[Morgan Stanley]***")
    st.image(os.path.join(general_directory, "..", "..", "data/images/robot_representation.png"))

    # Pie chart of Primary Use-case of robots
    st.subheader("   ", anchor="primary-use-case")
    st.subheader("Primary use case distribution for humanoid robots")

    with st.container(border=True):
        df["Primary Use-Case"] = df["Primary Use-Case"].fillna("Not specified").astype(str)
        frequency_primaryusecase = df["Primary Use-Case"].value_counts().reset_index(name="Count")
        fig = px.pie(frequency_primaryusecase,
                     names="Primary Use-Case",
                     values="Count",
                     )

        fig.update_layout(autosize=False, width=1000, height=500, xaxis_title="Primary Use-Case",
                          yaxis_title="Number of Robots")
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
    df_prod=df[df["Status"]=="In Production"]

    st.markdown(f"""
    <div style="
        background-color: #f1f3f6;
        border-left: 6px solid #4b8bbe;
        padding: 20px 25px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'Segoe UI', sans-serif;
    ">

      <h3 style="margin-top: 0; color: #2c3e50;">🤖 Global Humanoid Robots in Production</h3>

      <p style="font-size: 16px; color: #333;">
        🔧 <strong>This section presents the robots that are currently produced around the world.</strong>
      </p>

      <p style="font-size: 16px; color: #333;">
        📊 Currently, there are 
        <span style="font-weight: bold; color: #e74c3c;">{len(df_prod)}</span> 
        humanoid robots listed as <strong>'In Production'</strong>.
      </p>

      <ul style="font-size: 15px; line-height: 1.6; color: #333; padding-left: 20px;">
        <li>🧠 Key specifications: mobility, autonomy, sensors…</li>
        <li>💵 Estimated cost (if available)</li>
        <li>🏭 Production capacity per year</li>
        <li>📈 Radar chart summarizing technical specs</li>
      </ul>

    </div>
    """, unsafe_allow_html=True)

    for idx, robot in df_prod.iterrows():
        st.subheader("  ", anchor="robot-info" + str(idx))

        # Ignoring robots with too many NaN values
        categories = [
            "Height(cm)", "Weight (kg)", "Total Degrees of Freedom (DOF)",
            "Two Hand Payload (kg)", "Speed (m/s)", "Autonomy (hour)"
        ]

        missing_count = sum(
            robot[col] is None or pd.isna(robot[col]) for col in categories
        )

        missing_threshold = 4

        if missing_count > missing_threshold:
            st.subheader(f"**{robot['Robot Name']}** - :grey[*{robot['Company']}*] - not enough information")
        else:
            st.subheader(f"**{robot['Robot Name']}** - :grey[*{robot['Company']}*]")

            with st.container(border=True):
                anchor_ids.append("robot-info" + str(idx))
                subcol1, subcol2 = st.columns(2)

                with subcol1:
                    image_path = os.path.join(general_directory, "..",
                                              "../data/images/" + robot['Robot Name'] + ".png")
                    if os.path.exists(image_path):
                        st.image(image_path, width=115)
                    else:
                        st.write("No image found")

                    st.write("**SPECS INFO** :wrench::")
                    for col in categories:
                        if robot[col] is not None and pd.notna(robot[col]):
                            st.write(f"- **{col}:** {robot[col]}")

                with subcol2:
                    cost_robot = robot['Cost (USD)']
                    if cost_robot is not None and pd.notna(cost_robot):
                        st.metric(label="Cost :heavy_dollar_sign: :", value=cost_robot)
                    else:
                        st.metric(label="Cost :heavy_dollar_sign: :", value="Not found")

                    prod_cap = robot['Production Capacity (units per year)']
                    if prod_cap is not None and pd.notna(prod_cap):
                        st.metric(label="Quantity :", value=prod_cap + " units/year")
                    else:
                        st.metric(label="Quantity :", value="Not found")

                    # Radar chart
                    values = []
                    for cat in categories:
                        try:
                            val = float(robot.get(cat))
                        except:
                            val = np.nan
                        values.append(min_max_scale(df, cat, val))

                    values += values[:1]
                    categories += categories[:1]

                    fig = go.Figure(
                        data=[
                            go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill='toself',
                                name=robot['Robot Name']
                            )
                        ],
                        layout=go.Layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1])
                            ),
                            showlegend=False,
                            title="Radar Chart Technical Specs 🔧",
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    anchor_ids.extend(["s4","country-cost-comp","tot-fundraise","market-description","val-share"])

    st.subheader("    ")
    st.title("What is the current economic situation?", anchor="s4")
    st.subheader("    ")
    st.subheader("   ", anchor="country-cost-comp")

    # Bar plot of the price of humanoid robots according to the region they are from
    st.subheader(":dollar: Average Robot Cost by Country :")

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
        text=df_country_robot_cost_exp_europe["Average Cost (USD)"].round(2)  # Add rounded cost as text
    )

    fig_robot_cost.update_traces(textposition='outside')  # Puts text above the bars
    fig_robot_cost.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_robot_cost, use_container_width=True)
    st.divider()


    st.subheader("   ", anchor="tot-fundraise")
    st.subheader("📊 Total Fundraising in Humanoid Robotics (2018+)")
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
        text=df_total["Total Fundraising"].round(2)  # Add rounded cost as text
    )

    fig.update_layout(
        annotations=[
            dict(
                x=2025,
                y=df_total.loc[df_total["Year"] == 2025, "Total Fundraising"].values[0],
                text="Data only up to mid-2025",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color="red", size=12)
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
                font=dict(color="red", size=12),
                arrowcolor="black"
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)



    st.subheader(" ", anchor="market-description")
    st.subheader("Companies' Market Valuations in Humanoid Robotics (2025)")

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
            autosize=False,
            height=500,
            xaxis_title="Company",
            yaxis_title="Cumulative Fundraising (USD)",
            uniformtext_minsize=8,
            uniformtext_mode='hide'
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
            height=450,
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

    df_companies = pd.read_csv(os.path.join(general_directory, "../../data/companies_data.csv"))

    import locale


    locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')

    st.subheader("  ", anchor="global-numbers")
    st.subheader("  ")
    st.subheader("  ")
    st.subheader("📊 Market in numbers")
    anchor_ids.append("global-numbers")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sum of funding raised in humanoid robotics world**")
        total_funding = df_companies["Total Funding (USD)"].sum()
        st.metric(
            label="Total Funding",
            value=locale.format_string("%d", total_funding, grouping=True) + " $"
        )

    with col2:
        st.markdown(
            "*:grey[Sum of the valuation of all companies in the database]*  \n"
        )
        market_val = df_companies["Market Capitalization (USD)"].sum()
        st.metric(
            label="Market Valuation",
            value=locale.format_string("%d", market_val, grouping=True) + " $"
        )

    # Optional: add some spacing after the section
    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    anchor_ids.append("company-clients")
    st.subheader("   ",anchor="company-clients")
    st.subheader("   ")
    st.subheader("Top 5 higher market valuation companies' clients")
    st.subheader("   ")
    st.subheader("   ")
    top5 = df_companies.sort_values(by="Market Capitalization (USD)", ascending=False).head(5)
    for index, row in top5.iterrows():
        st.subheader("   ", anchor="clients-"+row["Company Name"])
        anchor_ids.append("clients-"+row["Company Name"])
        st.subheader(f"{row["Company Name"]} (Current market valuation: {row['Market Capitalization (USD)']} $)")
        clients = row["Partner Companies in automobile world"]

        if isinstance(clients, str):
            clients_list = clients.strip("[]").split(";")
            clients_list = [c.strip() for c in clients_list]
        else:
            clients_list = clients

        st.write("**Partner Companies in automobile world**")
        st.markdown("\n".join(f"- {client}" for client in clients_list))
        st.subheader("   ")
        st.subheader("   ")
        st.write("---")

    st.subheader("   ")
    st.subheader("   ")

    # Page title
    anchor_ids.extend(["warnings","poll"])
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

    sheet_url = "https://docs.google.com/document/d/1PSAj3k5GzgEFlDZVzpRXTalovhYxtbegK4NRkHYIS2M/edit?usp=sharing"

    st.subheader("   ", anchor="sources")
    st.subheader("   ")
    st.subheader("📄 Sources Used")

    anchor_ids.append("sources")
    # Generate the QR code
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(sheet_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    col1, col2, col3 = st.columns(3)
    with col2 :
        # Display the QR code
        st.image(byte_im, width=400)

    st.subheader("   ")
    st.subheader("   ")


    # javascript to allow autoscroll
    display_time_seconds = 2  # Display time of each section

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

