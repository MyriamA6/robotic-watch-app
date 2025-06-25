import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
from streamlit.components.v1 import html, iframe
import os
import json
import re

# Function to show the dashboard
def dashboard():

    # Retrieval of the path to the current directory
    general_directory = os.path.dirname(os.path.abspath(__file__))

    anchor_ids =["title","map-repartition","company-repartition","img-interv","physical-prop", "img-hum-flow", "means","top-robots-1","ai-tech","dashboard2","camera-type","company-vision", "autonomy-time","payload-time","speed-time","robots-produced"]

    # Retrieval of the dataset
    dataset_file=os.path.join(general_directory,"..","../data/updated_humanoid_data.csv")
    df = pd.read_csv(dataset_file)

    # Description of the content of the webpage
    st.title(":bar_chart: Market Analysis of humanoid Robots", anchor="title")
    st.divider()
    st.write("This page is dedicated to explore the features for the registered humanoid robots :robot_face: and their evolution."
             + "\n\nRankings :trophy:,  Average specs of prototypes :wrench:,  currently produced ones :mechanical_arm:,..."
             +"\n\n:information_source: Find many information below : ")
    st.divider()

    numerical_columns=df[["Cost(USD)","Weight (kg)","Height(cm)","Speed (m/s)","Autonomy (hour)","Total Degrees of Freedom (DOF)","Body Degrees of Freedom (DOF)","Hands Degrees of Freedom (DOF)","Two Hand Payload (kg)"]]

    # Converting the right columns to numeric
    for col in numerical_columns.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df=df.dropna(subset="Year Unveiled")
    df["Year Unveiled"]=df["Year Unveiled"].astype(int)

    country_dataset=os.path.join(general_directory,"..","../data/country_co.csv")
    country_data = pd.read_csv(country_dataset)

    country_data = country_data.merge(df,how='right',left_on='Country',right_on='Country')
    country_counts = country_data.groupby(["Country","Latitude","Longitude"]).size().reset_index(name="Company Count")


    # Showing the repartition of creators of robots
    # The bigger the point, the more the corresponding country has recently invented robots
    st.subheader("   ",anchor="map-repartition")
    st.subheader("World repartition of humanoid robots creators :earth_americas:")
    with st.container(border=True):
        fig = px.scatter_map(country_counts,
                             lat="Latitude", lon="Longitude",
                             size="Company Count",
                             hover_name="Country",
                             hover_data=["Company Count"],
                             color="Country",
                             size_max=50,
                             zoom=1,
                             width=1000,
                             height=600)

        fig.update_layout(mapbox_style="carto-positron")

        st.plotly_chart(fig)

    # Grouping countries that have not created many robots
    low_count_country = country_counts[country_counts["Company Count"] <= 1]["Company Count"].sum()
    distribution_companies_country = country_counts[country_counts["Company Count"] > 1]

    if low_count_country > 0:
        distribution_companies_country = pd.concat([
            distribution_companies_country,
            pd.DataFrame({"Country": ["Other countries"], "Company Count": [low_count_country]})
            # Others corresponds to companies having invented only one robot
        ], ignore_index=True)


    # This chart shows how companies building humanoid robots are distributed across different countries, based on their count.
    st.subheader("   ", anchor="company-repartition")
    st.subheader("Humanoid Robot Companies Around the World :earth_americas:: ")
    with st.container(border=True):
        fig_rep = px.pie(
            distribution_companies_country,
            values="Company Count",
            names="Country"
        )
        fig_rep.update_layout(autosize=False,
            width=1000,
            height=550)
        st.plotly_chart(fig_rep)


    # image from Morgan Stanley
    st.subheader("   ", anchor="img-interv")
    st.subheader("Humanoid Enablers - ***:grey[Morgan Stanley]***")
    st.image(os.path.join(general_directory,"..","..","data/images/humanoid_map.png"))

    st.subheader("   ", anchor="physical-prop")
    st.subheader("Physical properties based repartition of robots")

    # Scatter plot of the robots based on their weight and height

    with st.container(border=True):
        data_interactive=df.copy()
        data_interactive = data_interactive.dropna(subset=["Weight (kg)", "Height(cm)", "Two Hand Payload (kg)"])
        fig = px.scatter(
            data_interactive,
            x="Height(cm)",
            y="Weight (kg)",
            size="Two Hand Payload (kg)",
            color="Robot Name+A1:AB1",
            hover_name="Robot Name+A1:AB1",
            hover_data=["Robot Name+A1:AB1","Company","Country","Region","Year Unveiled", "Two Hand Payload (kg)"],
            title="The bigger the point, the heavier the robot can lift."
        )

        fig.update_layout(legend=dict(font=dict(size=10)))
        fig.update_layout(autosize=False, width=1000, height=500)

        st.plotly_chart(fig)



    # Function to show the n-best robots in a given category
    # The order of sort is also chosen by the dev
    def show_top(df, column, ascending=False, n=4, unit="",info_sup=None):
        top_df = df.copy()
        top_df[column] = pd.to_numeric(top_df[column], errors="coerce")
        top_df = top_df.dropna(subset=[column])
        top_df = top_df.sort_values(by=column, ascending=ascending).head(n)

        for _, row in top_df.iterrows():
            if info_sup is not None:
                st.markdown(f"- **{row['Robot Name+A1:AB1']}** ({row['Company']}): {row[column]} {unit}. NB: "+row[info_sup])
            else :
                st.markdown(f"- **{row['Robot Name+A1:AB1']}** ({row['Company']}): {row[column]} {unit}")



    # Print of the mean over years of a given column
    def plot_mean_over_years(df,column):
        plot_df = df.dropna(subset=column)

        market_df_cleaned = plot_df.groupby("Year Unveiled")[[column]].mean().reset_index().dropna()

        fig = px.line(
            market_df_cleaned,
            x="Year Unveiled",
            y=column,
            title=column+" of robots vs year unveiled",
            markers=True,
            text=column
        )

        fig.update_layout(autosize=False,width=1000, height=500)
        fig.update_traces(textposition="top center")

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("  ",anchor="img-hum-flow")
    st.subheader("Understanding humanoid robots - ***:grey[Morgan Stanley]***")
    st.image(os.path.join(general_directory,"..","..","data/images/robot_representation.png"))

    # Mean of several data
    weight_mean = round(df["Weight (kg)"].mean(), 2)
    height_mean = round(df["Height(cm)"].mean(), 2)
    payload_mean = round(df["Two Hand Payload (kg)"].mean(), 2)
    dof_mean = round(df["Total Degrees of Freedom (DOF)"].mean(), 2)

    st.subheader("   ", anchor = "means")
    st.subheader("A recap of some useful data")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col1.metric(label="Average weight in kg :scales:", value=weight_mean)
        col2.metric(label="Average height in cm :straight_ruler:", value=height_mean)
        col3.metric(label="Average payload in kg :package:", value=payload_mean)
        col4.metric(label="Average total degrees of freedom :feather:", value=dof_mean)


    st.subheader("   ",anchor="top-robots-1")
    st.subheader(":trophy: Top Robots by Category")
    c1,c2,c3 = st.columns(3)

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
            show_top(df, "Cost(USD)", ascending=True, unit="USD")

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

    st.subheader("  ", anchor="ai-tech")
    st.subheader(":computer: AI Analysis :")
    col3,col4=st.columns(2)
    with col3:

        with st.container(border=True):
            st.subheader("Repartition of AI technologies used in the robots :")
            # Pie chart highlighting the proportion of use of each AI technology
            ai_techno_count= df.groupby("AI Technology").size().reset_index(name="Count")
            fig1 = px.pie(
                    ai_techno_count,
                    values="Count",
                    names="AI Technology"
            )
            fig1.update_layout(autosize=False,width=500, height=450)
            st.plotly_chart(fig1, use_container_width=True)

    with col4 :

        with st.container(border=True):
            st.subheader("How many robots actually are able to converse naturally ? :speaking_head_in_silhouette:")
            df["Converse naturally ?"] = df["Converse naturally ?"].fillna("no").replace("", "no")
            df_able_to_talk = df.groupby("Converse naturally ?").size().reset_index(name="Count")
            fig_able_to_talk = px.pie(df_able_to_talk, values="Count",
                                      names="Converse naturally ?")
            fig_able_to_talk.update_layout(autosize=False,width=500, height=450)

            st.plotly_chart(fig_able_to_talk, use_container_width=True)




    st.subheader("  ", anchor="dashboard2")
    st.subheader("Mini Dashboard")
    col5, col6 = st.columns(2)
    with col5:
        with st.container(border=True):
            # Plot of the evolution of the cost of robots during the years
            st.subheader(":moneybag: Cost evolution : ")
            plot_mean_over_years(df,"Cost(USD)")


    with col6 :
        with st.container(border=True):
            # Regression line plotted between Lifting capacities and Total Degrees of Freedom
            st.subheader(":building_construction: Lifting capacities :")
            df_clean = df[["Robot Name+A1:AB1",
                                "Company",
                                "Total Degrees of Freedom (DOF)",
                                "Two Hand Payload (kg)",
                     ]].dropna(subset=["Total Degrees of Freedom (DOF)", "Two Hand Payload (kg)"])


            # Prediction of line based on the data
            model=LinearRegression()
            X=df_clean["Total Degrees of Freedom (DOF)"].values.reshape(-1,1)
            y=df_clean["Two Hand Payload (kg)"].values
            model.fit(X,y)
            df_clean["Predicted Payload (kg)"]=model.predict(X)


            # Plot of the actual data
            fig=go.Figure()
            fig.add_trace(go.Scatter(
                x=df_clean["Total Degrees of Freedom (DOF)"],
                y=df_clean["Two Hand Payload (kg)"],
                mode="markers",
                name="Real Data",
                text=df_clean["Robot Name+A1:AB1"]
            ))

            # Add of the regression line
            fig.add_trace(
                go.Scatter(
                    x=df_clean["Total Degrees of Freedom (DOF)"],
                    y=df_clean["Predicted Payload (kg)"],
                    mode="lines",
                    name="Regression Line",
                    line=dict(color="red", width=1)
                )
            )

            fig.update_layout(
                title="Two Hands Payload vs Total Degrees of Freedom",
                xaxis_title="Total Degrees of Freedom",
                yaxis_title="Two Hands Payload Payload",
                autosize= False,
                width=600,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)


    # Function that creates for each elem of "from_col" a new column with header "keyword"
    def derive_spe_columns(df,from_col,keyword):
        pattern = fr"\b{re.escape(keyword)}\b"
        df[keyword] = df[from_col].str.contains(pattern, case=False, na=False).astype(int)
        return df

    st.divider()

    st.subheader("    ", anchor="camera-type")
    st.subheader("Repartition of vision sensors type used by companies")
    df_camera = df.copy()
    df_camera= df_camera.dropna(subset=["Vision Sensors type"])

    # Preprocessing data to find all different types of cameras
    vision_sensors = list(df_camera["Vision Sensors type"].dropna())
    all_sensors = " , ".join(vision_sensors).replace("\n", " ").replace("\r", " ")
    separated_sensors = all_sensors.split(",")
    list_of_sensors = []

    # Retrieving different names of rgbd cameras to rename them
    same_cam_depth = ["rgbd", "rgb-d","depth camera","depth","intel", "depth cameras","rgb-d camera", "rgbd cameras"]
    for sensor in separated_sensors:
        tp=sensor.strip().lower()
        depth_cam =any([elem in tp for elem in same_cam_depth])
        if depth_cam:
            list_of_sensors.append("rgbd")
        else :
            list_of_sensors.append(tp)

    result = list(set(list_of_sensors))
    if '' in result:
        result.remove('')

    for camera in result:
        df_camera = derive_spe_columns(df_camera,"Vision Sensors type",camera)

    # Count the number of time each vision sensors appears for each robot
    camera_counts = {}
    for camera in result:
        if camera in df_camera.columns:
            camera_counts[camera] = df_camera[camera].sum()

    print(result)
    # Building a dataframe for the pie chart
    df_camera_count = pd.DataFrame({
        "Camera Type": list(camera_counts.keys()),
        "Count": list(camera_counts.values())
    })

    # Print of pie chart
    fig_camera = px.pie(df_camera_count, values="Count", names="Camera Type",
                        title="Distribution of Vision Sensor Types")
    fig_camera.update_layout(autosize=False, width=1000,height=600)

    st.plotly_chart(fig_camera, use_container_width=True)

    companies = df_camera["Company"].unique()

    df_table = pd.DataFrame(index=sorted(companies))
    st.subheader("   ", anchor="company-vision")
    st.subheader("Companies using each vision sensor")
    for camera in result:
        if camera in df_camera.columns:
            companies_using = df_camera[df_camera[camera] == 1]["Company"].unique()
            df_table[camera] = df_table.index.isin(companies_using).astype(int)

    df_table = df_table.replace({1: "yes", 0: ""})


    # function to color yes case in the dataframe
    def color_checkmark(val):
        if val == "yes":
            return 'background-color: #b2f2bb; color: black; font-weight: bold; text-align: center;'
        else:
            return 'background-color: lightgrey;'

    # Applying the style
    styled_df = df_table.style.applymap(color_checkmark)

    st.dataframe(styled_df, height=560)

    st.divider()

    # Some plot to analyse the evolution of features with time
    st.write("This section presents analysis of the evolution of some features throughout the years.")

    st.subheader("   ", anchor="autonomy-time")
    st.subheader(":battery: Autonomy evolution : ")
    # Line chart to show evolution of autonomy
    with st.container(border=True):
        st.write("Study of average autonomy evolution of robots based on the year they were unveiled.")
        plot_mean_over_years(df,"Autonomy (hour)")

    # Line chart to show evolution of payload
    st.subheader("   ", anchor="payload-time")
    st.subheader(":package: Payload evolution : ")
    with st.container(border=True):
        st.write("Study of average payload evolution of robots based on the year they were unveiled.")
        plot_mean_over_years(df, "Two Hand Payload (kg)")

    # Line chart to show evolution of speed
    st.subheader("   ", anchor="speed-time")
    st.subheader(":running_woman: Speed evolution : ")
    with st.container(border=True):
        st.write("Study of average speed evolution of robots based on the year they were unveiled.")
        plot_mean_over_years(df, "Speed (m/s)")


    st.subheader("   ", anchor="robots-produced")
    st.subheader("Robots currently produced", anchor="robots-produced")
    st.divider()


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
    st.write(":mechanic: This section presents the robots that are currently produced around the world.")
    df_prod=df[df["Status"]=="In Production"]

    for idx, robot in df_prod.iterrows():
        st.subheader("  ", anchor="robot-info" + str(idx))
        st.subheader(f"**{robot['Robot Name+A1:AB1']}** - :grey[*{robot['Company']}*]")
        with st.container(border=True):
            anchor_ids.append("robot-info" + str(idx))

            subcol1,subcol2=st.columns(2)


            # Retrieving the right image
            with subcol1:
                image_path = os.path.join(general_directory,"..",
                                          "../data/images/" + robot['Robot Name+A1:AB1']+ ".png")
                if os.path.exists(image_path):
                    st.image(image_path, width=115)
                else:
                    st.write("No image found")
                categories = ["Height(cm)",
                    "Weight (kg)",
                    "Total Degrees of Freedom (DOF)",
                    "Two Hand Payload (kg)",
                    "Speed (m/s)",
                    "Autonomy (hour)"]

                st.write("**SPECS INFO** :wrench::")
                for col in categories :
                    if robot[col] is not None:
                        st.write(f"- **{col}:** {robot[col]}")


            # Plus some info on the robot
            with subcol2 :
                cost_robot= robot['Cost(USD)']
                if cost_robot is not None and cost_robot != np.nan :
                    st.metric(label="Cost :heavy_dollar_sign: :", value=cost_robot)
                else :
                    st.metric(label="Cost :heavy_dollar_sign: :", value="Not found")

                prod_cap = robot['Production Capacity (units/year)']
                if prod_cap is not None and prod_cap != np.nan :
                    st.metric(label= "Quantity :", value= robot['Production Capacity (units/year)']+" units/year")
                else :
                    st.metric(label= "Quantity :", value="Not found")


                values = []
                for cat in categories:
                    try:
                        val = float(robot.get(cat))
                    except:
                        val = np.nan
                    values.append(min_max_scale(df,cat,val))

                values += values[:1]
                categories += categories[:1]

                fig = go.Figure(
                    data=[
                        go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=robot['Robot Name+A1:AB1']
                        )
                    ],
                    layout=go.Layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=False,
                        title="Radar Chart Technical Specs 🔧",
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    next_ids=["primary-use-case","humanoid-creators", "cost-vs-primary-use-case"]
    anchor_ids.extend(next_ids)

    # Histogram of Primary Use-case of robots
    st.subheader("   ", anchor= "primary-use-case")
    st.subheader("Frequency of primary use cases for humanoid robots")

    st.divider()

    df["Primary Use-Case"] = df["Primary Use-Case"].fillna("Unknown").astype(str)
    frequency_primaryusecase=df["Primary Use-Case"].value_counts().reset_index(name="Count")
    fig = px.pie(frequency_primaryusecase,
                 names="Primary Use-Case",
                 values="Count",
                 )

    fig.update_layout(autosize=False, width=1000,height=500,xaxis_title="Primary Use-Case", yaxis_title="Number of Robots")
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # Creation of a pie chart of the main companies in the humanoid robots market
    # On all companies, wheeled robots included

    distribution_companies = df.groupby("Company").size().reset_index(name="Count")
    low_count = distribution_companies[distribution_companies["Count"] <= 1]["Count"].sum()

    distribution_companies = distribution_companies[distribution_companies["Count"] > 1]


    st.subheader("   ", anchor="humanoid-creators")
    st.subheader("Main humanoid robots creators :")
    st.divider()
    names = distribution_companies["Company"].unique()
    for name in names:
        with st.container(border=True):
            cb1,cb2 = st.columns(2)
            with cb1:
                country=df[df["Company"] == name]["Country"].iloc[0]
                st.markdown(f"<p style='font-size:24px; font-weight:bold; color:rgb(46, 134, 193)'>{name} - Location : {country} </p>"
            , unsafe_allow_html=True)
            with cb2 :
                nb_robots=distribution_companies[distribution_companies["Company"]==name]["Count"].iloc[0]
                st.markdown("**Has recently created **"+str(nb_robots)+"** robots.** ")

    st.divider()

    st.subheader("   ", anchor="cost-vs-primary-use-case")

    st.subheader(":money_with_wings: Average robot cost per primary use case")
    st.divider()

    # Plot of the average cost of robots by primary use-case combined with the standard deviation
    grouped = df.groupby("Primary Use-Case")["Cost(USD)"].agg(["mean", "std"]).dropna().sort_values("mean", ascending=False).reset_index()
    fig = px.bar(
        grouped,
        x="Primary Use-Case",
        y="mean",
        error_y="std",
        labels={"mean": "Average Cost (USD)", "Primary Use-Case": "Primary Use-Case"},
        color="Primary Use-Case",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("   ", anchor = "more-analysis")
    st.subheader(":heavy_plus_sign: More analysis ")

    # function to preprocess data of a given column to plot the pie chart of the proportion of the different elements it contains
    def process_multivalue_column(df, column_name,personalised_comment=""):
        df = df.copy()
        df[column_name] = df[column_name].fillna(personalised_comment)

        all_values = " , ".join(df[column_name]).replace("\n", " ").replace("\r", " ")
        separated_values = [val.strip().lower() for val in all_values.split(",") if val.strip()]

        unique_values = sorted(set(separated_values))

        for val in unique_values:
            df = derive_spe_columns(df, column_name, val)

        value_counts = {val: df[val].sum() for val in unique_values if val in df.columns}

        return pd.DataFrame({
            column_name: list(value_counts.keys()),
            "Count": list(value_counts.values())
        })

    col8, col9 = st.columns(2)

    with col8:
        with st.container(border=True):
            st.subheader("🛡️ Different Security Standards Implemented")
            df_safety = process_multivalue_column(df, "Safety Features", personalised_comment= "Not specified")
            fig_safe = px.pie(df_safety, values="Count", names="Safety Features",
                              title="Distribution of Safety Features")
            fig_safe.update_layout(autosize=False, width=500, height =460)
            st.plotly_chart(fig_safe)


    anchor_ids.extend(["more-analysis","news"])

    st.subheader("   ", anchor="news")

    st.subheader(":newspaper: News in the humanoid world of the day   -   ***:grey[humanoidsdaily]***")
    from .website_retrieval import website_retrieval

    news = website_retrieval()
    html_blocks = ""
    for idx, (i, item) in enumerate(news.items()):
        html_blocks += f"""
                    <div class="news-item" id="news-{i}" 
                    style="border: 2px outset lightgrey; border-radius: 20px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
                    background-color: rgb(252, 252, 252); 
                    text-align: center; display:block; margin-top: 1rem; margin-bottom: 1rem; text-align:center; max-height=none; overflow :visible; font-family: Trebuchet MS, sans-serif;">
                    <h2 style = " margin-left:30px ; margin-right : 30px; ">{item['title']}</h2>
                    <p style = " margin-left:30px ; margin-right : 30px; ">{item['summary']}</p>
                    <p style="color:grey">{item['date']}</p>
                    <img src="{item['image']}" style="width: 90%; height:400px; object-fit: contain; border-radius: 20px; margin-bottom : 20px" />

                    </div>
                    """


    # javascript to allow autoscroll
    display_time_seconds = 2 # Display time of each section

    js_code = f"""
                <script>
                    // === Autoscroll logic ===
                    const sectionIds = {json.dumps(anchor_ids)};
                    let currentSectionIndex = 0;
                    const displayTime = {display_time_seconds * 1000};
                    const specialAnchor = "news";
                    const specialDelay = 60000;

                    function scrollToSection() {{
                        const targetId = sectionIds[currentSectionIndex];
                        const targetElement = window.parent.document.getElementById(targetId);

                        if (targetElement) {{
                            targetElement.scrollIntoView({{ behavior: 'smooth', block: 'start' }});

                        }}

                        const nextDelay = targetId === specialAnchor ? specialDelay : displayTime;
                        currentSectionIndex = (currentSectionIndex + 1) % sectionIds.length;
                        setTimeout(scrollToSection, nextDelay);
                    }}

                    setTimeout(scrollToSection, 500);


                    // === News slideshow logic ===
                    let newsIndex = 0;
                    const newsItems = document.querySelectorAll(".news-item");
                    const newsDelay = 7000; // each slide visible for 7 seconds
                    const lastNewsId = "news-{len(news)}"; 

                    function showNews(index) {{
                        newsItems.forEach((el, i) => {{
                            el.style.display = (i === index) ? "block" : "none";
                        }});
                    }}

                    function cycleNews() {{
                        showNews(newsIndex);
                        const currentId = newsItems[newsIndex]?.id;
                        const wait = newsDelay;
                        newsIndex = (newsIndex + 1) % newsItems.length;
                        setTimeout(cycleNews, wait);
                    }}

                    window.addEventListener("load", () => {{
                        cycleNews();
                    }});
                </script>
                """

    html(f"""<div>{html_blocks}</div>
         """, height=650)
    st.divider()














8