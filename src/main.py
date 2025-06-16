import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
from streamlit.components.v1 import html
import os
import json
from streamlit_scroll_navigation import scroll_navbar

st.set_page_config(
    layout="wide",
)

anchor_ids =["title","map-repartition","physical-prop", "dashboard1","dashboard2", "speaking-abilities", "autonomy-time","payload-time","speed-time","robots-produced"]
anchor_icons = ["info-circle", "lightbulb", "gear", "tag", "envelope", "moneybag"]

with st.sidebar:
    st.subheader("Data analysis")
    scroll_navbar(
        anchor_ids,
        anchor_labels=None, # Use anchor_ids as labels
        anchor_icons=anchor_icons)

# Retrieving the dataset
general_directory = os.path.dirname(os.path.abspath(__file__))

dataset_file=os.path.join(general_directory,"../data/humanoids_data.csv")
df = pd.read_csv(dataset_file, delimiter=";")

st.title(":bar_chart: Market Analysis of Humanoïd Robots", anchor="title")
st.write("This section is dedicated to explore the features for the recorded robots and their evolution.")

numerical_columns=df[["Cost(USD)","Weight (kg)","Height(cm)","Speed (m/s)","Autonomy (hour)","Total Degrees of Freedom (DOF)","Body Degrees of Freedom (DOF)","Hands Degrees of Freedom (DOF)","Two Hand Payload (kg)"]]

# Converting the right columns to numeric
for col in numerical_columns.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


country_dataset=os.path.join(general_directory,"..","data/country_co.csv")
country_data = pd.read_csv(country_dataset)

country_data = country_data.merge(df,how='right',left_on='Country',right_on='Country')
country_counts = country_data.groupby(["Country","Latitude","Longitude"]).size().reset_index(name="Company Count")

# Showing the repartition of inventors of robots
# The more a point is big the more the corresponding country has recently invented robots

st.subheader("World repartition of humanoid robots",anchor="map-repartition")
with st.container(border=True):
    fig = px.scatter_map(country_counts,
                         lat="Latitude", lon="Longitude",
                         size="Company Count",
                         hover_name="Country",
                         hover_data=["Company Count"],
                         color="Country",
                         size_max=50,
                         zoom=1,
                         width=800,
                         height=600)

    fig.update_layout(mapbox_style="carto-positron")

    st.plotly_chart(fig)


#Scatter plot of the robots based on their weight and height
with st.container(border=True):
    st.subheader("Physical properties based repartition of robots", anchor="physical-prop")
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
        title="Robot Specifications"
    )

    fig.update_layout(legend=dict(font=dict(size=10)))
    fig.update_layout(autosize=True)

    st.plotly_chart(fig)


    # Function to detect special robots according to a given feature
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
    tp=df.copy()
    if unique_robots.empty:
        st.info("None")
        return tp
    st.dataframe(unique_robots)

    tp[str(column)+"- spe"] = None
    tp.loc[mask,str(column)+"- spe"]= "Special"

    choice = st.radio(f"Keep or remove special robots in {column}?",
                      [":green[Keep]", ":red[Remove]"], index=None)
    if choice == ":red[Remove]":
        tp.loc[mask, column] = None
        st.success(f"special robots removed from **{column}**.")
    elif choice == ":green[Keep]":
        st.info("special robots kept unchanged.")
    else:
        st.warning("Please make a selection.")
    return tp

work_df=df.copy()
include_wheeled = True
if not include_wheeled :
    work_df=work_df[work_df["Mobility Type"]!="Wheeled"]



# Function to show the n-best robots in a given category
# The order of sort is also chosen by the dev
def show_top(df, column, ascending=False, n=7, unit="",info_sup=None):
    top_df = df.copy()
    top_df[column] = pd.to_numeric(top_df[column], errors="coerce")
    top_df = top_df.dropna(subset=[column])
    top_df = top_df.sort_values(by=column, ascending=ascending).head(n)

    for _, row in top_df.iterrows():
        if info_sup is not None:
            st.markdown(f"- **{row['Robot Name+A1:AB1']}** ({row['Company']}): {row[column]} {unit}. NB: "+row[info_sup])
        else :
            st.markdown(f"- **{row['Robot Name+A1:AB1']}** ({row['Company']}): {row[column]} {unit}")


def plot_mean_over_years(df,column):
    plot_df = df.dropna(subset=column)


    fig = px.scatter(
        plot_df,
        x="Year Unveiled",
        y=column,
        hover_name="Robot Name+A1:AB1",
        title=column+" of robots vs year unveiled",
        labels={column+"- spe": "Special robots"}
    )

    market_df_cleaned = plot_df.groupby("Year Unveiled")[[column]].mean().reset_index().dropna()
    fig.add_trace(px.line(market_df_cleaned, x="Year Unveiled", y=column).data[0].update(
        line=dict(color="blue", width=2)))

    fig.update_layout(
        xaxis_title="Year unveiled",
        yaxis_title=column,
        hovermode="closest",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

tab_ids=["b-auto","b-payload","b-money","b-speed","b-scales","b-dof"]
st.subheader("Mini Dashboard - part 1", anchor="dashboard1")
col3, col4 = st.columns(2)
# Mini dashboard on some interesting data on the recorded robots
with col3 :
    with st.container(border=True):

        # Board of the best robots in each selected category
        st.subheader(":trophy: Top Robots by Category")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Autonomy :battery:", "Payload :package:", "Cost :money_with_wings:", "Speed :woman-running:", "Weight :scales:", "DOF :wave:"
        ])

        with tab1:
            st.subheader(":battery: Best Autonomy", anchor="b-auto")
            show_top(work_df, "Autonomy (hour)", ascending=False, unit="hours")

        with tab2:
            st.subheader(":package: Best Payload", anchor="b-payload")
            show_top(work_df, "Two Hand Payload (kg)", ascending=False, unit="kg")

        with tab3:
            st.subheader(":money_with_wings: Cheapest Robots", anchor="b-money")
            show_top(work_df, "Cost(USD)", ascending=True, unit="USD")

        with tab4:
            st.subheader(":woman-running: Fastest Robots", anchor="b-speed")
            show_top(work_df, "Speed (m/s)", ascending=False, unit="m/s", info_sup="Mobility Type")

        with tab5:
            st.subheader(":scales: Lightest Robots", anchor="b-scales")
            show_top(work_df, "Weight (kg)", ascending=True, unit="kg")

        with tab6:
            st.subheader(":wave: Higher DOF", anchor="b-dof")
            show_top(work_df, "Total Degrees of Freedom (DOF)", ascending=False)


with col4 :
    with st.container(border=True):
        # Pie chart highlighting the proportion of use of each AI technology
        st.subheader(":computer: AI Techno Analysis :")
        ai_techno_count= work_df.groupby("AI Technology").size().reset_index(name="Count")
        fig1 = px.pie(
            ai_techno_count,
            values="Count",
            names="AI Technology"
        )
        st.write("Distribution of types of AI technology used")
        st.plotly_chart(fig1, use_container_width=True)

st.subheader("Mini Dashboard - part 2", anchor="dashboard2")
col5, col6 = st.columns(2)
with col5:
    with st.container(border=True):
        # Plot of the evolution of the cost of robots during the years
        st.subheader(":moneybag: Cost evolution : ")
        st.write("Study of average cost of robot based on the year they were unveiled.")
        plot_mean_over_years(work_df,"Cost(USD)")


with col6 :
    with st.container(border=True):
        # Regression line plotted between Lifting capacities and Total Degrees of Freedom
        st.subheader(":building_construction: Lifting capacities :")
        df_clean = work_df[["Robot Name+A1:AB1",
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
                line=dict(color="darkcyan", width=1)
            )
        )

        fig.update_layout(
            title="Two Hands Payload vs Total Degrees of Freedom",
            xaxis_title="Total Degrees of Freedom",
            yaxis_title="Two Hands Payload Payload"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.write("**:red[Red points] are :blue[Real Data] still, but are special robots too.**")

st.divider()
st.subheader("Speaking abilities", anchor="speaking-abilities")
st.write("How many robots actually are able to converse naturally ? :speaking_head_in_silhouette: ")
df["Converse naturally ?"]= df["Converse naturally ?"].fillna("no").replace("","no")
df_able_to_talk = df.groupby("Converse naturally ?").size().reset_index(name="Count")
fig_able_to_talk = px.pie(df_able_to_talk, values="Count",
                          names="Converse naturally ?",
                          title="Distribution of Humanoïd speaking ability")
st.plotly_chart(fig_able_to_talk, use_container_width=True)
st.divider()

# Some plot to analyse the evolution of features with time
st.write("This section presents analysis of the evolution of some features throughout the years.")

st.subheader("Evolution over time", anchor="evolution-over-time")
#Evolution of autonomy
with st.container(border=True):
    st.subheader(":battery: Autonomy evolution : ", anchor="autonomy-time")
    st.write("Study of average autonomy evolution of robots based on the year they were unveiled.")
    market_df=work_df.groupby("Year Unveiled")[["Autonomy (hour)"]].mean().reset_index()
    market_df_cleaned = market_df.dropna()
    figautonomy=px.line(
        market_df_cleaned,
        x="Year Unveiled",
        y="Autonomy (hour)",
    )
    st.plotly_chart(figautonomy, use_container_width=True)

#Evolution of payload
with st.container(border=True):
    st.subheader(":package: Payload evolution : ", anchor="payload-time")
    st.write("Study of average payload evolution of robots based on the year they were unveiled.")
    market_df = work_df.groupby("Year Unveiled")[["Two Hand Payload (kg)"]].mean().reset_index()
    market_df_cleaned = market_df.dropna()
    st.line_chart(market_df_cleaned, x="Year Unveiled", y="Two Hand Payload (kg)")

#Evolution of speed
with st.container(border=True):
    st.subheader(":running_woman: Speed evolution : ", anchor="speed-time")
    st.write("Study of average speed evolution of robots based on the year they were unveiled.")
    market_df = work_df.groupby("Year Unveiled")[["Speed (m/s)"]].mean().reset_index()
    market_df_cleaned = market_df.dropna()
    st.line_chart(market_df_cleaned, x="Year Unveiled", y="Speed (m/s)")




st.divider()
st.subheader("Robots currently produced", anchor="robots-produced")
# Retrieving robots currently produced
st.write(":mechanic: This section presents the robots that are currently produced around the world.")
df_prod=df[df["Status"]=="In Production"]
df_prod=df[df["Status"]=="In Production"]
for idx, robot in df_prod.iterrows():
    st.subheader(f"**{robot['Robot Name+A1:AB1']}**", anchor="robot-info"+str(idx))
    anchor_ids.append("robot-info"+str(idx))
    with st.container(border=True):
        subcol1,subcol2=st.columns(2)
        # Getting the right image
        with subcol1:
            image_path = os.path.join(general_directory,
                                      "../data/images/" + robot['Robot Name+A1:AB1']+ ".png")
            if os.path.exists(image_path):
                st.image(image_path, width=200)
            else:
                st.write("No image found")
            st.markdown(f"**{robot['Robot Name+A1:AB1']}**")

        # Plus some info on the robot
        with subcol2 :
            st.metric(label= "Quantity :", value= robot['Production Capacity (units/year)']+" units/year")


            categories = [
                "Height(cm)",
                "Weight (kg)",
                "Total Degrees of Freedom (DOF)",
                "Two Hand Payload (kg)",
                "Speed (m/s)",
                "Autonomy (hour)"
            ]

            values = []
            for cat in categories:
                try:
                    val = float(robot.get(cat))
                except:
                    val = np.nan

                col_min = df[cat].min()
                col_max = df[cat].max()

                if pd.isna(val) or pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
                    scaled_val = 0
                else:
                    scaled_val = (val - col_min) / (col_max - col_min)

                values.append(scaled_val)

            # Fermer le radar
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
# Histogram of the Primary Use-case of robots
# On all robots wheeled included
st.subheader("Frequency of primary Use-case of humanoïd robots", anchor= "primary-use-case")
df["Primary Use-Case"] = df["Primary Use-Case"].fillna("Unknown").astype(str)
frequency_primaryusecase=df["Primary Use-Case"].value_counts().reset_index(name="Count")
fig = px.bar(frequency_primaryusecase,
             x="Primary Use-Case",
             y="Count",
             )
fig.update_layout(xaxis_title="Primary Use-Case", yaxis_title="Number of Robots")
st.plotly_chart(fig, use_container_width=True)
st.divider()

# Creation of a pie chart of the main companies in the humanoïd robots market
# On all companies, wheeled robots included
distribution_companies = df.groupby("Company").size().reset_index(name="Count")
low_count = distribution_companies[distribution_companies["Count"] <= 1]["Count"].sum()
distribution_companies = distribution_companies[distribution_companies["Count"] > 1]

st.subheader("Main humanoïd robots creators :", anchor="humanoid-creators")
names = distribution_companies["Company"].unique()
for name in names:
    st.badge(name, color="blue")
st.divider()

st.subheader("Average robot cost by Primary Use-Case", anchor="cost-vs-primary-use-case")
# Plot of the average cost of robots by primary use-case combined with the standard deviation
grouped = work_df.groupby("Primary Use-Case")["Cost(USD)"].agg(["mean", "std"]).dropna().sort_values("mean", ascending=False).reset_index()
fig = px.bar(
    grouped,
    x="Primary Use-Case",
    y="mean",
    error_y="std",
    labels={"mean": "Average Cost (USD)", "Primary Use-Case": "Primary Use-Case"},
    title="Average Robot Cost by Primary Use-Case",
    color="Primary Use-Case",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

display_time_seconds = 5 # Le temps d'arrêt sur chaque section

# Convertir la liste Python en un tableau JSON JavaScript
js_sections_array = json.dumps(anchor_ids)

js_code = f"""
<script>
    const sectionIds = {js_sections_array}; // Tableau des IDs de sections
    let currentSectionIndex = 0;
    const displayTime = {display_time_seconds * 1000}; // Convertir en millisecondes

    function scrollToSection() {{
        const targetId = sectionIds[currentSectionIndex];
        // Streamlit encapsule son contenu dans un conteneur principal.
        // On doit cibler les éléments dans le DOM parent de l'iframe de st.html.
        const targetElement = window.parent.document.getElementById(targetId);

        if (targetElement) {{
            // Utiliser scrollIntoView sur l'élément lui-même pour un défilement doux
            targetElement.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            console.log(`Défilement vers #$targetId effectué.`);
        }} else {{
            console.warn(`Section #$targetId non trouvée.`);
            // Si une section n'est pas trouvée, passe à la suivante
        }}

        // Préparer la prochaine section (boucle)
        currentSectionIndex = (currentSectionIndex + 1) % sectionIds.length;
    }}

    // Démarre le défilement après un court délai pour laisser le rendu initial se faire
    // puis configure l'intervalle pour les défilements suivants.
    setTimeout(() => {{
        scrollToSection(); // Défile vers la première section immédiatement
        setInterval(scrollToSection, displayTime); // Puis défile toutes les 'displayTime'
    }}, 500); // Délai initial de 0.5 seconde pour assurer le chargement du DOM

    console.log("Autoscroll");
</script>
"""

# Injecte le code JavaScript dans la page
html(js_code, height=0)