import streamlit as st
import pages as pg
from streamlit_navigation_bar import st_navbar
import os

st.set_page_config(
    layout="wide"
)


general_directory = os.path.dirname(os.path.abspath(__file__))
logo = os.path.join(general_directory, 'RGlogo.svg')
page = st_navbar(pages=["hide"],logo_path=logo,
                 styles={"nav": {
                     "background-color": "black",
                     "font-family": "Helvetica",
                     "height": "4.5rem",
                     "text-color": "white !important",
                     # This will push the logo to the left, and the group of links to the right
                     "justify-content": "left",
                     "align-items": "flex-standard",  # Vertically center content in the nav bar
                     "padding": "0 2rem"  # Add some padding on the sides
                 },
                     "active": {
                         "background-color": "black",
                         "color": "black !important"},
                     "span": {
                         "border-radius": "0.5rem",
                         "color": "black",
                         "margin": "0 0.125rem",
                         "padding": "0.4375rem 0.625rem",
                     },
                     "img": {
                         "padding-right": "0px",
                         "height": "140px",
                         "width": "140px",
                     },
                     "hover": {
                         "background-color": "black",
                         "color": "black",
                     }
                 }
                 )


if page == "Home" :
    pg.home()
