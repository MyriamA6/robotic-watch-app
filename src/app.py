"""
Main page of the Streamlit application.

This script sets up the app layout and navigation bar, loads a logo,
and displays the selected page based on the navigation bar choice.

- Uses `streamlit_navigation_bar` for navigation with a custom logo.
- Defines CSS styles for the navigation bar.
- Loads and displays pages defined in the `pages` module depending on the selected page.

Available pages:
- "Home": the home page (function `home()` in `pages`)
- "web_scrap_file": web scraping page (function `webscrap()` in `pages`)
"""

import streamlit as st
import pages as pg
from streamlit_navigation_bar import st_navbar
import os

st.set_page_config(
    layout="wide"
)

general_directory = os.path.dirname(os.path.abspath(__file__))
logo = os.path.join(general_directory, 'RGlogo.svg')
page = st_navbar(pages=["web_scrap_file"],logo_path=logo,
                 styles={"nav": {
                     "background-color": "black",
                     "font-family": "Helvetica",
                     "height": "4.5rem",
                     "text-color": "black !important",
                     "justify-content": "left",
                     "align-items": "flex-standard",
                     "padding": "0 2rem"
                 },
                     "active": {
                         "background-color": "grey",
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
                         "background-color": "lightgrey",
                         "color": "black",
                     }
                 }
                 )


if page == "Home" :
    pg.home()
if page == "web_scrap_file" :
    pg.webscrap()
