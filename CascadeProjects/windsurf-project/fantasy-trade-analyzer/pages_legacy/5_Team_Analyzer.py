"""
This file contains the Streamlit page for the Team Analyzer feature.
"""

import streamlit as st
from modules.team_analyzer.ui import display_team_analyzer
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Team Analyzer", layout="wide")

# Ensure data is loaded before displaying the page
if not st.session_state.get('data_loaded', False):
    st.warning("Please load a league dataset on the Home page before using the analyzer.")
    st.page_link("Home.py", label="Go to Home", icon="🏠")
    st.stop()

display_global_sidebar()

st.title(":mag: Team Analyzer")

# Ensure data is loaded before displaying the page
if 'combined_data' not in st.session_state or not st.session_state.get('data_ranges'):
    st.warning("Please load data from the main page before using this feature.")
    st.stop()

display_team_analyzer()
