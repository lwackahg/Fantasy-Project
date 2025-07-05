"""
This file contains the Streamlit page for the Team Scouting feature.
"""

import streamlit as st
from modules.team_scouting.ui import display_team_scouting_page
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Team Scouting", layout="wide")

display_global_sidebar()

st.title(":blue[Team Scouting]")

# Ensure data is loaded before displaying the page
if 'combined_data' not in st.session_state or not st.session_state.get('data_ranges'):
    st.warning("Please load data from the main page before using this feature.")
    st.stop()

display_team_scouting_page(st.session_state.combined_data, st.session_state.data_ranges)
