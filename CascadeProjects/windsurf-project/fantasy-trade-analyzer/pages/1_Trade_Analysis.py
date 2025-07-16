"""
This file contains the Streamlit page for the Trade Analysis feature.
"""
import streamlit as st
from modules.trade_analysis.ui import display_trade_analysis_page
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Trade Analysis", layout="wide")

display_global_sidebar()

st.title(":violet[Trade Analysis]")

# Ensure data is loaded before displaying the page
if not st.session_state.get('data_loaded', False):
    st.warning("Please load a league dataset on the Home page before using the analyzer.")
    st.page_link("Home.py", label="Go to Home", icon="🏠")
    st.stop()

display_trade_analysis_page()
