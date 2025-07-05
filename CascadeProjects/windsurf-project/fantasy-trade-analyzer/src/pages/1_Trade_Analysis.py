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
if 'data_ranges' not in st.session_state or not st.session_state.data_ranges:
    st.warning("Please load data from the main page before using the analyzer.")
    st.stop()

display_trade_analysis_page()
