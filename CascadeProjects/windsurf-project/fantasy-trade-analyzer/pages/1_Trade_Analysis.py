"""
This file contains the Streamlit page for the Trade Analysis feature.
"""
import streamlit as st
from modules.trade_analysis.ui import display_trade_analysis_page
from modules.trade_suggestions.trade_suggestions_ui_tab import display_trade_suggestions_tab
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Trade Analysis", layout="wide")

display_global_sidebar()

st.title(":violet[Trade Analysis & Suggestions]")

tab_analysis, tab_suggestions = st.tabs(["Trade Analysis", "Trade Suggestions"])

with tab_analysis:
    # Ensure data is loaded before displaying the trade analysis UI
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load a league dataset on the Home page before using the analyzer.")
        st.page_link("Home.py", label="Go to Home", icon="🏠")
    else:
        display_trade_analysis_page()

with tab_suggestions:
    display_trade_suggestions_tab()
