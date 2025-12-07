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
    # Legacy Trade Analysis engine requires a CSV/YTD league dataset (combined_data).
    combined = st.session_state.get("combined_data", None)
    has_trade_data = bool(getattr(combined, "empty", False) is False and combined is not None)
    if not has_trade_data:
        st.info(
            "The **Trade Analysis** engine (legacy CSV/YTD-based) needs a league dataset loaded "
            "via **Admin Tools → League Data / CSV Loader** (or the default loader on Home).\n\n"
            "You can still use the **Trade Suggestions** tab below, which runs directly from DB "
            "game-log data and does not require this step."
        )
    else:
        display_trade_analysis_page()

with tab_suggestions:
    display_trade_suggestions_tab()
