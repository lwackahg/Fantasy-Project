"""
This module contains the global sidebar for the Fantasy Basketball Trade Analyzer.
"""

import streamlit as st
from pathlib import Path
from data_loader import csv_time

def display_global_sidebar():
    """Displays the global sidebar with navigation and high-level data status.

    League loading and management now live under the Admin Tools page; the
    sidebar only shows a read-only summary.
    """
    with st.sidebar:
        st.title(f":rainbow[Fantasy Trade Analyzer]")
        st.caption(f"Last Data Update: {csv_time()}")

        st.markdown("---")
        st.subheader("League Data")
        if st.session_state.get('data_loaded', False):
            st.success(f"Loaded: **{st.session_state.get('loaded_league_name', 'Default')}**")
        else:
            st.warning("Default dataset not found or failed to load.")
        st.caption("Manage and change leagues from the 🔐 Admin Tools page.")
        st.markdown("--- ")
        st.caption("CSV/DB data is updated regularly. Ping the commish if it looks stale.")
