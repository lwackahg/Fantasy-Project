"""
This module contains the global sidebar for the Fantasy Basketball Trade Analyzer.
"""

import streamlit as st
from pathlib import Path
from data_loader import csv_time
from modules.data_loader_ui.ui import display_data_loader_ui

def display_global_sidebar():
    """Displays the global sidebar with navigation and data loader."""
    with st.sidebar:
        st.title(f":rainbow[Fantasy Trade Analyzer]")
        st.caption(f"Last Data Update: {csv_time()}")

        st.markdown("---")

        # Display the data loader UI
        st.subheader("League Data")

        # Display status of the default data load
        if st.session_state.get('data_loaded', False):
            st.success(f"Loaded: **{st.session_state.get('loaded_league_name', 'Default')}**")
        else:
            st.warning("Default dataset not found or failed to load.")

        # Option to manually load a different dataset
        with st.expander("Load a different league"):
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
            display_data_loader_ui(data_dir)
        
        st.markdown("--- ")
        st.caption("CSV data is updated regularly. Please message me if you notice it's been too long.")
