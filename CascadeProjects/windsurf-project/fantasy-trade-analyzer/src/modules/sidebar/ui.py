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
        st.sidebar.title(":blue[Navigation]")
        st.sidebar.header(":rainbow[CSV Update Time]")
        st.sidebar.subheader(f":blue[{csv_time()}]")
        st.sidebar.markdown("---")
        
        # Display the data loader UI, now synced with central session state
        st.sidebar.subheader(":blue[Load League Data]")

        # Display status of the default data load
        if st.session_state.get('data_loaded', False):
            st.success("Default dataset loaded successfully.")
        else:
            st.warning("Default dataset not found or failed to load.")

        # Always provide an option to manually load a different dataset
        with st.expander("Load a different league dataset"):
            # Correct the path to point to the project's root /data directory
            data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
            display_data_loader_ui(data_dir)
        st.sidebar.markdown("---")
        
        # The main navigation is now handled by Streamlit's native multipage feature.
        # The pages are automatically discovered from the 'pages' directory.
        st.sidebar.markdown("---")
        st.sidebar.subheader(":orange[Note:]")  
        st.sidebar.write(":orange[The CSV data is updated regularly. Please message me if you notice it's been too long.]")
