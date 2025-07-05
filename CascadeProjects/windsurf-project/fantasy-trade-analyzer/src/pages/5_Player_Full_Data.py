"""
This file contains the Streamlit page for the Player Full Data feature.
"""

import streamlit as st
from pathlib import Path
from data_loader import load_draft_results
from modules.player_data.logic import merge_with_draft_results
from modules.player_data.ui import display_player_dataframe, display_player_metrics

from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Player Full Data", layout="wide")

display_global_sidebar()

st.title("Player Full Data")

# Check if data is loaded
if 'data_ranges' not in st.session_state or not st.session_state.data_ranges:
    st.warning("No data loaded. Please go to the main page and load a dataset.")
    st.stop()

# Get data from session state
data_ranges = st.session_state.get('data_ranges', {})

ranges = list(data_ranges.keys())
# Ensure there is a default selection that is safe
default_range_index = 0
if 'current_range' in st.session_state and st.session_state.current_range in ranges:
    default_range_index = ranges.index(st.session_state.current_range)

selected_range = st.selectbox("Select Time Range", ranges, index=default_range_index)

if selected_range:
    st.session_state.current_range = selected_range
    player_data = data_ranges[selected_range]

    # Construct absolute path for draft results
    project_root = Path(__file__).resolve().parent.parent.parent
    draft_results_path = project_root / 'data' / 'Fantrax-Draft-Results-Mr Squidward’s 69.csv'

    if draft_results_path.exists():
        # Load draft results
        draft_results = load_draft_results(draft_results_path)
        
        # Merge with draft results
        player_data_with_draft = merge_with_draft_results(player_data, draft_results)
        
        # Display data and metrics
        display_player_dataframe(player_data_with_draft)
        display_player_metrics(player_data)
    else:
        st.error(f"Draft results file not found at: {draft_results_path}")
        st.info("Displaying player data without draft information.")
        display_player_dataframe(player_data)
        display_player_metrics(player_data)
