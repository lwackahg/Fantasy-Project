"""
This file contains the Streamlit page for the Player Full Data feature.
"""
import streamlit as st
from pathlib import Path
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, MENUITEMS
from data_loader import load_draft_results
from modules.player_data.logic import merge_with_draft_results
from modules.player_data.ui import display_player_dataframe, display_player_metrics
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title=f"Player Full Data - {PAGE_TITLE}", page_icon=PAGE_ICON, layout=LAYOUT, menu_items=MENUITEMS)

# Ensure data is loaded before displaying the page
if not st.session_state.get('data_loaded', False):
    st.warning("Please load a league dataset on the Home page before using the analyzer.")
    st.page_link("Home.py", label="Go to Home", icon="🏠")
    st.stop()

display_global_sidebar()

st.title("Player Full Data")

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

    # Construct dynamic path for draft results
    project_root = Path(__file__).resolve().parent.parent
    league_name = st.session_state.get("league_name", "")
    draft_results_filename = f"Fantrax-Draft-Results-{league_name}.csv"
    draft_results_path = project_root / 'data' / draft_results_filename

    if draft_results_path.exists():
        draft_results = load_draft_results(draft_results_path)
        player_data_with_draft = merge_with_draft_results(player_data, draft_results)
        display_player_dataframe(player_data_with_draft)
        display_player_metrics(player_data)
    else:
        st.error(f"Draft results file not found at: {draft_results_path}")
        st.info("Displaying player data without draft information.")
        display_player_dataframe(player_data)
        display_player_metrics(player_data)

