import streamlit as st
from pathlib import Path
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, MENUITEMS
from data_loader import csv_time, load_schedule_data
from logic.schedule_analysis import calculate_all_schedule_swaps
from modules.sidebar.ui import display_global_sidebar
from modules.legacy.data_loader_ui.ui import find_default_files_to_load, process_files_into_session_state

import pandas as pd


def main():
    """Main application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT, menu_items=MENUITEMS)

    # Consolidate session state initialization
    session_defaults = {
        'data_ranges': {},
        'combined_data': None,
        'current_range': None,
        'debug_manager': type('DebugManager', (), {'debug_mode': False, 'toggle_debug': lambda: None}),
        'trade_analyzer': None,
        'trade_analysis': None,
        'csv_timestamp': "CSV timestamp not available"
    }
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Define data directory path relative to the Home.py script location
    data_dir = Path(__file__).resolve().parent / "data"

    def load_default_data():
        """Load all default data (players and schedule) into session state."""
        try:
            with st.spinner("Loading default dataset..."):
                # Load player data
                player_files_found = False
                default_files, league_name = find_default_files_to_load(data_dir)
                if default_files:
                    process_files_into_session_state(default_files)
                    st.session_state.loaded_league_name = league_name
                    player_files_found = True
                
                # Load schedule data (canonical schedule25_26.csv)
                needs_schedule = (
                    'schedule_data' not in st.session_state
                    or st.session_state.schedule_data is None
                    or getattr(st.session_state.schedule_data, 'empty', True)
                )
                if needs_schedule:
                    schedule_df = load_schedule_data()
                    if schedule_df is not None and not schedule_df.empty:
                        st.session_state.schedule_data = schedule_df
                    else:
                        # Ensure it's an empty DataFrame if load fails
                        st.session_state.schedule_data = pd.DataFrame()

                # Only set the loaded flag if both were successful
                if player_files_found and ('schedule_data' in st.session_state and st.session_state.schedule_data is not None and not st.session_state.schedule_data.empty):
                    st.session_state.data_loaded = True
                    st.toast(f"Default dataset '{league_name}' loaded successfully!", icon="✅")
                else:
                    # This will ensure it tries again on reload if one part failed
                    if 'data_loaded' in st.session_state:
                        del st.session_state['data_loaded']
                    if not player_files_found:
                        st.warning("Default player CSV data not found in /data folder.")
                    if 'schedule_data' not in st.session_state or st.session_state.schedule_data is None or st.session_state.schedule_data.empty:
                        st.warning("Default schedule file not found in /data/schedule folder (expected 'schedule25_26.csv').")


        except Exception as e:
            st.error(f"A critical error occurred while loading default data: {e}")
            if 'data_loaded' in st.session_state:
                del st.session_state['data_loaded']

    # Auto-load default data on first run if nothing is loaded
    if 'data_loaded' not in st.session_state:
        load_default_data()

    st.title(f":rainbow[{PAGE_TITLE}]")
    
    # Display the global sidebar
    display_global_sidebar()

    # Display welcome message and instructions
    st.markdown("""
    ### :wave: Welcome to the Fantasy Basketball Trade Analyzer!

    This application is designed to help you make informed decisions in your fantasy basketball league. 
    Use the navigation sidebar to explore the different features available.

    **Getting Started:**
    1.  **Load Your Data:** The app will load the default league data for you. If you want to use a different league, use the data loader in the sidebar to select and load your league's CSV files.
    2.  **Analyze Trades:** Once your data is loaded, head over to the `Trade Analysis` page to evaluate potential trades.
    3.  **Explore Other Features:** Check out the other pages for more in-depth analysis of your league.
    """)

    # Display a warning if no data is loaded
    if not st.session_state.get('data_ranges'):
        st.warning("Load player data from the sidebar to begin analysis.")

if __name__ == "__main__":
    main()