import streamlit as st
import os 
import time 
from pathlib import Path
from data_loader import load_data, csv_time, load_draft_results
from trade_analysis import display_trade_analysis_page
from trade_options import TradeAnalyzer
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, MENUITEMS
from player_data_display import (
    display_player_data, 
    display_metrics, 
    
    display_team_scouting,
)
from schedule_display import display_schedule_page
import pandas as pd

def main():
    """Main application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT, menu_items=MENUITEMS)
    # Load data

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

    st.title(f":rainbow[{PAGE_TITLE}]")
    
    # Setup sidebar
    with st.sidebar:
        if st.checkbox(":green[Enable Debug Mode]", value=st.session_state.debug_manager.debug_mode):
            st.session_state.debug_manager.toggle_debug()
        
        st.sidebar.title(":blue[Navigation]")
        st.sidebar.header(":rainbow[CSV Update Time]")
        st.sidebar.subheader(f":blue[{csv_time()}]")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio("Go to", [":violet[Trade Analysis]", ":blue[Team Scouting]", ":green[Schedule Swap]", ":rainbow[Player Full Data]", ":blue[Draft Results]"])
        st.sidebar.markdown("---")
        st.sidebar.subheader(":orange[Note:]")  
        st.sidebar.write(":orange[The CSV data is updated regularly. Please message me if you notice it's been too long.]")

    # Load draft results once
    draft_results = load_draft_results('data/Fantrax-Draft-Results-Mr Squidward’s 69.csv')

    # Get data directory
    data_dir = Path(__file__).parent.parent / "data"

    # Load data if not already loaded
    if not st.session_state.data_ranges:
        data_ranges, combined_data = load_data()  # Remove the data_dir parameter
        st.session_state.data_ranges = data_ranges
        st.session_state.combined_data = combined_data
        if combined_data is not None:
            st.session_state.trade_analyzer = TradeAnalyzer(combined_data)

    # Handle page navigation
    if st.session_state.data_ranges:
        if page == ":rainbow[Player Full Data]":
            ranges = list(st.session_state.data_ranges.keys())
            selected_range = st.selectbox("Select Time Range", ranges, index=0 if ranges else None)

            if selected_range:
                st.session_state.current_range = selected_range
                data = st.session_state.data_ranges[selected_range]
                display_player_data(data, st.session_state.combined_data, draft_results)
                display_metrics(data)

        elif page == ":blue[Team Scouting]":
            display_team_scouting(st.session_state.combined_data, st.session_state.data_ranges)

        elif page == ":violet[Trade Analysis]":
            display_trade_analysis_page()
            
        elif page == ":green[Schedule Swap]":
            display_schedule_page()

        elif page == ":blue[Draft Results]":
            def display_draft_results_page():
                """Display the draft results page in the app."""
                st.title(":blue[Draft Results]")
                
                # Load draft results
                draft_results = load_draft_results('data/Fantrax-Draft-Results-Mr Squidward’s 69.csv')
                
                if draft_results.empty:
                    st.warning("No draft results available.")
                else:
                    st.dataframe(draft_results)
            display_draft_results_page()


if __name__ == "__main__":
    main()