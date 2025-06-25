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
from player_csv_selector import select_player_csv_files, load_selected_csvs
from streamlit_downloader_ui import downloader_sidebar

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
        
        # Downloader UI (league/range selection and download)
        downloader_sidebar()
        # CSV Selection Section
        st.sidebar.subheader(":blue[Select Player CSV(s)]")
        data_dir = Path(__file__).parent.parent / "data"
        selected_files = select_player_csv_files(data_dir)
        load_btn = st.sidebar.button("Load Selected CSVs")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio("Go to", [":violet[Trade Analysis]", ":blue[Team Scouting]", ":green[Schedule Swap]", ":rainbow[Player Full Data]", ":blue[Draft Results]"])
        st.sidebar.markdown("---")
        st.sidebar.subheader(":orange[Note:]")  
        st.sidebar.write(":orange[The CSV data is updated regularly. Please message me if you notice it's been too long.")

    # Load draft results once
    draft_results = load_draft_results('data/Fantrax-Draft-Results-Mr Squidward’s 69.csv')

    # Only load data when user presses button
    if load_btn and selected_files:
        loaded_csvs = load_selected_csvs(selected_files)
        # Patch: temporarily replace load_data logic to only use selected files
        # We'll mimic load_data but only for selected files
        import re
        from data_loader import clean_data
        data_ranges = {}
        combined_data = pd.DataFrame()
        for file in selected_files:
            try:
                df = clean_data(loaded_csvs[file.name])
                if not df.empty:
                    timestamp_value = None
                    if "YTD" in file.name:
                        timestamp_value = 'YTD'
                        data_ranges['YTD'] = df
                    else:
                        match = re.search(r'\((\d+)\)', file.name)
                        if match:
                            days = match.group(1)
                            timestamp_value = f'{days} Days'
                            data_ranges[f'{days} Days'] = df
                    if timestamp_value:
                        df['Timestamp'] = timestamp_value
                    combined_data = pd.concat([combined_data, df], ignore_index=True)
            except Exception as e:
                st.error(f"Failed to process {file.name}: {e}")
        if not combined_data.empty:
            combined_data.set_index('Player', inplace=True)
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