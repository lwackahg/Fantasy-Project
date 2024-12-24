import streamlit as st
from pathlib import Path
from data_loader import load_data, csv_time
from trade_analysis import display_trade_analysis_page
from trade_options import TradeAnalyzer
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, MENUITEMS
from player_data_display import (
    display_player_data, 
    display_metrics, 
    display_player_trends,
    display_team_scouting,
)
from player_data_display import display_fantasy_managers_teams


def main():
    """Main application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT, menu_items=MENUITEMS)

    # Initialize session state if needed
    if 'data_ranges' not in st.session_state:
        st.session_state.data_ranges = {}
    if 'combined_data' not in st.session_state:
        st.session_state.combined_data = None
    if 'current_range' not in st.session_state:
        st.session_state.current_range = None
    if 'debug_manager' not in st.session_state:
        st.session_state.debug_manager = type('DebugManager', (), {'debug_mode': False, 'toggle_debug': lambda: None})
    if 'trade_analyzer' not in st.session_state:
        st.session_state.trade_analyzer = None
    if 'trade_analysis' not in st.session_state:
        st.session_state.trade_analysis = None

    st.title(":rainbow[" +  PAGE_TITLE + "]")
    
    # Debug mode toggle in sidebar
    with st.sidebar:
        if st.checkbox(":green[Enable Debug Mode]", value=st.session_state.debug_manager.debug_mode):
            st.session_state.debug_manager.toggle_debug()
        
        # Sidebar navigation
        st.sidebar.title(":blue[Navigation]")  # Title for navigation
        st.sidebar.header(":rainbow[CSV Update Time]")  # Header for CSV update time in blue
        st.sidebar.subheader(":blue[" + csv_time() + "]")  # Display the actual time
        
        # Add a divider
        st.sidebar.markdown("---")  # Horizontal line
        
       
        # Navigation options
        page = st.sidebar.radio(
            "Go to",
            [":violet[Trade Analysis]", ":violet[Team Details & Opportunities]", ":blue[Team Scouting]", ":blue[Player Trends]", ":rainbow[Player Full Data]"]
        )
        # Add another divider
        st.sidebar.markdown("---")  # Horizontal line
         # Note about CSV updates
        st.sidebar.subheader(":orange[Note:]")  
        st.sidebar.write(":orange[The CSV data is updated regularly. Please messsage me if you notice its been to long.]")  
        
    # Get data directory
    data_dir = Path(__file__).parent.parent / "data"

    # Load data if not already loaded
    if not st.session_state.data_ranges:
        data_ranges, combined_data = load_data(data_dir)
        st.session_state.data_ranges = data_ranges
        st.session_state.combined_data = combined_data
        if combined_data is not None:
            st.session_state.trade_analyzer = TradeAnalyzer(combined_data)

    if st.session_state.data_ranges:
        if page == ":rainbow[Player Full Data]":
            ranges = list(st.session_state.data_ranges.keys())
            selected_range = st.selectbox("Select Time Range", ranges, index=0 if ranges else None)

            if selected_range:
                st.session_state.current_range = selected_range
                data = st.session_state.data_ranges[selected_range]
                display_player_data(st.session_state.data_ranges, st.session_state.combined_data)
                display_metrics(data)

        elif page == ":blue[Player Trends]":
            current_data = st.session_state.combined_data.reset_index()
            selected_player = st.selectbox("Select Player to View Trends", current_data['Player'].unique().tolist())
            if selected_player:
                display_player_trends(selected_player, current_data)

        elif page == ":blue[Team Scouting]":
            display_team_scouting(st.session_state.combined_data, st.session_state.data_ranges)

        elif page == ":violet[Trade Analysis]":
            display_trade_analysis_page()

        elif page == ":violet[Team Details & Opportunities]":
            display_fantasy_managers_teams(st.session_state.combined_data)


if __name__ == "__main__":
    main()