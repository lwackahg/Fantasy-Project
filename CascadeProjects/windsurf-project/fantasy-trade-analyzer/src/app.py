import streamlit as st
from pathlib import Path
from data_loader import load_data
from config import PAGE_TITLE, PAGE_ICON, LAYOUT
from player_data_display import (
    display_player_data, 
    display_metrics, 
    display_player_trends,
    display_team_scouting,
    generate_trade_opportunities
)
from player_data_display import display_fantasy_managers_teams
from trade_analysis import display_trade_analysis


def main():
    """Main application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

    # Initialize session state if needed
    if 'data_ranges' not in st.session_state:
        st.session_state.data_ranges = {}
    if 'combined_data' not in st.session_state:
        st.session_state.combined_data = None
    if 'current_range' not in st.session_state:
        st.session_state.current_range = None
    if 'debug_manager' not in st.session_state:
        st.session_state.debug_manager = type('DebugManager', (), {'debug_mode': False, 'toggle_debug': lambda: None})

    st.title(PAGE_TITLE)
    
    # Debug mode toggle in sidebar
    with st.sidebar:
        if st.checkbox("Enable Debug Mode", value=st.session_state.debug_manager.debug_mode):
            st.session_state.debug_manager.toggle_debug()
        
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ["Player Overview", "Player Trends", "Team Scouting", "Trade Analysis", "Fantasy Managers' Teams"]
        )

    # Get data directory
    data_dir = Path(__file__).parent.parent / "data"

    # Load data if not already loaded
    if not st.session_state.data_ranges:
        data_ranges, combined_data = load_data(data_dir)
        st.session_state.data_ranges = data_ranges
        st.session_state.combined_data = combined_data

    if st.session_state.data_ranges:
        if page == "Player Overview":
            ranges = list(st.session_state.data_ranges.keys())
            selected_range = st.selectbox("Select Time Range", ranges, index=0 if ranges else None)

            if selected_range:
                st.session_state.current_range = selected_range
                data = st.session_state.data_ranges[selected_range]
                display_player_data(st.session_state.data_ranges, st.session_state.combined_data)
                display_metrics(data)

        elif page == "Player Trends":
            current_data = st.session_state.combined_data.reset_index()
            selected_player = st.selectbox("Select Player to View Trends", current_data['Player'].unique().tolist())
            if selected_player:
                display_player_trends(selected_player, current_data)

        elif page == "Team Scouting":
            display_team_scouting(st.session_state.combined_data, st.session_state.data_ranges)

        elif page == "Trade Analysis":
            # You need to define all_stats here based on the selected managers or players
            selected_managers = st.multiselect("Select your team:", options=list(st.session_state.data_ranges.keys()))
            if selected_managers:
                all_stats = {manager: st.session_state.data_ranges[manager] for manager in selected_managers}
                trade_suggestions = generate_trade_opportunities(all_stats, selected_managers)
                
                if trade_suggestions:
                    for suggestion in trade_suggestions:
                        player_name, targets, trade_details = suggestion
                        if st.button(player_name):
                            display_trade_analysis(player_name, trade_details)
                else:
                    st.warning("No trade suggestions available.")

        elif page == "Fantasy Managers' Teams":
            display_fantasy_managers_teams(st.session_state.combined_data)


if __name__ == "__main__":
    main()