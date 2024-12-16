import streamlit as st
from pathlib import Path
from session_manager import init_session_state
from data_loader import load_data
from config import PAGE_TITLE, PAGE_ICON, LAYOUT
from player_data_display import display_player_data, display_metrics, display_player_trends  # Import new function


def main():
    """Main application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
    init_session_state()

    st.title(PAGE_TITLE)
    
    # Debug mode toggle in sidebar
    with st.sidebar:
        if st.checkbox("Enable Debug Mode", value=st.session_state.debug_manager.debug_mode):
            st.session_state.debug_manager.toggle_debug()
        
        # Sidebar navigation
        page = st.selectbox("Select Page:", ["Home", "Player Trends"])  # Add new page option

    # Get data directory
    data_dir = Path(__file__).parent.parent / "data"

    # Load data if not already loaded
    if not st.session_state.data_ranges:
        data_ranges, combined_data = load_data(data_dir)
        st.session_state.data_ranges = data_ranges
        st.session_state.combined_data = combined_data

    if st.session_state.data_ranges:
        if page == "Home":
            ranges = list(st.session_state.data_ranges.keys())
            selected_range = st.selectbox("Select Time Range", ranges, index=0 if ranges else None)

            if selected_range:
                st.session_state.current_range = selected_range
                data = st.session_state.data_ranges[selected_range]
                display_metrics(data)
                display_player_data(st.session_state.data_ranges, st.session_state.combined_data)

        elif page == "Player Trends":
            player = st.selectbox("Select Player to View Trends", st.session_state.combined_data.index.tolist())
            if player:
                display_player_trends(player)  # Call the function to display trends for the selected player


if __name__ == "__main__":
    main()