import streamlit as st
from pathlib import Path
from session_manager import init_session_state
from data_loader import load_data
from metrics import display_metrics
from config import PAGE_TITLE, PAGE_ICON, LAYOUT

def main():
    """Main application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
    init_session_state()
    
    st.title(PAGE_TITLE)
    
    # Debug mode toggle in sidebar
    with st.sidebar:
        if st.checkbox("Enable Debug Mode", value=st.session_state.debug_manager.debug_mode):
            st.session_state.debug_manager.toggle_debug()
    
    # Get data directory
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load data if not already loaded
    if not st.session_state.data_ranges:
        st.session_state.data_ranges = load_data(data_dir)
        
    if st.session_state.data_ranges:
        ranges = list(st.session_state.data_ranges.keys())
        selected_range = st.selectbox("Select Time Range", ranges, index=0 if ranges else None)
        
        if selected_range:
            st.session_state.current_range = selected_range
            data = st.session_state.data_ranges[selected_range]
            display_metrics(data)

if __name__ == "__main__":
    main()