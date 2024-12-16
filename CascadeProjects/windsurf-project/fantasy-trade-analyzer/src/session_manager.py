import streamlit as st
from debug import debug_manager

def init_session_state():
    """Initialize session state variables."""
    if 'data_ranges' not in st.session_state:
        st.session_state.data_ranges = {}
    if 'combined_data' not in st.session_state:
        st.session_state.combined_data = None
    if 'current_range' not in st.session_state:
        st.session_state.current_range = None
    if 'debug_manager' not in st.session_state:
        st.session_state.debug_manager = debug_manager