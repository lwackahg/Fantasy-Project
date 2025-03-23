import streamlit as st
from debug import debug_manager

def init_session_state():
    """Initialize session state variables."""
    default_states = {
        'data_ranges': {},
        'current_range': None,
        'debug_manager': debug_manager,
        'schedule_view_type': "List View",
        'schedule_selected_period': "All Periods",
        'schedule_selected_team': "All Teams",
        'schedule_swap_team1': None,
        'schedule_swap_team2': None,
        'schedule_swap_performed': False
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value