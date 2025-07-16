"""
UI components for the player data feature.
"""

import streamlit as st
import pandas as pd

def display_player_dataframe(player_data: pd.DataFrame):
    """Displays the main player data table."""
    if player_data.empty:
        st.warning("No player data available.")
    else:
        st.dataframe(player_data)

def display_player_metrics(data: pd.DataFrame):
    """Displays basic statistics as metrics."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Players", len(data))
    with col2:
        st.metric("Teams", data['Team'].nunique())
    with col3:
        st.metric("Avg FP/G", f"{data['FP/G'].mean():.1f}")
