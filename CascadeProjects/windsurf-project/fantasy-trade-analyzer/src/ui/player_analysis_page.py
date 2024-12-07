"""Player analysis page component."""
import streamlit as st
import pandas as pd
from typing import Dict
from ..analyze_data import analyze_player_trends
from ..visualization import DataVisualizer

def display_player_analysis_page():
    """Display the player analysis interface."""
    st.header("Player Analysis")
    
    if not st.session_state.data_ranges:
        st.warning("Please upload player data files first.")
        return
        
    current_data = st.session_state.data_ranges[st.session_state.current_range]
    all_players = sorted(current_data['Player'].unique())
    
    selected_player = st.selectbox(
        "Select Player to Analyze",
        all_players,
        key="player_analysis_player"
    )
    
    if selected_player:
        display_player_stats(selected_player)

def display_player_stats(player: str):
    """Display comprehensive statistics for a selected player."""
    current_data = st.session_state.data_ranges[st.session_state.current_range]
    player_data = current_data[current_data['Player'] == player]
    
    if player_data.empty:
        st.warning(f"No data available for {player}")
        return
        
    # Calculate player metrics
    player_metrics = analyze_player_trends(player_data)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fantasy Points per Game", f"{player_metrics['mean_fp']:.1f}")
    with col2:
        st.metric("Total Fantasy Points", f"{player_metrics['total_fp']:.0f}")
        
    # Display performance chart
    visualizer = DataVisualizer()
    st.plotly_chart(visualizer.plot_player_performance(player_data))
    
    # Display detailed statistics
    st.subheader("Detailed Statistics")
    stats_df = player_data.drop(['Player', 'Team', 'Status'], axis=1)
    st.dataframe(stats_df)
