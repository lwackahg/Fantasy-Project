"""Team statistics page component."""
import streamlit as st
import pandas as pd
from typing import Dict
from ..analyze_data import analyze_team_performance, analyze_position_value
from ..visualization import DataVisualizer

def display_team_stats_page():
    """Display the team statistics analysis page."""
    st.header("Team Statistics Analysis")
    
    if not st.session_state.data_ranges:
        st.warning("Please upload player data files first.")
        return
        
    selected_team = st.selectbox(
        "Select Team to Analyze",
        get_all_teams(),
        key="team_stats_team"
    )
    
    if selected_team:
        display_team_analysis(selected_team)

def display_team_analysis(team: str):
    """Display comprehensive analysis for a selected team."""
    current_data = st.session_state.data_ranges[st.session_state.current_range]
    team_data = current_data[current_data['Status'] == team]
    
    if team_data.empty:
        st.warning(f"No data available for {team}")
        return
        
    # Calculate team metrics
    team_metrics = analyze_team_performance(team_data)
    position_metrics = analyze_position_value(team_data)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Fantasy Points", f"{team_metrics['avg_fp']:.1f}")
    with col2:
        st.metric("Team Depth", f"{team_metrics['depth']}")
    with col3:
        st.metric("Consistency Score", f"{team_metrics['consistency']:.2f}")
        
    # Display position analysis
    st.subheader("Position Analysis")
    position_df = pd.DataFrame(position_metrics).T
    st.dataframe(position_df)
    
    # Display visualizations
    visualizer = DataVisualizer()
    st.plotly_chart(visualizer.plot_team_performance(team_data))

def get_all_teams() -> list:
    """Get list of all teams from current data."""
    if not st.session_state.data_ranges or not st.session_state.current_range:
        return []
    
    current_data = st.session_state.data_ranges[st.session_state.current_range]
    return sorted(current_data['Status'].unique())
