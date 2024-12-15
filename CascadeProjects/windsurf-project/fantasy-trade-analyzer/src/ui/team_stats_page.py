"""Team statistics page component."""
import streamlit as st
import pandas as pd
from typing import Dict
from analyze_data import analyze_team_performance, analyze_position_value
from visualization import DataVisualizer
from shared_utils import calculate_z_scores
from visualization import create_team_radar_chart, plot_team_trends
from statistical_analysis import analyze_team_stats

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

def display_team_rankings():
    """Display team rankings based on various statistical categories."""
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload player data first.")
        return
        
    st.write("## Team Rankings")
    
    # Get team statistics
    team_stats = analyze_team_stats(st.session_state.data)
    
    # Display rankings table
    st.dataframe(
        team_stats.style.highlight_max(axis=0, color='lightgreen')
                       .highlight_min(axis=0, color='lightcoral')
    )
    
    # Create visualization
    create_team_radar_chart(team_stats, "Team Performance Comparison")

def display_team_trends():
    """Display team performance trends over time."""
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload player data first.")
        return
        
    st.write("## Team Performance Trends")
    
    # Get unique teams
    teams = sorted(st.session_state.data['Team'].unique())
    
    # Team selection
    selected_team = st.selectbox("Select Team", teams)
    
    # Get categories for analysis
    categories = st.multiselect(
        "Select Categories to Analyze",
        st.session_state.categories,
        default=st.session_state.categories[:3]
    )
    
    if categories:
        plot_team_trends(
            st.session_state.data,
            selected_team,
            categories
        )

def render_team_stats_page():
    """Main function to render the team statistics page."""
    st.write("# Team Statistics Analysis")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Team Rankings", "Team Trends"])
    
    with tab1:
        display_team_rankings()
        
    with tab2:
        display_team_trends()
