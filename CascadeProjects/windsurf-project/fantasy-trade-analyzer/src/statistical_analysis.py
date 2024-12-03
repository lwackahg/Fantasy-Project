"""Module for statistical analysis functionality"""
import streamlit as st
import pandas as pd
from src.visualization import plot_performance_trends, display_stats_table

class StatisticalAnalyzer:
    """Class for analyzing player and team statistics"""
    
    def __init__(self):
        """Initialize StatisticalAnalyzer"""
        self.data_ranges = st.session_state.data_ranges if hasattr(st.session_state, 'data_ranges') else {}

    def analyze_team_stats(self, team, n_top_players=5):
        """Analyze team statistics"""
        if not self.data_ranges:
            st.error("No data available for analysis")
            return
        
        # Get team data for each time range
        team_stats = {}
        top_players = {}
        
        for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
            if time_range not in self.data_ranges:
                continue
                
            data = self.data_ranges[time_range]
            team_data = data[data['Status'] == team]
            
            if team_data.empty:
                continue
            
            # Calculate team stats
            team_stats[time_range] = {
                'mean_fpg': team_data['FP/G'].mean(),
                'median_fpg': team_data['FP/G'].median(),
                'std_fpg': team_data['FP/G'].std(),
                'total_fpts': team_data['FPts'].sum(),
                'avg_gp': team_data['GP'].mean()
            }
            
            # Get top players by FP/G
            top_players[time_range] = team_data.nlargest(n_top_players, 'FP/G')[['Player', 'FP/G', 'FPts', 'GP']]
        
        # Display team performance trends
        st.write("### Team Performance Trends")
        metrics = ['mean_fpg', 'median_fpg', 'std_fpg', 'total_fpts', 'avg_gp']
        fig = plot_performance_trends(team_stats, metrics, f"{team} Performance Trends")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed stats
        st.write("### Team Statistics")
        stats_data = []
        for time_range, stats in team_stats.items():
            row_data = {'Time Range': time_range}
            row_data.update(stats)
            stats_data.append(row_data)
        
        display_stats_table(stats_data, time_ranges=['60 Days', '30 Days', '14 Days', '7 Days'],
                           metrics=['mean_fpg', 'median_fpg', 'std_fpg', 'total_fpts', 'avg_gp'])
        
        # Display top players for each time range
        st.write(f"### Top {n_top_players} Players by FP/G")
        for time_range, players in top_players.items():
            st.write(f"#### {time_range}")
            st.dataframe(players.style.format({
                'FP/G': '{:.1f}',
                'FPts': '{:.1f}',
                'GP': '{:.0f}'
            }))

    def analyze_player_stats(self, player_name):
        """Analyze player statistics"""
        if not self.data_ranges:
            st.error("No data available for analysis")
            return
            
        # Get available metrics
        sample_data = next(iter(self.data_ranges.values()))
        numeric_cols = sample_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Pre-select important metrics
        default_metrics = ['FPts', 'FP/G', 'GP']
        available_metrics = sorted(list(set(numeric_cols) - set(default_metrics)))
        
        # Create two columns for metric selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### Key Metrics")
            key_metrics_selected = []
            for metric in default_metrics:
                if metric in numeric_cols:
                    if st.checkbox(metric, value=True, key=f"key_{metric}_{player_name}"):
                        key_metrics_selected.append(metric)
        
        with col2:
            st.write("### Additional Metrics")
            additional_metrics = st.multiselect(
                "Select additional metrics to analyze",
                options=available_metrics,
                default=[],
                key=f"metric_selector_{player_name}"
            )
        
        selected_metrics = key_metrics_selected + additional_metrics
        if not selected_metrics:
            st.warning("Please select at least one metric to analyze")
            return
            
        # Gather player data across time ranges
        player_data = {}
        for time_range, data in self.data_ranges.items():
            player_stats = data[data['Player'] == player_name]
            if not player_stats.empty:
                player_data[time_range] = {
                    metric: player_stats[metric].iloc[0]
                    for metric in selected_metrics
                    if metric in player_stats.columns
                }
        
        # Create and display performance plot
        fig = plot_performance_trends(player_data, selected_metrics, f"{player_name}'s Performance Trends")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed stats table
        st.write("### Detailed Statistics")
        stats_data = []
        for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
            if time_range in player_data:
                row_data = {'Time Range': time_range}
                row_data.update(player_data[time_range])
                stats_data.append(row_data)
        
        display_stats_table(stats_data, time_ranges=['60 Days', '30 Days', '14 Days', '7 Days'],
                           metrics=selected_metrics)
