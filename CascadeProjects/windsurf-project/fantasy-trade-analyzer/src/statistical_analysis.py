"""Module for statistical analysis functionality"""
import streamlit as st
import pandas as pd
import logging
import numpy as np
from typing import Dict, List, Optional, Union

class StatisticalAnalyzer:
    """Class for analyzing player and team statistics"""
    
    def __init__(self):
        """Initialize StatisticalAnalyzer"""
        self.data_ranges = st.session_state.data_ranges if hasattr(st.session_state, 'data_ranges') else {}
        
    def calculate_team_stats(self, team_data: pd.DataFrame, top_x: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a team's roster.
        
        Args:
            team_data: DataFrame containing team player statistics
            top_x: Number of top players to consider (e.g., top 8)
            
        Returns:
            Dictionary containing calculated statistics
        """
        if team_data.empty:
            logging.warning("Empty team data provided")
            return {}

        # Sort players by fantasy points per game
        sorted_players = team_data.sort_values('FP/G', ascending=False)
        
        if top_x:
            sorted_players = sorted_players.head(top_x)

        try:
            stats = {
                'avg_fp': sorted_players['FP/G'].mean(),
                'total_fp': sorted_players['FPts'].sum(),
                'consistency': sorted_players['FP/G'].std(),
                'efficiency': (sorted_players['FPts'] / sorted_players['MIN']).mean(),
                'depth': len(sorted_players)
            }
            
            logging.debug(f"Calculated team stats: {stats}")
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating team stats: {str(e)}")
            return {}
    
    def calculate_player_value(self, player_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate a player's overall value based on performance across time ranges.
        
        The value score is computed using a weighted average of performance metrics:
        - Fantasy points per game (FP/G)
        - Consistency (inverse of standard deviation)
        - Minutes played (normalized to percentage of full game)
        
        Each time range (7, 14, 30, 60 days) is weighted differently, with recent
        performance having higher weight.
        
        Args:
            player_data: Dictionary mapping time ranges to player statistics DataFrames
            
        Returns:
            float: Calculated player value score between 0 and 100
            
        Raises:
            ValueError: If weights don't sum to 1.0 or if required columns are missing
        """
        if not player_data:
            logging.warning("Empty player data provided, returning 0.0")
            return 0.0

        weights = {
            '7 Days': 0.4,
            '14 Days': 0.3,
            '30 Days': 0.2,
            '60 Days': 0.1
        }
        
        # Validate weights
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        required_columns = ['FP/G', 'MIN']
        value_score = 0.0
        
        for time_range, data in player_data.items():
            if time_range not in weights:
                logging.debug(f"Skipping unknown time range: {time_range}")
                continue
                
            if data.empty:
                logging.warning(f"Empty data for time range: {time_range}")
                continue
                
            # Validate required columns
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            try:
                avg_fp = data['FP/G'].mean()
                # Add small epsilon to avoid division by zero
                consistency = 1 / (1 + data['FP/G'].std() + 1e-6)
                minutes = data['MIN'].mean() / 48  # Normalize minutes to percentage of full game
                
                period_score = avg_fp * consistency * minutes
                weight = weights[time_range]
                value_score += period_score * weight
                
                logging.debug(
                    f"Time range: {time_range}, "
                    f"Avg FP: {avg_fp:.2f}, "
                    f"Consistency: {consistency:.2f}, "
                    f"Minutes: {minutes:.2f}, "
                    f"Period Score: {period_score:.2f}"
                )
                
            except Exception as e:
                logging.error(f"Error calculating stats for {time_range}: {str(e)}")
                continue

        # Normalize value score to 0-100 range
        return min(max(value_score * 10, 0), 100)  # Scale and clamp value

    def calculate_trade_fairness(
        self,
        before_stats: Dict[str, Dict[str, float]],
        after_stats: Dict[str, Dict[str, float]],
        team_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate the fairness and impact of a proposed trade.
        
        Args:
            before_stats: Team statistics before the trade
            after_stats: Projected team statistics after the trade
            team_data: Current team rosters and player statistics
            
        Returns:
            Dictionary containing fairness analysis for each team
        """
        analysis = {}
        
        for team in before_stats.keys():
            before = before_stats[team]
            after = after_stats[team]
            
            # Calculate value changes
            fp_change = after['avg_fp'] - before['avg_fp']
            depth_change = after['depth'] - before['depth']
            efficiency_change = after['efficiency'] - before['efficiency']
            
            # Calculate risk based on consistency change
            consistency_change = after['consistency'] - before['consistency']
            risk_score = abs(consistency_change) / before['consistency']
            
            # Calculate overall fairness score (0-1)
            fairness_score = 1.0 - min(1.0, abs(fp_change) / before['avg_fp'])
            
            analysis[team] = {
                'fairness_score': fairness_score,
                'value_change': fp_change,
                'depth_impact': depth_change,
                'risk_score': risk_score
            }
        
        return analysis

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
        # Removed plot_performance_trends function call
        
        # Display detailed stats
        st.write("### Team Statistics")
        stats_data = []
        for time_range, stats in team_stats.items():
            row_data = {'Time Range': time_range}
            row_data.update(stats)
            stats_data.append(row_data)
        
        # Removed display_stats_table function call
        
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
        
        # Removed plot_performance_trends function call
        
        # Display detailed stats table
        st.write("### Detailed Statistics")
        stats_data = []
        for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
            if time_range in player_data:
                row_data = {'Time Range': time_range}
                row_data.update(player_data[time_range])
                stats_data.append(row_data)
        
        # Removed display_stats_table function call
