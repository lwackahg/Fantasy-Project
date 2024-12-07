"""Module for data visualization functionality.

This module provides functions for creating and displaying various visualizations
related to fantasy sports trade analysis, including performance trends, statistics tables,
and trade summaries.
"""
from typing import Dict, List, Optional, Union
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.shared_utils import (
    TIME_RANGES,
    DEFAULT_METRICS,
    ADVANCED_METRICS,
    format_stats_for_display
)

# Constants for styling
GRID_COLOR = 'rgba(128, 128, 128, 0.2)'
GRID_WIDTH = 1
PLOT_BG_COLOR = 'white'
DEFAULT_PLOT_HEIGHT = 400
DEFAULT_TABLE_HEIGHT = 200

class DataVisualizer:
    """
    Handles data visualization for the Fantasy Trade Analyzer
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """Initialize DataVisualizer with optional data"""
        self.data = data

    def create_base_figure(
        self,
        title: str,
        x_title: str = "",
        y_title: str = "",
        height: int = DEFAULT_PLOT_HEIGHT
    ) -> go.Figure:
        """
        Create a base plotly figure with standard styling.
        
        Args:
            title: The title of the figure
            x_title: The x-axis title
            y_title: The y-axis title
            height: Height of the figure in pixels
            
        Returns:
            A plotly Figure object with standard styling applied
        """
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            height=height,
            plot_bgcolor=PLOT_BG_COLOR,
            xaxis=dict(
                showgrid=True,
                gridcolor=GRID_COLOR,
                gridwidth=GRID_WIDTH
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=GRID_COLOR,
                gridwidth=GRID_WIDTH
            )
        )
        return fig

    def plot_performance_trends(
        self,
        data: Dict[str, Dict[str, float]],
        selected_metrics: List[str],
        title: str = "Performance Trends"
    ) -> go.Figure:
        """
        Create a performance trend plot with visible data points and tooltips.
        
        Args:
            data: Dictionary mapping time ranges to metric values
            selected_metrics: List of metrics to plot
            title: Title of the plot
            
        Returns:
            A plotly Figure object containing the performance trends
        """
        fig = self.create_base_figure(
            title=title,
            x_title="Time Period",
            y_title="Value"
        )
        
        for metric in selected_metrics:
            if not any(metric in period_data for period_data in data.values()):
                continue
                
            y_values = []
            x_values = []
            hover_text = []
            
            for period, period_data in data.items():
                if metric in period_data:
                    y_values.append(period_data[metric])
                    x_values.append(period)
                    hover_text.append(
                        f"{metric}: {period_data[metric]:.2f}<br>"
                        f"Period: {period}"
                    )
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=metric,
                hovertext=hover_text,
                hoverinfo='text'
            ))
        
        return fig

    def display_stats_table(
        self,
        stats_data: Optional[Dict],
        time_ranges: List[str] = TIME_RANGES,
        metrics: List[str] = DEFAULT_METRICS,
        height: int = DEFAULT_TABLE_HEIGHT
    ) -> Optional[pd.DataFrame]:
        """
        Display a formatted statistics table.
        
        Args:
            stats_data: Dictionary containing statistics data
            time_ranges: List of time ranges to display
            metrics: List of metrics to display
            height: Height of the table in pixels
            
        Returns:
            A Streamlit dataframe object if data is present, None otherwise
        """
        if not stats_data:
            st.warning("No statistics data available to display.")
            return None
            
        # Create DataFrame from stats data
        df = pd.DataFrame(stats_data)
        
        # Format numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col] = df[col].round(2)
            
        # Display the table
        st.dataframe(df, height=height)
        return df

    def get_trend_color(self, value: float, is_positive_good: bool = True) -> str:
        """
        Determine the color for visualizing trend changes.
        
        Args:
            value: The change in value to visualize
            is_positive_good: If True, positive changes are green
            
        Returns:
            Hex color code for visualization
        """
        if (value > 0 and is_positive_good) or (value < 0 and not is_positive_good):
            return '#2ecc71'  # Green
        elif (value < 0 and is_positive_good) or (value > 0 and not is_positive_good):
            return '#e74c3c'  # Red
        return '#f1c40f'  # Yellow for neutral/no change

    def get_fairness_color(self, fairness_score: float) -> str:
        """
        Get the color for visualizing trade fairness scores.
        
        Args:
            fairness_score: Trade fairness score (0-1)
            
        Returns:
            Hex color code based on fairness level
        """
        if fairness_score >= 0.8:
            return '#2ecc71'  # Green for very fair
        elif fairness_score >= 0.6:
            return '#f1c40f'  # Yellow for moderately fair
        return '#e74c3c'  # Red for potentially unfair

    def display_team_comparison(self, team_data: Dict[str, pd.DataFrame], team_name: str):
        """
        Display a comprehensive comparison of team statistics.
        
        Args:
            team_data: Dictionary containing team statistics
            team_name: Name of the team to display
        """
        if not team_data or team_name not in team_data:
            st.warning(f"No data available for team {team_name}")
            return

        df = team_data[team_name]
        
        # Create bar chart for key metrics
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['Player'],
            y=df['FP/G'],
            name='FP/G'
        ))
        fig.add_trace(go.Bar(
            x=df['Player'],
            y=df['PTS'],
            name='PTS'
        ))
        fig.add_trace(go.Bar(
            x=df['Player'],
            y=df['REB'],
            name='REB'
        ))
        fig.add_trace(go.Bar(
            x=df['Player'],
            y=df['AST'],
            name='AST'
        ))
        fig.update_layout(
            title=f"{team_name} - Player Performance Metrics",
            barmode='group'
        )
        st.plotly_chart(fig)

    def plot_player_trend(self, players_data: Dict[str, Dict[str, float]]):
        """
        Create performance trend plots for multiple players.
        
        Args:
            players_data: Dictionary containing player performance data
        """
        fig = go.Figure()
        
        for player, data in players_data.items():
            fig.add_trace(go.Scatter(
                x=list(data.keys()),
                y=[v['FP/G'] for v in data.values()],
                mode='lines+markers',
                name=player
            ))
        
        fig.update_layout(
            title="Player Performance Trends",
            xaxis_title="Time Period",
            yaxis_title="Fantasy Points per Game",
            showlegend=True
        )
        
        st.plotly_chart(fig)

    def display_trade_summary(
        self,
        team: str,
        trade_details: Dict[str, Dict[str, List[str]]],
        other_teams: List[str]
    ) -> None:
        """Display trade summary for a team.
        
        Args:
            team: The team name to display summary for
            trade_details: Dictionary containing incoming/outgoing players for each team
            other_teams: List of other teams involved in the trade
        """
        incoming = trade_details[team]['incoming']
        outgoing = trade_details[team]['outgoing']
        
        if incoming:
            incoming_details = []
            for player in incoming:
                value = calculate_player_value({
                    k: v[v['Player'] == player] 
                    for k, v in st.session_state.data_ranges.items()
                })
                from_team = next((t for t in other_teams if player in trade_details[t]['outgoing']), None)
                from_team_name = get_team_name(from_team) if from_team else "Unknown"
                incoming_details.append(f"{player} ({value:.1f}) from {from_team_name}")
            st.write("📥 Receiving:", ", ".join(incoming_details))
        
        if outgoing:
            outgoing_details = []
            for player in outgoing:
                value = calculate_player_value({
                    k: v[v['Player'] == player] 
                    for k, v in st.session_state.data_ranges.items()
                })
                to_team = next((t for t in other_teams if player in trade_details[t]['incoming']), None)
                to_team_name = get_team_name(to_team) if to_team else "Unknown"
                outgoing_details.append(f"{player} ({value:.1f}) to {to_team_name}")
            st.write("📤 Sending:", ", ".join(outgoing_details))

    def display_fairness_score(self, team_name: str, score: float) -> None:
        """Display a team's fairness score with appropriate styling.
        
        Args:
            team_name: Name of the team
            score: Fairness score between 0 and 1
        """
        color = 'green' if score > 0.7 else 'orange' if score > 0.5 else 'red'
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 16px;'>{team_name}</div>
            <div style='color: {color}; font-size: 20px;'>{score:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    def display_overall_fairness(self, min_fairness: float) -> None:
        """Display overall trade fairness with appropriate styling.
        
        Args:
            min_fairness: The minimum fairness score across all teams
        """
        fairness_color = 'green' if min_fairness > 0.7 else 'orange' if min_fairness > 0.5 else 'red'
        fairness_text = 'Fair' if min_fairness > 0.7 else 'Questionable' if min_fairness > 0.5 else 'Unfair'
        
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 16px;'>Overall</div>
            <div style='color: {fairness_color}; font-size: 20px;'>{min_fairness:.1%}</div>
            <div style='color: {fairness_color}; font-size: 16px;'>{fairness_text}</div>
        </div>
        """, unsafe_allow_html=True)
