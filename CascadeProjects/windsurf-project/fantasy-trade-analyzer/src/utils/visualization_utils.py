"""Utility functions for visualization in the Fantasy Basketball Trade Analyzer."""

from typing import Dict, List, Optional
import pandas as pd
from config.constants import ERROR_COLOR, SUCCESS_COLOR, WARNING_COLOR, NEUTRAL_COLOR

def get_trend_color(value: float, is_positive_good: bool = True) -> str:
    """Determine the color for visualizing trend changes."""
    if abs(value) < 0.05:
        return NEUTRAL_COLOR
    
    if (value > 0 and is_positive_good) or (value < 0 and not is_positive_good):
        return SUCCESS_COLOR
    return ERROR_COLOR

def get_fairness_color(fairness_score: float) -> str:
    """Get the color for visualizing trade fairness scores."""
    if fairness_score >= 0.8:
        return SUCCESS_COLOR
    elif fairness_score >= 0.6:
        return WARNING_COLOR
    return ERROR_COLOR

def plot_performance_trends(data: Dict[str, Dict[str, float]], 
                          selected_metrics: List[str], 
                          title: str) -> 'plotly.graph_objects.Figure':
    """Create a performance trend plot with visible data points."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    for metric in selected_metrics:
        values = [period_data.get(metric, 0) for period_data in data.values()]
        fig.add_trace(go.Scatter(
            x=list(data.keys()),
            y=values,
            mode='lines+markers',
            name=metric
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Value",
        showlegend=True
    )
    
    return fig

def plot_player_trend(players_data: Dict[str, Dict[str, float]]) -> 'plotly.graph_objects.Figure':
    """Create performance trend plots for multiple players."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    for player, data in players_data.items():
        fig.add_trace(go.Scatter(
            x=list(data.keys()),
            y=list(data.values()),
            mode='lines+markers',
            name=player
        ))
    
    fig.update_layout(
        title="Player Performance Trends",
        xaxis_title="Time Period",
        yaxis_title="Fantasy Points",
        showlegend=True
    )
    
    return fig
