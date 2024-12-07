"""Module for data visualization functionality.

This module provides functions for creating and displaying various visualizations
related to fantasy sports trade analysis, including performance trends, statistics tables,
and trade summaries.
"""
from typing import Dict, List, Optional, Union
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Constants for styling
GRID_COLOR = 'rgba(128, 128, 128, 0.2)'
GRID_WIDTH = 1
PLOT_BG_COLOR = 'white'
DEFAULT_PLOT_HEIGHT = 400
DEFAULT_TABLE_HEIGHT = 200
TIME_RANGES = ['60 Days', '30 Days', '14 Days', '7 Days']

def create_base_figure(
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = DEFAULT_PLOT_HEIGHT
) -> go.Figure:
    """Create a base plotly figure with standard styling.
    
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
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridwidth=GRID_WIDTH,
            gridcolor=GRID_COLOR,
            zeroline=False
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridwidth=GRID_WIDTH,
            gridcolor=GRID_COLOR,
            zeroline=False
        ),
        plot_bgcolor=PLOT_BG_COLOR,
        height=height,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        hovermode='closest'
    )
    return fig

def plot_performance_trends(
    data: Dict[str, Dict],
    selected_metrics: List[str],
    title: str = "Performance Trends"
) -> go.Figure:
    """Create a performance trend plot with visible data points and tooltips.
    
    Args:
        data: Dictionary mapping time ranges to metric values
        selected_metrics: List of metrics to plot
        title: Title of the plot
        
    Returns:
        A plotly Figure object containing the performance trends
    """
    fig = create_base_figure(title=title, x_title="Time Range", y_title="Value")
    x_positions = list(range(len(TIME_RANGES)))
    
    for metric in selected_metrics:
        values = []
        hover_texts = []
        
        for time_range in TIME_RANGES:
            if time_range in data:
                value = data[time_range].get(metric)
                values.append(value)
                hover_texts.append(
                    f"<b>{time_range}</b><br>" +
                    f"{metric}: {value:.2f}<br>" +
                    f"Time Period: Last {time_range}"
                )
            else:
                values.append(None)
                hover_texts.append(f"No data available for {time_range}")
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=values,
            name=metric,
            mode='lines',
            line=dict(width=2),
            showlegend=True
        ))
        
        # Add point trace for enhanced visibility
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=values,
            name=metric + ' (Points)',
            mode='markers',
            marker=dict(
                size=12,
                symbol='circle',
                line=dict(width=2, color='white'),
                color=fig.data[-1].line.color
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False
        ))
    
    fig.update_layout(
        xaxis_ticktext=TIME_RANGES,
        xaxis_tickvals=x_positions
    )
    
    return fig

def display_stats_table(
    stats_data: Optional[Dict],
    time_ranges: List[str],
    metrics: List[str],
    height: int = DEFAULT_TABLE_HEIGHT
) -> Optional[st.delta_generator.DeltaGenerator]:
    """Display a formatted statistics table.
    
    Args:
        stats_data: Dictionary containing statistics data
        time_ranges: List of time ranges to display
        metrics: List of metrics to display
        height: Height of the table in pixels
        
    Returns:
        A Streamlit dataframe object if data is present, None otherwise
    """
    if not stats_data:
        return None
        
    df = pd.DataFrame(stats_data)
    
    numeric_cols = [col for col in df.columns if col != 'Time Range']
    formatter = {col: '{:.1f}' for col in numeric_cols}
    
    return st.dataframe(
        df.style.format(formatter),
        hide_index=True,
        height=height
    )

def display_trade_summary(
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

def display_fairness_score(team_name: str, score: float) -> None:
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

def display_overall_fairness(min_fairness: float) -> None:
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
