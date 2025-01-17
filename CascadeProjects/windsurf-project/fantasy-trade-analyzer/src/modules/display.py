"""
Display Module
Handles all display-related functions for the application.
"""
import streamlit as st
import pandas as pd
import numpy as np
from .trend_analysis import identify_trend_groups, filter_trend_data
from .player_data import calculate_player_stats

def display_trend_settings():
    """Display and handle trend calculation settings."""
    col1, col2 = st.columns(2)
    
    with col1:
        trend_method = st.radio(
            "Trend Calculation Method",
            ['standard', 'exponential', 'custom'],
            help="Choose how trends should be calculated"
        )
    
    weights = None
    if trend_method == 'custom':
        with col2:
            w1 = st.number_input("Recent Weight", 0.0, 1.0, 0.6)
            w2 = st.number_input("Past Weight", 0.0, 1.0, 0.4)
            weights = [w1, w2]
    
    return trend_method, weights

def display_trending_players(data_ranges, metrics, trend_method='standard', weights=None):
    """Display trending players based on calculated trends."""
    trends_df = identify_trend_groups(data_ranges, metrics, trend_method, weights)
    
    st.subheader("Trending Players")
    
    tabs = st.tabs(["Uptrending", "Stable", "Downtrending"])
    
    for metric in metrics:
        with tabs[0]:
            _display_trend_group(trends_df, metric, "Uptrend")
        with tabs[1]:
            _display_trend_group(trends_df, metric, "Stable")
        with tabs[2]:
            _display_trend_group(trends_df, metric, "Downtrend")

def _display_trend_group(trends_df, metric, category):
    """Display a specific trend group for a metric."""
    filtered_df = trends_df[trends_df[f'{metric}_category'] == category]
    if not filtered_df.empty:
        st.write(f"{metric} - {category}")
        st.dataframe(
            filtered_df[['Player', f'{metric}_trend']]
            .sort_values(f'{metric}_trend', ascending=(category == "Downtrend"))
            .style.format({f'{metric}_trend': '{:.3f}'})
        )

def display_player_details(data_ranges, player, metrics):
    """Display detailed statistics for a specific player."""
    stats_df, std_dev_df = calculate_player_stats(data_ranges, player, metrics)
    
    if stats_df.empty:
        st.warning(f"No data available for {player}")
        return
    
    st.subheader(f"{player} Statistics")
    
    # Display time range statistics
    st.write("Performance by Time Range")
    st.dataframe(
        stats_df.style.format({metric: '{:.2f}' for metric in metrics})
    )
    
    # Display standard deviations
    st.write("Performance Volatility (Standard Deviation)")
    st.dataframe(
        std_dev_df.style.format({f'{metric}_STD': '{:.2f}' for metric in metrics})
    )

def display_metric_distribution(data_ranges, metrics):
    """Display distribution of metrics across all players."""
    st.subheader("Metric Distributions")
    
    for metric in metrics:
        data = []
        for range_name, df in data_ranges.items():
            data.extend(df[metric].tolist())
        
        fig = pd.Series(data).hist(bins=30)
        st.pyplot(fig.figure)
        st.write(f"{metric} Distribution Statistics")
        st.write({
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data)
        })
