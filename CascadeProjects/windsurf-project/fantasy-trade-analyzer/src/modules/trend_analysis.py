"""
Trend Analysis Module
Handles trend analysis functionalities and calculations.
"""
import pandas as pd
import numpy as np
from .player_data import get_ordered_time_ranges, get_available_time_ranges

def calculate_trend(data, method='standard', weights=None):
    """
    Calculate trend based on specified method.
    Methods: 'standard', 'exponential', 'custom'
    """
    if len(data) < 2:
        return 0

    if method == 'exponential':
        return _calculate_exponential_trend(data)
    elif method == 'custom' and weights is not None:
        return _calculate_custom_trend(data, weights)
    else:
        return _calculate_standard_trend(data)

def _calculate_standard_trend(data):
    """Calculate trend using standard method (linear regression slope)."""
    x = np.arange(len(data))
    slope, _ = np.polyfit(x, data, 1)
    return slope

def _calculate_exponential_trend(data):
    """Calculate trend using exponential weighting."""
    weights = np.exp(np.linspace(0, 1, len(data)))
    weights = weights / np.sum(weights)
    return np.sum(weights * data) - np.mean(data)

def _calculate_custom_trend(data, weights):
    """Calculate trend using custom weights."""
    if len(weights) != len(data):
        weights = np.linspace(weights[0], weights[-1], len(data))
    weights = weights / np.sum(weights)
    return np.sum(weights * data) - np.mean(data)

def identify_trend_groups(data_ranges, metrics, method='standard', weights=None):
    """
    Identify trending groups of players based on their performance metrics.
    Returns DataFrame with trend classifications.
    """
    trends = []
    available_ranges = get_available_time_ranges(data_ranges)
    
    for range_name, df in data_ranges.items():
        for _, player_data in df.groupby('Player'):
            player_trends = {}
            for metric in metrics:
                metric_data = player_data[metric].values
                trend = calculate_trend(metric_data, method, weights)
                player_trends[f'{metric}_trend'] = trend
            
            trends.append({
                'Player': player_data['Player'].iloc[0],
                'Time_Range': range_name,
                **player_trends
            })
    
    trends_df = pd.DataFrame(trends)
    return _categorize_trends(trends_df, metrics)

def _categorize_trends(trends_df, metrics):
    """Categorize players based on their trend values."""
    for metric in metrics:
        trend_col = f'{metric}_trend'
        mean_trend = trends_df[trend_col].mean()
        std_trend = trends_df[trend_col].std()
        
        trends_df[f'{metric}_category'] = pd.cut(
            trends_df[trend_col],
            bins=[-np.inf, mean_trend - std_trend, mean_trend + std_trend, np.inf],
            labels=['Downtrend', 'Stable', 'Uptrend']
        )
    
    return trends_df

def filter_trend_data(data_ranges, time_range=None, min_games=0):
    """Filter trend data based on time range and minimum games played."""
    if time_range:
        filtered_ranges = {k: v for k, v in data_ranges.items() if k in time_range}
    else:
        filtered_ranges = data_ranges.copy()
    
    if min_games > 0:
        filtered_ranges = {
            k: v[v['Games_Played'] >= min_games] 
            for k, v in filtered_ranges.items()
        }
    
    return filtered_ranges
