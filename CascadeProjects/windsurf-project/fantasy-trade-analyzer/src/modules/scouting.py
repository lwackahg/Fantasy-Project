"""
Scouting Module
Handles team scouting features and displays team metrics.
"""
import pandas as pd
import numpy as np
from .player_data import calculate_team_metrics, get_available_time_ranges

def analyze_team_strength(data_ranges, players, metrics, n_best=5):
    """
    Analyze team strength based on top N players' performance.
    Returns detailed metrics and summary statistics.
    """
    metric_tables = calculate_team_metrics(data_ranges, players, metrics, n_best)
    
    team_summary = {}
    for metric, data in metric_tables.items():
        team_summary[metric] = {
            'total': data['totals'].mean(),
            'avg': data['averages'].mean(),
            'std': data['std_devs'].mean(),
            'trend': _calculate_team_trend(data['totals'], data['available_ranges'])
        }
    
    return metric_tables, team_summary

def _calculate_team_trend(totals, available_ranges):
    """Calculate overall team trend based on total metrics."""
    if len(available_ranges) < 2:
        return 0
    
    trend = 0
    weights = np.linspace(1, 0.5, len(available_ranges)-1)
    
    for i in range(len(available_ranges)-1):
        curr = totals[available_ranges[i]]
        prev = totals[available_ranges[i+1]]
        if prev != 0:
            trend += weights[i] * ((curr - prev) / prev)
    
    return trend

def identify_team_needs(data_ranges, players, metrics, threshold=0.5):
    """
    Identify areas where team might need improvement.
    Returns a list of metrics where team performance is below threshold.
    """
    _, team_summary = analyze_team_strength(data_ranges, players, metrics)
    needs = []
    
    for metric, stats in team_summary.items():
        if stats['trend'] < threshold:
            needs.append({
                'metric': metric,
                'current_value': stats['total'],
                'trend': stats['trend'],
                'volatility': stats['std']
            })
    
    return sorted(needs, key=lambda x: x['trend'])

def compare_teams(data_ranges, team1_players, team2_players, metrics):
    """
    Compare two teams across various metrics.
    Returns comparative statistics and head-to-head matchup data.
    """
    team1_tables, team1_summary = analyze_team_strength(data_ranges, team1_players, metrics)
    team2_tables, team2_summary = analyze_team_strength(data_ranges, team2_players, metrics)
    
    comparison = {}
    for metric in metrics:
        comparison[metric] = {
            'team1': team1_summary[metric],
            'team2': team2_summary[metric],
            'difference': {
                'total': team1_summary[metric]['total'] - team2_summary[metric]['total'],
                'trend': team1_summary[metric]['trend'] - team2_summary[metric]['trend']
            }
        }
    
    return comparison
