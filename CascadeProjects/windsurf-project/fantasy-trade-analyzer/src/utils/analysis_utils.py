"""Utility functions for data analysis in the Fantasy Basketball Trade Analyzer."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from config.constants import TIME_RANGES

def calculate_team_stats(team_data: pd.DataFrame, top_x: Optional[int] = None) -> Dict[str, float]:
    """Calculate comprehensive statistics for a team's roster."""
    if team_data.empty:
        return {
            'avg_fp': 0.0,
            'total_fp': 0.0,
            'consistency': 0.0,
            'efficiency': 0.0,
            'depth': 0.0
        }
    
    # Sort players by fantasy points per game
    sorted_players = team_data.sort_values('FP/G', ascending=False)
    
    # Consider only top X players if specified
    if top_x:
        sorted_players = sorted_players.head(top_x)
    
    stats = {
        'avg_fp': sorted_players['FP/G'].mean(),
        'total_fp': sorted_players['FPts'].sum(),
        'consistency': sorted_players['FP/G'].std() / sorted_players['FP/G'].mean(),
        'depth': len(sorted_players[sorted_players['FP/G'] > sorted_players['FP/G'].mean()])
    }
    
    return stats

def calculate_player_value(player_data: Dict[str, pd.DataFrame]) -> float:
    """Calculate a player's overall value based on performance across time ranges."""
    if not player_data:
        return 0.0
    
    total_weight = 0.0
    weighted_value = 0.0
    
    for time_range, data in player_data.items():
        if time_range in TIME_RANGES:
            weight = TIME_RANGES[time_range]
            value = data['FP/G'].mean() if not data.empty else 0.0
            
            weighted_value += weight * value
            total_weight += weight
    
    return weighted_value / total_weight if total_weight > 0 else 0.0

def calculate_trade_fairness(
    before_stats: Dict[str, Dict[str, float]],
    after_stats: Dict[str, Dict[str, float]],
    team_data: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, float]]:
    """Calculate the fairness and impact of a proposed trade."""
    results = {}
    
    for team in before_stats:
        # Calculate value changes
        value_change = after_stats[team]['avg_fp'] - before_stats[team]['avg_fp']
        depth_impact = after_stats[team]['depth'] - before_stats[team]['depth']
        
        # Calculate fairness score (0-1)
        fairness_score = min(1.0, max(0.0, 0.5 + value_change / 10))
        
        # Calculate risk score based on consistency changes
        risk_score = abs(after_stats[team]['consistency'] - before_stats[team]['consistency'])
        
        results[team] = {
            'fairness_score': fairness_score,
            'value_change': value_change,
            'depth_impact': depth_impact,
            'risk_score': risk_score
        }
    
    return results
