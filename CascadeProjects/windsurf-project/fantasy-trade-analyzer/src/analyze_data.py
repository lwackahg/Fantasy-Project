import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_import import DataImporter

def analyze_team_performance(data: pd.DataFrame, team: str) -> Dict[str, float]:
    """
    Analyze performance metrics for a specific team.
    
    Args:
        data: DataFrame containing player statistics
        team: Team name to analyze
        
    Returns:
        Dictionary containing team performance metrics
    """
    team_data = data[data['Status'] == team]
    if team_data.empty:
        logging.warning(f"No data found for team: {team}")
        return {}
        
    try:
        metrics = {
            'avg_fp': team_data['FP/G'].mean(),
            'total_fp': team_data['FPts'].sum(),
            'consistency': team_data['FP/G'].std() / team_data['FP/G'].mean(),
            'depth': len(team_data[team_data['FP/G'] > team_data['FP/G'].mean()])
        }
        return metrics
    except Exception as e:
        logging.error(f"Error analyzing team performance: {str(e)}")
        return {}

def analyze_player_trends(data: pd.DataFrame, player_name: str) -> Dict[str, float]:
    """
    Analyze performance trends for a specific player.
    
    Args:
        data: DataFrame containing player statistics
        player_name: Name of the player to analyze
        
    Returns:
        Dictionary containing player performance metrics
    """
    player_data = data[data['Player'] == player_name]
    if player_data.empty:
        logging.warning(f"No data found for player: {player_name}")
        return {}
        
    try:
        metrics = {
            'fp_per_game': player_data['FP/G'].iloc[0],
            'total_points': player_data['FPts'].iloc[0],
            'position': player_data['Position'].iloc[0],
            'team': player_data['Status'].iloc[0]
        }
        return metrics
    except Exception as e:
        logging.error(f"Error analyzing player trends: {str(e)}")
        return {}

def analyze_position_value(data: pd.DataFrame, position: Optional[str] = None) -> Dict[str, float]:
    """
    Analyze value metrics for different positions.
    
    Args:
        data: DataFrame containing player statistics
        position: Optional specific position to analyze
        
    Returns:
        Dictionary containing position-based value metrics
    """
    try:
        if position:
            pos_data = data[data['Position'].str.contains(position, na=False)]
            if pos_data.empty:
                logging.warning(f"No data found for position: {position}")
                return {}
        else:
            pos_data = data
            
        metrics = {
            'avg_fp_per_game': pos_data['FP/G'].mean(),
            'median_fp_per_game': pos_data['FP/G'].median(),
            'std_fp_per_game': pos_data['FP/G'].std(),
            'total_players': len(pos_data)
        }
        return metrics
    except Exception as e:
        logging.error(f"Error analyzing position value: {str(e)}")
        return {}

def main():
    # Initialize the data importer
    importer = DataImporter()
    
    # Get the absolute path to your data file
    project_root = os.path.dirname(current_dir)
    data_file = os.path.join(project_root, "data", "Fantrax-Players-Mr Squidward s 69 (60).csv")
    
    # Import the data
    data = importer.import_csv(data_file)
    
    # Print data preview
    print("\n=== Data Preview ===")
    importer.preview_data()
    
    # Print some basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Total Players: {len(data)}")
    print(f"Average Fantasy Points: {data['FPts'].mean():.2f}")
    print(f"Average Fantasy Points per Game: {data['FP/G'].mean():.2f}")
    
    # Position distribution
    print("\n=== Position Distribution ===")
    position_counts = {}
    for pos_list in data['Position'].str.split(','):
        for pos in pos_list:
            position_counts[pos] = position_counts.get(pos, 0) + 1
    
    for pos, count in sorted(position_counts.items()):
        print(f"{pos}: {count} players")
    
    # Top 10 players by Fantasy Points
    print("\n=== Top 10 Players by Fantasy Points ===")
    top_players = data.nlargest(10, 'FPts')[['Player', 'Position', 'Status', 'FPts', 'FP/G']]
    print(top_players.to_string(index=False))

    # Add new analysis examples
    print("\n=== Team Analysis Example ===")
    team_metrics = analyze_team_performance(data, "Team1")  # Replace with actual team name
    if team_metrics:
        for metric, value in team_metrics.items():
            print(f"{metric}: {value:.2f}")
            
    print("\n=== Position Analysis Example ===")
    for position in ['PG', 'SG', 'SF', 'PF', 'C']:
        pos_metrics = analyze_position_value(data, position)
        if pos_metrics:
            print(f"\n{position} Metrics:")
            for metric, value in pos_metrics.items():
                print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()
