"""Checks for Trade Options, Not the actual Trade Analyzer"""
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from debug import debug_manager
from data_loader import TEAM_MAPPINGS

class TradeAnalyzer:
    """Analyzes fantasy basketball trades."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the trade analyzer with player data."""
        self.data = data
        self.trade_history = []
        debug_manager.log("TradeAnalyzer initialized", level='info')
    
    def update_data(self, data: pd.DataFrame):
        """Update the player data used for analysis."""
        if not isinstance(data, pd.DataFrame):
            debug_manager.log("Invalid data type provided to update_data", level='error')
            return
        self.data = data
        debug_manager.log("Data updated successfully", level='info')
    
    def get_team_players(self, team: str) -> pd.DataFrame:
        """Get all players for a given team."""
        if self.data is None:
            debug_manager.log("No data available", level='error')
            return pd.DataFrame()
            
        team_data = self.data[self.data['Status'] == team].copy()
        debug_manager.log(f"Retrieved {len(team_data)} players for team {team}", level='debug')
        return team_data
    
    def calculate_team_metrics(self, team_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for a team for different time ranges."""
        time_ranges = {
            '60 Days': 60,
            '30 Days': 30,
            '14 Days': 14,
            '7 Days': 7
        }
        
        metrics_by_range = {}
        team_data = team_data.reset_index()  # Reset index to access Player column
        
        for range_name, days in time_ranges.items():
            # Filter data for time range
            range_data = team_data[team_data['Timestamp'] == f'{days} Days'].copy()
            
            if range_data.empty:
                metrics_by_range[range_name] = {
                    'mean_fpg': 0.0,
                    'median_fpg': 0.0,
                    'std_dev': 0.0,
                    'total_fpts': 0.0,
                    'avg_gp': 0.0
                }
                continue
                
            metrics = {
                'mean_fpg': range_data['FP/G'].mean(),
                'median_fpg': range_data['FP/G'].median(),
                'std_dev': range_data['FP/G'].std(),
                'total_fpts': range_data['FPts'].sum(),
                'avg_gp': range_data['GP'].mean() if 'GP' in range_data.columns else 0.0
            }
            
            metrics_by_range[range_name] = {
                k: float(v) if not pd.isna(v) else 0.0 
                for k, v in metrics.items()
            }
        
        return metrics_by_range
    
    def evaluate_trade_fairness(self, trade_teams: Dict[str, Dict[str, str]], num_top_players: int = 10) -> Dict[str, Dict[str, Any]]:
        """Evaluate the fairness of a trade between teams."""
        analysis_results = {}
        
        for team, players in trade_teams.items():
            # Get current team data
            team_data = self.data[self.data['Status'] == team].copy()
            
            # Calculate pre-trade rosters for each time range
            time_ranges = ['60 Days', '30 Days', '14 Days', '7 Days']
            pre_trade_rosters = {}
            post_trade_rosters = {}
            
            # Get outgoing and incoming players
            outgoing_players = list(players.keys())
            incoming_players = []
            for other_team, other_players in trade_teams.items():
                for player, dest in other_players.items():
                    if dest == team:
                        incoming_players.append(player)
            
            # Calculate pre and post trade rosters for each time range
            for time_range in time_ranges:
                # Get the team's current roster for this time range
                range_data = team_data[team_data['Timestamp'] == time_range].reset_index()
                if not range_data.empty:
                    pre_trade_rosters[time_range] = range_data.nlargest(num_top_players, 'FP/G')[
                        ['Player', 'Team', 'FPts', 'FP/G', 'GP']
                    ].to_dict('records')
                    
                    # Create post-trade roster by removing outgoing players
                    post_trade_data = range_data[~range_data['Player'].isin(outgoing_players)].copy()
                    
                    # Add incoming players from their respective teams
                    incoming_data = []
                    for player in incoming_players:
                        player_data = self.data[
                            (self.data.index == player) & 
                            (self.data['Timestamp'] == time_range)
                        ].reset_index()
                        if not player_data.empty:
                            player_data['Status'] = team  # Update team
                            incoming_data.append(player_data)
                    
                    if incoming_data:
                        incoming_df = pd.concat(incoming_data, ignore_index=True)
                        post_trade_data = pd.concat([post_trade_data, incoming_df], ignore_index=True)
                    
                    # Get top N players from post-trade roster
                    post_trade_rosters[time_range] = post_trade_data.nlargest(num_top_players, 'FP/G')[
                        ['Player', 'Team', 'FPts', 'FP/G', 'GP']
                    ].to_dict('records')
            
            # Calculate metrics using the roster data
            pre_trade_metrics = {}
            post_trade_metrics = {}
            
            for time_range in time_ranges:
                if time_range in pre_trade_rosters:
                    pre_roster_df = pd.DataFrame(pre_trade_rosters[time_range])
                    pre_trade_metrics[time_range] = {
                        'mean_fpg': pre_roster_df['FP/G'].mean(),
                        'median_fpg': pre_roster_df['FP/G'].median(),
                        'std_dev': pre_roster_df['FP/G'].std(),
                        'total_fpts': pre_roster_df['FPts'].sum(),
                        'avg_gp': pre_roster_df['GP'].mean()
                    }
                
                if time_range in post_trade_rosters:
                    post_roster_df = pd.DataFrame(post_trade_rosters[time_range])
                    post_trade_metrics[time_range] = {
                        'mean_fpg': post_roster_df['FP/G'].mean(),
                        'median_fpg': post_roster_df['FP/G'].median(),
                        'std_dev': post_roster_df['FP/G'].std(),
                        'total_fpts': post_roster_df['FPts'].sum(),
                        'avg_gp': post_roster_df['GP'].mean()
                    }
            
            # Calculate value changes
            value_changes = {}
            for time_range in time_ranges:
                if time_range in pre_trade_metrics and time_range in post_trade_metrics:
                    value_changes[time_range] = {
                        'mean_fpg_change': post_trade_metrics[time_range]['mean_fpg'] - pre_trade_metrics[time_range]['mean_fpg'],
                        'total_fpts_change': post_trade_metrics[time_range]['total_fpts'] - pre_trade_metrics[time_range]['total_fpts'],
                        'avg_gp_change': post_trade_metrics[time_range]['avg_gp'] - pre_trade_metrics[time_range]['avg_gp']
                    }
            
            analysis_results[team] = {
                'outgoing_players': outgoing_players,
                'incoming_players': incoming_players,
                'pre_trade_metrics': pre_trade_metrics,
                'post_trade_metrics': post_trade_metrics,
                'value_changes': value_changes,
                'pre_trade_rosters': pre_trade_rosters,
                'post_trade_rosters': post_trade_rosters
            }
        
        return analysis_results

    def get_trade_history(self) -> List[Tuple[Dict[str, Dict[str, str]], str]]:
        """Get the history of analyzed trades."""
        return self.trade_history
    
    def _generate_trade_summary(self, analysis_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate a text summary of the trade analysis."""
        summary_parts = []
        
        for team_name, results in analysis_results.items():
            outgoing = ', '.join(results['outgoing_players']) if results['outgoing_players'] else 'none'
            incoming = ', '.join(results['incoming_players']) if results['incoming_players'] else 'none'
            
            for range_name, value_changes in results['value_changes'].items():
                fpg_change = value_changes['mean_fpg_change']
                total_change = value_changes['total_fpts_change']
                
                summary = f"{team_name} ({range_name}):\n"
                summary += f"  Giving: {outgoing}\n"
                summary += f"  Getting: {incoming}\n"
                summary += f"  Impact: {fpg_change:+.1f} FP/G, {total_change:+.0f} Total FPts\n"
                
                summary_parts.append(summary)
        
        return "\n".join(summary_parts)
