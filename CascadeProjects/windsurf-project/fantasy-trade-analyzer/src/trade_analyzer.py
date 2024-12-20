"""Trade analyzer class for Fantasy Basketball Trade Analyzer."""
from typing import Dict, List, Tuple, Any
import pandas as pd
from debug import debug_manager

class TradeAnalyzer:
    def __init__(self, data_ranges: Dict[str, pd.DataFrame]):
        """Initialize with all time range data."""
        self.data_ranges = data_ranges
        self.trade_history = []
        debug_manager.log("Trade Analyzer initialized", level='debug')
    
    def evaluate_trade_fairness(self, trade_teams: Dict[str, Dict[str, str]], time_range_weights: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the fairness of a trade between teams.
        
        Args:
            trade_teams: Dictionary mapping team names to their trade details
            time_range_weights: Dictionary mapping time ranges to their weights (e.g., {'60 Days': 0.5, '30 Days': 0.3, '14 Days': 0.2})
            
        Returns:
            Dictionary containing analysis results for each team
        """
        debug_manager.log(f"Evaluating trade fairness for {len(trade_teams)} teams", level='debug')
        analysis_results = {}
        
        for team, players in trade_teams.items():
            if isinstance(players, dict) and players:  # Only analyze teams with assigned players
                outgoing_players = list(players.keys())
                incoming_players = []
                
                # Find incoming players for this team
                for other_team, other_players in trade_teams.items():
                    if other_team != team and isinstance(other_players, dict):
                        for player, destination in other_players.items():
                            if destination == team:
                                incoming_players.append(player)
                
                # Calculate weighted values
                outgoing_value = self._calculate_weighted_player_values(outgoing_players, time_range_weights)
                incoming_value = self._calculate_weighted_player_values(incoming_players, time_range_weights)
                net_value = incoming_value - outgoing_value
                
                analysis_results[team] = {
                    'outgoing_players': outgoing_players,
                    'incoming_players': incoming_players,
                    'outgoing_value': outgoing_value,
                    'incoming_value': incoming_value,
                    'value_change': net_value,
                    'value_details': {
                        'outgoing': self._get_player_value_details(outgoing_players, time_range_weights),
                        'incoming': self._get_player_value_details(incoming_players, time_range_weights)
                    }
                }
                
                debug_manager.log(f"Analysis for {team}: Value change = {net_value}", level='debug')
        
        return analysis_results
    
    def _calculate_weighted_player_values(self, players: List[str], time_range_weights: Dict[str, float]) -> float:
        """Calculate the weighted value of players across all time ranges."""
        if not players:
            return 0.0
            
        total_value = 0.0
        for player in players:
            player_value = 0.0
            for time_range, weight in time_range_weights.items():
                if time_range in self.data_ranges:
                    player_data = self.data_ranges[time_range][self.data_ranges[time_range]['Player'] == player]
                    if not player_data.empty:
                        # Use FP/G for value calculation
                        value = player_data['FP/G'].mean() * weight
                        player_value += value
            total_value += player_value
                
        return total_value
        
    def _get_player_value_details(self, players: List[str], time_range_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Get detailed value breakdown for each player across time ranges."""
        details = {}
        for player in players:
            player_details = {}
            for time_range, weight in time_range_weights.items():
                if time_range in self.data_ranges:
                    player_data = self.data_ranges[time_range][self.data_ranges[time_range]['Player'] == player]
                    if not player_data.empty:
                        player_details[time_range] = {
                            'FP/G': player_data['FP/G'].mean(),
                            'weighted_value': player_data['FP/G'].mean() * weight
                        }
            details[player] = player_details
        return details
        
    def get_trade_history(self) -> List[Tuple[str, str]]:
        """Get the history of analyzed trades."""
        return self.trade_history

    def update_data(self, new_data_ranges: Dict[str, pd.DataFrame]):
        """Update the analyzer with new data when time range changes."""
        self.data_ranges = new_data_ranges
        debug_manager.log("Trade Analyzer data updated", level='debug')
