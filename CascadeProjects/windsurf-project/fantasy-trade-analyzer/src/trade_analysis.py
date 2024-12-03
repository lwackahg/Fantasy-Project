"""Module for trade analysis functionality"""
import streamlit as st
import pandas as pd
import numpy as np
from config.team_mappings import TEAM_MAPPINGS

def get_team_name(team_id):
    """Get full team name from team ID"""
    return TEAM_MAPPINGS.get(team_id, team_id)

class TradeAnalyzer:
    """Class for analyzing fantasy trades"""
    
    def __init__(self, data=None):
        """Initialize TradeAnalyzer with player data"""
        self.data = data if data is not None else {}
        self.data_ranges = st.session_state.data_ranges if hasattr(st.session_state, 'data_ranges') else {}
    
    def _calculate_player_value(self, player_name):
        """Calculate a player's value based on recent performance"""
        value = 0
        weights = {'7 Days': 0.4, '14 Days': 0.3, '30 Days': 0.2, '60 Days': 0.1}
        
        for time_range, weight in weights.items():
            if time_range in self.data_ranges:
                data = self.data_ranges[time_range]
                player_data = data[data['Player'] == player_name]
                
                if not player_data.empty:
                    # Calculate value based on FP/G and games played
                    fpg = player_data['FP/G'].iloc[0]
                    gp = player_data['GP'].iloc[0] if 'GP' in player_data.columns else 0
                    
                    # Value formula: FP/G * sqrt(GP) * weight
                    # This gives more weight to consistent performers
                    value += fpg * (gp ** 0.5) * weight
        
        return value
    
    def _calculate_fairness_score(self, before_stats, after_stats, incoming_value, outgoing_value, num_incoming, num_outgoing):
        """Calculate trade fairness score based on multiple factors"""
        # Base score from value comparison
        if max(incoming_value, outgoing_value) == 0:
            value_ratio = 1.0
        else:
            value_ratio = min(incoming_value, outgoing_value) / max(incoming_value, outgoing_value)
        
        # Calculate FP/G impact across time ranges
        fpg_impact = 0
        time_weights = {'7 Days': 0.4, '14 Days': 0.3, '30 Days': 0.2, '60 Days': 0.1}
        
        for time_range, weight in time_weights.items():
            if time_range in before_stats and time_range in after_stats:
                before_fpg = before_stats[time_range]['mean_fpg']
                after_fpg = after_stats[time_range]['mean_fpg']
                
                # Calculate normalized impact
                if before_fpg > 0:
                    impact = (after_fpg - before_fpg) / before_fpg
                    fpg_impact += impact * weight
        
        # Player count balance factor
        count_ratio = min(num_incoming, num_outgoing) / max(num_incoming, num_outgoing) if max(num_incoming, num_outgoing) > 0 else 1.0
        
        # Combine factors with weights
        fairness_score = (
            value_ratio * 0.4 +  # Value comparison
            (1 + fpg_impact) * 0.4 +  # Team performance impact
            count_ratio * 0.2  # Player count balance
        )
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, fairness_score))

    def evaluate_trade_fairness(self, trade_teams, top_x=None):
        """Evaluate trade fairness for all teams involved"""
        analysis = {}
        
        # Store initial team states
        initial_states = {}
        for team in list(trade_teams.keys()):
            initial_states[team] = self.calculate_team_stats(team, top_x)
        
        # First pass: Organize incoming/outgoing players for each team
        for team, players in trade_teams.items():
            outgoing_players = []
            incoming_players = []
            
            # Handle dictionary format where players have destinations
            if isinstance(players, dict):
                outgoing_players = list(players.keys())
                for player, dest_team in players.items():
                    if dest_team in trade_teams:
                        if dest_team not in analysis:
                            analysis[dest_team] = {'incoming_players': [], 'outgoing_players': []}
                        analysis[dest_team]['incoming_players'].append(player)
            else:
                outgoing_players = list(players) if players else []  # Convert to list and handle empty case
            
            if team not in analysis:
                analysis[team] = {'incoming_players': [], 'outgoing_players': []}
            analysis[team]['outgoing_players'].extend(outgoing_players)
        
        # Calculate statistics and value changes for each team
        for team, team_data in analysis.items():
            # Use stored initial state for before stats
            before_stats = initial_states[team]
            after_stats = self.calculate_team_stats_after_trade(team, team_data, top_x)
            
            # Calculate value changes
            incoming_value = sum(self._calculate_player_value(player) 
                               for player in team_data['incoming_players'])
            outgoing_value = sum(self._calculate_player_value(player) 
                               for player in team_data['outgoing_players'])
            
            # Store all analysis data
            analysis[team].update({
                'before_stats': before_stats['stats'],
                'after_stats': after_stats['stats'],
                'top_players_before': before_stats['player_data'],
                'top_players_after': after_stats['player_data'],
                'value_change': incoming_value - outgoing_value,
                'fairness_score': self._calculate_fairness_score(
                    before_stats['stats'],
                    after_stats['stats'],
                    incoming_value,
                    outgoing_value,
                    len(team_data['incoming_players']),
                    len(team_data['outgoing_players'])
                )
            })
        
        return analysis

    def calculate_player_value(self, player_data):
        """Calculate a player's value based on their stats across time ranges"""
        value = 0
        weights = {
            '7 Days': 0.1,
            '14 Days': 0.4,
            '30 Days': 0.4,
            '60 Days': 0.1
        }
        
        for time_range, weight in weights.items():
            if time_range not in player_data:
                continue
            stats = player_data[time_range]
            if len(stats) == 0:
                continue
            
            # Use mean FP/G as the primary value metric
            value += stats['FP/G'].mean() * weight
        
        return value

    def calculate_trade_fairness(self, before_stats, after_stats, team_data):
        """Calculate trade fairness based on multiple factors"""
        fairness_score = 0.0
        weights = {
            'value_ratio': 0.4,      # How close the player values are
            'stats_impact': 0.4,     # Impact on team's statistical performance
            'depth_impact': 0.2      # Impact on team depth/roster balance
        }
        
        # 1. Calculate value ratio (how close the incoming/outgoing values are)
        incoming_value = sum(self.calculate_player_value(self.data[player]) 
                           for player in team_data['incoming_players'])
        outgoing_value = sum(self.calculate_player_value(self.data[player]) 
                           for player in team_data['outgoing_players'])
        
        # Avoid division by zero
        max_value = max(incoming_value, outgoing_value)
        min_value = min(incoming_value, outgoing_value)
        if max_value == 0:
            value_ratio = 1.0  # If both values are 0, consider it neutral
        else:
            value_ratio = min_value / max_value
        
        # 2. Calculate statistical impact
        stats_changes = []
        key_metrics = ['mean_fpg', 'median_fpg', 'total_fpts']
        
        for metric in key_metrics:
            for time_range in ['7 Days', '14 Days', '30 Days']:
                if time_range in before_stats and time_range in after_stats:
                    before_val = before_stats[time_range][metric]
                    after_val = after_stats[time_range][metric]
                    if before_val != 0:
                        pct_change = (after_val - before_val) / abs(before_val)
                        stats_changes.append(pct_change)
        
        # Calculate average statistical impact
        if stats_changes:
            avg_stats_change = sum(stats_changes) / len(stats_changes)
            # Convert to a 0-1 scale, with 0.5 being neutral
            stats_impact = 0.5 + (avg_stats_change / 2)
            # Clamp between 0 and 1
            stats_impact = max(0, min(1, stats_impact))
        else:
            stats_impact = 0.5
        
        # 3. Calculate depth impact
        before_depth = len([p for p in before_stats['7 Days'].get('player_data', []) 
                          if p.get('FP/G', 0) > 20])  # Count solid contributors
        after_depth = len([p for p in after_stats['7 Days'].get('player_data', []) 
                         if p.get('FP/G', 0) > 20])
        
        depth_ratio = after_depth / before_depth if before_depth > 0 else 1.0
        depth_impact = min(1.0, depth_ratio)
        
        # Combine all factors with weights
        fairness_score = (
            weights['value_ratio'] * value_ratio +
            weights['stats_impact'] * stats_impact +
            weights['depth_impact'] * depth_impact
        )
        
        return fairness_score

    def calculate_team_stats(self, team, top_x=None):
        """Calculate team statistics from roster data"""
        team_stats = {}
        player_data = {}
        
        for time_range, data in self.data_ranges.items():
            if data is None:
                continue
            
            # Get all team data first
            team_data = data[data['Status'] == team].copy()
            
            if not team_data.empty:
                # Sort players by FP/G for selecting top players
                sorted_players = team_data.sort_values('FP/G', ascending=False)
                
                # If top_x is specified, use only top players for stats calculation
                if top_x and len(sorted_players) > top_x:
                    stats_players = sorted_players.head(top_x)
                else:
                    stats_players = sorted_players
                
                # Calculate stats using only the selected players
                all_stats = {
                    'mean_fpg': stats_players['FP/G'].mean(),
                    'median_fpg': stats_players['FP/G'].median(),
                    'std_fpg': stats_players['FP/G'].std(),
                    'total_fpts': stats_players['FPts'].sum() if 'FPts' in stats_players.columns else 0,
                    'avg_gp': stats_players['GP'].mean() if 'GP' in stats_players.columns else 0,
                    'player_data': stats_players.to_dict('records')  # Include full player data
                }
                
                team_stats[time_range] = all_stats
                player_data[time_range] = sorted_players  # Store full player DataFrame
        
        return {'stats': team_stats, 'player_data': player_data}

    def calculate_team_stats_after_trade(self, team, team_data, top_x=None):
        """Calculate team stats after the proposed trade"""
        after_stats = {}
        player_data = {}
        
        for time_range, data in self.data_ranges.items():
            if data is None:
                continue
            
            # Get current team roster
            current_roster = data[data['Status'] == team].copy()
            
            # Remove outgoing players
            remaining_players = current_roster[~current_roster['Player'].isin(team_data['outgoing_players'])]['Player'].tolist()
            
            # Combine remaining roster with incoming players
            new_roster = pd.concat([
                data[data['Player'].isin(remaining_players)],
                data[data['Player'].isin(team_data['incoming_players'])]
            ])
            
            if not new_roster.empty:
                # Sort players by FP/G for selecting top players
                sorted_players = new_roster.sort_values('FP/G', ascending=False)
                
                # If top_x is specified, use only top players for stats calculation
                if top_x and len(sorted_players) > top_x:
                    stats_players = sorted_players.head(top_x)
                else:
                    stats_players = sorted_players
                
                # Calculate stats using only the selected players
                all_stats = {
                    'mean_fpg': stats_players['FP/G'].mean(),
                    'median_fpg': stats_players['FP/G'].median(),
                    'std_fpg': stats_players['FP/G'].std(),
                    'total_fpts': stats_players['FPts'].sum() if 'FPts' in stats_players.columns else 0,
                    'avg_gp': stats_players['GP'].mean() if 'GP' in stats_players.columns else 0,
                    'player_data': stats_players.to_dict('records')  # Include full player data
                }
                
                after_stats[time_range] = all_stats
                player_data[time_range] = sorted_players  # Store full player DataFrame
        
        return {'stats': after_stats, 'player_data': player_data}

def get_trade_details(teams):
    """Get trade details for all teams involved"""
    trade_details = {}
    for team in teams:
        team_data = st.session_state.trade_analysis['teams'][team]
        trade_details[team] = {
            'incoming': team_data['incoming_players'],
            'outgoing': team_data['outgoing_players']
        }
    return trade_details

def calculate_team_stats(roster_data):
    """Calculate team statistics from roster data"""
    if roster_data.empty:
        return {
            'mean_fpg': 0,
            'median_fpg': 0,
            'std_fpg': 0,
            'total_fpts': 0,
            'avg_gp': 0
        }
    
    stats = {
        'mean_fpg': roster_data['FP/G'].mean(),
        'median_fpg': roster_data['FP/G'].median(),
        'std_fpg': roster_data['FP/G'].std(),
        'total_fpts': roster_data['FPts'].sum() if 'FPts' in roster_data.columns else 0,
        'avg_gp': roster_data['GP'].mean() if 'GP' in roster_data.columns else 0
    }
    
    return stats

def get_team_stats_before_after(team, team_data):
    """Calculate team stats before and after trade"""
    before_stats = {}
    after_stats = {}
    
    for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
        if time_range not in st.session_state.data_ranges:
            continue
            
        current_data = st.session_state.data_ranges[time_range]
        current_roster = current_data[current_data['Status'] == team]
        incoming = team_data['incoming_players']
        outgoing = team_data['outgoing_players']
        new_roster = current_roster[~current_roster['Player'].isin(outgoing)]
        incoming_roster = current_data[current_data['Player'].isin(incoming)]
        new_roster = pd.concat([new_roster, incoming_roster])
        
        # Calculate team stats
        before_stats[time_range] = calculate_team_stats(current_roster)
        after_stats[time_range] = calculate_team_stats(new_roster)
        
        # Add player data to stats
        before_stats[time_range]['player_data'] = current_roster.to_dict('records')
        after_stats[time_range]['player_data'] = new_roster.to_dict('records')
    
    return before_stats, after_stats
