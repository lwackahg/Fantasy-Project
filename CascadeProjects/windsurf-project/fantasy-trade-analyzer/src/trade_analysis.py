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

    def calculate_trade_fairness(self, trade_teams, top_x=None):
        """Calculate trade fairness for all teams involved in the trade"""
        analysis = {}
        
        for team, players in trade_teams.items():
            team_analysis = {
                'incoming_players': players,
                'outgoing_players': [],  # Will be filled based on other teams' incoming players
                'value_change': 0,
                'before_stats': None,
                'after_stats': None,
                'top_players_before': None,
                'top_players_after': None
            }
            analysis[team] = team_analysis
        
        # Fill in outgoing players
        for team, team_data in analysis.items():
            for other_team, other_data in analysis.items():
                if team != other_team:
                    team_data['outgoing_players'].extend(other_data['incoming_players'])
        
        # Calculate statistics and value changes
        for team, team_data in analysis.items():
            before_stats = self.calculate_team_stats(team, top_x)
            after_stats = self.calculate_team_stats_after_trade(team, team_data, top_x)
            
            team_data['before_stats'] = before_stats['stats']
            team_data['after_stats'] = after_stats['stats']
            team_data['top_players_before'] = before_stats['top_players']
            team_data['top_players_after'] = after_stats['top_players']
            
            # Calculate value changes
            incoming_value = sum(self.calculate_player_value(self.data[player]) 
                               for player in team_data['incoming_players'])
            outgoing_value = sum(self.calculate_player_value(self.data[player]) 
                               for player in team_data['outgoing_players'])
            team_data['value_change'] = incoming_value - outgoing_value
        
        return analysis

    def calculate_team_stats(self, team, top_x=None):
        """Calculate team statistics from roster data"""
        team_stats = {}
        top_players = {}
        
        for time_range, data in self.data_ranges.items():
            if data is None:
                continue
            
            team_data = data[data['Team'] == team].copy()
            
            # Store top players before filtering
            if top_x:
                top_players[time_range] = team_data.nlargest(top_x, 'FP/G')[['Player', 'FP/G', 'MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK']]
                team_data = top_players[time_range]
            
            if not team_data.empty:
                team_stats[time_range] = {
                    'FP/G': team_data['FP/G'].mean(),
                    'MIN': team_data['MIN'].mean(),
                    'PTS': team_data['PTS'].mean(),
                    'AST': team_data['AST'].mean(),
                    'REB': team_data['REB'].mean(),
                    'STL': team_data['STL'].mean(),
                    'BLK': team_data['BLK'].mean(),
                    'TOV': team_data['TOV'].mean() if 'TOV' in team_data.columns else 0,
                }
        
        return {'stats': team_stats, 'top_players': top_players}

    def calculate_team_stats_after_trade(self, team, team_data, top_x=None):
        """Calculate team stats after the proposed trade"""
        after_stats = {}
        top_players = {}
        
        for time_range, data in self.data_ranges.items():
            if data is None:
                continue
            
            # Get current team roster
            current_roster = data[data['Team'] == team].copy()
            
            # Remove outgoing players and add incoming players
            new_roster = current_roster[~current_roster['Player'].isin(team_data['outgoing_players'])]
            incoming_roster = data[data['Player'].isin(team_data['incoming_players'])]
            new_roster = pd.concat([new_roster, incoming_roster])
            
            # Store top players before filtering
            if top_x:
                top_players[time_range] = new_roster.nlargest(top_x, 'FP/G')[['Player', 'FP/G', 'MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK']]
                new_roster = top_players[time_range]
            
            if not new_roster.empty:
                after_stats[time_range] = {
                    'FP/G': new_roster['FP/G'].mean(),
                    'MIN': new_roster['MIN'].mean(),
                    'PTS': new_roster['PTS'].mean(),
                    'AST': new_roster['AST'].mean(),
                    'REB': new_roster['REB'].mean(),
                    'STL': new_roster['STL'].mean(),
                    'BLK': new_roster['BLK'].mean(),
                    'TOV': new_roster['TOV'].mean() if 'TOV' in new_roster.columns else 0,
                }
        
        return {'stats': after_stats, 'top_players': top_players}

    def evaluate_trade_fairness(self, trade_teams, top_x=None):
        """Evaluate the fairness of a trade by analyzing team changes"""
        return self.calculate_trade_fairness(trade_teams, top_x)

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
            'std_fpg': 0
        }
    
    return {
        'mean_fpg': roster_data['FP/G'].mean(),
        'median_fpg': roster_data['FP/G'].median(),
        'std_fpg': roster_data['FP/G'].std()
    }

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
        
        before_stats[time_range] = calculate_team_stats(current_roster)
        after_stats[time_range] = calculate_team_stats(new_roster)
    
    return before_stats, after_stats
