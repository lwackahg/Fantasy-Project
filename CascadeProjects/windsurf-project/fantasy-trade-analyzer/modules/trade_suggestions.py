"""Trade Suggestion Engine - Suggests optimal trades based on exponential value calculations."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations
import streamlit as st


def calculate_exponential_value(fpts: float) -> float:
	"""
	Calculate exponential value of a player.
	Accounts for the fact that elite players are worth exponentially more than their FPts suggest.
	
	Uses a power function to create exponential scaling:
	- 45 FPts player = ~45 value
	- 70 FPts player = ~140 value (3x more valuable than the difference suggests)
	- 95 FPts player = ~380 value (way more valuable than linear scaling)
	"""
	# Base exponential factor (1.5 creates good separation)
	# Higher = more exponential, lower = more linear
	exponent = 1.5
	
	# Scale factor to keep values reasonable
	scale = 0.5
	
	return (fpts ** exponent) * scale


def calculate_player_value(player_data: pd.Series, include_consistency: bool = True) -> float:
	"""
	Calculate comprehensive player value including production and consistency.
	
	Args:
		player_data: Series with 'Mean FPts', 'CV %', 'GP'
		include_consistency: Whether to factor in consistency bonus/penalty
	
	Returns:
		Total player value (exponential base + consistency adjustment)
	"""
	fpts = player_data.get('Mean FPts', 0)
	cv = player_data.get('CV %', 30)
	gp = player_data.get('GP', 0)
	
	# Base exponential value
	base_value = calculate_exponential_value(fpts)
	
	if not include_consistency:
		return base_value
	
	# Consistency multiplier (ranges from 0.85 to 1.15)
	# Very consistent (CV < 20): +15% value
	# Moderate (CV 20-30): 0% adjustment
	# Volatile (CV > 30): -15% value
	if cv < 20:
		consistency_mult = 1.15
	elif cv <= 30:
		consistency_mult = 1.0
	else:
		# Scale down more for very volatile players
		consistency_mult = max(0.85, 1.0 - (cv - 30) * 0.01)
	
	# Games played penalty for low sample size
	if gp < 10:
		gp_mult = 0.7
	elif gp < 20:
		gp_mult = 0.85
	else:
		gp_mult = 1.0
	
	return base_value * consistency_mult * gp_mult


def find_trade_suggestions(
	your_team: pd.DataFrame,
	other_teams: Dict[str, pd.DataFrame],
	trade_patterns: List[str] = ['1-for-1', '2-for-1', '2-for-2'],
	min_value_gain: float = 5.0,
	max_suggestions: int = 20,
	target_teams: List[str] = None,
	exclude_players: List[str] = None
) -> List[Dict]:
	"""
	Find optimal trade suggestions based on exponential value calculations.
	
	Args:
		your_team: DataFrame of your players
		other_teams: Dict of team_name -> DataFrame of their players
		trade_patterns: List of trade patterns to consider
		min_value_gain: Minimum value gain to suggest a trade
		max_suggestions: Maximum number of suggestions to return
		target_teams: Optional list of teams to target (None = all teams)
		exclude_players: Optional list of players to exclude from your side
	
	Returns:
		List of trade suggestions sorted by value gain
	"""
	suggestions = []
	
	# Filter teams if specified
	if target_teams:
		other_teams = {k: v for k, v in other_teams.items() if k in target_teams}
	
	# Exclude players if specified
	if exclude_players:
		your_team = your_team[~your_team['Player'].isin(exclude_players)].copy()
	
	# Calculate values for all players
	your_team['Value'] = your_team.apply(calculate_player_value, axis=1)
	
	for team_name, team_df in other_teams.items():
		if team_df.empty:
			continue
		
		team_df = team_df.copy()
		team_df['Value'] = team_df.apply(calculate_player_value, axis=1)
		
		# Try each trade pattern
		for pattern in trade_patterns:
			if pattern == '1-for-1':
				suggestions.extend(_find_1_for_1_trades(your_team, team_df, team_name, min_value_gain))
			elif pattern == '2-for-1':
				suggestions.extend(_find_2_for_1_trades(your_team, team_df, team_name, min_value_gain))
			elif pattern == '1-for-2':
				suggestions.extend(_find_1_for_2_trades(your_team, team_df, team_name, min_value_gain))
			elif pattern == '2-for-2':
				suggestions.extend(_find_2_for_2_trades(your_team, team_df, team_name, min_value_gain))
			elif pattern == '3-for-1':
				suggestions.extend(_find_3_for_1_trades(your_team, team_df, team_name, min_value_gain))
			elif pattern == '1-for-3':
				suggestions.extend(_find_1_for_3_trades(your_team, team_df, team_name, min_value_gain))
			elif pattern == '3-for-2':
				suggestions.extend(_find_3_for_2_trades(your_team, team_df, team_name, min_value_gain))
			elif pattern == '2-for-3':
				suggestions.extend(_find_2_for_3_trades(your_team, team_df, team_name, min_value_gain))
			elif pattern == '3-for-3':
				suggestions.extend(_find_3_for_3_trades(your_team, team_df, team_name, min_value_gain))
	
	# Sort by value gain and limit
	suggestions.sort(key=lambda x: x['value_gain'], reverse=True)
	return suggestions[:max_suggestions]


def _find_1_for_1_trades(your_team, other_team, team_name, min_gain):
	"""Find 1-for-1 trade opportunities."""
	trades = []
	
	for _, your_player in your_team.iterrows():
		for _, their_player in other_team.iterrows():
			value_gain = their_player['Value'] - your_player['Value']
			
			if value_gain >= min_gain and _is_realistic_trade([your_player], [their_player]):
				trades.append({
					'pattern': '1-for-1',
					'team': team_name,
					'you_give': [your_player['Player']],
					'you_get': [their_player['Player']],
					'your_value': your_player['Value'],
					'their_value': their_player['Value'],
					'value_gain': value_gain,
					'your_fpts': [your_player['Mean FPts']],
					'their_fpts': [their_player['Mean FPts']],
					'your_cv': [your_player['CV %']],
					'their_cv': [their_player['CV %']]
				})
	
	return trades


def _find_2_for_1_trades(your_team, other_team, team_name, min_gain):
	"""Find 2-for-1 trade opportunities (you give 2, get 1)."""
	trades = []
	
	# You give 2, get 1 elite
	for your_combo in combinations(your_team.iterrows(), 2):
		your_players = [p[1] for p in your_combo]
		your_total_value = sum(p['Value'] for p in your_players)
		
		for _, their_player in other_team.iterrows():
			value_gain = their_player['Value'] - your_total_value
			
			if value_gain >= min_gain and _is_realistic_trade(your_players, [their_player]):
				trades.append({
					'pattern': '2-for-1',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [their_player['Player']],
					'your_value': your_total_value,
					'their_value': their_player['Value'],
					'value_gain': value_gain,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [their_player['Mean FPts']],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [their_player['CV %']]
				})
	
	return trades


def _find_2_for_2_trades(your_team, other_team, team_name, min_gain):
	"""Find 2-for-2 trade opportunities."""
	trades = []
	
	for your_combo in combinations(your_team.iterrows(), 2):
		your_players = [p[1] for p in your_combo]
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_combo in combinations(other_team.iterrows(), 2):
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			
			value_gain = their_total_value - your_total_value
			
			if value_gain >= min_gain and _is_realistic_trade(your_players, their_players):
				trades.append({
					'pattern': '2-for-2',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_total_value,
					'their_value': their_total_value,
					'value_gain': value_gain,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades


def _find_3_for_1_trades(your_team, other_team, team_name, min_gain):
	"""Find 3-for-1 trade opportunities (you give 3, get 1 superstar)."""
	trades = []
	
	for your_combo in combinations(your_team.iterrows(), 3):
		your_players = [p[1] for p in your_combo]
		your_total_value = sum(p['Value'] for p in your_players)
		
		for _, their_player in other_team.iterrows():
			value_gain = their_player['Value'] - your_total_value
			
			if value_gain >= min_gain and _is_realistic_trade(your_players, [their_player]):
				trades.append({
					'pattern': '3-for-1',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [their_player['Player']],
					'your_value': your_total_value,
					'their_value': their_player['Value'],
					'value_gain': value_gain,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [their_player['Mean FPts']],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [their_player['CV %']]
				})
	
	return trades


def _find_3_for_2_trades(your_team, other_team, team_name, min_gain):
	"""Find 3-for-2 trade opportunities."""
	trades = []
	
	for your_combo in combinations(your_team.iterrows(), 3):
		your_players = [p[1] for p in your_combo]
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_combo in combinations(other_team.iterrows(), 2):
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			
			value_gain = their_total_value - your_total_value
			
			if value_gain >= min_gain and _is_realistic_trade(your_players, their_players):
				trades.append({
					'pattern': '3-for-2',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_total_value,
					'their_value': their_total_value,
					'value_gain': value_gain,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _find_1_for_2_trades(your_team, other_team, team_name, min_gain):
	"""Find 1-for-2 trades (you give 1, get 2)."""
	trades = []
	
	for _, your_player in your_team.iterrows():
		for their_combo in combinations(other_team.iterrows(), 2):
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			
			value_gain = their_total_value - your_player['Value']
			
			if value_gain >= min_gain and _is_realistic_trade([your_player], their_players):
				trades.append({
					'pattern': '1-for-2',
					'team': team_name,
					'you_give': [your_player['Player']],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_player['Value'],
					'their_value': their_total_value,
					'value_gain': value_gain,
					'your_fpts': [your_player['Mean FPts']],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [your_player['CV %']],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _find_1_for_3_trades(your_team, other_team, team_name, min_gain):
	"""Find 1-for-3 trades (you give 1, get 3)."""
	trades = []
	
	for _, your_player in your_team.iterrows():
		for their_combo in combinations(other_team.iterrows(), 3):
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			
			value_gain = their_total_value - your_player['Value']
			
			if value_gain >= min_gain and _is_realistic_trade([your_player], their_players):
				trades.append({
					'pattern': '1-for-3',
					'team': team_name,
					'you_give': [your_player['Player']],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_player['Value'],
					'their_value': their_total_value,
					'value_gain': value_gain,
					'your_fpts': [your_player['Mean FPts']],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [your_player['CV %']],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _find_2_for_3_trades(your_team, other_team, team_name, min_gain):
	"""Find 2-for-3 trades."""
	trades = []
	
	for your_combo in combinations(your_team.iterrows(), 2):
		your_players = [p[1] for p in your_combo]
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_combo in combinations(other_team.iterrows(), 3):
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			
			value_gain = their_total_value - your_total_value
			
			if value_gain >= min_gain and _is_realistic_trade(your_players, their_players):
				trades.append({
					'pattern': '2-for-3',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_total_value,
					'their_value': their_total_value,
					'value_gain': value_gain,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _find_3_for_3_trades(your_team, other_team, team_name, min_gain):
	"""Find 3-for-3 trades."""
	trades = []
	
	for your_combo in combinations(your_team.iterrows(), 3):
		your_players = [p[1] for p in your_combo]
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_combo in combinations(other_team.iterrows(), 3):
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			
			value_gain = their_total_value - your_total_value
			
			if value_gain >= min_gain and _is_realistic_trade(your_players, their_players):
				trades.append({
					'pattern': '3-for-3',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_total_value,
					'their_value': their_total_value,
					'value_gain': value_gain,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _is_realistic_trade(your_players, their_players):
	"""
	Check if a trade is realistic (not too lopsided).
	Prevents suggesting trades that no one would accept.
	Considers both FPts and consistency (CV%).
	"""
	your_avg_fpts = sum(p['Mean FPts'] for p in your_players) / len(your_players)
	their_avg_fpts = sum(p['Mean FPts'] for p in their_players) / len(their_players)
	your_avg_cv = sum(p['CV %'] for p in your_players) / len(your_players)
	their_avg_cv = sum(p['CV %'] for p in their_players) / len(their_players)
	
	# Don't suggest trades where average player quality is too different
	# Very tight restriction: max 15% difference in average FPts
	fpts_ratio = max(your_avg_fpts, their_avg_fpts) / min(your_avg_fpts, their_avg_fpts) if min(your_avg_fpts, their_avg_fpts) > 0 else 999
	
	if fpts_ratio > 1.15:  # More than 15% difference in average quality
		return False
	
	# Check for extreme value mismatches (catching buy-low/sell-high)
	your_total_fpts = sum(p['Mean FPts'] for p in your_players)
	their_total_fpts = sum(p['Mean FPts'] for p in their_players)
	
	# Very tight restrictions on total FPts imbalance
	if len(your_players) > len(their_players):
		# You're consolidating - you give more total FPts for better single player
		max_ratio = 1.15  # 15% max imbalance for consolidation
	elif len(their_players) > len(your_players):
		# They're consolidating - you get more total FPts for giving up better player
		max_ratio = 1.10  # 10% max imbalance
	else:
		# Equal player counts - extremely tight restrictions
		max_ratio = 1.05  # 5% max imbalance
	
	total_ratio = max(your_total_fpts, their_total_fpts) / min(your_total_fpts, their_total_fpts) if min(your_total_fpts, their_total_fpts) > 0 else 999
	
	if total_ratio > max_ratio:
		return False
	
	# CV% check: Don't trade consistent players for volatile ones unless getting significant upgrade
	# If you're giving up more consistent players (lower CV%), you better be getting a big FPts upgrade
	if your_avg_cv < their_avg_cv:
		# You're giving up consistency, need at least 5% FPts upgrade
		if their_avg_fpts < your_avg_fpts * 1.05:
			return False
	
	# Additional check: prevent trading elite players for scrubs
	your_max_fpts = max(p['Mean FPts'] for p in your_players)
	their_max_fpts = max(p['Mean FPts'] for p in their_players)
	
	# If you're giving up an elite player (70+), they must give back at least a solid player (50+)
	if your_max_fpts >= 70 and their_max_fpts < 50:
		return False
	
	# If they're giving up an elite player, you must give back at least a solid player
	if their_max_fpts >= 70 and your_max_fpts < 50:
		return False
	
	# Additional check: best player comparison
	# The best player in the trade shouldn't be too much better than the other side's best
	best_player_ratio = max(your_max_fpts, their_max_fpts) / min(your_max_fpts, their_max_fpts) if min(your_max_fpts, their_max_fpts) > 0 else 999
	
	# Max 8% difference in best players
	if best_player_ratio > 1.08:
		return False
	
	# Check individual player matchups - each player you give should have a comparable player you get
	your_fpts_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
	their_fpts_sorted = sorted([p['Mean FPts'] for p in their_players], reverse=True)
	
	# For each of your players, check if there's a comparable player on their side
	for i, your_fpts in enumerate(your_fpts_sorted):
		if i < len(their_fpts_sorted):
			their_fpts = their_fpts_sorted[i]
			# Each player matchup should be within 10%
			player_ratio = max(your_fpts, their_fpts) / min(your_fpts, their_fpts) if min(your_fpts, their_fpts) > 0 else 999
			if player_ratio > 1.10:
				return False
	
	return True
