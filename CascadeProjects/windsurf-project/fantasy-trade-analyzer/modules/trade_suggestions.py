"""Trade Suggestion Engine - Suggests optimal trades based on exponential value calculations."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import streamlit as st

# Tier thresholds (FP/G) and league settings extracted from existing logic.
# Changing these values will change behavior; they are defined here only
# to centralize configuration.
TIER_TOP5_FP = 90
TIER_TOP10_FP = 80
TIER_TOP20_FP = 70
TIER_SOLID_STARTER_FP = 50
TIER_STREAMER_FP = 35

ROSTER_SIZE = 10  # From league rules
REPLACEMENT_PERCENTILE = 0.85  # Top 85% of rostered players

def calculate_exponential_value(fpts: float) -> float:
	"""
	Calculate exponential value of a player.
	Accounts for the fact that elite players are worth exponentially more than their FPts suggest.
	
	Uses a power function to create exponential scaling:
	- 45 FPts player = ~200 value
	- 70 FPts player = ~550 value (2.75x more valuable)
	- 95 FPts player = ~1200 value (way more valuable - top 5 tier)
	
	Additional tier bonuses for truly elite players:
	- 90+ FPts (Top 5): +30% bonus (Jokic, Giannis tier)
	- 80-90 FPts (Top 10): +15% bonus
	- 70-80 FPts (Top 20): +8% bonus
	"""
	# Increased exponential factor (1.8 creates stronger separation for elite players)
	# Higher = more exponential, lower = more linear
	exponent = 1.8
	
	# Scale factor to keep values reasonable
	scale = 0.35
	
	base_value = (fpts ** exponent) * scale
	
	# Add tier bonuses for truly elite players (scarcity premium)
	if fpts >= 90:
		# Top 5 players - massive scarcity bonus
		base_value *= 1.30
	elif fpts >= 80:
		# Top 10 players - significant scarcity bonus
		base_value *= 1.15
	elif fpts >= 70:
		# Top 20 players - moderate scarcity bonus
		base_value *= 1.08
	
	return base_value


def calculate_league_scarcity_context(all_teams_data: Dict[str, pd.DataFrame]) -> Dict:
	"""
	Analyze the entire league player pool to calculate scarcity metrics.
	
	Args:
		all_teams_data: Dict of team_name -> DataFrame with all rostered players
	
	Returns:
		Dict with scarcity metrics: replacement_level, tier_counts, position_scarcity, etc.
	"""
	# Combine all players across all teams
	all_players = []
	for team_df in all_teams_data.values():
		if not team_df.empty:
			all_players.append(team_df)
	
	if not all_players:
		return {'replacement_level': 0, 'tier_counts': {}, 'position_scarcity': {}}
	
	league_df = pd.concat(all_players, ignore_index=True)
	
	# Calculate replacement level (top 85% of rostered players)
	num_teams = len(all_teams_data)
	roster_size = ROSTER_SIZE
	replacement_idx = int(num_teams * roster_size * REPLACEMENT_PERCENTILE)
	
	if len(league_df) >= replacement_idx:
		replacement_level = league_df.nlargest(replacement_idx, 'Mean FPts')['Mean FPts'].iloc[-1]
	else:
		replacement_level = league_df['Mean FPts'].min()
	
	# Tier classification based on FP/G
	def assign_tier(fpts):
		if fpts >= TIER_TOP5_FP:
			return 1  # Top 5 tier
		elif fpts >= TIER_TOP10_FP:
			return 2  # Top 10 tier
		elif fpts >= TIER_TOP20_FP:
			return 3  # Top 20 tier
		elif fpts >= TIER_SOLID_STARTER_FP:
			return 4  # Solid starters
		elif fpts >= TIER_STREAMER_FP:
			return 5  # Streamers
		else:
			return 6  # Bench/waiver
	
	league_df['Tier'] = league_df['Mean FPts'].apply(assign_tier)
	tier_counts = league_df['Tier'].value_counts().to_dict()
	
	# Calculate position scarcity if position data available
	position_scarcity = {}
	if 'Position' in league_df.columns:
		pos_counts = league_df['Position'].value_counts().to_dict()
		total_players = len(league_df)
		# Scarcity = 1 / (count / total) - rarer positions get higher multiplier
		for pos, count in pos_counts.items():
			position_scarcity[pos] = total_players / max(count, 1)
	
	return {
		'replacement_level': replacement_level,
		'tier_counts': tier_counts,
		'position_scarcity': position_scarcity,
		'total_rostered': len(league_df),
		'league_avg_fpts': league_df['Mean FPts'].mean(),
		'league_median_fpts': league_df['Mean FPts'].median()
	}


def calculate_player_value(player_data: pd.Series, include_consistency: bool = True, 
						   scarcity_context: Optional[Dict] = None) -> float:
	"""
	Calculate comprehensive player value including production, consistency, league scarcity, and trends.
	
	Args:
		player_data: Series with 'Mean FPts', 'CV %', 'GP', optionally 'Position', 'L7 FPts', 'L15 FPts', 'L30 FPts'
		include_consistency: Whether to factor in consistency bonus/penalty
		scarcity_context: Optional league-wide scarcity metrics from calculate_league_scarcity_context
	
	Returns:
		Total player value (exponential base + consistency + scarcity + trend adjustments)
	"""
	fpts = player_data.get('Mean FPts', 0)
	cv = player_data.get('CV %', 30)
	gp = player_data.get('GP', 0)
	
	# Get recent performance for trend analysis
	l7_fpts = player_data.get('L7 FPts', fpts)
	l15_fpts = player_data.get('L15 FPts', fpts)
	l30_fpts = player_data.get('L30 FPts', fpts)
	
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
	
	# Apply scarcity multiplier if context provided
	scarcity_mult = 1.0
	if scarcity_context:
		# VORP-style calculation: value above replacement
		replacement_level = scarcity_context.get('replacement_level', 0)
		if fpts > replacement_level:
			# Players above replacement get bonus based on how far above they are
			vorp_ratio = (fpts - replacement_level) / max(replacement_level, 1)
			# Cap the VORP bonus at 30%
			scarcity_mult = 1.0 + min(0.30, vorp_ratio * 0.15)
		
		# Position scarcity bonus (if position data available)
		if 'Position' in player_data and player_data['Position'] in scarcity_context.get('position_scarcity', {}):
			pos_scarcity = scarcity_context['position_scarcity'][player_data['Position']]
			# Normalize: if scarcity > 1.5, add up to 10% bonus
			if pos_scarcity > 1.5:
				scarcity_mult *= (1.0 + min(0.10, (pos_scarcity - 1.0) * 0.05))
	
	# Trend multiplier: reward players trending up, penalize trending down
	trend_mult = 1.0
	if l7_fpts != fpts or l15_fpts != fpts or l30_fpts != fpts:
		# Weight recent performance more heavily: L7 (50%), L15 (30%), L30 (20%)
		weighted_recent = (l7_fpts * 0.50) + (l15_fpts * 0.30) + (l30_fpts * 0.20)
		
		# Compare to YTD average
		if fpts > 0:
			trend_ratio = weighted_recent / fpts
			
			# Strong uptrend (>10% better): +8% value
			if trend_ratio >= 1.10:
				trend_mult = 1.08
			# Moderate uptrend (5-10% better): +4% value
			elif trend_ratio >= 1.05:
				trend_mult = 1.04
			# Moderate downtrend (5-10% worse): -4% value
			elif trend_ratio <= 0.90:
				trend_mult = 0.96
			# Strong downtrend (>10% worse): -8% value
			elif trend_ratio <= 0.95:
				trend_mult = 0.92
			# Stable (within 5%): no adjustment
	
	return base_value * consistency_mult * gp_mult * scarcity_mult * trend_mult


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
	
	# Calculate league-wide scarcity context
	all_teams = {**other_teams, 'Your Team': your_team}
	scarcity_context = calculate_league_scarcity_context(all_teams)
	
	# Calculate values for all players with scarcity awareness
	your_team['Value'] = your_team.apply(lambda row: calculate_player_value(row, scarcity_context=scarcity_context), axis=1)
	
	for team_name, team_df in other_teams.items():
		if team_df.empty:
			continue
		
		team_df = team_df.copy()
		team_df['Value'] = team_df.apply(lambda row: calculate_player_value(row, scarcity_context=scarcity_context), axis=1)
		
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
	
	# Sort by value gain first
	suggestions.sort(key=lambda x: x['value_gain'], reverse=True)

	# Deduplicate by (pattern, team, players you receive) so we keep the best version
	seen_keys = set()
	deduped = []
	for suggestion in suggestions:
		key = (suggestion['pattern'], suggestion['team'], tuple(sorted(suggestion['you_get'])))
		if key in seen_keys:
			continue
		seen_keys.add(key)
		deduped.append(suggestion)

	# Limit per opponent/pattern to keep variety (e.g., top 4 each)
	per_bucket_counts = {}
	filtered = []
	max_per_bucket = 4
	for suggestion in deduped:
		bucket = (suggestion['pattern'], suggestion['team'])
		count = per_bucket_counts.get(bucket, 0)
		if count >= max_per_bucket:
			continue
		per_bucket_counts[bucket] = count + 1
		filtered.append(suggestion)

	return filtered[:max_suggestions]


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
	your_count = len(your_players)
	their_count = len(their_players)
	is_consolidating = your_count > their_count  # You give more players than you receive
	is_expanding = their_count > your_count      # You receive more players than you give

	your_avg_fpts = sum(p['Mean FPts'] for p in your_players) / len(your_players)
	their_avg_fpts = sum(p['Mean FPts'] for p in their_players) / len(their_players)
	your_avg_cv = sum(p['CV %'] for p in your_players) / len(your_players)
	their_avg_cv = sum(p['CV %'] for p in their_players) / len(their_players)
	
	# Don't suggest trades where average player quality is too different
	# Even for consolidations, the average quality can't be too far apart
	fpts_ratio = max(your_avg_fpts, their_avg_fpts) / min(your_avg_fpts, their_avg_fpts) if min(your_avg_fpts, their_avg_fpts) > 0 else 999
	
	if is_consolidating:
		# Consolidations: your average should be at least ~62.5% of theirs
		# (e.g., if they give 70 FP/G, your avg must be about 44+)
		if fpts_ratio > 1.60:  # 1 / 0.625
			return False
	elif is_expanding:
		# Expansions: stay tight
		if fpts_ratio > 1.25:
			return False
	else:
		# Equal-count: very strict
		if fpts_ratio > 1.15:
			return False
	
	# Check for extreme value mismatches (catching buy-low/sell-high)
	your_total_fpts = sum(p['Mean FPts'] for p in your_players)
	their_total_fpts = sum(p['Mean FPts'] for p in their_players)
	
	# Dynamic restrictions on total FPts imbalance
	if is_consolidating:
		# You're consolidating - keep trades tight but allow fair depth-for-elite swaps
		max_ratio = 1.22  # Default: up to 22% gap allowed
		their_max = max(p['Mean FPts'] for p in their_players)
		
		# Even for elite targets, total production must remain bounded
		if their_max >= 80:
			max_ratio = 1.35  # Top-10 player: allow up to 35% gap
		elif their_max >= 70:
			max_ratio = 1.30  # Top-20 player: allow up to 30% gap
		
		# CRITICAL: Check quality of pieces you're giving
		# Both players in a 2-for-1 must be decent, not one good + one scrub
		if your_count == 2:
			your_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
			# Your second-best player must be at least 55% of the target's value
			if your_sorted[1] < their_max * 0.55:
				return False
			# Your best player must be at least 72% of the target
			if your_sorted[0] < their_max * 0.72:
				return False
		elif your_count == 3:
			your_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
			# All three must be quality pieces (at least 48% of target each)
			if any(p < their_max * 0.48 for p in your_sorted):
				return False
			# Your best must be at least 68% of target
			if your_sorted[0] < their_max * 0.68:
				return False
	elif is_expanding:
		# You're expanding the roster - keep things fairly tight
		max_ratio = 1.15
	else:
		# Equal player counts - remain strict
		max_ratio = 1.08
	
	total_ratio = max(your_total_fpts, their_total_fpts) / min(your_total_fpts, their_total_fpts) if min(your_total_fpts, their_total_fpts) > 0 else 999
	
	if total_ratio > max_ratio:
		return False
	
	# CV% check: Don't trade consistent players for volatile ones unless getting significant upgrade
	# If you're giving up more consistent players (lower CV%), you better be getting a big FPts upgrade
	if your_avg_cv < their_avg_cv:
		consistency_upgrade_needed = 1.05
		their_max_fpts = max(p['Mean FPts'] for p in their_players)
		your_max_fpts = max(p['Mean FPts'] for p in your_players)
		if is_consolidating and their_max_fpts >= your_max_fpts + 8:
			consistency_upgrade_needed = 1.0  # Elite upgrade offsets volatility
		elif is_consolidating:
			consistency_upgrade_needed = 1.03
		if their_avg_fpts < your_avg_fpts * consistency_upgrade_needed:
			return False
	else:
		your_max_fpts = max(p['Mean FPts'] for p in your_players)
		their_max_fpts = max(p['Mean FPts'] for p in their_players)
	
	# Additional check: prevent trading elite players for scrubs
	if 'your_max_fpts' not in locals():
		your_max_fpts = max(p['Mean FPts'] for p in your_players)
	if 'their_max_fpts' not in locals():
		their_max_fpts = max(p['Mean FPts'] for p in their_players)
	
	# TIER-BASED PROTECTION: Elite players require elite return
	# Top 5 tier (90+ FP/G) - Jokic, Giannis, etc.
	if your_max_fpts >= 90:
		# Must get back another top-tier player (75+) or this is a robbery
		if their_max_fpts < 75:
			return False
		# Even with a 75+ player, need strong secondary pieces
		if is_consolidating and their_count == 1:
			# 2-for-1 or 3-for-1: their single player must be 80+ to take a top-5 guy
			if their_max_fpts < 80:
				return False
	elif their_max_fpts >= 90:
		# They're giving up a top-5 player, you must give back elite talent
		if your_max_fpts < 75:
			return False
		if your_count == 1 and your_max_fpts < 80:
			return False
	
	# Top 10-20 tier (70-90 FP/G) - Stars
	if your_max_fpts >= 70 and their_max_fpts < 50:
		return False
	if their_max_fpts >= 70 and your_max_fpts < 50:
		return False
	
	# When consolidating, ensure the incoming player meaningfully upgrades your top end
	# Stricter requirements for elite player targets
	if is_consolidating:
		if their_max_fpts >= 90:
			# Targeting a top-5 player: your best must be at least 70+ or this won't work
			if your_max_fpts < 70:
				return False
			# And the upgrade must be significant (at least +12 FP/G)
			if their_max_fpts < your_max_fpts + 12:
				return False
		elif their_max_fpts >= 80:
			# Targeting top-10: need at least +10 FP/G upgrade
			if their_max_fpts < your_max_fpts + 10:
				return False
		elif their_max_fpts >= 70:
			# Targeting top-20: need at least +8 FP/G upgrade
			if their_max_fpts < your_max_fpts + 8:
				return False
		else:
			# Regular consolidation: at least +5 FP/G
			if their_max_fpts < your_max_fpts + 5:
				return False
	
	# Additional check: best player comparison
	# The best player in the trade shouldn't be too much better than the other side's best
	best_player_ratio = max(your_max_fpts, their_max_fpts) / min(your_max_fpts, their_max_fpts) if min(your_max_fpts, their_max_fpts) > 0 else 999
	
	if is_consolidating:
		# Stricter limits when targeting elite players
		if their_max_fpts >= 90:
			best_ratio_limit = 1.20  # Top-5 player: very strict
		elif their_max_fpts >= 80:
			best_ratio_limit = 1.22  # Top-10: strict
		elif their_max_fpts >= 70:
			best_ratio_limit = 1.25  # Top-20: moderate
		else:
			best_ratio_limit = 1.18  # Regular consolidation
	elif is_expanding:
		best_ratio_limit = 1.12
	else:
		best_ratio_limit = 1.08
	
	if best_player_ratio > best_ratio_limit:
		return False
	
	# Check individual player matchups - each player you give should have a comparable player you get
	your_fpts_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
	their_fpts_sorted = sorted([p['Mean FPts'] for p in their_players], reverse=True)
	
	# For each of your players, check if there's a comparable player on their side
	for i, your_fpts in enumerate(your_fpts_sorted):
		if i < len(their_fpts_sorted):
			their_fpts = their_fpts_sorted[i]
			# Allow larger gaps when consolidating into a star, but stricter for elite targets
			if is_consolidating:
				if their_max_fpts >= 90:
					player_ratio_limit = 1.28  # Top-5: tighter individual matchups
				elif their_max_fpts >= 80:
					player_ratio_limit = 1.32  # Top-10
				elif their_max_fpts >= 70:
					player_ratio_limit = 1.35  # Top-20
				else:
					player_ratio_limit = 1.20  # Regular
			elif is_expanding:
				player_ratio_limit = 1.15
			else:
				player_ratio_limit = 1.10
			player_ratio = max(your_fpts, their_fpts) / min(your_fpts, their_fpts) if min(your_fpts, their_fpts) > 0 else 999
			if player_ratio > player_ratio_limit:
				return False
	
	return True
