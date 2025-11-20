"""Trade Suggestion Engine - Suggests optimal trades based on exponential value calculations."""
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import streamlit as st

from modules.trade_suggestions_config import (
	ROSTER_SIZE,
	REPLACEMENT_PERCENTILE,
	MIN_GAMES_REQUIRED,
	AVG_GAMES_PER_PLAYER,
	MAX_CANDIDATES_YOUR,
	MAX_CANDIDATES_THEIR,
	MAX_COMBINATIONS_PER_PATTERN,
	ENABLE_VALUE_FAIRNESS_GUARD,
	MAX_OPP_WEEKLY_LOSS,
	EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS,
	MAX_OPP_CORE_AVG_DROP,
	MIN_GP_SHARE_OF_MAX,
	MIN_TRADE_FP_G,
	MAX_COMPLEXITY_OPS,
	EQUAL_COUNT_MAX_AVG_FPTS_RATIO,
	EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO,
	TRADE_BALANCE_LEVEL,
	SHOW_COMPLEXITY_DEBUG,
	MAX_ACCEPTED_TRADES_PER_PATTERN_TEAM,
	set_trade_balance_preset,
)
from modules.trade_suggestions_realism import (
	_is_realistic_trade,
	_check_1_for_n_package_ratio,
	_check_3_for_2_package_ratio,
)
from modules.trade_suggestions_core import (
	_update_realism_caps_from_league,
	_get_core_size,
	_check_opponent_core_avg_drop,
	_calculate_core_value,
	_simulate_core_value_gain,
	_calculate_floor_impact,
	_determine_trade_reasoning,
)
from modules.trade_suggestions_search import (
	_find_1_for_1_trades,
	_find_2_for_1_trades,
	_find_2_for_2_trades,
	_find_3_for_1_trades,
	_find_3_for_2_trades,
	_find_1_for_2_trades,
	_find_1_for_3_trades,
	_find_2_for_3_trades,
	_find_3_for_3_trades,
)


def _calculate_league_percentile_tiers(all_teams: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate percentile-based FP/G tiers from actual league rosters.
    This adapts to league scoring, roster construction, and scarcity.
    
    Returns dict with tier thresholds:
    - elite: Top ~5% (16 teams × 10 roster = 160 players, top 8 = elite)
    - star: Top ~10% (top 16)
    - quality: Top ~30% (top 48, roughly top 3 per team)
    - starter: Top ~60% (top 96, roughly top 6 per team)
    - streamer: Top ~85% (replacement level)
    """
    all_fpts = []
    for team_df in all_teams.values():
        if not team_df.empty and 'Mean FPts' in team_df.columns:
            all_fpts.extend(team_df['Mean FPts'].tolist())
    
    if not all_fpts:
        # Fallback to hardcoded if no data
        return {
            'elite': 90,
            'star': 70,
            'quality': 50,
            'starter': 40,
            'streamer': 30,
        }
    
    all_fpts = sorted(all_fpts, reverse=True)
    n = len(all_fpts)
    
    return {
        'elite': all_fpts[max(0, int(n * 0.05) - 1)] if n > 0 else 90,
        'star': all_fpts[max(0, int(n * 0.10) - 1)] if n > 0 else 70,
        'quality': all_fpts[max(0, int(n * 0.30) - 1)] if n > 0 else 50,
        'starter': all_fpts[max(0, int(n * 0.60) - 1)] if n > 0 else 40,
        'streamer': all_fpts[max(0, int(n * 0.85) - 1)] if n > 0 else 30,
    }


def estimate_trade_search_complexity(
	your_team: pd.DataFrame,
	other_teams: Dict[str, pd.DataFrame],
	trade_patterns: List[str],
	target_teams: Optional[List[str]] = None,
	exclude_players: Optional[List[str]] = None,
	exclude_teams: Optional[List[str]] = None,
	exclude_opposing_players: Optional[List[str]] = None,
) -> int:
	if your_team is None or your_team.empty or not other_teams:
		return 0

	teams = other_teams
	if target_teams:
		teams = {k: v for k, v in teams.items() if k in target_teams}
	if exclude_teams:
		teams = {k: v for k, v in teams.items() if k not in exclude_teams}
	if not teams:
		return 0

	your_trade_team = your_team
	if exclude_players:
		your_trade_team = your_trade_team[~your_trade_team['Player'].isin(exclude_players)]

	len_y = len(your_trade_team)
	if len_y <= 0:
		return 0
	if len_y > MAX_CANDIDATES_YOUR:
		len_y = MAX_CANDIDATES_YOUR

	total_ops = 0
	for team_name, df in teams.items():
		if df is None or df.empty:
			continue
		team_df = df
		if exclude_opposing_players and 'Player' in team_df.columns:
			team_df = team_df[~team_df['Player'].isin(exclude_opposing_players)]
		len_t = len(team_df)
		if len_t <= 0:
			continue
		if len_t > MAX_CANDIDATES_THEIR:
			len_t = MAX_CANDIDATES_THEIR

		team_ops = 0

		for pattern in trade_patterns:
			if pattern == '1-for-1':
				combos = len_y * len_t
			elif pattern == '2-for-1':
				if len_y < 2:
					continue
				combos = math.comb(len_y, 2) * len_t
			elif pattern == '1-for-2':
				if len_t < 2:
					continue
				combos = len_y * math.comb(len_t, 2)
			elif pattern == '2-for-2':
				if len_y < 2 or len_t < 2:
					continue
				combos = math.comb(len_y, 2) * math.comb(len_t, 2)
			elif pattern == '3-for-1':
				if len_y < 3:
					continue
				combos = math.comb(len_y, 3) * len_t
			elif pattern == '3-for-2':
				if len_y < 3 or len_t < 2:
					continue
				combos = math.comb(len_y, 3) * math.comb(len_t, 2)
			elif pattern == '1-for-3':
				if len_t < 3:
					continue
				combos = len_y * math.comb(len_t, 3)
			elif pattern == '2-for-3':
				if len_y < 2 or len_t < 3:
					continue
				combos = math.comb(len_y, 2) * math.comb(len_t, 3)
			elif pattern == '3-for-3':
				if len_y < 3 or len_t < 3:
					continue
				combos = math.comb(len_y, 3) * math.comb(len_t, 3)
			else:
				continue

			if combos <= 0:
				continue

			combos = min(combos, MAX_COMBINATIONS_PER_PATTERN)
			team_ops += combos

		total_ops += team_ops
		if SHOW_COMPLEXITY_DEBUG:
			st.write(f"Estimated combinations for {team_name}: {team_ops}")

	return total_ops

def calculate_exponential_value(fpts: float) -> float:
	"""
	Calculate the base value of a player from FP/G.
	Uses a slight non-linear exponent (1.05) to encode the "superstar premium"
	where elite players have irreplaceable ceiling value beyond their average FP/G.
	This makes a 100 FP/G player worth ~2.15x a 50 FP/G player instead of exactly 2x,
	reflecting the strategic reality that consolidating into stars is powerful.
	"""
	exponent = 1.3  # Moderately stronger superstar premium
	anchor_fpts = 50.0
	scale = anchor_fpts / (anchor_fpts ** exponent) if anchor_fpts > 0 else 1.0
	base_value = (fpts ** exponent) * scale
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
		return {
			'replacement_level': 0,
			'tier_counts': {},
			'position_scarcity': {},
			'total_rostered': 0,
			'league_avg_fpts': 0.0,
			'league_median_fpts': 0.0,
			'league_avg_cv': None,
		}
	
	league_df = pd.concat(all_players, ignore_index=True)
	
	# Calculate replacement level (top 85% of rostered players)
	num_teams = len(all_teams_data)
	roster_size = ROSTER_SIZE
	replacement_idx = int(num_teams * roster_size * REPLACEMENT_PERCENTILE)
	
	if len(league_df) >= replacement_idx and replacement_idx > 0:
		replacement_level = league_df.nlargest(replacement_idx, 'Mean FPts')['Mean FPts'].iloc[-1]
	else:
		replacement_level = league_df['Mean FPts'].min()
	
	# Use percentile-based tiers derived from the actual league distribution
	league_tiers = _calculate_league_percentile_tiers(all_teams_data)
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	starter_threshold = league_tiers['starter']
	streamer_threshold = league_tiers['streamer']
	
	def assign_tier(fpts: float) -> int:
		"""Map FP/G into a scarcity tier using dynamic thresholds.
		1 = elite, 2 = star, 3 = quality, 4 = starter, 5 = streamer, 6 = bench.
		"""
		if fpts >= elite_threshold:
			return 1  # Elite
		elif fpts >= star_threshold:
			return 2  # Star
		elif fpts >= quality_threshold:
			return 3  # Quality starter
		elif fpts >= starter_threshold:
			return 4  # Starter
		elif fpts >= streamer_threshold:
			return 5  # Streamer / fringe
		else:
			return 6  # Bench / deep waiver
	
	league_df['Tier'] = league_df['Mean FPts'].apply(assign_tier)
	tier_counts = league_df['Tier'].value_counts().to_dict()
	
	# Calculate percentile rank for each player (for dynamic realism checks)
	# Sort by FP/G descending, then assign percentile (0.0 = best, 1.0 = worst)
	league_df_sorted = league_df.sort_values('Mean FPts', ascending=False).reset_index(drop=True)
	league_df_sorted['LeagueRank'] = league_df_sorted.index + 1
	league_df_sorted['LeaguePercentile'] = league_df_sorted['LeagueRank'] / len(league_df_sorted)
	
	# Create a lookup dict: player name -> percentile
	percentile_lookup = {}
	if 'Player' in league_df_sorted.columns:
		percentile_lookup = dict(zip(league_df_sorted['Player'], league_df_sorted['LeaguePercentile']))
	
	# Calculate position scarcity if position data available
	position_scarcity = {}
	if 'Position' in league_df.columns:
		pos_counts = league_df['Position'].value_counts().to_dict()
		total_players = len(league_df)
		# Scarcity = 1 / (count / total) - rarer positions get higher multiplier
		for pos, count in pos_counts.items():
			position_scarcity[pos] = total_players / max(count, 1)

	league_avg_cv = None
	if 'CV %' in league_df.columns:
		league_avg_cv = float(league_df['CV %'].mean())
	
	return {
		'replacement_level': replacement_level,
		'tier_counts': tier_counts,
		'position_scarcity': position_scarcity,
		'total_rostered': len(league_df),
		'league_avg_fpts': league_df['Mean FPts'].mean(),
		'league_median_fpts': league_df['Mean FPts'].median(),
		'percentile_lookup': percentile_lookup,
		'league_avg_cv': league_avg_cv,
	}


def calculate_player_value(player_data: pd.Series, include_consistency: bool = True, 
			   scarcity_context: Optional[Dict] = None) -> float:
	"""\
	Calculate player value as FP/G with minimal adjustments.
	
	Value ≈ FP/G with small penalties for:
	- High volatility (CV > 35%)
	- (Optionally) small, league-aware scarcity adjustments
	
	This keeps value interpretable and aligned with 901/902's FP/G-centric philosophy.
	
	Args:
		player_data: Series with 'Mean FPts', 'CV %', 'GP'
		include_consistency: Whether to apply CV penalty
		scarcity_context: Optional league context dict from calculate_league_scarcity_context
	
	Returns:
		Player value ≈ effective FP/G
	"""
	fpts = float(player_data.get('Mean FPts', 0) or 0.0)
	cv = float(player_data.get('CV %', 30) or 30.0)
	gp = player_data.get('GP', 0)
	
	# Base value = FP/G
	base_value = fpts
	
	if not include_consistency:
		return base_value
	
	# Small consistency penalty for very volatile players only
	# CV > 35%: penalize up to -10%
	if cv > 35:
		consistency_mult = max(0.90, 1.0 - (cv - 35) * 0.005)
	else:
		consistency_mult = 1.0
	
	# Optional scarcity adjustment based on league-wide context
	scarcity_mult = 1.0
	if scarcity_context:
		# Percentile-based adjustment: top players get a small boost, deep bench a small downgrade.
		percentile_lookup = scarcity_context.get('percentile_lookup') or {}
		player_name = player_data.get('Player')
		if percentile_lookup and player_name in percentile_lookup:
			pct_raw = percentile_lookup[player_name]
			try:
				pct = float(pct_raw)  # 0.0 = best, 1.0 = worst
			except (TypeError, ValueError):
				pct = 1.0
			pct = max(0.0, min(1.0, pct))
			# Stronger superstar premium: top ~1-3% get a large multiplier, tapering by tier.
			if pct <= 0.01:
				tier_mult = 2.5
			elif pct <= 0.03:
				tier_mult = 2.0
			elif pct <= 0.08:
				tier_mult = 1.7
			elif pct <= 0.15:
				tier_mult = 1.4
			elif pct <= 0.30:
				tier_mult = 1.2
			elif pct <= 0.60:
				tier_mult = 1.05
			else:
				tier_mult = 1.0
			scarcity_mult *= tier_mult
		
		# Position scarcity: rarer positions get a slight bump, common ones a slight trim.
		position_scarcity = scarcity_context.get('position_scarcity') or {}
		pos = player_data.get('Position')
		if position_scarcity and pos in position_scarcity:
			values = list(position_scarcity.values())
			avg_scarcity = sum(values) / len(values) if values else 1.0
			raw = position_scarcity[pos] / avg_scarcity if avg_scarcity > 0 else 1.0
			# Center around 1.0 and squash into roughly [0.95, 1.05]
			pos_mult = 1.0 + max(-0.05, min(0.05, (raw - 1.0) * 0.1))
			scarcity_mult *= pos_mult
	
	return base_value * consistency_mult * scarcity_mult


def find_trade_suggestions(
	your_team: pd.DataFrame,
	other_teams: Dict[str, pd.DataFrame],
	trade_patterns: List[str] = ['1-for-1', '2-for-1', '2-for-2'],
	min_value_gain: float = 5.0,
	max_suggestions: int = 20,
	target_teams: List[str] = None,
	exclude_players: List[str] = None,
	include_players: List[str] = None,
	exclude_teams: List[str] = None,
	target_opposing_players: List[str] = None,
	exclude_opposing_players: List[str] = None,
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
	if exclude_teams:
		other_teams = {k: v for k, v in other_teams.items() if k not in exclude_teams}
	
	# Preserve your full roster for core / context calculations
	your_full_team = your_team.copy()
	
	# GP-based eligibility filter using share of max GP in league
	all_teams_raw = {**other_teams, 'Your Team': your_full_team}
	max_gp = 0
	for team_df in all_teams_raw.values():
		if team_df is not None and not team_df.empty and 'GP' in team_df.columns:
			team_max = team_df['GP'].max()
			if team_max > max_gp:
				max_gp = team_max
	if max_gp > 0:
		gp_threshold = max_gp * MIN_GP_SHARE_OF_MAX
		if 'GP' in your_full_team.columns:
			filtered_your = your_full_team[your_full_team['GP'] >= gp_threshold]
			if not filtered_your.empty:
				your_full_team = filtered_your.copy()
		filtered_other_teams = {}
		for name, team_df in other_teams.items():
			if team_df is not None and not team_df.empty and 'GP' in team_df.columns:
				filtered_team = team_df[team_df['GP'] >= gp_threshold]
				if filtered_team.empty:
					filtered_team = team_df
				filtered_other_teams[name] = filtered_team.copy()
			else:
				filtered_other_teams[name] = team_df
		other_teams = filtered_other_teams
	
	# Calculate league-wide scarcity context using full rosters
	all_teams = {**other_teams, 'Your Team': your_full_team}
	scarcity_context = calculate_league_scarcity_context(all_teams)
	
	# Calculate percentile-based tiers from actual league data
	league_tiers = _calculate_league_percentile_tiers(all_teams)
	# Update realism caps (opponent loss / ratios) from league stats + balance level
	_update_realism_caps_from_league(scarcity_context, league_tiers)
	
	# Calculate values for all of your players with scarcity awareness
	your_full_team['Value'] = your_full_team.apply(
		lambda row: calculate_player_value(row, scarcity_context=scarcity_context),
		axis=1,
	)
	
	# Determine which of your players are actually tradable (respect exclusions)
	if exclude_players:
		your_trade_team = your_full_team[~your_full_team['Player'].isin(exclude_players)].copy()
	else:
		your_trade_team = your_full_team.copy()
	# Drop clearly replacement-level pieces from the trade candidate pool
	if 'Mean FPts' in your_trade_team.columns:
		eligible = your_trade_team[your_trade_team['Mean FPts'] >= MIN_TRADE_FP_G]
		if not eligible.empty:
			your_trade_team = eligible.copy()
	if len(your_trade_team) > MAX_CANDIDATES_YOUR:
		your_trade_team = your_trade_team.nlargest(MAX_CANDIDATES_YOUR, 'Value').copy()
	
	# Dynamically prune trade patterns if the estimated search space is too large.
	effective_patterns = list(trade_patterns) if isinstance(trade_patterns, list) else list(trade_patterns)
	try:
		est_ops = estimate_trade_search_complexity(
			your_full_team,
			other_teams,
			effective_patterns,
			target_teams=target_teams,
			exclude_players=exclude_players,
			exclude_teams=exclude_teams,
			exclude_opposing_players=exclude_opposing_players,
		)
	except Exception:
		est_ops = 0
	
	if est_ops > MAX_COMPLEXITY_OPS and len(effective_patterns) > 1:
		# Drop the most combinatorially expensive patterns first until we are under budget
		pattern_priority = [
			'3-for-3', '2-for-3', '3-for-2', '1-for-3',
			'3-for-1', '1-for-2', '2-for-2', '2-for-1', '1-for-1',
		]
		while len(effective_patterns) > 1 and est_ops > MAX_COMPLEXITY_OPS:
			for p in pattern_priority:
				if p in effective_patterns and len(effective_patterns) > 1:
					effective_patterns.remove(p)
					break
			est_ops = estimate_trade_search_complexity(
				your_full_team,
				other_teams,
				effective_patterns,
				target_teams=target_teams,
				exclude_players=exclude_players,
				exclude_teams=exclude_teams,
				exclude_opposing_players=exclude_opposing_players,
			)
	
	# Derive core size and baseline core value; these will be used by downstream logic
	core_size = _get_core_size()
	baseline_core_value = _calculate_core_value(your_full_team, core_size)
	
	# Precompute opponent core values for symmetric evaluation
	opponent_core_values = {}
	for team_name, team_df in other_teams.items():
		if not team_df.empty:
			# Calculate values for opponent's full roster
			opp_full = team_df.copy()
			opp_full['Value'] = opp_full.apply(
				lambda row: calculate_player_value(row, scarcity_context=scarcity_context),
				axis=1,
			)
			opponent_core_values[team_name] = {
				'full_team': opp_full,
				'baseline_core': _calculate_core_value(opp_full, core_size),
			}
	
	for team_name, team_df in other_teams.items():
		if team_df.empty:
			continue
		
		# Get opponent's precomputed data
		opp_data = opponent_core_values[team_name]
		opp_full_team = opp_data['full_team']
		opp_baseline_core = opp_data['baseline_core']
		
		team_df = opp_full_team.copy()
		# Apply opposing player exclusion early to reduce search space
		if exclude_opposing_players and 'Player' in team_df.columns:
			team_df = team_df[~team_df['Player'].isin(exclude_opposing_players)].copy()
		# Exclude obvious drop-tier pieces from the opponent candidate pool
		if 'Mean FPts' in team_df.columns:
			eligible_opp = team_df[team_df['Mean FPts'] >= MIN_TRADE_FP_G]
			if not eligible_opp.empty:
				team_df = eligible_opp.copy()
		if len(team_df) > MAX_CANDIDATES_THEIR:
			team_df = team_df.nlargest(MAX_CANDIDATES_THEIR, 'Value').copy()
		
		# Try each trade pattern
		for pattern in effective_patterns:
			if pattern == '1-for-1':
				suggestions.extend(
					_find_1_for_1_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
			elif pattern == '2-for-1':
				suggestions.extend(
					_find_2_for_1_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
			elif pattern == '1-for-2':
				suggestions.extend(
					_find_1_for_2_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
			elif pattern == '2-for-2':
				suggestions.extend(
					_find_2_for_2_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
			elif pattern == '3-for-1':
				suggestions.extend(
					_find_3_for_1_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
			elif pattern == '1-for-3':
				suggestions.extend(
					_find_1_for_3_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
			elif pattern == '3-for-2':
				suggestions.extend(
					_find_3_for_2_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
			elif pattern == '2-for-3':
				suggestions.extend(
					_find_2_for_3_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
			elif pattern == '3-for-3':
				suggestions.extend(
					_find_3_for_3_trades(
						your_trade_team,
						team_df,
						team_name,
						min_value_gain,
						your_full_team,
						core_size,
						baseline_core_value,
						include_players,
						opp_full_team,
						opp_baseline_core,
						league_tiers,
						target_opposing_players,
					)
				)
	
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
