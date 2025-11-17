"""Trade Suggestion Engine - Suggests optimal trades based on exponential value calculations."""
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import streamlit as st

ROSTER_SIZE = 10  # From league rules
REPLACEMENT_PERCENTILE = 0.85  # Top 85% of rostered players
MIN_GAMES_REQUIRED = 25  # Weekly minimum games in current configuration
AVG_GAMES_PER_PLAYER = 3.5  # Approximate NBA games per player per fantasy week
# Hard caps on how many players per team enter the trade engine. These were
# originally 10/10; tightening them reduces combinatorial explosion while still
# considering the key pieces on each roster.
MAX_CANDIDATES_YOUR = 8
MAX_CANDIDATES_THEIR = 8
# Per-pattern cap on combinations evaluated for a given team pair.
MAX_COMBINATIONS_PER_PATTERN = 1000
ENABLE_VALUE_FAIRNESS_GUARD = False
MAX_OPP_WEEKLY_LOSS = 15.0  # Max weekly core FP we assume an opponent would accept losing (slightly loosened)
EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS = 10
MAX_OPP_CORE_AVG_DROP = 1.5  # Max FP/G drop in opponent's core average (league philosophy: FP/G > total FP)
MIN_GP_SHARE_OF_MAX = 0.25
MAX_COMPLEXITY_OPS = 200_000  # Rough cap on estimated combination evaluations
EQUAL_COUNT_MAX_AVG_FPTS_RATIO = 1.15
EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO = 1.5
TRADE_BALANCE_LEVEL = 5  # 1-10 scale, 5 = standard
SHOW_COMPLEXITY_DEBUG = False  # Toggle to print per-team combination estimates
MAX_ACCEPTED_TRADES_PER_PATTERN_TEAM = 12  # Early-exit once we have plenty of good trades for a pattern vs team


def _apply_trade_balance_level(level: int) -> None:
	"""Apply threshold settings for a 1-10 trade balance level.

	This function now acts as a fallback static ramp. League-aware caps are
	derived later from calculate_league_scarcity_context and the stored
	TRADE_BALANCE_LEVEL inside find_trade_suggestions.
	"""
	global EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS, EQUAL_COUNT_MAX_AVG_FPTS_RATIO, EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO
	level = max(1, min(10, int(level)))

	# Linear ramps between conservative (1) and aggressive (10)
	EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS = 4.0 + (level - 1) * 1.5  # 4 → 17.5
	EQUAL_COUNT_MAX_AVG_FPTS_RATIO = 1.05 + (level - 1) * 0.025  # 1.05 → 1.275
	EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO = 1.02 + (level - 1) * 0.02   # 1.02 → 1.20


def set_trade_balance_preset(preset) -> None:
	"""
	Adjust equal-count realism thresholds.

	Accepts either legacy labels ("conservative", "standard", "aggressive")
	or a numeric balance level 1-10 (5 = standard).
	"""
	global TRADE_BALANCE_LEVEL
	if isinstance(preset, (int, float)):
		level = int(preset)
	else:
		key = (preset or 'standard').lower()
		legacy_map = {
			'very conservative': 2,
			'conservative': 3,
			'standard': 5,
			'aggressive': 7,
			'ultra aggressive': 9,
		}
		level = legacy_map.get(key, 5)

	TRADE_BALANCE_LEVEL = max(1, min(50, int(level)))
	_apply_trade_balance_level(TRADE_BALANCE_LEVEL)


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


def _update_realism_caps_from_league(scarcity_context: Dict, league_tiers: Dict[str, float]) -> None:
	"""Derive opponent-loss and ratio caps from league FP/G and CV plus balance level.

	This keeps the thresholds scaled to your actual league while letting the
	Trade Balance slider act as a simple 1–10 strictness control.
	"""
	global MAX_OPP_CORE_AVG_DROP, MAX_OPP_WEEKLY_LOSS, EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS
	global EQUAL_COUNT_MAX_AVG_FPTS_RATIO, EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO

	if not league_tiers:
		return

	quality = float(league_tiers.get('quality', 50.0))
	star = float(league_tiers.get('star', quality + 10.0))
	fp_unit = max(5.0, star - quality)

	# Base core FP/G drop the opponent might tolerate in a "typical" league.
	base_core_drop = fp_unit * 0.3  # ~30% of the quality→star gap

	# Volatility factor: swingier leagues allow slightly larger FP/G loss.
	league_avg_cv = scarcity_context.get('league_avg_cv')
	vol_factor = 1.0
	if isinstance(league_avg_cv, (int, float)) and league_avg_cv > 0:
		# Around 30% CV = neutral; clamp to [0.85, 1.15]
		vol_factor = max(0.85, min(1.15, league_avg_cv / 30.0))

	# Map trade balance level 1–10 into a 0.7–1.3 multiplier.
	level = TRADE_BALANCE_LEVEL
	strict_mult = 0.7 + (level - 1) * (0.6 / 9.0)

	core_drop = base_core_drop * vol_factor * strict_mult

	# Core FP/G drop cap (master guard)
	MAX_OPP_CORE_AVG_DROP = core_drop

	# Weekly loss caps derived from core drop.
	equal_weekly = core_drop * MIN_GAMES_REQUIRED
	EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS = min(equal_weekly, 60.0)
	MAX_OPP_WEEKLY_LOSS = min(EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS * 1.5, 90.0)

	# Ratio caps for equal-count trades.
	if quality <= 0:
		quality = 50.0
	delta_ratio = min(0.25, core_drop / quality)  # never allow absurd ratios

	EQUAL_COUNT_MAX_AVG_FPTS_RATIO = 1.05 + delta_ratio * 0.5   # ≈ [1.05, ~1.17]
	EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO = 1.02 + delta_ratio * 0.4  # ≈ [1.02, ~1.12]

	EQUAL_COUNT_MAX_AVG_FPTS_RATIO = min(EQUAL_COUNT_MAX_AVG_FPTS_RATIO, 1.30)
	EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO = min(EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO, 1.20)


def _get_core_size(min_games: float = MIN_GAMES_REQUIRED, g_avg: float = AVG_GAMES_PER_PLAYER) -> int:
	"""Approximate number of core roster spots based on min games and avg games per player."""
	if g_avg <= 0:
		return ROSTER_SIZE
	size = math.ceil(min_games / g_avg)
	return max(1, min(size, ROSTER_SIZE))


def _check_opponent_core_avg_drop(
	opp_full_team: pd.DataFrame,
	opp_baseline_core: float,
	players_they_give: List[pd.Series],
	players_they_get: List[pd.Series],
	core_size: int,
) -> bool:
	"""
	Check if opponent's core FP/G average drops too much.
	Returns True if the drop is acceptable, False if it exceeds MAX_OPP_CORE_AVG_DROP.
	
	This enforces the league philosophy from 901: FP/G > total FP.
	Prevents trades where opponent sacrifices core quality for depth they don't need.
	"""
	# Simulate opponent's roster after trade
	opp_after_team = opp_full_team.copy()
	
	# Remove players they're giving
	give_names = [p.get('Player') for p in players_they_give if 'Player' in p]
	if give_names:
		opp_after_team = opp_after_team[~opp_after_team['Player'].isin(give_names)]
	
	# Add players they're getting
	for p in players_they_get:
		opp_after_team = pd.concat([opp_after_team, pd.DataFrame([p])], ignore_index=True)
	
	# Calculate core FP/G average before and after
	opp_core_after = _calculate_core_value(opp_after_team, core_size)
	opp_core_avg_before = opp_baseline_core / core_size if core_size > 0 else 0
	opp_core_avg_after = opp_core_after / core_size if core_size > 0 else 0
	opp_core_avg_drop = opp_core_avg_before - opp_core_avg_after
	
	return opp_core_avg_drop <= MAX_OPP_CORE_AVG_DROP


def _calculate_core_value(team_df: pd.DataFrame, core_size: int) -> float:
	"""Sum of value for top N players by FP/G, representing core roster value."""
	if team_df is None or team_df.empty or core_size <= 0:
		return 0.0
	sorted_df = team_df.sort_values('Mean FPts', ascending=False)
	core_df = sorted_df.head(core_size)
	if 'Value' not in core_df.columns:
		return 0.0
	return float(core_df['Value'].sum())


def _simulate_core_value_gain(
	your_full_team: pd.DataFrame,
	your_players_give: List[pd.Series],
	their_players_get: List[pd.Series],
	core_size: int,
	baseline_core_value: float,
) -> float:
	"""Simulate change in your *weekly core output* if a given trade is executed.

	We treat the core sum of Value as an FP/G-like quantity for your top
	`core_size` players. To approximate weekly impact under the league's
	MinGames rules (see 901/902), we convert the change in core value
	into an estimated weekly FP change as:

		ΔWeeklyCoreFP ≈ (CoreValue_after − CoreValue_before)
		                 × (MIN_GAMES_REQUIRED / core_size)

	This roughly corresponds to ΔCorePPG × MinGames, assuming each core
	spot contributes ~MinGames/core_size games per week.
	"""
	if your_full_team is None or your_full_team.empty or core_size <= 0:
		return 0.0

	after_team = your_full_team.copy()

	# Remove players you are giving up
	if your_players_give:
		give_names = [p.get('Player') for p in your_players_give if 'Player' in p]
		if give_names:
			after_team = after_team[~after_team['Player'].isin(give_names)].copy()

	# Add players you are receiving
	if their_players_get:
		new_rows = pd.DataFrame(their_players_get)
		if not new_rows.empty:
			# Avoid duplicate player rows if you already have someone
			if 'Player' in after_team.columns and 'Player' in new_rows.columns:
				existing = set(after_team['Player'].tolist())
				new_rows = new_rows[~new_rows['Player'].isin(existing)]
			if not new_rows.empty:
				after_team = pd.concat([after_team, new_rows], ignore_index=True)

	core_after = _calculate_core_value(after_team, core_size)
	core_delta = core_after - baseline_core_value
	# Approximate weekly FP change from this delta in core value
	scale_weeks = MIN_GAMES_REQUIRED / float(core_size) if core_size > 0 else 0.0
	return core_delta * scale_weeks


def _calculate_floor_impact(
	your_full_team: pd.DataFrame,
	your_players_give: List[pd.Series],
	their_players_get: List[pd.Series],
	floor_size: int = 2,
) -> float:
	"""Calculate change in roster floor (bottom N players' average FP/G).
	
	This measures how the trade affects your roster's depth and flexibility.
	Positive values mean your floor improved (better depth).
	Negative values mean your floor worsened (worse depth).
	
	For consolidation trades (giving more than getting), we simulate adding
	a 40 FP/G waiver wire player to fill the open roster spot.
	"""
	if your_full_team is None or your_full_team.empty:
		return 0.0
	
	# Calculate floor before trade
	floor_before = your_full_team.nsmallest(floor_size, 'Mean FPts')['Mean FPts'].mean()
	
	# Simulate roster after trade
	after_team = your_full_team.copy()
	
	# Remove players you are giving up
	if your_players_give:
		give_names = [p.get('Player') for p in your_players_give if 'Player' in p]
		if give_names:
			after_team = after_team[~after_team['Player'].isin(give_names)].copy()
	
	# Add players you are receiving
	if their_players_get:
		new_rows = pd.DataFrame(their_players_get)
		if not new_rows.empty:
			if 'Player' in after_team.columns and 'Player' in new_rows.columns:
				existing = set(after_team['Player'].tolist())
				new_rows = new_rows[~new_rows['Player'].isin(existing)]
			if not new_rows.empty:
				after_team = pd.concat([after_team, new_rows], ignore_index=True)
	
	# For consolidation trades, add waiver wire replacement (40 FP/G)
	num_give = len(your_players_give) if your_players_give else 0
	num_get = len(their_players_get) if their_players_get else 0
	if num_give > num_get:
		# Add waiver wire players to fill open spots
		num_waiver = num_give - num_get
		for i in range(num_waiver):
			waiver_player = pd.Series({
				'Player': f'Waiver_Wire_{i}',
				'Mean FPts': 40.0,  # Typical best available in 16-team league
				'Value': 40.0,
			})
			after_team = pd.concat([after_team, pd.DataFrame([waiver_player])], ignore_index=True)
	
	# Calculate floor after trade
	if len(after_team) >= floor_size:
		floor_after = after_team.nsmallest(floor_size, 'Mean FPts')['Mean FPts'].mean()
	else:
		floor_after = after_team['Mean FPts'].mean() if not after_team.empty else 0.0
	
	return floor_after - floor_before


def _determine_trade_reasoning(core_gain: float, floor_delta: float) -> str:
	"""Determine the strategic purpose of a trade based on core and floor impacts.
	
	Returns a human-readable string explaining the trade's strategic value:
	- "Consolidation" = Core upgrade at the expense of floor (trading up)
	- "Deconstruction" = Floor upgrade at the expense of core (trading down for depth)
	- "Overall Improvement" = Both core and floor improve
	- "Lateral Move" = Minimal impact on both core and floor
	"""
	# Thresholds for what counts as "significant"
	CORE_THRESHOLD = 2.0  # Weekly core FP gain
	FLOOR_THRESHOLD = 1.0  # FP/G floor change
	
	if core_gain > CORE_THRESHOLD and floor_delta < -FLOOR_THRESHOLD:
		return "Consolidation"  # Trading up: better core, worse floor
	elif core_gain < CORE_THRESHOLD and floor_delta > FLOOR_THRESHOLD:
		return "Deconstruction"  # Trading down: worse core, better floor
	elif core_gain > CORE_THRESHOLD and floor_delta > FLOOR_THRESHOLD:
		return "Overall Improvement"  # Win-win: both improve
	elif abs(core_gain) < CORE_THRESHOLD and abs(floor_delta) < FLOOR_THRESHOLD:
		return "Lateral Move"  # Sideways trade
	else:
		return "Mixed Impact"  # Some combination


def calculate_player_value(player_data: pd.Series, include_consistency: bool = True, 
				   scarcity_context: Optional[Dict] = None) -> float:
	"""
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
	fpts = player_data.get('Mean FPts', 0)
	cv = player_data.get('CV %', 30)
	gp = player_data.get('GP', 0)
	
	# Base value = FP/G
	base_value = calculate_exponential_value(fpts)
	
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
		if player_name in percentile_lookup:
			pct = float(percentile_lookup[player_name])  # 0.0 = best, 1.0 = worst
			# Map percentile into ~[0.95, 1.05] range so impact stays modest.
			tier_mult = 1.05 - 0.10 * max(0.0, min(1.0, pct))
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


def _find_1_for_1_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 1-for-1 trade opportunities with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	for your_player in your_rows:
		for their_player in their_rows:
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			# Enforce target_opposing_players constraint (their side)
			if target_opposing_players:
				if their_player.get('Player') not in target_opposing_players:
					continue
			# Enforce must-include constraint if provided (your side)
			if include_players is not None and len(include_players) > 0:
				if your_player.get('Player') not in include_players:
					continue
			# Enforce target_opposing_players constraint (their side)
			if target_opposing_players:
				if their_player.get('Player') not in target_opposing_players:
					continue
			
			# Your core gain
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				[your_player],
				[their_player],
				core_size,
				baseline_core_value,
			)
			
			# Opponent core gain (they give their_player, get your_player)
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				[their_player],
				[your_player],
				core_size,
				opp_baseline_core,
			)
			
			# Check opponent's core FP/G average drop (league philosophy: FP/G > total FP)
			opp_after_team = opp_full_team.copy()
			opp_after_team = opp_after_team[~opp_after_team['Player'].isin([their_player['Player']])]
			opp_after_team = pd.concat([opp_after_team, pd.DataFrame([your_player])], ignore_index=True)
			opp_core_after = _calculate_core_value(opp_after_team, core_size)
			opp_core_avg_before = opp_baseline_core / core_size if core_size > 0 else 0
			opp_core_avg_after = opp_core_after / core_size if core_size > 0 else 0
			opp_core_avg_drop = opp_core_avg_before - opp_core_avg_after
			
			# Filter: you must gain enough, opponent can't lose too much (weekly OR avg FP/G)
			if your_core_gain >= min_gain and opp_core_gain >= -EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS and opp_core_avg_drop <= MAX_OPP_CORE_AVG_DROP and _is_realistic_trade([your_player], [their_player], league_tiers):
				# Calculate floor impact only for accepted trades
				floor_delta = _calculate_floor_impact(
					your_full_team,
					[your_player],
					[their_player],
				)
				# Determine trade reasoning
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '1-for-1',
					'team': team_name,
					'you_give': [your_player['Player']],
					'you_get': [their_player['Player']],
					'your_value': your_player['Value'],
					'their_value': their_player['Value'],
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,  # NEW
					'reasoning': reasoning,  # NEW
					'your_fpts': [your_player['Mean FPts']],
					'their_fpts': [their_player['Mean FPts']],
					'your_cv': [your_player['CV %']],
					'their_cv': [their_player['CV %']]
				})
				accepted += 1
				if accepted >= MAX_ACCEPTED_TRADES_PER_PATTERN_TEAM:
					return trades
	
	return trades


def _find_2_for_1_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 2-for-1 trade opportunities (you give 2, get 1) with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	# You give 2, get 1 elite
	for your_combo in combinations(your_rows, 2):
		your_players = list(your_combo)
		# Must include at least one of the include_players, if specified
		if include_players is not None and len(include_players) > 0:
			if not any(p.get('Player') in include_players for p in your_players):
				continue
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_player in their_rows:
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				your_players,
				[their_player],
				core_size,
				baseline_core_value,
			)
			
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				[their_player],
				your_players,
				core_size,
				opp_baseline_core,
			)
			
			# Check core avg drop for opponent (they expand: give 1, get 2)
			if (your_core_gain >= min_gain and 
				opp_core_gain >= -MAX_OPP_WEEKLY_LOSS and 
				_check_opponent_core_avg_drop(opp_full_team, opp_baseline_core, [their_player], your_players, core_size) and 
				_is_realistic_trade(your_players, [their_player], league_tiers)):
				floor_delta = _calculate_floor_impact(your_full_team, your_players, [their_player])
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '2-for-1',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [their_player['Player']],
					'your_value': your_total_value,
					'their_value': their_player['Value'],
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,
					'reasoning': reasoning,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [their_player['Mean FPts']],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [their_player['CV %']]
				})
	
	return trades


def _find_2_for_2_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 2-for-2 trade opportunities with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	for your_combo in combinations(your_rows, 2):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get('Player') in include_players for p in your_players):
				continue
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_combo in combinations(their_rows, 2):
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			# Enforce target_opposing_players constraint: at least one of their players must be targeted
			if target_opposing_players:
				if not any(p.get('Player') in target_opposing_players for p in their_players):
					continue
			their_total_value = sum(p['Value'] for p in their_players)
			
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				your_players,
				their_players,
				core_size,
				baseline_core_value,
			)
			
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				their_players,
				your_players,
				core_size,
				opp_baseline_core,
			)
			
			if your_core_gain >= min_gain and opp_core_gain >= -EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS and _is_realistic_trade(your_players, their_players, league_tiers):
				floor_delta = _calculate_floor_impact(your_full_team, your_players, their_players)
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '2-for-2',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_total_value,
					'their_value': their_total_value,
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,
					'reasoning': reasoning,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades


def _find_3_for_1_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 3-for-1 trade opportunities (you give 3, get 1 superstar) with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	for your_combo in combinations(your_rows, 3):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get('Player') in include_players for p in your_players):
				continue
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_player in their_rows:
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				your_players,
				[their_player],
				core_size,
				baseline_core_value,
			)
			
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				[their_player],
				your_players,
				core_size,
				opp_baseline_core,
			)
			
			opp_after_team = opp_full_team.copy()
			opp_after_team = opp_after_team[~opp_after_team['Player'].isin([their_player['Player']])]
			opp_after_team = pd.concat([opp_after_team, pd.DataFrame(your_players)], ignore_index=True)
			opp_core_after = _calculate_core_value(opp_after_team, core_size)
			opp_core_avg_before = opp_baseline_core / core_size if core_size > 0 else 0
			opp_core_avg_after = opp_core_after / core_size if core_size > 0 else 0
			opp_core_avg_drop = opp_core_avg_before - opp_core_avg_after
			
			if your_core_gain >= min_gain and opp_core_gain >= -MAX_OPP_WEEKLY_LOSS and opp_core_avg_drop <= MAX_OPP_CORE_AVG_DROP and _is_realistic_trade(your_players, [their_player], league_tiers):
				floor_delta = _calculate_floor_impact(your_full_team, your_players, [their_player])
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '3-for-1',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [their_player['Player']],
					'your_value': your_total_value,
					'their_value': their_player['Value'],
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,
					'reasoning': reasoning,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [their_player['Mean FPts']],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [their_player['CV %']]
				})
	
	return trades


def _find_3_for_2_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 3-for-2 trade opportunities with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	for your_combo in combinations(your_rows, 3):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get('Player') in include_players for p in your_players):
				continue
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_combo in combinations(other_team.iterrows(), 2):
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			# 3-for-2 consolidation: guard against giving up too much total FP/G
			if not _check_3_for_2_package_ratio(your_players, their_players):
				continue
			
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				your_players,
				their_players,
				core_size,
				baseline_core_value,
			)
			
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				their_players,
				your_players,
				core_size,
				opp_baseline_core,
			)
			
			opp_after_team = opp_full_team.copy()
			opp_after_team = opp_after_team[~opp_after_team['Player'].isin([p['Player'] for p in their_players])]
			opp_after_team = pd.concat([opp_after_team, pd.DataFrame(your_players)], ignore_index=True)
			opp_core_after = _calculate_core_value(opp_after_team, core_size)
			opp_core_avg_before = opp_baseline_core / core_size if core_size > 0 else 0
			opp_core_avg_after = opp_core_after / core_size if core_size > 0 else 0
			opp_core_avg_drop = opp_core_avg_before - opp_core_avg_after
			
			if your_core_gain >= min_gain and opp_core_gain >= -MAX_OPP_WEEKLY_LOSS and opp_core_avg_drop <= MAX_OPP_CORE_AVG_DROP and _is_realistic_trade(your_players, their_players, league_tiers):
				floor_delta = _calculate_floor_impact(your_full_team, your_players, their_players)
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '3-for-2',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_total_value,
					'their_value': their_total_value,
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,
					'reasoning': reasoning,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _find_1_for_2_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 1-for-2 trades (you give 1, get 2) with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	for your_player in your_rows:
		# Must include player if include_players specified
		if include_players is not None and len(include_players) > 0:
			if your_player.get('Player') not in include_players:
				continue
		for their_combo in combinations(their_rows, 2):
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			# For 1-for-2 depth trades, your single player should be higher FP/G
			# than each incoming player.
			your_fpts = your_player['Mean FPts']
			if any(your_fpts <= p['Mean FPts'] for p in their_players):
				continue
			# Tier- and slider-aware package FP/G ratio check for 1-for-2 expansions
			if not _check_1_for_n_package_ratio(your_player, their_players, league_tiers):
				continue
			their_total_value = sum(p['Value'] for p in their_players)
			
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				[your_player],
				their_players,
				core_size,
				baseline_core_value,
			)
			
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				their_players,
				[your_player],
				core_size,
				opp_baseline_core,
			)
			
			if your_core_gain >= min_gain and opp_core_gain >= -MAX_OPP_WEEKLY_LOSS and _is_realistic_trade([your_player], their_players, league_tiers):
				floor_delta = _calculate_floor_impact(your_full_team, [your_player], their_players)
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '1-for-2',
					'team': team_name,
					'you_give': [your_player['Player']],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_player['Value'],
					'their_value': their_total_value,
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,
					'reasoning': reasoning,
					'your_fpts': [your_player['Mean FPts']],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [your_player['CV %']],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _find_1_for_3_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 1-for-3 trades (you give 1, get 3) with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	for your_player in your_rows:
		if include_players is not None and len(include_players) > 0:
			if your_player.get('Player') not in include_players:
				continue
		for their_combo in combinations(their_rows, 3):
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			if target_opposing_players:
				if not any(p.get('Player') in target_opposing_players for p in their_players):
					continue
			# For 1-for-3 depth trades, your single player should be higher FP/G
			# than each incoming player.
			your_fpts = your_player['Mean FPts']
			if any(your_fpts <= p['Mean FPts'] for p in their_players):
				continue
			# Tier- and slider-aware package FP/G ratio check for 1-for-3 expansions
			if not _check_1_for_n_package_ratio(your_player, their_players, league_tiers):
				continue
			their_total_value = sum(p['Value'] for p in their_players)
			
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				[your_player],
				their_players,
				core_size,
				baseline_core_value,
			)
			
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				their_players,
				[your_player],
				core_size,
				opp_baseline_core,
			)
			
			if your_core_gain >= min_gain and opp_core_gain >= -MAX_OPP_WEEKLY_LOSS and _is_realistic_trade([your_player], their_players, league_tiers):
				floor_delta = _calculate_floor_impact(your_full_team, [your_player], their_players)
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '1-for-3',
					'team': team_name,
					'you_give': [your_player['Player']],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_player['Value'],
					'their_value': their_total_value,
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,
					'reasoning': reasoning,
					'your_fpts': [your_player['Mean FPts']],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [your_player['CV %']],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _find_2_for_3_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 2-for-3 trades with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	for your_combo in combinations(your_rows, 2):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get('Player') in include_players for p in your_players):
				continue
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_combo in combinations(other_team.iterrows(), 3):
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				your_players,
				their_players,
				core_size,
				baseline_core_value,
			)
			
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				their_players,
				your_players,
				core_size,
				opp_baseline_core,
			)
			
			if your_core_gain >= min_gain and opp_core_gain >= -MAX_OPP_WEEKLY_LOSS and _is_realistic_trade(your_players, their_players, league_tiers):
				floor_delta = _calculate_floor_impact(your_full_team, your_players, their_players)
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '2-for-3',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_total_value,
					'their_value': their_total_value,
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,
					'reasoning': reasoning,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades

def _find_3_for_3_trades(your_team, other_team, team_name, min_gain, your_full_team, core_size, baseline_core_value, include_players, opp_full_team, opp_baseline_core, league_tiers, target_opposing_players=None):
	"""Find 3-for-3 trades with symmetric core evaluation."""
	trades = []
	accepted = 0
	combo_counter = 0
	
	your_rows = your_team.to_dict('records')
	their_rows = other_team.to_dict('records')
	
	for your_combo in combinations(your_rows, 3):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get('Player') in include_players for p in your_players):
				continue
		your_total_value = sum(p['Value'] for p in your_players)
		
		for their_combo in combinations(other_team.iterrows(), 3):
			combo_counter += 1
			if combo_counter > MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = [p[1] for p in their_combo]
			their_total_value = sum(p['Value'] for p in their_players)
			
			your_core_gain = _simulate_core_value_gain(
				your_full_team,
				your_players,
				their_players,
				core_size,
				baseline_core_value,
			)
			
			opp_core_gain = _simulate_core_value_gain(
				opp_full_team,
				their_players,
				your_players,
				core_size,
				opp_baseline_core,
			)
			
			if your_core_gain >= min_gain and opp_core_gain >= -MAX_OPP_WEEKLY_LOSS and _is_realistic_trade(your_players, their_players, league_tiers):
				floor_delta = _calculate_floor_impact(your_full_team, your_players, their_players)
				reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
				trades.append({
					'pattern': '3-for-3',
					'team': team_name,
					'you_give': [p['Player'] for p in your_players],
					'you_get': [p['Player'] for p in their_players],
					'your_value': your_total_value,
					'their_value': their_total_value,
					'value_gain': your_core_gain,
					'opp_core_gain': opp_core_gain,
					'floor_impact': floor_delta,
					'reasoning': reasoning,
					'your_fpts': [p['Mean FPts'] for p in your_players],
					'their_fpts': [p['Mean FPts'] for p in their_players],
					'your_cv': [p['CV %'] for p in your_players],
					'their_cv': [p['CV %'] for p in their_players]
				})
	
	return trades


def _check_avg_fpts_ratio(your_avg_fpts: float, their_avg_fpts: float, is_consolidating: bool, is_expanding: bool) -> bool:
	"""Average FP/G sanity check for package quality similarity."""
	if min(your_avg_fpts, their_avg_fpts) <= 0:
		fpts_ratio = 999
	else:
		fpts_ratio = max(your_avg_fpts, their_avg_fpts) / min(your_avg_fpts, their_avg_fpts)
	if is_consolidating:
		if fpts_ratio > 1.35:
			return False
	elif is_expanding:
		if fpts_ratio > 1.25:
			return False
	else:
		if fpts_ratio > EQUAL_COUNT_MAX_AVG_FPTS_RATIO:
			return False
	return True


def _check_total_fpts_and_piece_quality(
	your_players,
	their_players,
	your_total_fpts: float,
	their_total_fpts: float,
	your_count: int,
	is_consolidating: bool,
	is_expanding: bool,
	league_tiers: Dict[str, float],
) -> bool:
	if is_consolidating:
		max_ratio = 1.25
		their_max = max(p['Mean FPts'] for p in their_players)
		elite_threshold = league_tiers.get('elite', 90)
		star_threshold = league_tiers.get('star', 70)
		if their_max >= elite_threshold:
			max_ratio = 1.35
		elif their_max >= star_threshold:
			max_ratio = 1.30
		if your_count == 2:
			your_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
			if your_sorted[1] < their_max * 0.45:
				return False
			if your_sorted[0] < their_max * 0.60:
				return False
		elif your_count == 3:
			your_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
			if any(p < their_max * 0.38 for p in your_sorted):
				return False
			if your_sorted[0] < their_max * 0.55:
				return False
		your_min = min(p['Mean FPts'] for p in your_players)
		your_avg = sum(p['Mean FPts'] for p in your_players) / len(your_players)
		min_piece_ratio = 0.52
		avg_piece_ratio = 0.64
		if your_min < their_max * min_piece_ratio:
			return False
		if your_avg < their_max * avg_piece_ratio:
			return False
		elite_threshold = league_tiers.get('elite', 90)
		star_threshold = league_tiers.get('star', 70)
		package_loss_for_them = their_total_fpts - your_total_fpts
		if package_loss_for_them > 0:
			if their_max >= elite_threshold:
				max_loss = 30
			elif their_max >= star_threshold:
				max_loss = 25
			else:
				max_loss = 20
			if package_loss_for_them > max_loss:
				return False
	elif is_expanding:
		max_ratio = 1.25
		your_max = max(p['Mean FPts'] for p in your_players)
		their_min = min(p['Mean FPts'] for p in their_players)
		their_avg = sum(p['Mean FPts'] for p in their_players) / len(their_players)
		min_piece_ratio = 0.52
		avg_piece_ratio = 0.64
		if their_min < your_max * min_piece_ratio:
			return False
		if their_avg < your_max * avg_piece_ratio:
			return False
	else:
		max_ratio = EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO
	if min(your_total_fpts, their_total_fpts) <= 0:
		total_ratio = 999
	else:
		total_ratio = max(your_total_fpts, their_total_fpts) / min(your_total_fpts, their_total_fpts)
	if total_ratio > max_ratio:
		return False
	return True


def _check_value_fairness_guard(
	your_players,
	their_players,
	is_consolidating: bool,
	is_expanding: bool,
	league_tiers: Dict[str, float],
) -> bool:
	if not ENABLE_VALUE_FAIRNESS_GUARD:
		return True
	if 'Value' not in your_players[0] or 'Value' not in their_players[0]:
		return True
	your_total_value = sum(float(p['Value']) for p in your_players)
	their_total_value = sum(float(p['Value']) for p in their_players)
	if your_total_value <= 0 or their_total_value <= 0:
		return True
	value_ratio = max(your_total_value, their_total_value) / min(your_total_value, their_total_value)
	if is_consolidating:
		elite_threshold = league_tiers.get('elite', 90)
		star_threshold = league_tiers.get('star', 70)
		incoming_max_fpts = max(p['Mean FPts'] for p in their_players)
		if incoming_max_fpts >= elite_threshold:
			max_value_ratio = 1.33
		elif incoming_max_fpts >= star_threshold:
			max_value_ratio = 1.30
		else:
			max_value_ratio = 1.25
	elif is_expanding:
		max_value_ratio = 1.18
	else:
		max_value_ratio = 1.12
	if value_ratio > max_value_ratio:
		return False
	return True


def _check_cv_tradeoff(
	your_players,
	their_players,
	your_avg_cv: float,
	their_avg_cv: float,
	your_avg_fpts: float,
	their_avg_fpts: float,
	is_consolidating: bool,
) -> bool:
	if your_avg_cv < their_avg_cv:
		cv_loss = their_avg_cv - your_avg_cv
		their_max_fpts = max(p['Mean FPts'] for p in their_players)
		your_max_fpts = max(p['Mean FPts'] for p in your_players)
		if cv_loss > 10:
			consistency_upgrade_needed = 1.05
		elif cv_loss > 5:
			consistency_upgrade_needed = 1.02
		else:
			consistency_upgrade_needed = 1.0
		if is_consolidating and their_max_fpts >= your_max_fpts + 5:
			consistency_upgrade_needed = 1.0
		if their_avg_fpts < your_avg_fpts * consistency_upgrade_needed:
			return False
	return True


def _derive_max_fpts(your_players, their_players) -> Tuple[float, float]:
	your_max_fpts = max(p['Mean FPts'] for p in your_players)
	their_max_fpts = max(p['Mean FPts'] for p in their_players)
	return your_max_fpts, their_max_fpts


def _check_tier_protection(
	your_players,
	their_players,
	your_max_fpts: float,
	their_max_fpts: float,
	your_count: int,
	their_count: int,
	is_consolidating: bool,
	is_expanding: bool,
	league_tiers: Dict[str, float],
) -> bool:
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	starter_threshold = league_tiers['starter']
	if your_max_fpts >= elite_threshold:
		if is_expanding and their_count >= 3:
			if not all(p['Mean FPts'] >= quality_threshold for p in their_players):
				return False
		elif their_max_fpts < star_threshold:
			return False
		if is_consolidating and their_count == 1:
			if their_max_fpts < elite_threshold * 0.90:
				return False
	elif their_max_fpts >= elite_threshold:
		if is_consolidating and your_count >= 3:
			if not all(p['Mean FPts'] >= quality_threshold for p in your_players):
				return False
		elif your_max_fpts < star_threshold:
			return False
		if your_count == 1 and your_max_fpts < elite_threshold * 0.90:
			return False
	if your_max_fpts >= star_threshold:
		if is_expanding and their_count >= 3:
			if not all(p['Mean FPts'] >= starter_threshold for p in their_players):
				return False
		elif their_max_fpts < quality_threshold:
			return False
	if their_max_fpts >= star_threshold:
		if is_consolidating and your_count >= 3:
			if not all(p['Mean FPts'] >= starter_threshold for p in your_players):
				return False
		elif your_max_fpts < quality_threshold:
			return False
	return True


def _check_consolidation_upgrade(
	is_consolidating: bool,
	your_max_fpts: float,
	their_max_fpts: float,
	league_tiers: Dict[str, float],
) -> bool:
	if not is_consolidating:
		return True
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	if their_max_fpts >= elite_threshold:
		if your_max_fpts < star_threshold:
			return False
		min_upgrade = elite_threshold * 0.15
		if their_max_fpts < your_max_fpts + min_upgrade:
			return False
	elif their_max_fpts >= star_threshold:
		min_upgrade = star_threshold * 0.12
		if their_max_fpts < your_max_fpts + min_upgrade:
			return False
	elif their_max_fpts >= quality_threshold:
		min_upgrade = quality_threshold * 0.10
		if their_max_fpts < your_max_fpts + min_upgrade:
			return False
	else:
		min_upgrade = their_max_fpts * 0.08
		if their_max_fpts < your_max_fpts + min_upgrade:
			return False
	return True


def _check_best_player_ratio(
	is_consolidating: bool,
	is_expanding: bool,
	your_max_fpts: float,
	their_max_fpts: float,
	league_tiers: Dict[str, float],
) -> bool:
	if min(your_max_fpts, their_max_fpts) <= 0:
		best_player_ratio = 999
	else:
		best_player_ratio = max(your_max_fpts, their_max_fpts) / min(your_max_fpts, their_max_fpts)
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	if is_consolidating:
		if their_max_fpts >= elite_threshold:
			best_ratio_limit = 1.20
		elif their_max_fpts >= star_threshold:
			best_ratio_limit = 1.22
		elif their_max_fpts >= quality_threshold:
			best_ratio_limit = 1.25
		else:
			best_ratio_limit = 1.18
	elif is_expanding:
		best_ratio_limit = 1.12
	else:
		best_ratio_limit = 1.08
	if best_player_ratio > best_ratio_limit:
		return False
	return True


def _check_individual_matchups(
	your_players,
	their_players,
	is_consolidating: bool,
	is_expanding: bool,
	their_max_fpts: float,
	league_tiers: Dict[str, float],
) -> bool:
	your_fpts_sorted = sorted([p['Mean FPts'] for p in your_players], reverse=True)
	their_fpts_sorted = sorted([p['Mean FPts'] for p in their_players], reverse=True)
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	quality_threshold = league_tiers['quality']
	for i, your_fpts in enumerate(your_fpts_sorted):
		if i < len(their_fpts_sorted):
			their_fpts = their_fpts_sorted[i]
			if is_consolidating:
				if their_max_fpts >= elite_threshold:
					player_ratio_limit = 1.28
				elif their_max_fpts >= star_threshold:
					player_ratio_limit = 1.32
				elif their_max_fpts >= quality_threshold:
					player_ratio_limit = 1.35
				else:
					player_ratio_limit = 1.20
			elif is_expanding:
				player_ratio_limit = 1.15
			else:
				player_ratio_limit = 1.10
			if min(your_fpts, their_fpts) <= 0:
				player_ratio = 999
			else:
				player_ratio = max(your_fpts, their_fpts) / min(your_fpts, their_fpts)
			if player_ratio > player_ratio_limit:
				return False
	return True


def _check_1_for_n_package_ratio(
	your_player,
	their_players,
	league_tiers: Dict[str, float],
) -> bool:
	"""Tier- and slider-aware total FP/G ratio guard specifically for 1-for-2/1-for-3.

	At normal/strict settings, keep ratios fairly modest; at very loose settings,
	allow bigger upgrades, especially when you're giving up an elite.
	"""
	your_fpts = float(your_player['Mean FPts'])
	their_total_fpts = float(sum(p['Mean FPts'] for p in their_players))
	if your_fpts <= 0 or their_total_fpts <= 0:
		return True
	# You are expanding: giving 1, getting >1
	total_ratio = max(your_fpts, their_total_fpts) / min(your_fpts, their_total_fpts)
	# Base caps by tier of your outgoing player
	elite_threshold = league_tiers['elite']
	star_threshold = league_tiers['star']
	if your_fpts >= elite_threshold:
		# Elite outgoing: allow strong but not absurd depth upgrades
		base_cap = 1.85
	elif your_fpts >= star_threshold:
		# Star outgoing: consolidating into multiple solid pieces can be more dramatic
		base_cap = 2.25
	else:
		base_cap = 1.8
	# Trade-balance slider adjustment: higher looseness => higher cap
	# TRADE_BALANCE_LEVEL ~ 1..50; map to multiplier ~ [0.9, 1.2]
	looseness = TRADE_BALANCE_LEVEL
	looseness_factor = 0.9 + (min(max(looseness, 1), 50) - 1) * (0.3 / 49.0)
	max_ratio = base_cap * looseness_factor
	return total_ratio <= max_ratio


def _check_3_for_2_package_ratio(
	your_players,
	their_players,
)	-> bool:
	"""Slider-aware total FP/G ratio guard for 3-for-2 consolidations.

	At strict settings, you should not give up much more total FP/G than you
	receive when consolidating 3 pieces into 2. At looser settings, allow
	slightly larger FP/G sacrifices for meaningful consolidation upgrades.
	"""
	your_total_fpts = float(sum(p['Mean FPts'] for p in your_players))
	their_total_fpts = float(sum(p['Mean FPts'] for p in their_players))
	if your_total_fpts <= 0 or their_total_fpts <= 0:
		return True
	# Only guard cases where you are giving more total FP/G than you get.
	if your_total_fpts <= their_total_fpts:
		return True
	total_ratio = your_total_fpts / their_total_fpts
	# Base cap: at strictest setting, allow only a very small FP/G sacrifice.
	base_cap = 1.03
	# TRADE_BALANCE_LEVEL ~ 1..50; map to multiplier ~ [1.0, 1.2]
	looseness = TRADE_BALANCE_LEVEL
	looseness_factor = 1.0 + (min(max(looseness, 1), 50) - 1) * (0.2 / 49.0)
	max_ratio = base_cap * looseness_factor
	return total_ratio <= max_ratio


def _is_realistic_trade(your_players, their_players, league_tiers: Dict[str, float]):
	"""Check if a trade is realistic (not too lopsided).
	Prevents suggesting trades that no one would accept.
	Considers both FPts and consistency (CV%).

	Now more lenient to allow:
	- Consistency upgrades (lower CV%) even if FP/G is sideways
	- Consolidation trades that improve core FP/G

	Uses percentile-based tiers from actual league data instead of hardcoded thresholds.
	"""
	your_count = len(your_players)
	their_count = len(their_players)
	is_consolidating = your_count > their_count
	is_expanding = their_count > your_count
	if TRADE_BALANCE_LEVEL >= 40 and (is_consolidating or is_expanding):
		return True
	your_avg_fpts = sum(p['Mean FPts'] for p in your_players) / len(your_players)
	their_avg_fpts = sum(p['Mean FPts'] for p in their_players) / len(their_players)
	your_avg_cv = sum(p['CV %'] for p in your_players) / len(your_players)
	their_avg_cv = sum(p['CV %'] for p in their_players) / len(their_players)
	# NOTE: Consistency (CV) is mostly your edge. Opponent realism is driven by FP/G.
	# We do NOT let "consistency upgrades" justify huge FP/G gaps an opponent
	# would never consciously accept.
	# 1) Average FP/G sanity check: don't allow trades where the average quality
	# of pieces is wildly different.
	if not _check_avg_fpts_ratio(your_avg_fpts, their_avg_fpts, is_consolidating, is_expanding):
		return False
	# 2) Total FP/G sanity check (package totals). This is lenient for
	# consolidations into true elite players, but independent of CV.
	your_total_fpts = sum(p['Mean FPts'] for p in your_players)
	their_total_fpts = sum(p['Mean FPts'] for p in their_players)
	if not _check_total_fpts_and_piece_quality(
		your_players,
		their_players,
		your_total_fpts,
		their_total_fpts,
		your_count,
		is_consolidating,
		is_expanding,
		league_tiers,
	):
		return False
	if not _check_value_fairness_guard(
		your_players,
		their_players,
		is_consolidating,
		is_expanding,
		league_tiers,
	):
		return False
	# CV% check: Only block if trading consistency for volatility WITHOUT FP/G upgrade
	# Now more lenient - consistency for FP/G is a valid tradeoff
	if not _check_cv_tradeoff(
		your_players,
		their_players,
		your_avg_cv,
		their_avg_cv,
		your_avg_fpts,
		their_avg_fpts,
		is_consolidating,
	):
		return False
	# Additional check: prevent trading elite players for scrubs
	your_max_fpts, their_max_fpts = _derive_max_fpts(your_players, their_players)
	# TIER-BASED PROTECTION: Elite players require elite return
	# BUT: Allow 1-for-3 expansions as a valid depth strategy
	# Now uses percentile-based tiers instead of hardcoded thresholds
	if not _check_tier_protection(
		your_players,
		their_players,
		your_max_fpts,
		their_max_fpts,
		your_count,
		their_count,
		is_consolidating,
		is_expanding,
		league_tiers,
	):
		return False
	# When consolidating, ensure the incoming player meaningfully upgrades your top end
	# Now uses percentile-based upgrade requirements
	if not _check_consolidation_upgrade(
		is_consolidating,
		your_max_fpts,
		their_max_fpts,
		league_tiers,
	):
		return False
	# Additional check: best player comparison
	# The best player in the trade shouldn't be too much better than the other side's best
	if not _check_best_player_ratio(
		is_consolidating,
		is_expanding,
		your_max_fpts,
		their_max_fpts,
		league_tiers,
	):
		return False
	# Check individual player matchups - each player you give should have a comparable player you get
	if not _check_individual_matchups(
		your_players,
		their_players,
		is_consolidating,
		is_expanding,
		their_max_fpts,
		league_tiers,
	):
		return False
	return True
