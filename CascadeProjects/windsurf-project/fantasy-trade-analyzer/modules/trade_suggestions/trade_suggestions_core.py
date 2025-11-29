"""Core impact helpers for the trade suggestion engine.

This module centralizes core-size, core-value, floor-impact, and realism-cap
updates so they can be reused by both the main engine and auxiliary tools
without importing the entire trade_suggestions module.
"""

from typing import Dict, List, Optional

import math
import pandas as pd

from .trade_suggestions_config import (
	ROSTER_SIZE,
	AVG_GAMES_PER_PLAYER,
	MIN_GAMES_REQUIRED,
	MAX_OPP_CORE_AVG_DROP,
	MAX_OPP_WEEKLY_LOSS,
	EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS,
	EQUAL_COUNT_MAX_AVG_FPTS_RATIO,
	EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO,
	TRADE_BALANCE_LEVEL,
)


def _update_realism_caps_from_league(scarcity_context: Dict, league_tiers: Dict[str, float]) -> None:
	"""Derive opponent-loss and ratio caps from league FP/G and CV plus balance level.

	This keeps the thresholds scaled to your actual league while letting the
	Trade Balance slider act as a simple 1–10 strictness control.
	"""
	global MAX_OPP_CORE_AVG_DROP, MAX_OPP_WEEKLY_LOSS, EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS
	global EQUAL_COUNT_MAX_AVG_FPTS_RATIO, EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO

	if not league_tiers:
		return

	quality = float(league_tiers.get("quality", 50.0))
	star = float(league_tiers.get("star", quality + 10.0))
	fp_unit = max(5.0, star - quality)

	# Base core FP/G drop the opponent might tolerate in a "typical" league.
	base_core_drop = fp_unit * 0.3  # ~30% of the quality→star gap

	# Volatility factor: swingier leagues allow slightly larger FP/G loss.
	league_avg_cv = scarcity_context.get("league_avg_cv")
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


def _calculate_core_value(team_df: pd.DataFrame, core_size: int) -> float:
	"""Sum of value for top N players by FP/G, representing core roster value."""
	if team_df is None or team_df.empty or core_size <= 0:
		return 0.0
	sorted_df = team_df.sort_values("Mean FPts", ascending=False)
	core_df = sorted_df.head(core_size)
	if "Value" not in core_df.columns:
		return 0.0
	return float(core_df["Value"].sum())


def _check_opponent_core_avg_drop(
	opp_full_team: pd.DataFrame,
	opp_baseline_core: float,
	players_they_give: List[pd.Series],
	players_they_get: List[pd.Series],
	core_size: int,
) -> bool:
	"""Check if opponent's core FP/G average drops too much.

	Returns True if the drop is acceptable, False if it exceeds MAX_OPP_CORE_AVG_DROP.
	This enforces the league philosophy: FP/G > total FP.
	"""
	opp_after_team = opp_full_team.copy()

	# Remove players they're giving
	give_names = [p.get("Player") for p in players_they_give if "Player" in p]
	if give_names:
		opp_after_team = opp_after_team[~opp_after_team["Player"].isin(give_names)]

	# Add players they're getting
	for p in players_they_get:
		opp_after_team = pd.concat([opp_after_team, pd.DataFrame([p])], ignore_index=True)

	opp_core_after = _calculate_core_value(opp_after_team, core_size)
	opp_core_avg_before = opp_baseline_core / core_size if core_size > 0 else 0
	opp_core_avg_after = opp_core_after / core_size if core_size > 0 else 0
	opp_core_avg_drop = opp_core_avg_before - opp_core_avg_after

	return opp_core_avg_drop <= MAX_OPP_CORE_AVG_DROP


def _simulate_core_value_gain(
	your_full_team: pd.DataFrame,
	your_players_give: List[pd.Series],
	their_players_get: List[pd.Series],
	core_size: int,
	baseline_core_value: float,
) -> float:
	"""Simulate change in your weekly core output if a given trade is executed."""
	if your_full_team is None or your_full_team.empty or core_size <= 0:
		return 0.0

	after_team = your_full_team.copy()

	# Remove players you are giving up
	if your_players_give:
		give_names = [p.get("Player") for p in your_players_give if "Player" in p]
		if give_names:
			after_team = after_team[~after_team["Player"].isin(give_names)].copy()

	# Add players you are receiving
	if their_players_get:
		new_rows = pd.DataFrame(their_players_get)
		if not new_rows.empty:
			if "Player" in after_team.columns and "Player" in new_rows.columns:
				existing = set(after_team["Player"].tolist())
				new_rows = new_rows[~new_rows["Player"].isin(existing)]
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
	"""Calculate change in roster floor (bottom N players' average FP/G)."""
	if your_full_team is None or your_full_team.empty:
		return 0.0

	# Floor before trade
	floor_before = your_full_team.nsmallest(floor_size, "Mean FPts")["Mean FPts"].mean()

	after_team = your_full_team.copy()

	# Remove players you are giving up
	if your_players_give:
		give_names = [p.get("Player") for p in your_players_give if "Player" in p]
		if give_names:
			after_team = after_team[~after_team["Player"].isin(give_names)].copy()

	# Add players you are receiving
	if their_players_get:
		new_rows = pd.DataFrame(their_players_get)
		if not new_rows.empty:
			if "Player" in after_team.columns and "Player" in new_rows.columns:
				existing = set(after_team["Player"].tolist())
				new_rows = new_rows[~new_rows["Player"].isin(existing)]
			if not new_rows.empty:
				after_team = pd.concat([after_team, new_rows], ignore_index=True)

	# For consolidation trades, add waiver wire replacement (40 FP/G)
	num_give = len(your_players_give) if your_players_give else 0
	num_get = len(their_players_get) if their_players_get else 0
	if num_give > num_get:
		num_waiver = num_give - num_get
		for i in range(num_waiver):
			waiver_player = pd.Series({
				"Player": f"Waiver_Wire_{i}",
				"Mean FPts": 40.0,
				"Value": 40.0,
			})
			after_team = pd.concat([after_team, pd.DataFrame([waiver_player])], ignore_index=True)

	if len(after_team) >= floor_size:
		floor_after = after_team.nsmallest(floor_size, "Mean FPts")["Mean FPts"].mean()
	else:
		floor_after = after_team["Mean FPts"].mean() if not after_team.empty else 0.0

	return floor_after - floor_before


def _determine_trade_reasoning(core_gain: float, floor_delta: float) -> str:
	"""Determine the strategic purpose of a trade based on core and floor impacts."""
	CORE_THRESHOLD = 2.0  # Weekly core FP gain
	FLOOR_THRESHOLD = 1.0  # FP/G floor change

	if core_gain > CORE_THRESHOLD and floor_delta < -FLOOR_THRESHOLD:
		return "Consolidation"
	elif core_gain < CORE_THRESHOLD and floor_delta > FLOOR_THRESHOLD:
		return "Deconstruction"
	elif core_gain > CORE_THRESHOLD and floor_delta > FLOOR_THRESHOLD:
		return "Overall Improvement"
	elif abs(core_gain) < CORE_THRESHOLD and abs(floor_delta) < FLOOR_THRESHOLD:
		return "Lateral Move"
	else:
		return "Mixed Impact"
