"""Pattern search helpers for the trade suggestion engine.

This module contains the _find_* helpers that enumerate concrete trade
packages for each pattern (1-for-1, 2-for-1, ..., 3-for-3). It is
intentionally free of Streamlit/UI concerns so it can be imported by the
core engine and any offline tools.
"""

from typing import List, Dict, Optional

import pandas as pd
from itertools import combinations

import modules.trade_suggestions_config as cfg
from modules.trade_suggestions_core import (
	_simulate_core_value_gain,
	_calculate_core_value,
	_calculate_floor_impact,
	_determine_trade_reasoning,
	_check_opponent_core_avg_drop,
)
from modules.trade_suggestions_realism import (
	_is_realistic_trade,
	_check_1_for_n_package_ratio,
	_check_3_for_2_package_ratio,
)


def _get_expansion_min_core_gain(base_min_gain: float) -> float:
	"""Adjust the minimum core gain requirement for expansion patterns.

	At strict/normal Trade Balance levels, require non-negative core gain as usual.
	At very loose settings, allow a small core downgrade to surface more depth trades.
	"""
	# At the absolute loosest setting, effectively remove your_core_gain as a gate
	# for expansion patterns by allowing a very large negative threshold.
	level = cfg.TRADE_BALANCE_LEVEL
	if level >= 50:
		return base_min_gain - 9999.0
	elif level >= 45:
		return base_min_gain - 5.0
	elif level >= 35:
		return base_min_gain - 2.5
	return base_min_gain


def _is_ultra_elite_single(players, league_tiers):
	if len(players) != 1:
		return False
	elite = float(league_tiers.get("elite", 0) or 0.0)
	star = float(league_tiers.get("star", elite + 10.0))
	fpts = float(players[0].get("Mean FPts", 0) or 0.0)
	gap = max(3.0, star - elite)
	threshold = elite + gap
	return fpts >= threshold


def _passes_score_guards(
	your_players,
	their_players,
	your_weekly_gain: float,
	opp_weekly_gain: float,
	min_gain: float,
	league_tiers,
):
	if not your_players or not their_players:
		return False
	if min_gain > 0 and your_weekly_gain < min_gain:
		return False
	your_avg_fpts = sum(float(p["Mean FPts"]) for p in your_players) / len(your_players)
	their_avg_fpts = sum(float(p["Mean FPts"]) for p in their_players) / len(their_players)
	your_avg_cv = sum(float(p["CV %"]) for p in your_players) / len(your_players)
	their_avg_cv = sum(float(p["CV %"]) for p in their_players) / len(their_players)
	if cfg.MIN_GAMES_REQUIRED > 0:
		core_ppg_you = your_weekly_gain / float(cfg.MIN_GAMES_REQUIRED)
		core_ppg_opp = opp_weekly_gain / float(cfg.MIN_GAMES_REQUIRED)
		base_you = min_gain / float(cfg.MIN_GAMES_REQUIRED)
	else:
		core_ppg_you = 0.0
		core_ppg_opp = 0.0
		base_you = 0.0
	pkg_fp_change = their_avg_fpts - your_avg_fpts
	pkg_cv_change = their_avg_cv - your_avg_cv
	cv_gain_you = -pkg_cv_change
	cv_gain_opp = your_avg_cv - their_avg_cv
	score_you = core_ppg_you + cv_gain_you / 5.0
	score_opp = core_ppg_opp + cv_gain_opp / 5.0
	spec_margin = 0.5
	if score_you < base_you - spec_margin:
		return False
	level = max(1, min(50, int(cfg.TRADE_BALANCE_LEVEL)))
	t_val = (level - 1) / 49.0
	opp_tol = 0.1 + 1.4 * t_val
	if score_opp < -opp_tol:
		return False
	if opp_weekly_gain < -cfg.MAX_OPP_WEEKLY_LOSS:
		return False
	if _is_ultra_elite_single(their_players, league_tiers) and score_opp < 0.0:
		return False
	return True


def _find_1_for_1_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 1-for-1 trade opportunities with symmetric core evaluation."""
	trades: List[Dict] = []
	accepted = 0
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")

	for your_player in your_rows:
		for their_player in their_rows:
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			# Enforce target_opposing_players constraint (their side)
			if target_opposing_players:
				if their_player.get("Player") not in target_opposing_players:
					continue
			# Enforce must-include constraint if provided (your side)
			if include_players is not None and len(include_players) > 0:
				if your_player.get("Player") not in include_players:
					continue
			# Enforce target_opposing_players constraint (their side) again (defensive)
			if target_opposing_players:
				if their_player.get("Player") not in target_opposing_players:
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

			if not _passes_score_guards(
				[your_player],
				[their_player],
				your_core_gain,
				opp_core_gain,
				min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(
				your_full_team,
				[your_player],
				[their_player],
			)
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "1-for-1",
					"team": team_name,
					"you_give": [your_player["Player"]],
					"you_get": [their_player["Player"]],
					"your_value": your_player["Value"],
					"their_value": their_player["Value"],
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [your_player["Mean FPts"]],
					"their_fpts": [their_player["Mean FPts"]],
					"your_cv": [your_player["CV %"]],
					"their_cv": [their_player["CV %"]],
				}
			)
			accepted += 1
			if accepted >= cfg.MAX_ACCEPTED_TRADES_PER_PATTERN_TEAM:
				return trades

	return trades


def _find_2_for_1_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 2-for-1 trade opportunities (you give 2, get 1) with symmetric core evaluation."""
	trades: List[Dict] = []
	accepted = 0
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")

	for your_combo in combinations(your_rows, 2):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get("Player") in include_players for p in your_players):
				continue
		your_total_value = sum(p["Value"] for p in your_players)

		for their_player in their_rows:
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			if target_opposing_players:
				if their_player.get("Player") not in target_opposing_players:
					continue

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

			if not _passes_score_guards(
				your_players,
				[their_player],
				your_core_gain,
				opp_core_gain,
				min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(your_full_team, your_players, [their_player])
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "2-for-1",
					"team": team_name,
					"you_give": [p["Player"] for p in your_players],
					"you_get": [their_player["Player"]],
					"your_value": your_total_value,
					"their_value": their_player["Value"],
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [p["Mean FPts"] for p in your_players],
					"their_fpts": [their_player["Mean FPts"]],
					"your_cv": [p["CV %"] for p in your_players],
					"their_cv": [their_player["CV %"]],
				}
			)
			accepted += 1
			if accepted >= cfg.MAX_ACCEPTED_TRADES_PER_PATTERN_TEAM:
				return trades

	return trades


def _find_1_for_2_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 1-for-2 trades (you give 1, get 2) with symmetric core evaluation."""
	trades: List[Dict] = []
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")
	expansion_min_gain = _get_expansion_min_core_gain(min_gain)

	for your_player in your_rows:
		if include_players is not None and len(include_players) > 0:
			if your_player.get("Player") not in include_players:
				continue
		for their_combo in combinations(their_rows, 2):
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			# your single player should be higher FP/G than each incoming player
			your_fpts = your_player["Mean FPts"]
			if cfg.TRADE_BALANCE_LEVEL < 45:
				if any(your_fpts <= p["Mean FPts"] for p in their_players):
					continue
			# At Trade Balance 50, treat as idea generator and skip the 1-for-n ratio guard
			if cfg.TRADE_BALANCE_LEVEL < 50:
				if not _check_1_for_n_package_ratio(your_player, their_players, league_tiers):
					continue
			if target_opposing_players:
				if not any(p.get("Player") in target_opposing_players for p in their_players):
					continue
			their_total_value = sum(p["Value"] for p in their_players)

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

			if not _passes_score_guards(
				[your_player],
				their_players,
				your_core_gain,
				opp_core_gain,
				expansion_min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(your_full_team, [your_player], their_players)
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "1-for-2",
					"team": team_name,
					"you_give": [your_player["Player"]],
					"you_get": [p["Player"] for p in their_players],
					"your_value": your_player["Value"],
					"their_value": their_total_value,
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [your_player["Mean FPts"]],
					"their_fpts": [p["Mean FPts"] for p in their_players],
					"your_cv": [your_player["CV %"]],
					"their_cv": [p["CV %"] for p in their_players],
				}
			)

	return trades


def _find_1_for_3_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 1-for-3 trades (you give 1, get 3) with symmetric core evaluation."""
	trades: List[Dict] = []
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")
	expansion_min_gain = _get_expansion_min_core_gain(min_gain)

	for your_player in your_rows:
		if include_players is not None and len(include_players) > 0:
			if your_player.get("Player") not in include_players:
				continue
		for their_combo in combinations(their_rows, 3):
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			if target_opposing_players:
				if not any(p.get("Player") in target_opposing_players for p in their_players):
					continue
			your_fpts = your_player["Mean FPts"]
			if cfg.TRADE_BALANCE_LEVEL < 45:
				if any(your_fpts <= p["Mean FPts"] for p in their_players):
					continue
			# At Trade Balance 50, treat as idea generator and skip the 1-for-n ratio guard
			if cfg.TRADE_BALANCE_LEVEL < 50:
				if not _check_1_for_n_package_ratio(your_player, their_players, league_tiers):
					continue
			their_total_value = sum(p["Value"] for p in their_players)

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

			if not _passes_score_guards(
				[your_player],
				their_players,
				your_core_gain,
				opp_core_gain,
				expansion_min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(your_full_team, [your_player], their_players)
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "1-for-3",
					"team": team_name,
					"you_give": [your_player["Player"]],
					"you_get": [p["Player"] for p in their_players],
					"your_value": your_player["Value"],
					"their_value": their_total_value,
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [your_player["Mean FPts"]],
					"their_fpts": [p["Mean FPts"] for p in their_players],
					"your_cv": [your_player["CV %"]],
					"their_cv": [p["CV %"] for p in their_players],
				}
			)

	return trades


def _find_2_for_3_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 2-for-3 trades with symmetric core evaluation."""
	trades: List[Dict] = []
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")

	for your_combo in combinations(your_rows, 2):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get("Player") in include_players for p in your_players):
				continue
		your_total_value = sum(p["Value"] for p in your_players)

		expansion_min_gain = _get_expansion_min_core_gain(min_gain)
		for their_combo in combinations(their_rows, 3):
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			if target_opposing_players:
				if not any(p.get("Player") in target_opposing_players for p in their_players):
					continue
			their_total_value = sum(p["Value"] for p in their_players)

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

			if not _passes_score_guards(
				your_players,
				their_players,
				your_core_gain,
				opp_core_gain,
				expansion_min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(your_full_team, your_players, their_players)
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "2-for-3",
					"team": team_name,
					"you_give": [p["Player"] for p in your_players],
					"you_get": [p["Player"] for p in their_players],
					"your_value": your_total_value,
					"their_value": their_total_value,
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [p["Mean FPts"] for p in your_players],
					"their_fpts": [p["Mean FPts"] for p in their_players],
					"your_cv": [p["CV %"] for p in your_players],
					"their_cv": [p["CV %"] for p in their_players],
				}
			)

	return trades


def _find_2_for_2_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 2-for-2 trade opportunities with symmetric core evaluation."""
	trades: List[Dict] = []
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")

	for your_combo in combinations(your_rows, 2):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get("Player") in include_players for p in your_players):
				continue
		your_total_value = sum(p["Value"] for p in your_players)

		for their_combo in combinations(their_rows, 2):
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			if target_opposing_players:
				if not any(p.get("Player") in target_opposing_players for p in their_players):
					continue
			their_total_value = sum(p["Value"] for p in their_players)

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

			if not _passes_score_guards(
				your_players,
				their_players,
				your_core_gain,
				opp_core_gain,
				min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(your_full_team, your_players, their_players)
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "2-for-2",
					"team": team_name,
					"you_give": [p["Player"] for p in your_players],
					"you_get": [p["Player"] for p in their_players],
					"your_value": your_total_value,
					"their_value": their_total_value,
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [p["Mean FPts"] for p in your_players],
					"their_fpts": [p["Mean FPts"] for p in their_players],
					"your_cv": [p["CV %"] for p in your_players],
					"their_cv": [p["CV %"] for p in their_players],
				}
			)

	return trades


def _find_3_for_1_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 3-for-1 trade opportunities (you give 3, get 1 superstar)."""
	trades: List[Dict] = []
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")

	for your_combo in combinations(your_rows, 3):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get("Player") in include_players for p in your_players):
				continue
		your_total_value = sum(p["Value"] for p in your_players)

		for their_player in their_rows:
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			if target_opposing_players:
				if their_player.get("Player") not in target_opposing_players:
					continue

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

			if not _passes_score_guards(
				your_players,
				[their_player],
				your_core_gain,
				opp_core_gain,
				min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(your_full_team, your_players, [their_player])
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "3-for-1",
					"team": team_name,
					"you_give": [p["Player"] for p in your_players],
					"you_get": [their_player["Player"]],
					"your_value": your_total_value,
					"their_value": their_player["Value"],
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [p["Mean FPts"] for p in your_players],
					"their_fpts": [their_player["Mean FPts"]],
					"your_cv": [p["CV %"] for p in your_players],
					"their_cv": [their_player["CV %"]],
				}
			)

	return trades


def _find_3_for_2_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 3-for-2 trade opportunities with symmetric core evaluation."""
	trades: List[Dict] = []
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")

	for your_combo in combinations(your_rows, 3):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get("Player") in include_players for p in your_players):
				continue
		your_total_value = sum(p["Value"] for p in your_players)

		for their_combo in combinations(their_rows, 2):
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			if target_opposing_players:
				if not any(p.get("Player") in target_opposing_players for p in their_players):
					continue
			their_total_value = sum(p["Value"] for p in their_players)

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

			# Optional 3-for-2 FP/G ratio sanity check
			if cfg.TRADE_BALANCE_LEVEL < 40:
				if not _check_3_for_2_package_ratio(your_players, their_players):
					continue

			if not _passes_score_guards(
				your_players,
				their_players,
				your_core_gain,
				opp_core_gain,
				min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(your_full_team, your_players, their_players)
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "3-for-2",
					"team": team_name,
					"you_give": [p["Player"] for p in your_players],
					"you_get": [p["Player"] for p in their_players],
					"your_value": your_total_value,
					"their_value": their_total_value,
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [p["Mean FPts"] for p in your_players],
					"their_fpts": [p["Mean FPts"] for p in their_players],
					"your_cv": [p["CV %"] for p in your_players],
					"their_cv": [p["CV %"] for p in their_players],
				}
			)

	return trades


def _find_3_for_3_trades(
	your_team,
	other_team,
	team_name,
	min_gain,
	your_full_team,
	core_size,
	baseline_core_value,
	include_players,
	opp_full_team,
	opp_baseline_core,
	league_tiers,
	target_opposing_players: Optional[List[str]] = None,
):
	"""Find 3-for-3 trades with symmetric core evaluation."""
	trades: List[Dict] = []
	combo_counter = 0

	your_rows = your_team.to_dict("records")
	their_rows = other_team.to_dict("records")

	for your_combo in combinations(your_rows, 3):
		your_players = list(your_combo)
		if include_players is not None and len(include_players) > 0:
			if not any(p.get("Player") in include_players for p in your_players):
				continue
		your_total_value = sum(p["Value"] for p in your_players)

		for their_combo in combinations(their_rows, 3):
			combo_counter += 1
			if combo_counter > cfg.MAX_COMBINATIONS_PER_PATTERN:
				return trades
			their_players = list(their_combo)
			if target_opposing_players:
				if not any(p.get("Player") in target_opposing_players for p in their_players):
					continue
			their_total_value = sum(p["Value"] for p in their_players)

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

			if not _passes_score_guards(
				your_players,
				their_players,
				your_core_gain,
				opp_core_gain,
				min_gain,
				league_tiers,
			):
				continue

			floor_delta = _calculate_floor_impact(your_full_team, your_players, their_players)
			reasoning = _determine_trade_reasoning(your_core_gain, floor_delta)
			trades.append(
				{
					"pattern": "3-for-3",
					"team": team_name,
					"you_give": [p["Player"] for p in your_players],
					"you_get": [p["Player"] for p in their_players],
					"your_value": your_total_value,
					"their_value": their_total_value,
					"value_gain": your_core_gain,
					"opp_core_gain": opp_core_gain,
					"floor_impact": floor_delta,
					"reasoning": reasoning,
					"your_fpts": [p["Mean FPts"] for p in your_players],
					"their_fpts": [p["Mean FPts"] for p in their_players],
					"your_cv": [p["CV %"] for p in your_players],
					"their_cv": [p["CV %"] for p in their_players],
				}
			)

	return trades
