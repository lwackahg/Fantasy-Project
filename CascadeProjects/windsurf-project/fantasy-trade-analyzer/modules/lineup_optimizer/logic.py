from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from modules.player_game_log_scraper.logic import (
	get_player_code_by_name,
	load_cached_player_log,
	calculate_multi_range_stats,
	calculate_variability_stats,
)


@dataclass
class PlayerComparisonResult:
	"""Container for a single player's comparison metrics for a given date.
	
	This is returned primarily for programmatic use; the public helper returns
	a pandas DataFrame built from a list of these objects.
	"""

	player_name: str
	player_code: Optional[str]
	league_id: str
	season: Optional[str]
	status: str
	games_sampled: int = 0
	base_mean_fpts: float = 0.0
	baseline_mean_fpts: float = 0.0
	blended_projection: float = 0.0
	cv_percent: float = 0.0
	std_dev: float = 0.0
	form_stability: float = 0.0
	matchup: Optional[str] = None
	matchup_favorability: float = 0.5
	situational_boost: float = 0.0
	trend_last7_vs_30: Optional[float] = None
	expected_fpts: float = 0.0
	confidence: float = 0.0
	utility: float = 0.0
	games_before: Optional[int] = None
	games_after_if_started: Optional[int] = None
	ppg_before: Optional[float] = None
	ppg_after_if_started: Optional[float] = None
	ppg_delta: Optional[float] = None
	ppg_tag: Optional[str] = None
	ppg_impact_score: Optional[float] = None
	decision: str = "UNKNOWN"
	reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Projection & Utility Constants
# ─────────────────────────────────────────────────────────────────────────────
# Regression constant for Empirical Bayes blending. Higher = more regression
# toward baseline. 10 means "treat baseline as if it were 10 games of data".
REGRESSION_CONSTANT_K = 10

# Risk aversion coefficient for utility calculation. Higher = more penalty for
# variance. 0.4 is moderate risk aversion suitable for H2H weekly formats.
RISK_AVERSION_LAMBDA = 0.4

# Minimum CV% floor to prevent overconfidence in small samples
MIN_CV_FLOOR = 20.0


def _normalize_confidence(value: float) -> float:
	"""Clamp a confidence-like score to [0, 1]."""
	if pd.isna(value):
		return 0.5
	return float(max(0.0, min(1.0, value)))


def _compute_form_stability(cv_percent: float) -> float:
	"""Convert CV% into a [0,1] stability score (lower CV = more stable).
	
	We assume typical CV ranges roughly from 20% (very stable) to 80% (very
	volatile). Anything outside this band is clipped for the purposes of the
	score so that outliers do not dominate.
	"""
	if pd.isna(cv_percent):
		return 0.5
	# Map 20% -> 1.0, 80% -> 0.0 linearly and clamp
	scaled = 1.0 - (cv_percent - 20.0) / (80.0 - 20.0)
	return _normalize_confidence(scaled)


def _compute_matchup_favorability(
	player_name: str,
	opponent_by_player: Optional[Dict[str, str]],
	opponent_strength: Optional[Dict[str, float]],
) -> Tuple[Optional[str], float]:
	"""Return (opponent, favorability[0-1]) for a player.

	This is intentionally simple for v1:
	- opponent_strength is expected to map team code -> numeric score where
	  larger = softer matchup (e.g., 1-5). If not provided, we default to 0.5.
	"""
	if not opponent_by_player:
		return None, 0.5

	opp = opponent_by_player.get(player_name)
	if not opp:
		return None, 0.5

	if not opponent_strength or opp not in opponent_strength:
		return opp, 0.5

	score = float(opponent_strength[opp])
	# Assume scores in [1,5], map linearly to [0,1]
	favor = (score - 1.0) / 4.0
	return opp, _normalize_confidence(favor)


def _compute_blended_projection(
	multi_stats: Dict,
	overall_stats: Dict,
) -> Tuple[float, float, float, float, int, str]:
	"""Compute Empirical Bayes blended projection from recent and baseline data.

	Uses shrinkage to regress recent hot/cold streaks toward the longer-term
	baseline. This prevents overreacting to small-sample variance while still
	respecting genuine form changes.

	Returns:
		(blended_mean, baseline_mean, recent_mean, cv_percent, games_sampled, range_key)
	"""
	# Determine "recent" window (prefer Last 14, fall back to Last 7)
	recent = None
	range_key = "YTD"
	for key in ("Last 14", "Last 7"):
		if key in multi_stats and multi_stats[key].get("games_played", 0) >= 3:
			recent = multi_stats[key]
			range_key = key
			break

	# Determine "baseline" window (prefer Last 30, fall back to YTD)
	baseline = None
	for key in ("Last 30", "YTD"):
		if key in multi_stats and multi_stats[key].get("games_played", 0) >= 10:
			baseline = multi_stats[key]
			break

	# If no baseline from multi_stats, use overall_stats
	if baseline is None:
		baseline_mean = float(overall_stats.get("mean_fpts", 0))
	else:
		baseline_mean = float(baseline.get("mean_fpts", 0))

	if recent is None:
		# Not enough recent data, use baseline only
		return (
			baseline_mean,
			baseline_mean,
			baseline_mean,
			float(overall_stats.get("coefficient_of_variation", 40)),
			int(overall_stats.get("games_played", 0)),
			"YTD",
		)

	recent_n = int(recent.get("games_played", 0))
	recent_mean = float(recent.get("mean_fpts", 0))
	recent_cv = float(recent.get("coefficient_of_variation", 40))

	# Empirical Bayes shrinkage:
	# blended = (n * recent + k * baseline) / (n + k)
	# where k is the regression constant
	k = REGRESSION_CONSTANT_K
	blended_mean = (recent_n * recent_mean + k * baseline_mean) / (recent_n + k)

	# Use the higher of recent CV or floor to prevent overconfidence
	cv_percent = max(recent_cv, MIN_CV_FLOOR)

	return (blended_mean, baseline_mean, recent_mean, cv_percent, recent_n, range_key)


def _compute_risk_adjusted_utility(
	expected_fpts: float,
	cv_percent: float,
	situational_boost: float = 0.0,
) -> float:
	"""Compute risk-adjusted utility using mean-variance framework.

	For H2H fantasy, we penalize variance but don't obsess over it. This uses
	the formula: utility = E[X] - λ * σ(X) + boost

	This is fundamentally different from the old multiplicative approach
	(E[X] * confidence) which over-penalized high-variance players.

	Args:
		expected_fpts: Blended projection
		cv_percent: Coefficient of variation (std/mean * 100)
		situational_boost: Additional points from situational factors (injuries, etc.)

	Returns:
		Risk-adjusted utility score
	"""
	# Convert CV% to standard deviation
	std_dev = (cv_percent / 100.0) * expected_fpts

	# Risk-adjusted utility: E[X] - λ * σ(X) + situational boost
	utility = expected_fpts - (RISK_AVERSION_LAMBDA * std_dev) + situational_boost

	return max(utility, 0.0)


def _compute_situational_boost(
	player_name: str,
	teammates_out: Optional[Dict[str, bool]] = None,
	opponent_injuries: Optional[Dict[str, List[str]]] = None,
	player_position: Optional[str] = None,
	opponent: Optional[str] = None,
) -> Tuple[float, List[str]]:
	"""Compute situational boost from contextual factors.

	This is a placeholder for future enhancements. Currently returns 0 boost
	but the framework is in place for:

	1. Teammate injuries → usage/minutes boost
	   - If key teammates are out, player may see increased usage
	   - Example: "Half the team is injured, this guy gets more minutes"

	2. Opponent injuries → matchup boost
	   - If opponent's key defenders are out, player may have easier matchup
	   - Example: "Their centers are hurt, this center could pop off"

	3. Back-to-back detection → fatigue penalty (negative boost)

	4. Home/away splits → small adjustment

	Args:
		player_name: Player being evaluated
		teammates_out: Dict mapping player name -> bool for injured teammates
		opponent_injuries: Dict mapping team code -> list of injured player names
		player_position: Player's position (C, PG, etc.)
		opponent: Opponent team code

	Returns:
		(boost_amount, list of reason strings explaining the boost)
	"""
	boost = 0.0
	reasons: List[str] = []

	# Teammate injuries boost (future implementation)
	if teammates_out:
		injured_count = sum(1 for v in teammates_out.values() if v)
		if injured_count >= 2:
			# Significant teammate injuries → usage boost
			# This would need to be calibrated based on who is out
			# For now, just note it in reasons
			pass

	# Opponent injuries boost (future implementation)
	if opponent_injuries and opponent and opponent in opponent_injuries:
		injured_opponents = opponent_injuries[opponent]
		if injured_opponents:
			# Check if position-relevant players are out
			# For now, just note it in reasons
			pass

	return boost, reasons


def get_actual_fpts_for_date(
	player_name: str,
	game_date: date,
	league_id: str,
	season: Optional[str] = None,
	opponent: Optional[str] = None,
) -> Optional[float]:
	"""Return actual FPts from cached logs for a specific date/opponent, if available.

	This is used by the weekly planner to backfill completed games once the
	schedule is known. It looks for a row in the cached log whose `Date` matches
	`game_date` (month/day) and whose opponent matches the supplied code.
	"""
	player_code = get_player_code_by_name(league_id, player_name)
	if not player_code:
		return None

	df, meta = load_cached_player_log(player_code, league_id, season)
	if df is None or df.empty:
		return None

	if "Date" not in df.columns:
		return None

	month_abbr = game_date.strftime("%b")
	day_str = str(int(game_date.strftime("%d")))
	target_label = f"{month_abbr} {day_str}"
	date_series = df["Date"].astype(str).str.strip()
	mask = date_series == target_label

	if opponent and "Opp" in df.columns:
		opp_series = df["Opp"].astype(str).str.replace("@", "", regex=False).str.strip()
		target_opp = opponent.replace("@", "").strip()
		if target_opp:
			mask &= opp_series == target_opp

	matched = df[mask]
	if matched.empty and opponent:
		# Fallback: try matching on date only in case of subtle opponent code mismatches.
		mask = date_series == target_label
		matched = df[mask]
	if matched.empty:
		return None

	try:
		return float(matched.iloc[0].get("FPts"))
	except Exception:
		return None


def _compute_ppg_impact(
	expected_fpts: float,
	min_games_required: Optional[int],
	games_played_so_far: Optional[int],
	current_week_ppg: Optional[float],
) -> Tuple[
	Optional[int],
	Optional[int],
	Optional[float],
	Optional[float],
	Optional[float],
	Optional[str],
	Optional[float],
]:
	if min_games_required is None or games_played_so_far is None or current_week_ppg is None:
		return None, None, None, None, None, None, None
	if games_played_so_far < 0:
		return None, None, None, None, None, None, None
	games_before = games_played_so_far
	games_after = games_before + 1
	if games_before == 0 or current_week_ppg <= 0:
		return games_before, games_after, None, None, None, None, None
	ppg_before = float(current_week_ppg)
	total_before = ppg_before * games_before
	ppg_after = (total_before + float(expected_fpts)) / games_after
	ppg_delta = ppg_after - ppg_before
	if games_before < min_games_required:
		ppg_tag = "NEED_GAMES"
	else:
		if ppg_delta > 0:
			ppg_tag = "PPG_UP"
		elif ppg_delta < 0:
			ppg_tag = "PPG_DOWN"
		else:
			ppg_tag = "PPG_NEUTRAL"
	ppg_impact_score = ppg_delta
	return games_before, games_after, ppg_before, ppg_after, ppg_delta, ppg_tag, ppg_impact_score


def _classify_decision(expected_fpts: float, injured: bool = False) -> str:
	"""Classify into MUST_PLAY / STRONG_START / BORDERLINE / RISKY / MUST_SIT.

	Uses the blended projection to determine decision tier. Thresholds are
	calibrated for typical fantasy basketball scoring where:
	- 60+ FPts = elite performance
	- 50-60 FPts = strong performance
	- 40-50 FPts = average/borderline
	- 35-40 FPts = below average, risky
	- <35 FPts = likely not worth starting
	"""
	if injured:
		return "INJURED"
	if expected_fpts >= 60.0:
		return "MUST_PLAY"
	if expected_fpts >= 50.0:
		return "STRONG_START"
	if expected_fpts >= 40.0:
		return "BORDERLINE"
	if expected_fpts >= 35.0:
		return "RISKY"
	return "MUST_SIT"


def compare_players_for_date(
	player_names: List[str],
	game_date: str,
	league_id: str,
	season: Optional[str] = None,
	opponent_by_player: Optional[Dict[str, str]] = None,
	opponent_strength: Optional[Dict[str, float]] = None,
	injured_flags: Optional[Dict[str, bool]] = None,
	min_games_required: Optional[int] = None,
	games_played_so_far: Optional[int] = None,
	current_week_ppg: Optional[float] = None,
) -> pd.DataFrame:
	"""Compare a small set of players for a specific date.

	This helper is designed for "tonight's decision" scenarios, e.g. choosing
	between CJ McCollum and Kyshawn George. It relies entirely on cached game
	logs via `load_cached_player_log` and does *not* perform any scraping.

	Args:
		player_names: List of display names to evaluate.
		game_date: Date string; currently informational but kept for future use
		    (e.g., integrating true schedule or back-to-back detection).
		league_id: Fantrax league id for resolving player codes.
		season: Optional season string (e.g. "2025-26"). If omitted, the latest
		    available season for each player will be used.
		opponent_by_player: Optional mapping from player name -> opponent code
		    for the given game date (e.g., {"CJ McCollum": "CLE"}).
		opponent_strength: Optional mapping from opponent team code -> numeric
		    strength score where higher = softer matchup (e.g., 1-5). Used to
		    derive matchup_favorability.
		injured_flags: Optional mapping from player name -> bool indicating
		    manual injury status for tonight.

	Returns:
		A pandas DataFrame sorted by descending `utility`, with one row per
		player containing:
		- expected_fpts
		- confidence
		- utility
		- decision (MUST_PLAY / MUST_SIT / BORDERLINE)
		- various diagnostic fields used in the explanation.
	"""
	# Parse date mainly for future extension; we don't use it in the current
	# calculations beyond including it in the output.
	try:
		parsed_date = datetime.fromisoformat(game_date).date()
	except Exception:
		parsed_date = None

	injured_flags = injured_flags or {}

	results: List[PlayerComparisonResult] = []

	for name in player_names:
		injured = bool(injured_flags.get(name, False))

		player_code = get_player_code_by_name(league_id, name)
		if not player_code:
			results.append(
				PlayerComparisonResult(
					player_name=name,
					player_code=None,
					league_id=league_id,
					season=season,
					status="missing_code",
					decision="MUST_SIT" if injured else "UNKNOWN",
					reason="Player code not found in league index.",
				)
			)
			continue

		df, meta = load_cached_player_log(player_code, league_id, season)
		if df is None or df.empty:
			results.append(
				PlayerComparisonResult(
					player_name=name,
					player_code=player_code,
					league_id=league_id,
					season=meta.get("season") if meta else season,
					status="no_log",
					decision="MUST_SIT" if injured else "UNKNOWN",
					reason="No cached game log available. Run Bulk Scrape in Admin Tools.",
				)
			)
			continue

		# Use multi-range stats to derive recent form and stability
		multi_stats = calculate_multi_range_stats(df)
		if not multi_stats:
			results.append(
				PlayerComparisonResult(
					player_name=name,
					player_code=player_code,
					league_id=league_id,
					season=meta.get("season"),
					status="no_stats",
					decision="MUST_SIT" if injured else "UNKNOWN",
					reason="Unable to compute recent form from game log.",
				)
			)
			continue

		# Season-level variability stats (boom/bust, overall mean, minutes, FPPM).
		overall_stats = calculate_variability_stats(df.copy()) or {}

		# ─────────────────────────────────────────────────────────────────────
		# NEW: Empirical Bayes blended projection
		# ─────────────────────────────────────────────────────────────────────
		(
			blended_projection,
			baseline_mean,
			recent_mean,
			cv_percent,
			games_sampled,
			range_key,
		) = _compute_blended_projection(multi_stats, overall_stats)

		# For display, keep base_mean as the recent window mean
		base_mean = recent_mean

		# The expected_fpts is now the blended projection
		expected_fpts = blended_projection

		# Compute standard deviation for risk adjustment
		std_dev = (cv_percent / 100.0) * expected_fpts

		form_stability = _compute_form_stability(cv_percent)
		matchup, matchup_favorability = _compute_matchup_favorability(
			name, opponent_by_player, opponent_strength
		)

		# Situational boost (placeholder for future teammate/opponent injury logic)
		situational_boost, boost_reasons = _compute_situational_boost(
			player_name=name,
			teammates_out=None,  # Future: pass from UI
			opponent_injuries=None,  # Future: pass from UI
			player_position=None,  # Future: get from player data
			opponent=matchup,
		)

		# Optional trend: Last 7 vs Last 30, if available
		trend_last7_vs_30 = None
		if "Last 7" in multi_stats and "Last 30" in multi_stats:
			trend_last7_vs_30 = float(
				multi_stats["Last 7"]["mean_fpts"] - multi_stats["Last 30"]["mean_fpts"]
			)

		# ─────────────────────────────────────────────────────────────────────
		# NEW: Risk-adjusted utility (replaces multiplicative confidence)
		# ─────────────────────────────────────────────────────────────────────
		# Confidence is now purely informational (stability + matchup)
		confidence = 0.5 * form_stability + 0.5 * matchup_favorability
		confidence = _normalize_confidence(confidence)

		# Utility uses mean-variance framework: E[X] - λ*σ(X) + boost
		utility = _compute_risk_adjusted_utility(
			expected_fpts=expected_fpts,
			cv_percent=cv_percent,
			situational_boost=situational_boost,
		)

		(
			games_before,
			games_after_if_started,
			ppg_before,
			ppg_after_if_started,
			ppg_delta,
			ppg_tag,
			ppg_impact_score,
		) = _compute_ppg_impact(
			expected_fpts,
			min_games_required,
			games_played_so_far,
			current_week_ppg,
		)

		decision = _classify_decision(expected_fpts, injured=injured)

		# ─────────────────────────────────────────────────────────────────────
		# Build reason string with new projection logic explanation
		# ─────────────────────────────────────────────────────────────────────
		reason_parts = []

		# Show the blending logic
		if abs(recent_mean - baseline_mean) > 2.0:
			# Significant difference between recent and baseline
			reason_parts.append(
				f"Recent {range_key}: {recent_mean:.1f} FPts ({games_sampled}g) → "
				f"Baseline: {baseline_mean:.1f} → Blended: {blended_projection:.1f}."
			)
		else:
			reason_parts.append(
				f"Projection: {blended_projection:.1f} FPts (from {range_key}, {games_sampled}g)."
			)

		# Stability info
		reason_parts.append(f"CV {cv_percent:.1f}% (σ={std_dev:.1f}) → stability {form_stability:.2f}.")

		# Season context
		if overall_stats:
			boom_rate = overall_stats.get("boom_rate")
			bust_rate = overall_stats.get("bust_rate")
			if boom_rate is not None and bust_rate is not None:
				try:
					reason_parts.append(
						f"Boom {float(boom_rate):.1f}% / bust {float(bust_rate):.1f}%."
					)
				except Exception:
					pass
			mean_minutes = overall_stats.get("mean_minutes")
			fppm_mean = overall_stats.get("fppm_mean")
			if mean_minutes is not None and fppm_mean is not None:
				try:
					reason_parts.append(
						f"~{float(mean_minutes):.1f} min, {float(fppm_mean):.2f} FPts/min."
					)
				except Exception:
					pass

		# Matchup
		if matchup:
			reason_parts.append(f"vs {matchup} (matchup: {matchup_favorability:.2f}).")
		if trend_last7_vs_30 is not None:
			if trend_last7_vs_30 > 0:
				reason_parts.append(f"Trending up (+{trend_last7_vs_30:.1f} vs 30-game avg).")
			elif trend_last7_vs_30 < 0:
				reason_parts.append(f"Trending down ({trend_last7_vs_30:.1f} vs 30-game avg).")
		if injured:
			reason_parts.append("Flagged as injured for tonight.")

		results.append(
			PlayerComparisonResult(
				player_name=name,
				player_code=player_code,
				league_id=league_id,
				season=meta.get("season"),
				status="ok",
				games_sampled=games_sampled,
				base_mean_fpts=base_mean,
				baseline_mean_fpts=baseline_mean,
				blended_projection=blended_projection,
				cv_percent=cv_percent,
				std_dev=std_dev,
				form_stability=form_stability,
				matchup=matchup,
				matchup_favorability=matchup_favorability,
				situational_boost=situational_boost,
				trend_last7_vs_30=trend_last7_vs_30,
				expected_fpts=expected_fpts,
				confidence=confidence,
				utility=utility,
				games_before=games_before,
				games_after_if_started=games_after_if_started,
				ppg_before=ppg_before,
				ppg_after_if_started=ppg_after_if_started,
				ppg_delta=ppg_delta,
				ppg_tag=ppg_tag,
				ppg_impact_score=ppg_impact_score,
				decision=decision,
				reason=" ".join(reason_parts),
			)
		)

	# Build DataFrame for easy UI consumption
	if not results:
		return pd.DataFrame()

	df_out = pd.DataFrame([r.__dict__ for r in results])
	# Keep most relevant columns first
	column_order = [
		"player_name",
		"decision",
		"expected_fpts",
		"confidence",
		"utility",
		"base_mean_fpts",
		"cv_percent",
		"games_sampled",
		"matchup",
		"matchup_favorability",
		"trend_last7_vs_30",
		"games_before",
		"games_after_if_started",
		"ppg_before",
		"ppg_after_if_started",
		"ppg_delta",
		"ppg_tag",
		"ppg_impact_score",
		"status",
		"reason",
	]
	df_out = df_out[[c for c in column_order if c in df_out.columns]]

	# Sort by decision priority then utility
	decision_rank = {
		"MUST_PLAY": 0,
		"STRONG_START": 1,
		"BORDERLINE": 2,
		"RISKY": 3,
		"MUST_SIT": 4,
		"INJURED": 5,
		"UNKNOWN": 6,
	}
	df_out["_decision_rank"] = df_out["decision"].map(decision_rank).fillna(6)
	df_out = df_out.sort_values(["_decision_rank", "utility"], ascending=[True, False]).reset_index(drop=True)
	return df_out.drop(columns=["_decision_rank"], errors="ignore")
