from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
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
    """Container for a single player's comparison metrics for a given date."""

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
    returning_override: bool = False
    heater: bool = False
    returning_note: Optional[str] = None
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Projection & Utility Constants
# ─────────────────────────────────────────────────────────────────────────────
REGRESSION_CONSTANT_K = 10
RISK_AVERSION_LAMBDA = 0.4
MIN_CV_FLOOR = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# NBA Team Defense Data
# ─────────────────────────────────────────────────────────────────────────────
_NBA_DEFENSE_CACHE: Optional[Dict[str, Dict]] = None
NBA_DEFENSE_PA_G_MIN = 106.0
NBA_DEFENSE_PA_G_MAX = 127.0

TEAM_NAME_TO_CODE = {
    "Detroit Pistons": "DET",
    "New York Knicks": "NYK",
    "Boston Celtics": "BOS",
    "Philadelphia 76ers": "PHI",
    "Toronto Raptors": "TOR",
    "Orlando Magic": "ORL",
    "Cleveland Cavaliers": "CLE",
    "Miami Heat": "MIA",
    "Atlanta Hawks": "ATL",
    "Chicago Bulls": "CHI",
    "Milwaukee Bucks": "MIL",
    "Charlotte Hornets": "CHA",
    "Brooklyn Nets": "BKN",
    "Indiana Pacers": "IND",
    "Washington Wizards": "WAS",
    "Oklahoma City Thunder": "OKC",
    "San Antonio Spurs": "SAS",
    "Denver Nuggets": "DEN",
    "Los Angeles Lakers": "LAL",
    "Minnesota Timberwolves": "MIN",
    "Houston Rockets": "HOU",
    "Phoenix Suns": "PHX",
    "Golden State Warriors": "GSW",
    "Memphis Grizzlies": "MEM",
    "Portland Trail Blazers": "POR",
    "Dallas Mavericks": "DAL",
    "Utah Jazz": "UTA",
    "New Orleans Pelicans": "NOP",
    "Los Angeles Clippers": "LAC",
    "Sacramento Kings": "SAC",
}


def _parse_team_name(raw_name: str) -> Optional[str]:
    """Extract team code from raw name like 'Detroit Pistons (1) '."""
    if not raw_name or not isinstance(raw_name, str):
        return None
    clean_name = re.sub(r"\s*\(\d+\)\s*", "", raw_name).strip()
    return TEAM_NAME_TO_CODE.get(clean_name)


def _load_nba_defense_data() -> Dict[str, Dict]:
    """Load NBA team defense data from CSV. Cached after first load."""
    global _NBA_DEFENSE_CACHE

    if _NBA_DEFENSE_CACHE is not None:
        return _NBA_DEFENSE_CACHE

    csv_path = Path(__file__).parent.parent.parent / "data" / "nba_2025_standings.csv"

    if not csv_path.exists():
        _NBA_DEFENSE_CACHE = {}
        return _NBA_DEFENSE_CACHE

    try:
        df = pd.read_csv(csv_path)
        _NBA_DEFENSE_CACHE = {}

        first_col = df.columns[0]

        for _, row in df.iterrows():
            raw_name = row.get(first_col, "")
            team_code = _parse_team_name(raw_name)

            if not team_code:
                continue

            try:
                pa_g = float(row.get("PA/G", 116.0))
                ps_g = float(row.get("PS/G", 115.0))
                srs = float(row.get("SRS", 0.0))
            except (ValueError, TypeError):
                continue

            _NBA_DEFENSE_CACHE[team_code] = {
                "team_name": raw_name.strip(),
                "pa_g": pa_g,
                "ps_g": ps_g,
                "srs": srs,
            }

        aliases = {
            "SA": "SAS",
            "PHO": "PHX",
            "GS": "GSW",
            "NO": "NOP",
            "NY": "NYK",
            "BRK": "BKN",
        }
        for alias, canonical in aliases.items():
            if canonical in _NBA_DEFENSE_CACHE and alias not in _NBA_DEFENSE_CACHE:
                _NBA_DEFENSE_CACHE[alias] = _NBA_DEFENSE_CACHE[canonical]

    except Exception:
        _NBA_DEFENSE_CACHE = {}

    return _NBA_DEFENSE_CACHE


def get_team_defense_rating(team_code: str) -> Optional[Dict]:
    """Get defense stats for a team. Returns None if not found."""
    if not team_code:
        return None
    code = team_code.upper().replace("@", "").strip()
    data = _load_nba_defense_data()
    return data.get(code)


def get_matchup_favorability_from_defense(opponent: str) -> Tuple[float, bool]:
    """
    Return (favorability, found) based on opponent's PA/G.
    Higher PA/G = softer defense = better fantasy matchup.
    """
    if not opponent:
        return 0.5, False

    team_data = get_team_defense_rating(opponent)
    if not team_data:
        return 0.5, False

    pa_g = team_data.get("pa_g", 116.0)
    favorability = (pa_g - NBA_DEFENSE_PA_G_MIN) / (NBA_DEFENSE_PA_G_MAX - NBA_DEFENSE_PA_G_MIN)
    return max(0.0, min(1.0, favorability)), True


def reload_nba_defense_data() -> Dict[str, Dict]:
    """Force reload of NBA defense data from CSV."""
    global _NBA_DEFENSE_CACHE
    _NBA_DEFENSE_CACHE = None
    return _load_nba_defense_data()


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_confidence(value: float) -> float:
    """Clamp a confidence-like score to [0, 1]."""
    if pd.isna(value):
        return 0.5
    return float(max(0.0, min(1.0, value)))


def _compute_form_stability(cv_percent: float) -> float:
    """Convert CV% into a [0,1] stability score (lower CV = more stable)."""
    if pd.isna(cv_percent):
        return 0.5
    scaled = 1.0 - (cv_percent - 20.0) / (80.0 - 20.0)
    return _normalize_confidence(scaled)


def _compute_matchup_favorability(
    player_name: str,
    opponent_by_player: Optional[Dict[str, str]],
    opponent_strength: Optional[Dict[str, float]],
) -> Tuple[Optional[str], float]:
    """Return (opponent, favorability[0-1]) for a player.

    Uses real NBA defense data (PA/G) from CSV when available.
    Falls back to manual opponent_strength if provided.
    Default 0.5 if no data.
    """
    if not opponent_by_player:
        return None, 0.5

    opp = opponent_by_player.get(player_name)
    if not opp:
        return None, 0.5

    real_favorability, found = get_matchup_favorability_from_defense(opp)
    if found:
        return opp, real_favorability

    if opponent_strength and opp in opponent_strength:
        score = float(opponent_strength[opp])
        favor = (score - 1.0) / 4.0
        return opp, _normalize_confidence(favor)

    return opp, 0.5


def _compute_blended_projection(
	multi_stats: Dict,
	overall_stats: Dict,
	returning_override: bool = False,
	min_minutes_returner: float = 20.0,
	cv_outlier_threshold: float = 45.0,
	outlier_median_weight: float = 0.7,
) -> Tuple[float, float, float, float, int, str, bool]:
	"""Compute Empirical Bayes blended projection from recent and baseline data.

	Uses shrinkage to regress recent hot/cold streaks toward the longer-term
	baseline. This prevents overreacting to small-sample variance while still
	respecting genuine form changes.

	When recent CV is high (volatile window), blends mean toward median to
	reduce outlier impact.

	Returns:
		(blended_mean, baseline_mean, recent_mean, cv_percent, games_sampled, range_key, outlier_adjusted)
	"""
	# Determine "recent" window (prefer Last 14, fall back to Last 7)
	recent = None
	range_key = "YTD"
	for key in ("Last 14", "Last 7"):
		if key in multi_stats and multi_stats[key].get("games_played", 0) >= 3:
			# For returners: skip recent windows with suppressed minutes (e.g., ramp-up games)
			if returning_override:
				try:
					mean_mins_candidate = float(multi_stats[key].get("mean_minutes", 0))
				except Exception:
					mean_mins_candidate = 0.0
				if mean_mins_candidate < min_minutes_returner:
					continue
			recent = multi_stats[key]
			range_key = key
			break

	# If returning and we skipped too many low-minute recent windows, fall back to baseline directly
	if returning_override and recent is None and "Last 30" in multi_stats:
		recent = multi_stats.get("Last 30")
		range_key = "Last 30"

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
			False,
		)

	recent_n = int(recent.get("games_played", 0))
	recent_mean_raw = float(recent.get("mean_fpts", 0))
	recent_cv = float(recent.get("coefficient_of_variation", 40))

	# Outlier handling: blend mean toward median when CV is high
	outlier_adjusted = False
	recent_mean = recent_mean_raw
	recent_median = recent.get("median_fpts")

	if recent_median is not None and recent_cv > cv_outlier_threshold:
		# High variance - blend toward median to reduce outlier impact
		recent_mean = (outlier_median_weight * float(recent_median)) + ((1 - outlier_median_weight) * recent_mean_raw)
		outlier_adjusted = True

	# Empirical Bayes shrinkage:
	# blended = (n * recent + k * baseline) / (n + k)
	# where k is the regression constant
	k = REGRESSION_CONSTANT_K
	blended_mean = (recent_n * recent_mean + k * baseline_mean) / (recent_n + k)

	# Use the higher of recent CV or floor to prevent overconfidence
	cv_percent = max(recent_cv, MIN_CV_FLOOR)

	return (blended_mean, baseline_mean, recent_mean, cv_percent, recent_n, range_key, outlier_adjusted)


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


def _detect_heater(
	multi_stats: Dict,
	overall_stats: Dict,
	manual_flag: bool,
	returning_override: bool,
	min_minutes_floor: float = 26.0,
	min_minutes_bump: float = 4.0,
	fpts_pct_bump: float = 0.12,
	fpts_abs_bump: float = 5.0,
) -> tuple[bool, Dict]:
	"""Detect a heater (rising role) based on recent vs season minutes/FPts.

	Heaters are skipped if returner logic is active (no stacking). Manual flag
	can force heater.
	"""
	if returning_override:
		return False, {}

	season_mean_fpts = overall_stats.get("mean_fpts")
	season_mean_minutes = overall_stats.get("mean_minutes")
	if season_mean_fpts is None or season_mean_minutes is None:
		return False, {}

	# Prefer Last 7; fall back to Last 14
	recent_block = None
	recent_key = None
	for key in ("Last 7", "Last 14"):
		if key in multi_stats and multi_stats[key].get("games_played", 0) >= 3:
			recent_block = multi_stats[key]
			recent_key = key
			break

	if recent_block is None:
		return False, {}

	try:
		recent_mean = float(recent_block.get("mean_fpts", None))
	except Exception:
		recent_mean = None
	try:
		recent_minutes = float(recent_block.get("mean_minutes", None))
	except Exception:
		recent_minutes = None
	recent_games = int(recent_block.get("games_played", 0)) if recent_block else 0

	if recent_mean is None or recent_minutes is None:
		return False, {}

	# Manual override
	if manual_flag:
		return True, {
			"recent_mean": recent_mean,
			"recent_minutes": recent_minutes,
			"recent_key": recent_key,
			"recent_games": recent_games,
			"minutes_bump": recent_minutes - season_mean_minutes,
			"fpts_bump_pct": (recent_mean - season_mean_fpts) / season_mean_fpts if season_mean_fpts > 0 else 0,
			"confidence_in_role": 0.7,
		}

	minute_thresh = max(float(season_mean_minutes) + min_minutes_bump, min_minutes_floor)
	fpts_thresh = max(float(season_mean_fpts) * (1.0 + fpts_pct_bump), float(season_mean_fpts) + fpts_abs_bump)

	cond_minutes = recent_minutes >= minute_thresh
	cond_fpts = recent_mean >= fpts_thresh
	if not (cond_minutes and cond_fpts):
		return False, {}

	minutes_bump = recent_minutes - season_mean_minutes
	fpts_bump_pct = (recent_mean - season_mean_fpts) / season_mean_fpts if season_mean_fpts > 0 else 0
	role_change_magnitude = min(1.0, (minutes_bump / 10.0) + (fpts_bump_pct / 0.5))
	sample_confidence = min(1.0, recent_games / 5.0)
	confidence_in_role = 0.5 * role_change_magnitude + 0.5 * sample_confidence

	return True, {
		"recent_mean": recent_mean,
		"recent_minutes": recent_minutes,
		"recent_key": recent_key,
		"recent_games": recent_games,
		"minutes_bump": minutes_bump,
		"fpts_bump_pct": fpts_bump_pct,
		"confidence_in_role": confidence_in_role,
	}


def _compute_heater_projection(
	heater_data: Dict,
	blended_projection: float,
	baseline_mean: float,
	overall_stats: Dict,
) -> Tuple[float, str]:
	"""Compute a heater projection that trusts the minutes (role) while regressing efficiency."""
	recent_mean = heater_data["recent_mean"]
	recent_minutes = heater_data["recent_minutes"]
	recent_games = heater_data.get("recent_games", 0)
	minutes_bump = heater_data.get("minutes_bump", 0.0)
	fpts_bump_pct = heater_data.get("fpts_bump_pct", 0.0)
	confidence_in_role = heater_data.get("confidence_in_role", 0.7)

	# Baseline efficiency and minutes
	season_fppm = float(overall_stats.get("fppm_mean", 1.0))
	season_minutes = float(overall_stats.get("mean_minutes", 25.0))

	# Recent efficiency
	recent_fppm = recent_mean / recent_minutes if recent_minutes > 0 else season_fppm

	# Trust minutes fully for role change
	projected_minutes = recent_minutes

	# Regress efficiency: weight recent 55-70% based on confidence
	efficiency_weight = 0.55 + (0.15 * confidence_in_role)
	blended_fppm = (efficiency_weight * recent_fppm) + ((1 - efficiency_weight) * season_fppm)

	heater_projection = projected_minutes * blended_fppm

	# Floor: at least 10% above baseline; Ceiling: modestly above recent mean
	floor = baseline_mean * 1.10
	heater_projection = max(heater_projection, floor)
	heater_projection = min(heater_projection, recent_mean * 1.05)

	reason = (
		f"Heater: {recent_games}g at {recent_mean:.1f} FPts, {recent_minutes:.1f} mpg "
		f"(+{minutes_bump:.1f} min, +{fpts_bump_pct*100:.0f}% vs season). "
		f"Proj: {heater_projection:.1f} ({projected_minutes:.0f} min × {blended_fppm:.2f} FPPM; "
		f"eff wt {efficiency_weight*100:.0f}% recent, base fppm {season_fppm:.2f}, base mpg {season_minutes:.1f})."
	)
	return heater_projection, reason


def _compute_vs_team_delta(df: pd.DataFrame, opponent: Optional[str], baseline_mean: float) -> Optional[float]:
	"""Compute delta vs a specific opponent from recent history (last ~8 games)."""
	if opponent is None or "Opp" not in df.columns or baseline_mean <= 0:
		return None
	opp_series = df["Opp"].astype(str).str.replace("@", "", regex=False).str.strip()
	mask = opp_series == opponent.replace("@", "").strip()
	if not mask.any():
		return None
	df_vs = df[mask].head(8)  # recent history vs this opponent
	if df_vs.empty or len(df_vs) < 2:
		return None
	fpts = pd.to_numeric(df_vs.get("FPts"), errors="coerce").dropna()
	if fpts.empty or len(fpts) < 2:
		return None
	mean_vs = float(fpts.mean())
	std_vs = float(fpts.std())
	cv_vs = (std_vs / mean_vs * 100) if mean_vs > 0 else 0.0
	# Discount very high variance samples
	if cv_vs > 80:
		mean_vs = baseline_mean + (mean_vs - baseline_mean) * 0.5
	return mean_vs - baseline_mean


def _compute_situational_boost(
	matchup_favorability: float,
	opponent: Optional[str],
	player_log_df: Optional[pd.DataFrame],
	baseline_mean: float,
) -> Tuple[float, List[str]]:
	"""Compute situational boost from matchup favorability and recent history vs opponent."""
	boost = 0.0
	reasons: List[str] = []

	# Matchup favorability: small additive nudge, about -1.5 to +1.5 FPts, capped ±2.
	if matchup_favorability is not None:
		matchup_nudge = (float(matchup_favorability) - 0.5) * 6.0
		matchup_nudge = max(-2.0, min(2.0, matchup_nudge))
		boost += matchup_nudge
		if abs(matchup_nudge) > 0.05:
			reasons.append(f"Matchup nudge {matchup_nudge:+.1f} FPts.")

	# Past games vs opponent: gentle, capped ±2, discounted for high variance.
	vs_delta = None
	if player_log_df is not None:
		vs_delta = _compute_vs_team_delta(player_log_df, opponent, baseline_mean)
		if vs_delta is not None:
			vs_nudge = max(-2.0, min(2.0, vs_delta * 0.4))
			boost += vs_nudge
			if abs(vs_nudge) > 0.05:
				reasons.append(f"History vs {opponent}: {vs_nudge:+.1f} FPts.")

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
	returning_flags: Optional[Dict[str, bool]] = None,
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
	returning_flags = returning_flags or {}

	results: List[PlayerComparisonResult] = []

	for name in player_names:
		injured = bool(injured_flags.get(name, False))
		returning_override = bool(returning_flags.get(name, False))
		# Auto heater detection only; no manual flag required
		manual_heater = False

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
		mean_minutes = overall_stats.get("mean_minutes")

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
			outlier_adjusted,
		) = _compute_blended_projection(
			multi_stats,
			overall_stats,
			returning_override=returning_override,
		)

		# Minutes context for returner guardrails
		recent_minutes = None
		try:
			if range_key in multi_stats:
				recent_minutes = float(multi_stats[range_key].get("mean_minutes", None))
		except Exception:
			recent_minutes = None

		# For display, keep base_mean as the recent window mean
		base_mean = recent_mean

		# The expected_fpts is now the blended projection
		expected_fpts = blended_projection

		# If explicitly returning/cleared from injury, use baseline as a floor
		# and trim volatility penalty so recent suppressed minutes don't drag.
		if returning_override and baseline_mean > 0:
			# Use full baseline as floor (not regressed) to avoid suppressed recent mins.
			expected_fpts = max(expected_fpts, baseline_mean * 1.0)
			# Trim volatility more aggressively for cleared returners.
			cv_percent = max(MIN_CV_FLOOR * 0.6, cv_percent * 0.55)

		# Compute standard deviation for risk adjustment
		std_dev = (cv_percent / 100.0) * expected_fpts

		form_stability = _compute_form_stability(cv_percent)
		matchup, matchup_favorability = _compute_matchup_favorability(
			name, opponent_by_player, opponent_strength
		)

		# Heater detection (skip if returner to avoid stacking)
		heater_flag, heater_data = _detect_heater(
			multi_stats=multi_stats,
			overall_stats=overall_stats,
			manual_flag=manual_heater,
			returning_override=returning_override,
		)

		# Apply heater projection if detected (trust recent heavily)
		heater_reason = ""
		if heater_flag and heater_data:
			expected_fpts, heater_reason = _compute_heater_projection(
				heater_data=heater_data,
				blended_projection=blended_projection,
				baseline_mean=baseline_mean,
				overall_stats=overall_stats,
			)
			# Recompute std_dev with same cv_percent
			std_dev = (cv_percent / 100.0) * expected_fpts

		# Situational boost: matchup favorability and past games vs opponent
		situational_boost, boost_reasons = _compute_situational_boost(
			matchup_favorability=matchup_favorability,
			opponent=matchup,
			player_log_df=df,
			baseline_mean=baseline_mean,
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

		# Promotion logic for cleared returners: ensure at least STRONG_START if
		# baseline season form supports it; push to MUST_PLAY for elite baselines.
		# Guardrail: if season minutes are low (<26), cap at STRONG_START to avoid
		# over-promoting players without starter minutes.
		minute_guardrail = mean_minutes is not None and mean_minutes < 26
		if minute_guardrail and recent_minutes is not None and recent_minutes >= 27:
			# If recent minutes show full workloads despite low season mpg, relax guardrail.
			minute_guardrail = False
		if returning_override and not injured:
			if decision in {"BORDERLINE", "RISKY", "MUST_SIT"} and baseline_mean >= 45:
				decision = "STRONG_START"
			if baseline_mean >= 55 and decision != "MUST_PLAY" and not minute_guardrail:
				decision = "MUST_PLAY"

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

		if outlier_adjusted:
			reason_parts.append("High variance (CV>45%); recent blended toward median.")

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
		if boost_reasons:
			reason_parts.extend(boost_reasons)
		if injured:
			reason_parts.append("Flagged as injured for tonight.")
		elif returning_override:
			if minute_guardrail:
				reason_parts.append(
					"Return-to-form override applied with minutes guardrail (<26 mpg): baseline floor + trimmed volatility; capped at STRONG_START."
				)
			else:
				reason_parts.append(
					"Return-to-form override applied: baseline used as floor, volatility penalty trimmed."
				)
		elif heater_flag:
			if heater_reason:
				reason_parts.append(heater_reason)
			else:
				reason_parts.append(
					"Heater boost: minutes and FPts trending up (role change). Projection leans on recent form."
				)

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
				returning_override=returning_override,
				heater=heater_flag,
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
		"returning_override",
		"heater",
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
