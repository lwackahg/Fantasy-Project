import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from modules.player_game_log_scraper.logic import (
	get_cache_directory,
	load_league_cache_index,
	calculate_variability_stats,
	calculate_multi_range_stats,
)


def _compute_season_profile(game_log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
	"""Compute per-season production and consistency metrics from a raw game_log list."""
	if not game_log:
		return None

	df = pd.DataFrame(game_log)
	if "FPts" not in df.columns:
		return None

	stats = calculate_variability_stats(df.copy())
	if not stats:
		return None

	multi_range = calculate_multi_range_stats(df.copy()) or {}
	last15 = multi_range.get("Last 15", {})

	profile = {
		"games_played": stats.get("games_played", 0),
		"mean_fpts": stats.get("mean_fpts", 0.0),
		"median_fpts": stats.get("median_fpts", 0.0),
		"std_dev": stats.get("std_dev", 0.0),
		"cv_pct": stats.get("coefficient_of_variation", 0.0),
		"boom_rate": stats.get("boom_rate", 0.0),
		"bust_rate": stats.get("bust_rate", 0.0),
		"min_fpts": stats.get("min_fpts", 0.0),
		"max_fpts": stats.get("max_fpts", 0.0),
		"last15_mean_fpts": last15.get("mean_fpts", None),
	}
	return profile


def _normalize_series_to_0_100(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
	"""Normalize a numeric series to 0-100. Handles constant or empty series safely."""
	if series.empty:
		return pd.Series(dtype="float64")

	valid = series.replace([pd.NA, pd.NaT], pd.NA).dropna()
	if valid.empty:
		return pd.Series([50.0] * len(series), index=series.index)

	min_val = valid.min()
	max_val = valid.max()
	if max_val == min_val:
		return pd.Series([50.0] * len(series), index=series.index)

	if higher_is_better:
		norm = (series - min_val) / (max_val - min_val)
	else:
		norm = (max_val - series) / (max_val - min_val)

	return (norm * 100).clip(0, 100)


def build_player_value_profiles(
	league_id: str,
	seasons: Optional[List[str]] = None,
	min_games_per_season: int = 20,
	min_seasons: int = 1,
	max_games_per_season: int = 82,
) -> pd.DataFrame:
	"""Build cross-season player value & reliability profiles for a league.

	This uses the league cache index plus per-player-season JSONs to compute
	per-season production/consistency metrics and aggregate them into a
	multi-season value profile per player.
	"""
	index = load_league_cache_index(league_id, rebuild_if_missing=True)
	if not index or not index.get("players"):
		return pd.DataFrame()

	cache_dir = Path(get_cache_directory())

	rows = []
	for player_code, pdata in index["players"].items():
		player_name = pdata.get("player_name", "Unknown")
		seasons_dict = pdata.get("seasons", {})

		# Filter seasons
		candidate_seasons = list(seasons_dict.keys())
		if seasons is not None:
			candidate_seasons = [s for s in candidate_seasons if s in seasons]
		candidate_seasons = sorted(candidate_seasons, reverse=True)

		season_profiles = []
		for season in candidate_seasons:
			meta = seasons_dict.get(season, {})
			status = meta.get("status", "success")
			games = int(meta.get("games", 0) or 0)

			if status == "no_games_played" or games < min_games_per_season:
				continue

			cache_file_name = meta.get("cache_file")
			if not cache_file_name:
				continue
			cache_path = cache_dir / cache_file_name
			if not cache_path.exists():
				continue

			try:
				with open(cache_path, "r") as f:
					cache_data = json.load(f)
			except Exception:
				continue

			game_log = cache_data.get("data", cache_data.get("game_log", []))
			profile = _compute_season_profile(game_log)
			if not profile:
				continue

			profile["season"] = season
			season_profiles.append(profile)

		if len(season_profiles) < min_seasons:
			continue

		spf = pd.DataFrame(season_profiles)

		# Aggregate across seasons with simple averages
		total_games = int(spf["games_played"].sum())
		avg_mean_fpts = float(spf["mean_fpts"].mean())
		avg_cv = float(spf["cv_pct"].mean())
		avg_boom = float(spf["boom_rate"].mean())
		avg_bust = float(spf["bust_rate"].mean())
		avg_games_per_season = float(spf["games_played"].mean())
		seasons_included = len(spf)

		# Availability proxy: games played vs max_games_per_season
		availability_ratio = min(1.0, avg_games_per_season / max_games_per_season) if max_games_per_season > 0 else 0.0

		rows.append(
			{
				"Player": player_name,
				"player_code": player_code,
				"SeasonsIncluded": seasons_included,
				"TotalGames": total_games,
				"AvgGamesPerSeason": avg_games_per_season,
				"AvgMeanFPts": avg_mean_fpts,
				"AvgCV%": avg_cv,
				"AvgBoomRate%": avg_boom,
				"AvgBustRate%": avg_bust,
				"AvailabilityRatio": availability_ratio,
			}
		)

	if not rows:
		return pd.DataFrame()

	df = pd.DataFrame(rows)

	# Derive normalized scores
	df["ProductionScore"] = _normalize_series_to_0_100(df["AvgMeanFPts"], higher_is_better=True)
	df["ConsistencyScore"] = _normalize_series_to_0_100(df["AvgCV%"], higher_is_better=False)
	df["AvailabilityScore"] = (df["AvailabilityRatio"] * 100).clip(0, 100)

	# For now, use AvailabilityScore as a proxy for playoff reliability as well
	df["PlayoffReliabilityScore"] = df["AvailabilityScore"]

	# Composite value score
	df["ValueScore"] = (
		0.40 * df["ProductionScore"]
		+ 0.25 * df["ConsistencyScore"]
		+ 0.20 * df["AvailabilityScore"]
		+ 0.15 * df["PlayoffReliabilityScore"]
	)

	# Durability tiers based on AvailabilityRatio
	def _durability_tier(ratio: float) -> str:
		if ratio >= 0.9:
			return "Ironman"
		if ratio >= 0.75:
			return "Reliable"
		if ratio >= 0.6:
			return "Fragile"
		return "Landmine"

	df["DurabilityTier"] = df["AvailabilityRatio"].apply(_durability_tier)

	# Sort by ValueScore descending
	df = df.sort_values("ValueScore", ascending=False).reset_index(drop=True)

	return df
