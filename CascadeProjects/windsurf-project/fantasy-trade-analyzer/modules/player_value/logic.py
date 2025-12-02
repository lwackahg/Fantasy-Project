import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from modules.player_game_log_scraper.logic import (
	calculate_variability_stats,
	calculate_multi_range_stats,
)
from modules.player_game_log_scraper import db_store


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
	last14 = multi_range.get("Last 14", {})

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
		"last14_mean_fpts": last14.get("mean_fpts", None),
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


def _aggregate_player_season_profiles(
	player_name: str,
	player_code: str,
	season_profiles: List[Dict[str, Any]],
	min_seasons: int,
	max_games_per_season: int,
) -> Optional[Dict[str, Any]]:
	"""Aggregate per-season profiles into a single cross-season row for one player."""
	if len(season_profiles) < min_seasons:
		return None

	spf = pd.DataFrame(season_profiles)
	if spf.empty:
		return None

	# Aggregate across seasons with simple averages
	total_games = int(spf["games_played"].sum())
	avg_mean_fpts = float(spf["mean_fpts"].mean())
	avg_cv = float(spf["cv_pct"].mean())
	avg_boom = float(spf["boom_rate"].mean())
	avg_bust = float(spf["bust_rate"].mean())
	avg_games_per_season = float(spf["games_played"].mean())
	seasons_included = len(spf)

	# Availability proxy: games played vs max_games_per_season
	availability_ratio = (
		min(1.0, avg_games_per_season / max_games_per_season)
		if max_games_per_season > 0
		else 0.0
	)

	return {
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


def _build_player_value_profiles_from_db(
	league_id: str,
	seasons: Optional[List[str]],
	min_games_per_season: int,
	min_seasons: int,
	max_games_per_season: int,
) -> List[Dict[str, Any]]:
	"""Build player value profile rows using the SQLite DB as primary source.

	Returns a list of row dicts compatible with the final DataFrame schema.
	"""
	try:
		meta = db_store.get_league_player_seasons(league_id)
	except Exception:
		meta = []

	if not meta:
		return []

	# Group seasons by player_code
	players: Dict[str, Dict[str, Any]] = {}
	for m in meta:
		season = m.get("season")
		if not season:
			continue
		if seasons is not None and season not in seasons:
			continue

		player_code = m.get("player_code")
		player_name = m.get("player_name", "Unknown")
		if not player_code:
			continue

		pentry = players.setdefault(
			player_code,
			{"player_name": player_name, "seasons": []},
		)
		pentry["player_name"] = player_name
		pentry["seasons"].append(m)

	rows: List[Dict[str, Any]] = []
	for player_code, pdata in players.items():
		player_name = pdata.get("player_name", "Unknown")
		season_metas = pdata.get("seasons", [])
		if not season_metas:
			continue

		# Sort seasons newest-first for consistency (not strictly required for stats)
		season_metas = sorted(
			season_metas,
			key=lambda m: m.get("season", ""),
			reverse=True,
		)

		season_profiles: List[Dict[str, Any]] = []
		player_name_db_latest: Optional[str] = None
		for meta_row in season_metas:
			season = meta_row.get("season")
			status = meta_row.get("status", "success")
			games = int(meta_row.get("games", 0) or 0)

			if status == "no_games_played" or games < min_games_per_season:
				continue

			try:
				loaded = db_store.load_player_season(
					player_code=player_code,
					league_id=league_id,
					season=season,
				)
			except Exception:
				loaded = None
			if not loaded:
				continue

			records, status_db, player_name_db = loaded
			if player_name_db:
				player_name_db_latest = player_name_db
			if status_db == "no_games_played" or not records:
				continue

			profile = _compute_season_profile(records)
			if not profile:
				continue

			profile["season"] = season
			season_profiles.append(profile)

		row = _aggregate_player_season_profiles(
			player_name=player_name_db_latest or player_name,
			player_code=player_code,
			season_profiles=season_profiles,
			min_seasons=min_seasons,
			max_games_per_season=max_games_per_season,
		)
		if row is not None:
			rows.append(row)

	return rows


def _build_player_value_profiles_from_cache(
	league_id: str,
	seasons: Optional[List[str]],
	min_games_per_season: int,
	min_seasons: int,
	max_games_per_season: int,
) -> List[Dict[str, Any]]:
	"""Fallback implementation using JSON league index + per-player JSON files."""
	from modules.player_game_log_scraper.logic import (  # local import to avoid cycles at startup
		get_cache_directory,
		load_league_cache_index,
	)

	index = load_league_cache_index(league_id, rebuild_if_missing=True)
	if not index or not index.get("players"):
		return []

	cache_dir = Path(get_cache_directory())

	rows: List[Dict[str, Any]] = []
	for player_code, pdata in index["players"].items():
		player_name = pdata.get("player_name", "Unknown")
		seasons_dict = pdata.get("seasons", {})

		# Filter seasons
		candidate_seasons = list(seasons_dict.keys())
		if seasons is not None:
			candidate_seasons = [s for s in candidate_seasons if s in seasons]
		candidate_seasons = sorted(candidate_seasons, reverse=True)

		season_profiles: List[Dict[str, Any]] = []
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

		row = _aggregate_player_season_profiles(
			player_name=player_name,
			player_code=player_code,
			season_profiles=season_profiles,
			min_seasons=min_seasons,
			max_games_per_season=max_games_per_season,
		)
		if row is not None:
			rows.append(row)

	return rows


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
	# Primary path: use DB-backed player_seasons + game logs.
	rows = _build_player_value_profiles_from_db(
		league_id=league_id,
		seasons=seasons,
		min_games_per_season=min_games_per_season,
		min_seasons=min_seasons,
		max_games_per_season=max_games_per_season,
	)

	# Fallback path: use JSON league index + per-player JSON cache if DB has no usable rows.
	if not rows:
		rows = _build_player_value_profiles_from_cache(
			league_id=league_id,
			seasons=seasons,
			min_games_per_season=min_games_per_season,
			min_seasons=min_seasons,
			max_games_per_season=max_games_per_season,
		)

	if not rows:
		return pd.DataFrame()

	df = pd.DataFrame(rows)

	# Derive normalized scores
	df["ProductionScore"] = _normalize_series_to_0_100(df["AvgMeanFPts"], higher_is_better=True)
	df["ConsistencyScore"] = _normalize_series_to_0_100(df["AvgCV%"], higher_is_better=False)
	ratio = df["AvailabilityRatio"].clip(0, 1)
	df["AvailabilityScore"] = (ratio.pow(1.5) * 100).clip(0, 100)

	# For now, use AvailabilityScore as a proxy for playoff reliability as well
	df["PlayoffReliabilityScore"] = df["AvailabilityScore"]

	# Composite value score
	df["ValueScore"] = (
		0.40 * df["ProductionScore"]
		+ 0.25 * df["ConsistencyScore"]
		+ 0.20 * df["AvailabilityScore"]
		+ 0.15 * df["PlayoffReliabilityScore"]
	)

	# Apply tier-based scarcity premium
	def _assign_tier(fpg: float) -> str:
		if fpg >= 110:
			return "Elite (110+)"
		elif fpg >= 95:
			return "Star (95-110)"
		elif fpg >= 80:
			return "High-End (80-95)"
		elif fpg >= 65:
			return "Solid (65-80)"
		elif fpg >= 50:
			return "Rotation (50-65)"
		else:
			return "Depth (< 50)"
	
	df["ProductionTier"] = df["AvgMeanFPts"].apply(_assign_tier)
	tier_counts = df["ProductionTier"].value_counts().to_dict()
	
	def _scarcity_multiplier(tier: str, count: int) -> float:
		if count == 0:
			return 1.0
		if tier == "Elite (110+)":
			if count == 1:
				return 1.20
			elif count <= 3:
				return 1.15
			else:
				return 1.10
		elif tier == "Star (95-110)":
			if count <= 3:
				return 1.10
			elif count <= 7:
				return 1.05
			else:
				return 1.02
		elif tier == "High-End (80-95)":
			if count <= 5:
				return 1.05
			elif count <= 12:
				return 1.02
			else:
				return 1.0
		else:
			return 1.0
	
	df["ScarcityMultiplier"] = df.apply(
		lambda row: _scarcity_multiplier(row["ProductionTier"], tier_counts.get(row["ProductionTier"], 0)),
		axis=1
	)
	df["ValueScore"] = df["ValueScore"] * df["ScarcityMultiplier"]

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
