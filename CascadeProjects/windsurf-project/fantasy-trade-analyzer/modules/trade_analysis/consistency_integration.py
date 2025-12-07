"""Integration module for player consistency data in trade analysis."""
import json
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache
import pandas as pd
from modules.player_game_log_scraper.logic import calculate_variability_stats, get_player_code_by_name, load_cached_player_log, get_cache_directory
from modules.player_game_log_scraper import db_store

CONSISTENCY_VERY_MAX_CV = 25.0
CONSISTENCY_MODERATE_MAX_CV = 40.0

# No longer need get_consistency_cache_directory here if we rely on logic.py


def get_consistency_cache_directory() -> Path:
	"""Return the directory where player game log cache files are stored."""
	return get_cache_directory()

@lru_cache(maxsize=2048)
def _load_raw_player_data(player_name: str, league_id: str, season: Optional[str] = None) -> Optional[Dict]:
	"""Load raw player data, finding the best season match using logic.py helpers."""
	player_code = get_player_code_by_name(league_id, player_name)
	if not player_code:
		# If name resolution fails, we can't load by code.
		# Legacy fallback: glob by name if absolutely necessary, but condensing logic suggests we rely on the index.
		# If the index is stale, the scraper logic handles rebuilding it.
		return None
	
	df, meta = load_cached_player_log(player_code, league_id, season)
	if df is None or df.empty:
		return None
		
	# Pack into dictionary format expected by consumers
	return {
		'player_name': meta.get('player_name', player_name),
		'player_code': player_code,
		'league_id': league_id,
		'season': meta.get('season', ''),
		'timestamp': meta.get('timestamp'),
		'data': df.to_dict('records'),
		'game_log': df.to_dict('records') # support legacy key
	}

@lru_cache(maxsize=2048)
def _load_player_consistency_internal(player_name: str, league_id: str) -> Optional[Dict]:
	# Use get_player_game_log_df directly? No, we need metrics here.
	# But we can use _load_raw_player_data which now uses the centralized loader.
	data = _load_raw_player_data(player_name, league_id)
	if data is None:
		return None
	
	game_log = data.get('data', [])
	if not game_log:
		return None
		
	df = pd.DataFrame(game_log)
	# ... rest of calculation logic ...
	if 'FPts' not in df.columns or len(df) == 0:
		return None
	fpts = df['FPts']
	mean_fpts = fpts.mean()
	std_dev = fpts.std()
	cv = (std_dev / mean_fpts * 100) if mean_fpts > 0 else 0
	boom_threshold = mean_fpts + std_dev
	bust_threshold = mean_fpts - std_dev
	boom_games = len(fpts[fpts > boom_threshold])
	bust_games = len(fpts[fpts < bust_threshold])
	total_games = len(fpts)
	return {
		'player_name': player_name,
		'games_played': total_games,
		'mean_fpts': mean_fpts,
		'median_fpts': fpts.median(),
		'std_dev': std_dev,
		'cv_percent': cv,
		'min_fpts': fpts.min(),
		'max_fpts': fpts.max(),
		'range': fpts.max() - fpts.min(),
		'boom_games': boom_games,
		'boom_rate': (boom_games / total_games * 100) if total_games > 0 else 0,
		'bust_games': bust_games,
		'bust_rate': (bust_games / total_games * 100) if total_games > 0 else 0,
		'consistency_tier': get_consistency_tier(cv)
	}


def load_player_consistency(player_name: str, league_id: str) -> Optional[Dict]:
	"""Public wrapper around the cached consistency loader."""
	return _load_player_consistency_internal(player_name, league_id)

def get_player_game_log_df(player_name: str, league_id: str, season: Optional[str] = None) -> Optional[pd.DataFrame]:
	"""
	Get the raw game log DataFrame for a player from the cache.
	
	Args:
		player_name: Name of the player
		league_id: Fantrax league ID
		season: Optional specific season to retrieve
		
	Returns:
		DataFrame containing game log or None if not found
	"""
	data = _load_raw_player_data(player_name, league_id, season)
	if not data:
		return None
	
	game_log = data.get('data', data.get('game_log', []))
	if not game_log:
		return None
		
	return pd.DataFrame(game_log)

def get_consistency_tier(cv_percent: float) -> str:
	"""
	Get consistency tier based on CV%.
	
	Args:
		cv_percent: Coefficient of variation percentage
		
	Returns:
		Tier string with emoji
	"""
	if cv_percent < CONSISTENCY_VERY_MAX_CV:
		return "🟢 Very Consistent"
	elif cv_percent <= CONSISTENCY_MODERATE_MAX_CV:
		return "🟡 Solid / Moderate"
	else:
		return "🔴 Volatile / Boom-Bust"

def load_all_player_consistency(league_id: str) -> Dict[str, Dict]:
	"""
	Load consistency metrics for all cached players.
	
	Args:
		league_id: Fantrax league ID
		
	Returns:
		Dict mapping player names to their consistency metrics
	"""
	cache_dir = get_consistency_cache_directory()
	cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
	
	all_consistency = {}
	
	for cache_file in cache_files:
		try:
			with open(cache_file, 'r') as f:
				cache_data = json.load(f)
			
			player_name = cache_data.get('player_name', '')
			if player_name:
				metrics = load_player_consistency(player_name, league_id)
				if metrics:
					all_consistency[player_name] = metrics
		except Exception:
			continue
	
	return all_consistency

def build_league_consistency_index(league_id: str) -> Dict[str, Dict]:
	"""Build a league-wide consistency index, preferring DB-backed stats.

	Tries player_season_stats in the SQLite DB first and falls back to the
	JSON cache files only if no DB-backed stats are available.
	"""
	try:
		last_updated = db_store.get_league_last_updated(league_id)
	except Exception:
		last_updated = None
	return _build_league_consistency_index_cached(league_id, last_updated)


@lru_cache(maxsize=32)
def _build_league_consistency_index_cached(league_id: str, last_updated: Optional[str]) -> Dict[str, Dict]:
	all_consistency: Dict[str, Dict] = {}

	# First try to build from DB-backed season stats.
	try:
		rows = db_store.get_league_player_season_stats(league_id)
	except Exception:
		rows = []
	if rows:
		best_by_player: Dict[str, Dict] = {}
		for row in rows:
			player_name = row.get('player_name') or ''
			if not player_name:
				continue
			season_str = str(row.get('season', '')).strip()
			existing = best_by_player.get(player_name)
			if existing is None or season_str > str(existing.get('season', '')):
				best_by_player[player_name] = {
					'season': season_str,
					'stats': row,
				}
		for player_name, meta in best_by_player.items():
			stats_row = meta.get('stats') or {}
			mean_fpts = stats_row.get('mean_fpts')
			if mean_fpts is None:
				continue
			cv = stats_row.get('cv_percent') or 0.0
			all_consistency[player_name] = {
				'player_name': player_name,
				'games_played': stats_row.get('games_played') or 0,
				'mean_fpts': mean_fpts,
				'median_fpts': stats_row.get('median_fpts'),
				'std_dev': stats_row.get('std_dev'),
				'cv_percent': cv,
				'min_fpts': stats_row.get('min_fpts'),
				'max_fpts': stats_row.get('max_fpts'),
				'range': stats_row.get('range'),
				'boom_games': stats_row.get('boom_games'),
				'boom_rate': stats_row.get('boom_rate'),
				'bust_games': stats_row.get('bust_games'),
				'bust_rate': stats_row.get('bust_rate'),
				'consistency_tier': get_consistency_tier(cv),
				'mean_minutes': stats_row.get('mean_minutes'),
				'fppm_mean': stats_row.get('fppm_mean'),
			}
		if all_consistency:
			return all_consistency
	
	# Fallback: build from JSON cache if DB stats are unavailable.
	cache_dir = get_consistency_cache_directory()
	cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))

	best_by_player: Dict[str, Dict] = {}

	for cache_file in cache_files:
		try:
			with open(cache_file, 'r') as f:
				cache_data = json.load(f)
			player_name = cache_data.get('player_name', '')
			if not player_name:
				continue
			season_str = str(cache_data.get('season', '')).strip()
			existing = best_by_player.get(player_name)
			if existing is None or season_str > str(existing.get('season', '')):
				best_by_player[player_name] = {
					'season': season_str,
					'data': cache_data,
				}
		except Exception:
			continue

	for player_name, meta in best_by_player.items():
		cache_data = meta.get('data') or {}
		game_log = cache_data.get('data', cache_data.get('game_log', []))
		if not game_log:
			continue
		df = pd.DataFrame(game_log)
		if 'FPts' not in df.columns or len(df) == 0:
			continue
		fpts = df['FPts']
		mean_fpts = fpts.mean()
		std_dev = fpts.std()
		cv = (std_dev / mean_fpts * 100) if mean_fpts > 0 else 0
		boom_threshold = mean_fpts + std_dev
		bust_threshold = mean_fpts - std_dev
		boom_games = len(fpts[fpts > boom_threshold])
		bust_games = len(fpts[fpts < bust_threshold])
		total_games = len(fpts)

		mean_minutes = None
		fppm_mean = None
		try:
			stats = calculate_variability_stats(df.copy())
			if stats:
				mean_minutes = stats.get('mean_minutes')
				fppm_mean = stats.get('fppm_mean')
		except Exception:
			stats = None

		all_consistency[player_name] = {
			'player_name': player_name,
			'games_played': total_games,
			'mean_fpts': mean_fpts,
			'median_fpts': fpts.median(),
			'std_dev': std_dev,
			'cv_percent': cv,
			'min_fpts': fpts.min(),
			'max_fpts': fpts.max(),
			'range': fpts.max() - fpts.min(),
			'boom_games': boom_games,
			'boom_rate': (boom_games / total_games * 100) if total_games > 0 else 0,
			'bust_games': bust_games,
			'bust_rate': (bust_games / total_games * 100) if total_games > 0 else 0,
			'consistency_tier': get_consistency_tier(cv),
			'mean_minutes': mean_minutes,
			'fppm_mean': fppm_mean,
		}

	return all_consistency

def enrich_roster_with_consistency(roster_df: pd.DataFrame, league_id: str, consistency_index: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
	"""
	Add consistency metrics to a roster DataFrame.
	
	Args:
		roster_df: DataFrame with player roster data
		league_id: Fantrax league ID
		
	Returns:
		Enhanced DataFrame with consistency columns
	"""
	if roster_df.empty or 'Player' not in roster_df.columns:
		return roster_df
	
	# Add consistency columns
	roster_df['CV%'] = None
	roster_df['Consistency'] = None
	roster_df['Boom%'] = None
	roster_df['Bust%'] = None
	if 'Min' not in roster_df.columns:
		roster_df['Min'] = None
	
	index = consistency_index or {}
	use_index_only = consistency_index is not None
	
	for idx, row in roster_df.iterrows():
		player_name = row['Player']
		consistency = None
		if index and player_name in index:
			consistency = index[player_name]
		elif not use_index_only:
			consistency = load_player_consistency(player_name, league_id)
		
		if consistency:
			roster_df.at[idx, 'CV%'] = round(consistency['cv_percent'], 1)
			roster_df.at[idx, 'Consistency'] = consistency['consistency_tier']
			roster_df.at[idx, 'Boom%'] = round(consistency['boom_rate'], 1)
			roster_df.at[idx, 'Bust%'] = round(consistency['bust_rate'], 1)
			mean_minutes = consistency.get('mean_minutes')
			if mean_minutes is not None:
				roster_df.at[idx, 'Min'] = round(mean_minutes, 1)
	
	return roster_df
