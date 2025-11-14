"""Integration module for player consistency data in trade analysis."""
import json
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache
import pandas as pd

def get_consistency_cache_directory():
	"""Get the player game log cache directory."""
	cache_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'player_game_log_cache'
	cache_dir.mkdir(parents=True, exist_ok=True)
	return cache_dir

@lru_cache(maxsize=2048)
def _load_player_consistency_internal(player_name: str, league_id: str) -> Optional[Dict]:
	cache_dir = get_consistency_cache_directory()
	cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
	best_candidate = None
	best_season = None
	for cache_file in cache_files:
		try:
			with open(cache_file, 'r') as f:
				cache_data = json.load(f)
			cached_player_name = cache_data.get('player_name', '')
			if str(cached_player_name).strip().lower() != str(player_name).strip().lower():
				continue
			season_str = str(cache_data.get('season', '')).strip()
			if best_season is None or season_str > best_season:
				best_candidate = cache_data
				best_season = season_str
		except Exception:
			continue

	if best_candidate is None:
		return None
	game_log = best_candidate.get('data', best_candidate.get('game_log', []))
	if not game_log:
		return None
	df = pd.DataFrame(game_log)
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

def get_consistency_tier(cv_percent: float) -> str:
	"""
	Get consistency tier based on CV%.
	
	Args:
		cv_percent: Coefficient of variation percentage
		
	Returns:
		Tier string with emoji
	"""
	if cv_percent < 25:
		return "🟢 Very Consistent"
	elif cv_percent <= 40:
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

	all_consistency: Dict[str, Dict] = {}
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
	
	index = consistency_index or {}
	
	for idx, row in roster_df.iterrows():
		player_name = row['Player']
		consistency = None
		if index and player_name in index:
			consistency = index[player_name]
		else:
			consistency = load_player_consistency(player_name, league_id)
		
		if consistency:
			roster_df.at[idx, 'CV%'] = round(consistency['cv_percent'], 1)
			roster_df.at[idx, 'Consistency'] = consistency['consistency_tier']
			roster_df.at[idx, 'Boom%'] = round(consistency['boom_rate'], 1)
			roster_df.at[idx, 'Bust%'] = round(consistency['bust_rate'], 1)
	
	return roster_df
