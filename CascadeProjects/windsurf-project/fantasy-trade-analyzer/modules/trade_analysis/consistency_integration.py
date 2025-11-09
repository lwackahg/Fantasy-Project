"""Integration module for player consistency data in trade analysis."""
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

def get_consistency_cache_directory():
	"""Get the player game log cache directory."""
	cache_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'player_game_log_cache'
	cache_dir.mkdir(parents=True, exist_ok=True)
	return cache_dir

def load_player_consistency(player_name: str, league_id: str) -> Optional[Dict]:
	"""
	Load consistency metrics for a specific player from cache.
	
	Args:
		player_name: Name of the player
		league_id: Fantrax league ID
		
	Returns:
		Dict with consistency metrics or None if not found
	"""
	cache_dir = get_consistency_cache_directory()
	
	# Find cache file for this player
	# Cache files are named: player_game_log_{player_code}_{league_id}.json
	cache_files = list(cache_dir.glob(f"player_game_log_*_{league_id}.json"))
	
	for cache_file in cache_files:
		try:
			with open(cache_file, 'r') as f:
				cache_data = json.load(f)
			
			cached_player_name = cache_data.get('player_name', '')
			if cached_player_name == player_name:
				# Calculate metrics from game log
				game_log = cache_data.get('data', cache_data.get('game_log', []))
				if game_log:
					df = pd.DataFrame(game_log)
					if 'FPts' in df.columns and len(df) > 0:
						fpts = df['FPts']
						mean_fpts = fpts.mean()
						std_dev = fpts.std()
						cv = (std_dev / mean_fpts * 100) if mean_fpts > 0 else 0
						
						# Boom/Bust thresholds (±1 std dev)
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
		except Exception:
			continue
	
	return None

def get_consistency_tier(cv_percent: float) -> str:
	"""
	Get consistency tier based on CV%.
	
	Args:
		cv_percent: Coefficient of variation percentage
		
	Returns:
		Tier string with emoji
	"""
	if cv_percent < 20:
		return "🟢 Very Consistent"
	elif cv_percent <= 30:
		return "🟡 Moderate"
	else:
		return "🔴 Volatile"

def load_all_player_consistency(league_id: str) -> Dict[str, Dict]:
	"""
	Load consistency metrics for all cached players.
	
	Args:
		league_id: Fantrax league ID
		
	Returns:
		Dict mapping player names to their consistency metrics
	"""
	cache_dir = get_consistency_cache_directory()
	cache_files = list(cache_dir.glob(f"player_game_log_*_{league_id}.json"))
	
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

def enrich_roster_with_consistency(roster_df: pd.DataFrame, league_id: str) -> pd.DataFrame:
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
	
	for idx, row in roster_df.iterrows():
		player_name = row['Player']
		consistency = load_player_consistency(player_name, league_id)
		
		if consistency:
			roster_df.at[idx, 'CV%'] = round(consistency['cv_percent'], 1)
			roster_df.at[idx, 'Consistency'] = consistency['consistency_tier']
			roster_df.at[idx, 'Boom%'] = round(consistency['boom_rate'], 1)
			roster_df.at[idx, 'Bust%'] = round(consistency['bust_rate'], 1)
	
	return roster_df
