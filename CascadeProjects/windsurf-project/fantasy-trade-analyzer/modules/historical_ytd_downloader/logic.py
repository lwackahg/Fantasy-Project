"""
Historical YTD Downloader - Download past season YTD stats for year-over-year analysis.
Based on fantrax_downloader but modified to fetch historical seasons.
"""

import os
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from modules.fantrax_downloader.logic import get_chrome_driver, login_to_fantrax

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent.parent / 'fantrax.env'
load_dotenv(env_path)

FANTRAX_USERNAME = os.getenv('FANTRAX_USERNAME')
FANTRAX_PASSWORD = os.getenv('FANTRAX_PASSWORD')
DOWNLOAD_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'historical_ytd'

# Ensure download directory exists
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Historical season codes (Fantrax API codes)
# Format: (season_label, season_code)
HISTORICAL_SEASONS = [
	('2024-25', 'SEASON_41j_YEAR_TO_DATE'),  # Current season
	('2023-24', 'SEASON_41h_YEAR_TO_DATE'),
	('2022-23', 'SEASON_41f_YEAR_TO_DATE'),
	('2021-22', 'SEASON_41d_YEAR_TO_DATE'),
	('2020-21', 'SEASON_41b_YEAR_TO_DATE'),  # COVID shortened
	('2019-20', 'SEASON_40z_YEAR_TO_DATE'),  # COVID bubble
	('2018-19', 'SEASON_40x_YEAR_TO_DATE'),
]


def download_historical_ytd(league_id: str, season_label: str, season_code: str, driver=None):
	"""
	Download YTD stats for a specific historical season.
	
	Args:
		league_id: Fantrax league ID
		season_label: Season label (e.g., '2023-24')
		season_code: Fantrax season code (e.g., 'SEASON_41h_YEAR_TO_DATE')
		driver: Optional existing Chrome driver (for batch operations)
	
	Returns:
		Tuple of (success: bool, message: str, file_path: str)
	"""
	if not FANTRAX_USERNAME or not FANTRAX_PASSWORD:
		return False, "Set FANTRAX_USERNAME and FANTRAX_PASSWORD in fantrax.env", ""
	
	# Create driver if not provided
	close_driver = False
	if driver is None:
		driver = get_chrome_driver(str(DOWNLOAD_DIR))
		login_to_fantrax(driver, FANTRAX_USERNAME, FANTRAX_PASSWORD)
		close_driver = True
	
	try:
		# Build CSV download URL using season code (matching working fantrax_downloader logic)
		csv_url = (
			f"https://www.fantrax.com/fxpa/downloadPlayerStats?"
			f"leagueId={league_id}&pageNumber=1&statusOrTeamFilter=ALL"
			f"&seasonOrProjection={season_code}&timeframeTypeCode=YEAR_TO_DATE"
			f"&view=STATS&positionOrGroup=BASKETBALL_PLAYER"
			f"&transactionPeriod=22&miscDisplayType=1&sortType=SCORE&maxResultsPerPage=500"
			f"&scoringCategoryType=5&timeStartType=PERIOD_ONLY"
			f"&schedulePageAdj=0&searchName=&datePlaying=ALL&"
		)
		
		# Generate filename
		def get_league_name_map():
			ids = os.environ.get("FANTRAX_LEAGUE_IDS", "")
			names = os.environ.get("FANTRAX_LEAGUE_NAMES", "")
			return dict(zip([i.strip() for i in ids.split(",") if i.strip()], 
						   [n.strip() for n in names.split(",") if n.strip()]))
		
		def sanitize_name(name):
			import re
			return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().replace(" ", "_"))
		
		league_name = sanitize_name(get_league_name_map().get(league_id, league_id))
		csv_filename = f"Fantrax-Players-{league_name}-YTD-{season_label}.csv"
		download_path = DOWNLOAD_DIR / csv_filename
		
		# Download using session with cookies
		cookies = driver.get_cookies()
		session = requests.Session()
		for cookie in cookies:
			session.cookies.set(cookie['name'], cookie['value'])
		
		print(f'Downloading {season_label} YTD ({season_code})...')
		resp = session.get(csv_url)
		
		if resp.status_code == 200 and resp.content:
			with open(download_path, 'wb') as f:
				f.write(resp.content)
			print(f'Success: Downloaded {csv_filename}')
			return True, f'Success: Downloaded {season_label} YTD', str(download_path)
		else:
			print(f'Error: Failed to download (HTTP {resp.status_code})')
			return False, f'Error: Failed to download (HTTP {resp.status_code})', ""
	
	finally:
		if close_driver:
			driver.quit()


def download_all_historical_seasons(league_id: str, seasons_to_download: list = None, progress_callback=None):
	"""
	Download YTD stats for multiple historical seasons.
	
	Args:
		league_id: Fantrax league ID
		seasons_to_download: List of season labels to download (e.g., ['2023-24', '2022-23'])
						   If None, downloads all available seasons
		progress_callback: Optional callback function(current, total, season_label)
	
	Returns:
		List of result messages
	"""
	if not FANTRAX_USERNAME or not FANTRAX_PASSWORD:
		return ["Set FANTRAX_USERNAME and FANTRAX_PASSWORD in fantrax.env"]
	
	# Filter seasons if specific ones requested
	if seasons_to_download:
		seasons = [s for s in HISTORICAL_SEASONS if s[0] in seasons_to_download]
	else:
		seasons = HISTORICAL_SEASONS
	
	if not seasons:
		return ["No valid seasons selected"]
	
	messages = []
	driver = get_chrome_driver(str(DOWNLOAD_DIR))
	
	try:
		login_to_fantrax(driver, FANTRAX_USERNAME, FANTRAX_PASSWORD)
		
		total = len(seasons)
		for idx, (season_label, season_code) in enumerate(seasons, 1):
			if progress_callback:
				progress_callback(idx, total, season_label)
			
			success, message, _ = download_historical_ytd(
				league_id, season_label, season_code, driver
			)
			messages.append(f"{season_label}: {message}")
			
			# Small delay between downloads
			if idx < total:
				time.sleep(1)
	
	finally:
		driver.quit()
	
	return messages


def get_available_seasons():
	"""Return list of available historical seasons."""
	return [season[0] for season in HISTORICAL_SEASONS]


def get_season_code(season_label: str):
	"""Get Fantrax season code for a specific season."""
	for label, code in HISTORICAL_SEASONS:
		if label == season_label:
			return code
	return None


def load_and_compare_seasons(league_name: str, seasons_to_compare: list = None):
	"""
	Load historical YTD data and create year-over-year comparison.
	
	Args:
		league_name: League name (sanitized, as used in filenames)
		seasons_to_compare: List of season labels to compare (e.g., ['2024-25', '2023-24'])
	
	Returns:
		pandas DataFrame with YoY comparison, or None if files not found
	"""
	import pandas as pd
	
	if not seasons_to_compare or len(seasons_to_compare) < 2:
		return None
	
	# Load all season data
	season_data = {}
	for season in seasons_to_compare:
		filename = f"Fantrax-Players-{league_name}-YTD-{season}.csv"
		filepath = DOWNLOAD_DIR / filename
		
		if not filepath.exists():
			print(f"Warning: File not found for {season}: {filepath}")
			continue
		
		try:
			df = pd.read_csv(filepath)
			
			# Filter out free agents (N/A team) to avoid duplicate names
			if 'Team' in df.columns:
				df = df[df['Team'] != '(N/A)'].copy()
			
			# Keep only Player, ID, and FP/G columns
			if 'Player' in df.columns and 'FP/G' in df.columns and 'ID' in df.columns:
				# Create unique identifier using ID + Player name
				df['Player_ID'] = df['ID'].astype(str) + '|' + df['Player'].astype(str)
				season_data[season] = df[['Player', 'Player_ID', 'FP/G']].copy()
				season_data[season].rename(columns={'FP/G': f'FP/G_{season}'}, inplace=True)
			else:
				print(f"Warning: Required columns not found in {season}")
		except Exception as e:
			print(f"Error loading {season}: {e}")
	
	if len(season_data) < 2:
		return None
	
	# Merge all seasons on Player_ID (unique identifier)
	comparison_df = None
	for season, df in season_data.items():
		if comparison_df is None:
			comparison_df = df
		else:
			# Merge on Player_ID, but keep Player name from first df
			comparison_df = comparison_df.merge(
				df.drop(columns=['Player']), 
				on='Player_ID', 
				how='outer'
			)
	
	# Calculate YoY changes
	seasons_sorted = sorted(seasons_to_compare, reverse=True)  # Most recent first
	
	for i in range(len(seasons_sorted) - 1):
		current_season = seasons_sorted[i]
		previous_season = seasons_sorted[i + 1]
		
		current_col = f'FP/G_{current_season}'
		previous_col = f'FP/G_{previous_season}'
		change_col = f'YoY_Change_{current_season}_vs_{previous_season}'
		pct_col = f'YoY_Pct_{current_season}_vs_{previous_season}'
		
		if current_col in comparison_df.columns and previous_col in comparison_df.columns:
			# Calculate absolute change
			comparison_df[change_col] = comparison_df[current_col] - comparison_df[previous_col]
			
			# Calculate percentage change
			comparison_df[pct_col] = (
				(comparison_df[current_col] - comparison_df[previous_col]) / 
				comparison_df[previous_col] * 100
			).round(1)
	
	# Sort by most recent season FP/G
	most_recent = f'FP/G_{seasons_sorted[0]}'
	if most_recent in comparison_df.columns:
		comparison_df = comparison_df.sort_values(most_recent, ascending=False, na_position='last')
	
	# Drop Player_ID column (internal use only)
	if 'Player_ID' in comparison_df.columns:
		comparison_df = comparison_df.drop(columns=['Player_ID'])
	
	return comparison_df
