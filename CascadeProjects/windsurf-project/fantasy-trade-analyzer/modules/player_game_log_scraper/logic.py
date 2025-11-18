import os
import time
import json
import pandas as pd
import re
import subprocess
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import atexit
from . import db_store

# --- CONFIGURATION ---
load_dotenv(find_dotenv('fantrax.env'))

FANTRAX_USERNAME = os.getenv('FANTRAX_USERNAME')
FANTRAX_PASSWORD = os.getenv('FANTRAX_PASSWORD')
FANTRAX_LOGIN_URL = 'https://www.fantrax.com/login'
FANTRAX_DEFAULT_LEAGUE_ID = os.getenv('FANTRAX_DEFAULT_LEAGUE_ID')
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'player_game_log_cache'

def get_cache_directory():
	"""Returns the cache directory path."""
	return CACHE_DIR

def get_league_index_path(league_id: str) -> Path:
	"""Return the path to the league-level cache index file.

	The index summarizes which players and seasons exist for a given league,
	and where their underlying cache files live on disk.
	"""
	return CACHE_DIR / f"player_game_log_index_{league_id}.json"

def build_league_cache_index(league_id: str) -> dict:
	"""Scan all player_game_log_full JSONs for a league and build an index.

	Structure:
	{
	  "league_id": str,
	  "generated_at": ISO timestamp,
	  "players": {
	    player_code: {
	      "player_name": str,
	      "seasons": {
	        season_str: {
	          "status": str,
	          "games": int,
	          "cache_file": str,  # filename only
	          "last_modified": ISO timestamp
	        }
	      }
	    }
	  }
	}
	"""
	cache_dir = get_cache_directory()
	index = {
		"league_id": league_id,
		"generated_at": datetime.utcnow().isoformat(),
		"players": {}
	}

	pattern = f"player_game_log_full_*_{league_id}_*.json"
	for cache_file in cache_dir.glob(pattern):
		try:
			with open(cache_file, 'r') as f:
				cache_data = json.load(f)
		except Exception:
			continue

		player_code = cache_data.get('player_code')
		player_name = cache_data.get('player_name', 'Unknown')
		season = cache_data.get('season')
		if not player_code or not season:
			# Derive season from filename as fallback
			parts = cache_file.stem.split('_')
			if len(parts) >= 2:
				season_part = '_'.join(parts[-2:])
				season = season_part.replace('_', '-')
			else:
				continue

		status = cache_data.get('status', 'success')
		data = cache_data.get('data', cache_data.get('game_log', []))
		games = len(data) if isinstance(data, list) else 0
		last_modified = datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()

		player_entry = index["players"].setdefault(player_code, {
			"player_name": player_name,
			"seasons": {}
		})
		player_entry["player_name"] = player_name
		seasons = player_entry["seasons"]
		seasons[season] = {
			"status": status,
			"games": games,
			"cache_file": cache_file.name,
			"last_modified": last_modified
		}

	index_path = get_league_index_path(league_id)
	try:
		CACHE_DIR.mkdir(parents=True, exist_ok=True)
		with open(index_path, 'w') as f:
			json.dump(index, f, indent=2)
	except Exception:
		# Index generation failure should not break callers; they can fall back
		pass

	return index

def load_league_cache_index(league_id: str, rebuild_if_missing: bool = True) -> dict | None:
	"""Load the league cache index, optionally rebuilding it if missing/invalid."""
	index_path = get_league_index_path(league_id)
	if index_path.exists():
		try:
			with open(index_path, 'r') as f:
				return json.load(f)
		except Exception:
			# Fall through to optional rebuild
			pass

	if rebuild_if_missing:
		return build_league_cache_index(league_id)

	return None

def clean_html_from_text(text):
	"""Remove HTML tags from text strings."""
	if not isinstance(text, str):
		return text
	# Remove HTML tags using regex
	clean_text = re.sub(r'<[^>]+>', '', text)
	return clean_text.strip()

def clean_cached_game_log(game_log):
	"""Clean HTML tags from cached game log data."""
	if not game_log:
		return game_log
	
	cleaned_log = []
	for game in game_log:
		cleaned_game = {}
		for key, value in game.items():
			cleaned_game[key] = clean_html_from_text(value)
		cleaned_log.append(cleaned_game)
	
	return cleaned_log

def get_chrome_driver():
	options = webdriver.ChromeOptions()
	#options.add_argument('--headless')
	options.add_argument('--no-sandbox')
	options.add_argument('--disable-dev-shm-usage')
	options.add_argument('--log-level=3')
	options.add_experimental_option('excludeSwitches', ['enable-logging'])
	service = Service(ChromeDriverManager().install())
	driver = webdriver.Chrome(service=service, options=options)
	return driver

def login_to_fantrax(driver, username, password):
	"""Logs into Fantrax using the provided credentials."""
	driver.get(FANTRAX_LOGIN_URL)
	time.sleep(2)
	username_box = driver.find_element(By.ID, 'mat-input-0')
	password_box = driver.find_element(By.ID, 'mat-input-1')
	username_box.send_keys(username)
	password_box.send_keys(password)
	password_box.send_keys(Keys.RETURN)
	time.sleep(3)
	if 'login' in driver.current_url:
		raise Exception('Login failed. Check credentials.')

def clear_player_game_log_cache():
	"""Deletes all .json files from the player game log cache directory."""
	if not CACHE_DIR.exists():
		return "Cache directory does not exist.", False
	
	files_deleted = 0
	try:
		for file_path in CACHE_DIR.glob("*.json"):
			file_path.unlink()
			files_deleted += 1
		if files_deleted > 0:
			return f"Successfully deleted {files_deleted} file(s) from the cache.", True
		else:
			return "Cache is already empty.", True
	except Exception as e:
		return f"An error occurred while clearing the cache: {e}", False

def kill_chromedriver_processes(also_chrome=False):
	try:
		if os.name == 'nt':
			subprocess.run(['taskkill','/F','/IM','chromedriver.exe','/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			if also_chrome:
				subprocess.run(['taskkill','/F','/IM','chrome.exe','/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		else:
			subprocess.run(['pkill','-f','chromedriver'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			if also_chrome:
				subprocess.run(['pkill','-f','chrome'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except Exception:
		pass


def cleanup_on_exit():
	try:
		kill_chromedriver_processes(also_chrome=False)
	except Exception:
		pass

atexit.register(cleanup_on_exit)

def get_player_game_log(player_code, league_id, username, password, force_refresh=False):
	"""
	Scrapes player game log using Games (Fntsy) tab for current season.
	Now standardized to always use full Games tab approach.
	Returns a tuple (DataFrame, from_cache_boolean, player_name).
	"""
	# Always use current season and full format
	return get_player_game_log_full(player_code, league_id, username, password, "2025-26", force_refresh)

def get_player_game_log_full(player_code, league_id, username, password, season="2025-26", force_refresh=False):
	"""
	Enhanced version that navigates to Games (Fntsy) tab and allows season selection.
	Returns a tuple (DataFrame, from_cache_boolean, player_name).
	
	Args:
		player_code: Player ID code
		league_id: Fantrax league ID
		username: Fantrax username
		password: Fantrax password
		season: Season to scrape (e.g., "2024-25", defaults to "2025-26")
		force_refresh: Force refresh from web instead of cache
	"""
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	cache_file = CACHE_DIR / f"player_game_log_full_{player_code}_{league_id}_{season.replace('-', '_')}.json"

	# Cache logic:
	# - Past seasons: Use cache unless force_refresh=True  
	# - Current season (2025-26): Always refresh unless force_refresh=False
	if not force_refresh:
		try:
			loaded = db_store.load_player_season(
				player_code=player_code,
				league_id=league_id,
				season=season,
			)
		except Exception:
			loaded = None
		if loaded is not None:
			records, status, player_name = loaded
			if records is not None:
				df = pd.DataFrame.from_records(records)
				return df, True, player_name

	if cache_file.exists():
		if season != "2025-26":
			# Past season: use cache unless forcing refresh
			if not force_refresh:
				with open(cache_file, 'r') as f:
					cache_data = json.load(f)
					df = pd.DataFrame.from_records(cache_data['data'])
					player_name = cache_data.get('player_name', 'Unknown Player')
					return df, True, player_name
		else:
			# Current season: only use cache if explicitly not forcing refresh
			if force_refresh == False:
				with open(cache_file, 'r') as f:
					cache_data = json.load(f)
					df = pd.DataFrame.from_records(cache_data['data'])
					player_name = cache_data.get('player_name', 'Unknown Player')
					return df, True, player_name

	kill_chromedriver_processes(also_chrome=False)
	driver = get_chrome_driver()
	login_to_fantrax(driver, username, password)

	try:
		player_url = f"https://www.fantrax.com/player/{player_code}/{league_id}"
		# Retry navigation to the player page a few times in case Fantrax is flaky
		max_attempts = 3
		last_error = None
		for attempt in range(1, max_attempts + 1):
			try:
				driver.get(player_url)
				time.sleep(1.5)  # Short wait for initial load
				break
			except WebDriverException as e:
				last_error = e
				if attempt == max_attempts:
					raise Exception(f"Failed to load player page after {max_attempts} attempts: {e}")
				# Brief backoff before retrying
				time.sleep(2.0)

		page_source = driver.page_source
		soup = BeautifulSoup(page_source, 'html.parser')

		# Extract player name
		player_name = 'Unknown Player'
		for selector in [('h1', {'class': 'player-name'}), ('h1', {}), ('div', {'class': 'player-header__name'})]:
			player_name_elem = soup.find(selector[0], selector[1])
			if player_name_elem:
				player_name = player_name_elem.text.strip()
				break

		# Navigate to Games (Fntsy) tab
		try:
			# Find the Games (Fntsy) tab button
			games_tab = driver.find_element(By.XPATH, "//button[contains(text(), 'Games (Fntsy)')]")
			games_tab.click()
			time.sleep(1.5)  # Short wait for tab load
		except Exception as e:
			driver.quit()
			raise Exception(f"Could not find or click Games (Fntsy) tab: {e}")

		# Handle season selection if not default
		if season != "2025-26":
			try:
				# Click the season dropdown
				season_dropdown = driver.find_element(By.CSS_SELECTOR, "mat-select")
				season_dropdown.click()
				time.sleep(0.6)  # Short wait for dropdown open
				
				# Find and click the desired season option
				season_text = f"{season} Reg Season"
				season_option = driver.find_element(By.XPATH, f"//mat-option//span[contains(text(), '{season_text}')]")
				season_option.click()
				time.sleep(1)  # Short wait for data load
			except Exception as e:
				print(f"Warning: Could not select season {season}, using default: {e}")

		# Get updated page source after navigation
		page_source = driver.page_source
		soup = BeautifulSoup(page_source, 'html.parser')

		# Find the game log table (single table in Games tab)
		game_log_table = soup.find('div', class_='i-table minimal-scrollbar i-table--standard i-table--outside-sticky')
		if not game_log_table:
			driver.quit()
			raise Exception("Could not find the game log table in Games (Fntsy) tab.")

		# Find the table body and header
		table_body = game_log_table.find('div', {'itablebody': ''})
		table_header = game_log_table.find('div', {'itableheader': ''})
		
		if not table_body or not table_header:
			driver.quit()
			raise Exception("Could not find the table body or header in Games (Fntsy) tab.")

		# Extract header information
		headers = []
		header_cells = table_header.find_all('div', class_='i-table__cell')
		for cell in header_cells:
			header_text = cell.text.strip()
			headers.append(header_text)

		# Extract game rows
		game_rows = table_body.find_all('div', {'itablerow': ''})
		
		all_games = []
		for row in game_rows:
			cells = row.find_all('div', class_='i-table__cell')
			game_data = {}
			
			for i, cell in enumerate(cells):
				if i >= len(headers):
					break
				
				header = headers[i]
				
				# Check if there's a link (for Score column)
				link = cell.find('a')
				if link:
					value = link.get_text(strip=True)
				else:
					span = cell.find('span')
					if span:
						value = span.get_text(strip=True)
					else:
						value = cell.get_text(strip=True)
				
				game_data[header] = value
			
			if game_data:
				all_games.append(game_data)

		if not all_games:
			driver.quit()
			raise Exception("No game data found for this player in Games (Fntsy) tab.")

		df = pd.DataFrame(all_games)
		
		# Convert numeric columns
		numeric_columns = ['FPts', 'FGM', 'FGA', 'FG%', '3PTM', '3PTA', '3PT%', 
						   'FTM', 'FTA', 'FT%', 'REB', 'AST', 'ST', 'BLK', 'TO', 'PF', 'PTS']
		
		for col in numeric_columns:
			if col in df.columns:
				df[col] = pd.to_numeric(df[col], errors='coerce')
		
		# Save to cache
		cache_data = {
			'player_name': player_name,
			'player_code': player_code,
			'league_id': league_id,
			'season': season,
			'data': df.to_dict('records')
		}
		with open(cache_file, 'w') as f:
			json.dump(cache_data, f, indent=4)
		try:
			db_store.store_player_season(
				player_code=player_code,
				player_name=player_name,
				league_id=league_id,
				season=season,
				status='success',
				game_log_records=df.to_dict('records'),
			)
		except Exception:
			pass

		return df, False, player_name

	finally:
		driver.quit()

def calculate_variability_stats(game_log_df):
	"""Calculate variability statistics from a game log DataFrame."""
	if game_log_df.empty or 'FPts' not in game_log_df.columns:
		return None
	
	# Clean HTML from all string columns
	for col in game_log_df.columns:
		if game_log_df[col].dtype == 'object':
			game_log_df[col] = game_log_df[col].apply(clean_html_from_text)
	
	# Convert FPts to numeric, handling any non-numeric values
	fpts = pd.to_numeric(game_log_df['FPts'], errors='coerce').dropna()
	
	if len(fpts) == 0:
		return None
	
	stats = {
		'games_played': len(fpts),
		'mean_fpts': fpts.mean(),
		'median_fpts': fpts.median(),
		'std_dev': fpts.std(),
		'min_fpts': fpts.min(),
		'max_fpts': fpts.max(),
		'range': fpts.max() - fpts.min(),
		'coefficient_of_variation': (fpts.std() / fpts.mean() * 100) if fpts.mean() > 0 else 0,
		'q1': fpts.quantile(0.25),
		'q3': fpts.quantile(0.75),
		'iqr': fpts.quantile(0.75) - fpts.quantile(0.25)
	}
	
	# Calculate boom/bust rates (games above/below 1 std dev from mean)
	mean = stats['mean_fpts']
	std = stats['std_dev']
	stats['boom_games'] = len(fpts[fpts > mean + std])
	stats['bust_games'] = len(fpts[fpts < mean - std])
	stats['boom_rate'] = (stats['boom_games'] / len(fpts) * 100) if len(fpts) > 0 else 0
	stats['bust_rate'] = (stats['bust_games'] / len(fpts) * 100) if len(fpts) > 0 else 0
	
	return stats


def calculate_multi_range_stats(game_log_df):
	"""Calculate stats for multiple time ranges: Last 7, Last 15, Last 30, YTD."""
	if game_log_df.empty or 'FPts' not in game_log_df.columns:
		return None
	
	# Clean HTML from all string columns
	for col in game_log_df.columns:
		if game_log_df[col].dtype == 'object':
			game_log_df[col] = game_log_df[col].apply(clean_html_from_text)
	
	# Convert FPts to numeric
	game_log_df['FPts'] = pd.to_numeric(game_log_df['FPts'], errors='coerce')
	game_log_df = game_log_df.dropna(subset=['FPts'])
	
	if game_log_df.empty:
		return None
	
	# Calculate stats for each time range
	time_ranges = {
		'Last 7': 7,
		'Last 15': 15,
		'Last 30': 30,
		'YTD': len(game_log_df)
	}
	
	results = {}
	for range_name, num_games in time_ranges.items():
		# Take the most recent N games
		range_df = game_log_df.head(min(num_games, len(game_log_df)))
		fpts = range_df['FPts']
		
		if len(fpts) == 0:
			continue
		
		mean_fpts = fpts.mean()
		std_fpts = fpts.std()
		
		results[range_name] = {
			'games_played': len(fpts),
			'mean_fpts': mean_fpts,
			'median_fpts': fpts.median(),
			'std_dev': std_fpts,
			'coefficient_of_variation': (std_fpts / mean_fpts * 100) if mean_fpts > 0 else 0,
			'min_fpts': fpts.min(),
			'max_fpts': fpts.max()
		}
	
	return results

def get_available_players_from_csv():
	"""
	Reads player IDs and names from the most recent Fantrax-Players CSV file.
	Only includes rostered players (Status != 'FA').
	Returns a dictionary mapping player names to player codes.
	"""
	data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
	
	# Look for the most recent Fantrax-Players CSV file (YTD preferred)
	player_files = sorted(data_dir.glob("Fantrax-Players-*-(YTD).csv"), key=lambda f: f.stat().st_mtime, reverse=True)
	
	if not player_files:
		# Fallback: try any Fantrax-Players file
		player_files = sorted(data_dir.glob("Fantrax-Players-*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
	
	if not player_files:
		return {}
	
	csv_file = player_files[0]  # Use the most recent file
	
	try:
		df = pd.read_csv(csv_file)
		if 'ID' not in df.columns or 'Player' not in df.columns or 'Status' not in df.columns:
			return {}
		
		# Filter out Free Agents (only keep rostered players)
		df = df[df['Status'] != 'FA']
		
		# Remove asterisks from player IDs
		df['ID'] = df['ID'].str.replace('*', '', regex=False)
		
		# Create a dictionary mapping player names to IDs
		player_dict = dict(zip(df['Player'], df['ID']))
		return player_dict
		
	except Exception as e:
		print(f"Error reading {csv_file.name}: {e}")
		return {}

def bulk_scrape_all_players_full(league_id, username, password, seasons=None, player_dict=None, progress_callback=None, force_refresh=False):
	"""
	Enhanced bulk scraper that uses Games (Fntsy) tab and can scrape multiple seasons per player.
	
	Args:
		league_id: Fantrax league ID
		username: Fantrax username
		password: Fantrax password
		seasons: List of seasons to scrape (e.g., ['2025-26', '2024-25']). If None, defaults to current season only.
		player_dict: Dictionary of player names to codes. If None, loads from CSV.
		progress_callback: Optional callback function(current, total, player_name, season)
		force_refresh: If True, refresh all seasons. If False, smart caching (past seasons cached, current season refreshed)
	
	Returns:
		Dictionary with success/failure counts and details
	"""
	if seasons is None:
		seasons = ["2025-26"]  # Default to current season
	
	if player_dict is None:
		player_dict = get_available_players_from_csv()
	
	if not player_dict:
		return {"error": "No players found to scrape"}
	
	kill_chromedriver_processes(also_chrome=False)
	driver = get_chrome_driver()
	
	try:
		login_to_fantrax(driver, username, password)
		
		success_count = 0
		fail_count = 0
		failed_items = []
		total_operations = len(player_dict) * len(seasons)
		current_operation = 0
		
		for player_name, player_code in player_dict.items():
			# Check which seasons need scraping for this player
			seasons_to_scrape = []
			for season in seasons:
				cache_file = CACHE_DIR / f"player_game_log_full_{player_code}_{league_id}_{season.replace('-', '_')}.json"
				
				# Smart caching logic (unless force_refresh=True)
				if force_refresh:
					# Force refresh: scrape all seasons
					seasons_to_scrape.append(season)
				elif season == "2025-26":
					# Current season: always scrape (it's updating)
					seasons_to_scrape.append(season)
				elif not cache_file.exists():
					# Past season not cached: scrape it
					seasons_to_scrape.append(season)
				else:
					# Past season already cached: skip
					success_count += 1
			
			# Skip player if all seasons are cached
			if not seasons_to_scrape:
				current_operation += len(seasons)
				continue
			
			# Navigate to player page once per player
			try:
				player_url = f"https://www.fantrax.com/player/{player_code}/{league_id}"
				driver.get(player_url)
				time.sleep(3)
				
				# Navigate to Games (Fntsy) tab once
				try:
					games_tab = driver.find_element(By.XPATH, "//button[contains(text(), 'Games (Fntsy)')]")
					games_tab.click()
					time.sleep(1.5)  # Wait for Games tab to fully load
					
					# Games tab typically defaults to current season
					current_season_on_page = "2025-26"
						
				except Exception:
					# If Games tab fails, skip all seasons for this player
					for season in seasons_to_scrape:
						current_operation += 1
						failed_items.append((player_name, season, "Games tab not found"))
						fail_count += 1
					continue
				
				consecutive_empty_seasons = 0  # Track consecutive seasons with 0 games
				
				for season_index, season in enumerate(seasons_to_scrape):
					current_operation += 1
					
					# Skip remaining seasons if we've had 3 consecutive empty seasons
					if consecutive_empty_seasons >= 2:
						# Skip all remaining seasons for this player
						remaining_seasons = len(seasons_to_scrape) - season_index
						current_operation += remaining_seasons - 1  # Adjust operation count
						break
					
					if progress_callback:
						progress_callback(current_operation, total_operations, player_name, season)
					
					try:
						
						# Handle season selection - always select the season we want
						if season != current_season_on_page:
							try:
								season_dropdown = driver.find_element(By.CSS_SELECTOR, "mat-select")
								season_dropdown.click()
								time.sleep(1)
								
								season_text = f"{season} Reg Season"
								season_option = driver.find_element(By.XPATH, f"//mat-option//span[contains(text(), '{season_text}')]")
								season_option.click()
								
								# Wait for season data to load
								time.sleep(2)
								
								current_season_on_page = season  # Update what's currently displayed
							except Exception as e:
								# If season selection fails, skip this season
								failed_items.append((player_name, season, "Season selection failed"))
								fail_count += 1
								continue
						
						# Get page source and parse
						time.sleep(1)  # Brief wait for data to load
						page_source = driver.page_source
						soup = BeautifulSoup(page_source, 'html.parser')
						
						# Find the game log table
						game_log_table = soup.find('div', class_='i-table minimal-scrollbar i-table--standard i-table--outside-sticky')
						if not game_log_table:
							failed_items.append((player_name, season, "Table not found"))
							fail_count += 1
							continue
						
						table_body = game_log_table.find('div', {'itablebody': ''})
						table_header = game_log_table.find('div', {'itableheader': ''})
						
						if not table_body or not table_header:
							failed_items.append((player_name, season, "Table structure missing"))
							fail_count += 1
							continue
						
						# Extract headers
						headers = []
						header_cells = table_header.find_all('div', class_='i-table__cell')
						for cell in header_cells:
							headers.append(cell.text.strip())
						
						# Extract game rows
						game_rows = table_body.find_all('div', {'itablerow': ''})
						all_games = []
						
						
						for row in game_rows:
							cells = row.find_all('div', class_='i-table__cell')
							game_data = {}
							
							for i, cell in enumerate(cells):
								if i >= len(headers):
									break
								header = headers[i]
								link = cell.find('a')
								if link:
									value = link.get_text(strip=True)
								else:
									span = cell.find('span')
									if span:
										value = span.get_text(strip=True)
									else:
										value = cell.get_text(strip=True)
								game_data[header] = value
							
							if game_data:
								all_games.append(game_data)
						
						
						# Handle different scenarios for empty/missing data
						if not all_games:
							# Check if this might be a season the player didn't play
							# Look for indicators like "No games found" or empty table with headers
							if headers:
								# Table structure exists but no games - player didn't play this season
								# Create empty cache file to avoid re-scraping
								cache_file = CACHE_DIR / f"player_game_log_full_{player_code}_{league_id}_{season.replace('-', '_')}.json"
								cache_data = {
									'player_name': player_name,
									'player_code': player_code,
									'league_id': league_id,
									'season': season,
									'data': [],  # Empty games list
									'status': 'no_games_played'
								}
								CACHE_DIR.mkdir(parents=True, exist_ok=True)
								with open(cache_file, 'w') as f:
									json.dump(cache_data, f, indent=4)
								try:
									db_store.store_player_season(
										player_code=player_code,
										player_name=player_name,
										league_id=league_id,
										season=season,
										status='no_games_played',
										game_log_records=[],
									)
								except Exception:
									pass
								
								success_count += 1  # Count as success (valid empty season)
								consecutive_empty_seasons += 1  # Increment empty season counter
								continue
							else:
								# No table structure - likely season doesn't exist or scraping failed
								failed_items.append((player_name, season, "Season not available or scraping failed"))
								fail_count += 1
								continue
						
						# Convert to DataFrame and save (normal case with games)
						df = pd.DataFrame(all_games)
						numeric_columns = ['FPts', 'FGM', 'FGA', 'FG%', '3PTM', '3PTA', '3PT%', 
										   'FTM', 'FTA', 'FT%', 'REB', 'AST', 'ST', 'BLK', 'TO', 'PF', 'PTS']
						for col in numeric_columns:
							if col in df.columns:
								df[col] = pd.to_numeric(df[col], errors='coerce')
						
						# Save to cache
						CACHE_DIR.mkdir(parents=True, exist_ok=True)
						cache_file = CACHE_DIR / f"player_game_log_full_{player_code}_{league_id}_{season.replace('-', '_')}.json"
						cache_data = {
							'player_name': player_name,
							'player_code': player_code,
							'league_id': league_id,
							'season': season,
							'data': df.to_dict('records'),
							'status': 'success'
						}
						with open(cache_file, 'w') as f:
							json.dump(cache_data, f, indent=4)
						try:
							db_store.store_player_season(
								player_code=player_code,
								player_name=player_name,
								league_id=league_id,
								season=season,
								status='success',
								game_log_records=df.to_dict('records'),
							)
						except Exception:
							pass
						
						success_count += 1
						consecutive_empty_seasons = 0  # Reset counter when we find games
						
					except Exception as e:
						failed_items.append((player_name, season, str(e)))
						fail_count += 1
			
			except Exception as e:
				# If player-level navigation fails, skip all seasons for this player
				for season in seasons_to_scrape:
					if current_operation < total_operations:
						current_operation += 1
					failed_items.append((player_name, season, f"Player navigation failed: {str(e)}"))
					fail_count += 1
		
		return {
			"success_count": success_count,
			"fail_count": fail_count,
			"total": total_operations,
			"failed_items": failed_items
		}
	finally:
		if driver:
			driver.quit()

def bulk_scrape_all_players(league_id, username, password, player_dict=None, progress_callback=None):
	"""
	Scrapes game logs for all players in the provided dictionary.
	Returns a summary of successes and failures.
	"""
	if player_dict is None:
		player_dict = get_available_players_from_csv()
	
	if not player_dict:
		return {"error": "No players found to scrape"}
	
	kill_chromedriver_processes(also_chrome=False)
	driver = get_chrome_driver()
	try:
		login_to_fantrax(driver, username, password)
		
		success_count = 0
		fail_count = 0
		failed_players = []
		total = len(player_dict)
		
		for idx, (player_name, player_code) in enumerate(player_dict.items(), 1):
			if progress_callback:
				progress_callback(idx, total, player_name)
			
			try:
				# Navigate to player page
				player_url = f"https://www.fantrax.com/player/{player_code}/{league_id}"
				driver.get(player_url)
				time.sleep(1.0)  # Shorter wait for bulk operations
				
				# Navigate to Games (Fntsy) tab
				try:
					games_tab = driver.find_element(By.XPATH, "//button[contains(text(), 'Games (Fntsy)')]")
					games_tab.click()
					time.sleep(1.0)
				except Exception:
					failed_players.append((player_name, "Games tab not found"))
					fail_count += 1
					continue
				
				# Get updated page source after navigation
				page_source = driver.page_source
				soup = BeautifulSoup(page_source, 'html.parser')
				
				# Find the game log table (single table in Games tab)
				game_log_table = soup.find('div', class_='i-table minimal-scrollbar i-table--standard i-table--outside-sticky')
				if not game_log_table:
					failed_players.append((player_name, "Table not found"))
					fail_count += 1
					continue
				
				table_body = game_log_table.find('div', {'itablebody': ''})
				table_header = game_log_table.find('div', {'itableheader': ''})
				
				if not table_body or not table_header:
					failed_players.append((player_name, "Table structure missing"))
					fail_count += 1
					continue
				
				# Extract headers
				headers = []
				header_cells = table_header.find_all('div', class_='i-table__cell')
				for cell in header_cells:
					headers.append(cell.text.strip())
				
				# Extract game rows
				game_rows = table_body.find_all('div', {'itablerow': ''})
				all_games = []
				
				for row in game_rows:
					cells = row.find_all('div', class_='i-table__cell')
					game_data = {}
					
					for i, cell in enumerate(cells):
						if i >= len(headers):
							break
						header = headers[i]
						link = cell.find('a')
						if link:
							value = link.get_text(strip=True)
						else:
							span = cell.find('span')
							if span:
								value = span.get_text(strip=True)
							else:
								value = cell.get_text(strip=True)
						game_data[header] = value
					
					if game_data:
						all_games.append(game_data)
				
				if not all_games:
					failed_players.append((player_name, "No game data"))
					fail_count += 1
					continue
				
				# Convert to DataFrame and save
				df = pd.DataFrame(all_games)
				numeric_columns = ['FPts', 'FGM', 'FGA', 'FG%', '3PTM', '3PTA', '3PT%', 
								   'FTM', 'FTA', 'FT%', 'REB', 'AST', 'ST', 'BLK', 'TO', 'PF', 'PTS']
				for col in numeric_columns:
					if col in df.columns:
						df[col] = pd.to_numeric(df[col], errors='coerce')
				
				# Save to cache (use new format for consistency)
				CACHE_DIR.mkdir(parents=True, exist_ok=True)
				cache_file = CACHE_DIR / f"player_game_log_full_{player_code}_{league_id}_2025_26.json"
				cache_data = {
					'player_name': player_name,
					'player_code': player_code,
					'league_id': league_id,
					'season': '2025-26',
					'data': df.to_dict('records')
				}
				with open(cache_file, 'w') as f:
					json.dump(cache_data, f, indent=4)
				try:
					db_store.store_player_season(
						player_code=player_code,
						player_name=player_name,
						league_id=league_id,
						season='2025-26',
						status='success',
						game_log_records=df.to_dict('records'),
					)
				except Exception:
					pass
				
				success_count += 1
				
			except Exception as e:
				failed_players.append((player_name, str(e)))
				fail_count += 1
		
		return {
			"success_count": success_count,
			"fail_count": fail_count,
			"total": total,
			"failed_players": failed_players
		}
	finally:
		if driver:
			driver.quit()

def clear_all_cache():
	"""Clears all cached player game log files."""
	if not CACHE_DIR.exists():
		return 0
	
	cache_files = list(CACHE_DIR.glob("player_game_log_*.json"))
	count = len(cache_files)
	
	for cache_file in cache_files:
		try:
			cache_file.unlink()
		except Exception:
			pass
	
	return count
