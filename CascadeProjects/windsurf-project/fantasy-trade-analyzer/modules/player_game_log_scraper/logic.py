import os
import time
import json
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv, find_dotenv

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

def get_chrome_driver():
	"""Initializes and returns a headless Chrome WebDriver with suppressed logging."""
	options = webdriver.ChromeOptions()
	options.add_argument('--headless')
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
	time.sleep(5)
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

def get_player_game_log(player_code, league_id, username, password, force_refresh=False):
	"""
	Scrapes or loads from cache the player game log data from Fantrax.
	Returns a tuple (DataFrame, from_cache_boolean, player_name).
	"""
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	cache_file = CACHE_DIR / f"player_game_log_{player_code}_{league_id}.json"

	if not force_refresh and cache_file.exists():
		with open(cache_file, 'r') as f:
			cache_data = json.load(f)
			df = pd.DataFrame.from_records(cache_data['data'])
			player_name = cache_data.get('player_name', 'Unknown Player')
			return df, True, player_name

	driver = get_chrome_driver()
	login_to_fantrax(driver, username, password)

	player_url = f"https://www.fantrax.com/player/{player_code}/{league_id}"
	driver.get(player_url)
	time.sleep(5)  # Wait for the page to load

	page_source = driver.page_source
	soup = BeautifulSoup(page_source, 'html.parser')

	# Extract player name - try multiple selectors
	player_name = 'Unknown Player'
	# Try common player name selectors
	for selector in [('h1', {'class': 'player-name'}), ('h1', {}), ('div', {'class': 'player-header__name'})]:
		player_name_elem = soup.find(selector[0], selector[1])
		if player_name_elem:
			player_name = player_name_elem.text.strip()
			break

	# Find all i-table elements and get the second one (game log table)
	all_tables = soup.find_all('div', class_='i-table minimal-scrollbar i-table--standard i-table--outside-sticky')
	if len(all_tables) < 2:
		driver.quit()
		raise Exception(f"Could not find the game log table. Found {len(all_tables)} table(s), expected at least 2.")
	
	game_log_table = all_tables[1]  # Second table (0-indexed)

	# Find the game log table body within the second table
	table_body = game_log_table.find('div', {'itablebody': ''})
	if not table_body:
		driver.quit()
		raise Exception("Could not find the game log table body in the second table.")

	# Extract header information from the second table
	table_header = game_log_table.find('div', {'itableheader': ''})
	if not table_header:
		driver.quit()
		raise Exception("Could not find the table header in the second table.")
	
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
				value = link.text.strip()
			else:
				span = cell.find('span')
				value = span.text.strip() if span else cell.text.strip()
			
			game_data[header] = value
		
		if game_data:
			all_games.append(game_data)

	driver.quit()

	if not all_games:
		raise Exception("No game data found for this player.")

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
		'data': df.to_dict('records')
	}
	with open(cache_file, 'w') as f:
		json.dump(cache_data, f, indent=4)

	return df, False, player_name

def calculate_variability_stats(df):
	"""
	Calculates variability and consistency metrics for the player's game log.
	"""
	if df.empty or 'FPts' not in df.columns:
		return {}
	
	fpts = df['FPts'].dropna()
	
	if len(fpts) == 0:
		return {}
	
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

def bulk_scrape_all_players(league_id, username, password, player_dict=None, progress_callback=None):
	"""
	Scrapes game logs for all players in the provided dictionary.
	Returns a summary of successes and failures.
	"""
	if player_dict is None:
		player_dict = get_available_players_from_csv()
	
	if not player_dict:
		return {"error": "No players found to scrape"}
	
	# Initialize driver once for all players
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
				# Use the existing scraping logic but with shared driver
				player_url = f"https://www.fantrax.com/player/{player_code}/{league_id}"
				driver.get(player_url)
				time.sleep(2)  # Shorter wait for bulk operations
				
				page_source = driver.page_source
				soup = BeautifulSoup(page_source, 'html.parser')
				
				# Find the game log table
				all_tables = soup.find_all('div', class_='i-table minimal-scrollbar i-table--standard i-table--outside-sticky')
				if len(all_tables) < 2:
					failed_players.append((player_name, "Table not found"))
					fail_count += 1
					continue
				
				game_log_table = all_tables[1]
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
							value = link.text.strip()
						else:
							span = cell.find('span')
							value = span.text.strip() if span else cell.text.strip()
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
				
				# Save to cache
				CACHE_DIR.mkdir(parents=True, exist_ok=True)
				cache_file = CACHE_DIR / f"player_game_log_{player_code}_{league_id}.json"
				cache_data = {
					'player_name': player_name,
					'player_code': player_code,
					'league_id': league_id,
					'data': df.to_dict('records')
				}
				with open(cache_file, 'w') as f:
					json.dump(cache_data, f, indent=4)
				
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
