import os
import time
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv, find_dotenv

# --- CONFIGURATION ---
load_dotenv(find_dotenv('fantrax.env'))

FANTRAX_USERNAME = os.getenv('FANTRAX_USERNAME')
FANTRAX_PASSWORD = os.getenv('FANTRAX_PASSWORD')
FANTRAX_LOGIN_URL = 'https://www.fantrax.com/login'
FANTRAX_DEFAULT_LEAGUE_ID = os.getenv('FANTRAX_DEFAULT_LEAGUE_ID')
FANTRAX_LEAGUE_IDS = os.getenv('FANTRAX_LEAGUE_IDS', '')
FANTRAX_LEAGUE_NAMES = os.getenv('FANTRAX_LEAGUE_NAMES', '')
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'weekly_standings_cache')

def get_league_name_map():
    """Parses league IDs and names from env variables into a map."""
    ids = [id.strip() for id in FANTRAX_LEAGUE_IDS.split(',') if id.strip()]
    names = [name.strip() for name in FANTRAX_LEAGUE_NAMES.split(',') if name.strip()]
    # Pair names and IDs, padding with IDs if names run out
    league_map = {names[i] if i < len(names) else ids[i]: ids[i] for i in range(len(ids))}
    return league_map

def get_chrome_driver():
    """Initializes and returns a headless Chrome WebDriver with suppressed logging."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--log-level=3')  # Suppress logs except fatal ones
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Exclude logging switch
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

def clear_weekly_standings_cache():
    """Deletes all .json files from the weekly standings cache directory."""
    if not os.path.exists(CACHE_DIR):
        return "Cache directory does not exist.", False
    
    files_deleted = 0
    try:
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(CACHE_DIR, filename)
                os.remove(file_path)
                files_deleted += 1
        if files_deleted > 0:
            return f"Successfully deleted {files_deleted} file(s) from the cache.", True
        else:
            return "Cache is already empty.", True
    except Exception as e:
        return f"An error occurred while clearing the cache: {e}", False

def get_weekly_standings(league_id, period, username, password, min_games, force_refresh=False):
    """
    Scrapes or loads from cache the weekly standings data from Fantrax.
    Returns a tuple (DataFrame, from_cache_boolean).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"weekly_standings_{league_id}_{period}.json")

    if not force_refresh and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            df = pd.DataFrame.from_records(cache_data['data'])
            return df, True

    driver = get_chrome_driver()
    login_to_fantrax(driver, username, password)

    standings_url = (
        f"https://www.fantrax.com/fantasy/league/{league_id}/standings;"
        f"view=SEASON_STATS;timeframeType=BY_PERIOD;period={period}"
    )
    driver.get(standings_url)
    time.sleep(5)  # Wait for the page to load

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    ultimate_table = soup.find('ultimate-table')
    if not ultimate_table:
        raise Exception("Could not find the standings table on the page.")

    # --- Extract Team Information ---
    aside = ultimate_table.find('aside', class_='_ut__aside')
    if not aside:
        raise Exception("Could not find the team information section.")

    teams_data = []
    team_cells = aside.find_all('td', recursive=False)
    for cell in team_cells:
        rank_tag = cell.find('b')
        rank = rank_tag.text.strip() if rank_tag else 'N/A'

        link_tag = cell.find('a')
        if link_tag:
            team_name = link_tag.text.strip()
            href = link_tag.get('href', '')
            team_id = href.split('teamId=')[-1] if 'teamId=' in href else 'N/A'
        else:
            team_name = 'N/A'
            team_id = 'N/A'

        teams_data.append({'rank': rank, 'team_name': team_name, 'team_id': team_id})

    # --- Extract Stats ---
    content_div = ultimate_table.find('div', class_='_ut__content')
    if not content_div:
        raise Exception("Could not find the stats section.")

    stats_rows = content_div.find('table').find_all('tr')

    header_row = ultimate_table.find('header').find('tr', class_='_ut__head')
    headers = [th.text.strip() for th in header_row.find_all('th')]

    all_stats = []
    for row in stats_rows:
        stats = [td.text.strip().replace(',', '') for td in row.find_all('td')]
        if stats:
            all_stats.append(stats)

    # --- Combine Data ---
    if len(teams_data) != len(all_stats):
        # Fantrax sometimes includes empty rows, so we'll try to filter them out
        teams_data = [t for t in teams_data if t['team_name'] != 'N/A']
        if len(teams_data) != len(all_stats):
             raise Exception(f"Mismatch between number of teams ({len(teams_data)}) and stats rows ({len(all_stats)}).")

    for i, team in enumerate(teams_data):
        for j, header in enumerate(headers):
            try:
                team[header] = float(all_stats[i][j])
            except (ValueError, IndexError):
                team[header] = all_stats[i][j] if i < len(all_stats) and j < len(all_stats[i]) else 'N/A'

    df = pd.DataFrame(teams_data)
    
    # Save both data and metadata to a single JSON file
    cache_data = {
        'metadata': {'min_games': min_games},
        'data': df.to_dict('records')
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=4)

    return df, False

def calculate_adjusted_scores(df, min_games):
    """
    Calculates the adjusted scores based on the games played limit.
    """
    df['Games Over'] = df['GP'].apply(lambda x: max(0, x - min_games))
    df['Adjustment'] = df['Games Over'] * df['FPts'] / df['GP']
    df['Adjusted FPts'] = df['FPts'] - df['Adjustment']

    # Round calculated columns to 2 decimal points
    for col in ['Adjustment', 'Adjusted FPts']:
        if col in df.columns:
            df[col] = df[col].round(2)

    return df
