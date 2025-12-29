import os
import time
import requests
import pandas as pd
import json
import concurrent.futures
import subprocess
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
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
    
    # Set strict timeouts to fail fast rather than hang for 120s
    driver.set_page_load_timeout(25)
    driver.set_script_timeout(25)
    
    return driver

def login_to_fantrax(driver, username, password):
    """Logs into Fantrax using the provided credentials."""
    # Use a concurrent timeout for driver.get to prevent hanging on zombie driver
    def get_login():
        driver.get(FANTRAX_LOGIN_URL)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        try:
            executor.submit(get_login).result(timeout=25)
        except concurrent.futures.TimeoutError:
            raise Exception("Timeout navigating to login page")

    # Wait for login fields instead of static sleep
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, 'mat-input-0'))
    )
    
    username_box = driver.find_element(By.ID, 'mat-input-0')
    password_box = driver.find_element(By.ID, 'mat-input-1')
    username_box.send_keys(username)
    password_box.send_keys(password)
    password_box.send_keys(Keys.RETURN)
    
    # Wait for login to complete by checking URL change or specific element
    time.sleep(3) # Brief wait for redirect to start
    if 'login' in driver.current_url:
        # Check if still on login page after a bit more time
        time.sleep(2)
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
    def kill_chromedriver_processes():
        """Kills any orphaned chromedriver and chrome processes."""
        try:
            # On Windows we use taskkill
            subprocess.run(['taskkill', '/F', '/IM', 'chromedriver.exe', '/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe', '/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
        except Exception:
            pass

    def safe_quit(d):
        """Attempts to quit the driver safely with a timeout."""
        if d:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(d.quit).result(timeout=5)
            except Exception:
                pass

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"weekly_standings_{league_id}_{period}.json")

    if not force_refresh and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            df = pd.DataFrame.from_records(cache_data['data'])
            return df, True

    driver = None
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            if not driver:
                driver = get_chrome_driver()
                login_to_fantrax(driver, username, password)

            standings_url = (
                f"https://www.fantrax.com/fantasy/league/{league_id}/standings;"
                f"view=SEASON_STATS;timeframeType=BY_PERIOD;period={period}"
            )
            def get_standings():
                driver.get(standings_url)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    executor.submit(get_standings).result(timeout=25)
                except concurrent.futures.TimeoutError:
                    raise Exception("Timeout navigating to standings page")
            
            # Wait for the standings table to be present
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, 'ultimate-table'))
            )
            
            # Small buffer for the table content to populate
            time.sleep(1)

            def get_source():
                return driver.page_source
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    page_source = executor.submit(get_source).result(timeout=10)
                except concurrent.futures.TimeoutError:
                    raise Exception("Timeout getting page source (likely zombie driver)")

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

            if driver:
                safe_quit(driver)
            return df, False

        except Exception as e:
            retry_count += 1
            print(f"Error scraping standings (attempt {retry_count}/{max_retries + 1}): {e}")
            if driver:
                safe_quit(driver)
                driver = None
            
            # If it's a persistent driver issue, try killing processes
            if retry_count == max_retries:
                kill_chromedriver_processes()
                
            if retry_count > max_retries:
                raise e
            time.sleep(2)

def calculate_adjusted_scores(df, min_games):
    """
    Calculates the adjusted scores based on the games played limit.
    """
    df['Games Over'] = df['GP'].apply(lambda x: max(0, x - min_games))
    df['Adjustment'] = df['Games Over'] * df['FPts'] / df['GP']
    df['Adjusted FPts'] = df['FPts'] - df['Adjustment']

    # Round calculated columns to 0 decimal points and convert to integer
    for col in ['Adjustment', 'Adjusted FPts']:
        if col in df.columns:
            df[col] = df[col].round(0).astype(int)

    return df
