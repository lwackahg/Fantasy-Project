import os
import time
import json
import pandas as pd
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv, find_dotenv

# --- CONFIGURATION ---
load_dotenv(find_dotenv('fantrax.env'))

FANTRAX_USERNAME = os.getenv('FANTRAX_USERNAME')
FANTRAX_PASSWORD = os.getenv('FANTRAX_PASSWORD')
FANTRAX_LOGIN_URL = 'https://www.fantrax.com/login'
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'weekly_standings_cache')

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

def get_cached_standings_and_min_games(league_id, period):
    """Loads the cached weekly standings and the min_games value from a single JSON file."""
    cache_file = os.path.join(CACHE_DIR, f"weekly_standings_{league_id}_{period}.json")

    if not os.path.exists(cache_file):
        return None, None

    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            df = pd.DataFrame.from_records(cache_data['data'])
            min_games = cache_data.get('metadata', {}).get('min_games', 35)
            return df, min_games
    except (json.JSONDecodeError, KeyError):
        # Handle corrupt or old format files gracefully
        return None, None

def calculate_adjusted_scores(df, min_games):
    """
    Calculates the adjusted scores based on the games played limit.
    """
    if 'GP' not in df.columns or 'FPts' not in df.columns:
        raise ValueError("DataFrame must contain 'GP' and 'FPts' columns.")

    # Ensure 'Calc FP/G' exists
    if 'Calc FP/G' not in df.columns:
        df['Calc FP/G'] = df.apply(lambda row: row['FPts'] / row['GP'] if row['GP'] > 0 else 0, axis=1)

    df['Games Over'] = df['GP'].apply(lambda x: max(0, x - min_games))
    df['Adjustment'] = (df['Games Over'] * df['Calc FP/G']).round(2)
    df['Adjusted FPts'] = (df['FPts'] - df['Adjustment']).round(2)

    return df

def submit_adjustments_to_fantrax(league_id, period, username, password, adjustments_df):
    """Navigates to the Fantrax scoring adjustment page and submits the calculated adjustments."""
    driver = get_chrome_driver()
    try:
        login_to_fantrax(driver, username, password)
        
        adjustment_url = f"https://www.fantrax.com/newui/fantasy/scoringAdjustment.go?leagueId={league_id}"
        driver.get(adjustment_url)
        time.sleep(5)

        period_dropdown = Select(driver.find_element(By.NAME, 'period'))
        period_dropdown.select_by_value(str(period))
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        team_rows = soup.find('div', {'id': 'rosterContainer'}).find_all('tr')[1:]

        for row in team_rows:
            team_name_cell = row.find('td', class_='leftCol')
            input_cell = row.find('td', class_='rightCol')
            if not team_name_cell or not input_cell:
                continue

            team_name = team_name_cell.text.strip()
            input_tag = input_cell.find('input')
            if not input_tag:
                continue

            team_adjustment_row = adjustments_df[adjustments_df['team_name'] == team_name]
            if not team_adjustment_row.empty:
                # Adjustment should be a negative, rounded whole number
                adjustment_value = int(round(team_adjustment_row['Final Adjustment'].iloc[0], 0))
                
                input_element = driver.find_element(By.NAME, input_tag['name'])
                input_element.clear()
                input_element.send_keys(str(adjustment_value))

        submit_button = driver.find_element(By.XPATH, "//div[contains(@class, 'filterButton') and .//span[text()='Submit']]")
        submit_button.click()
        time.sleep(5)
        return True, "Adjustments submitted successfully!"

    except Exception as e:
        return False, f"An error occurred: {e}"
    finally:
        driver.quit()

def get_cached_periods(league_id):
    """Scans the cache directory for .json files and returns a list of available periods."""
    if not league_id or not os.path.exists(CACHE_DIR):
        return {}
    
    cached_periods = {}
    pattern = re.compile(f"weekly_standings_{re.escape(league_id)}_(\d+)\.json")
    
    for filename in os.listdir(CACHE_DIR):
        match = pattern.match(filename)
        if match:
            period = int(match.group(1))
            cached_periods[period] = period # Simple mapping, min_games is loaded later
            
    return cached_periods
