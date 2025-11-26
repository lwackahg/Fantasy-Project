import os
import time
import requests
import subprocess
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv, find_dotenv
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# Load .env from the project root
load_dotenv(find_dotenv('fantrax.env'))

FANTRAX_USERNAME = os.getenv('FANTRAX_USERNAME')
FANTRAX_PASSWORD = os.getenv('FANTRAX_PASSWORD')
DOWNLOAD_DIR = os.getenv('FANTRAX_DOWNLOAD_DIR', os.getcwd())
FANTRAX_LOGIN_URL = 'https://www.fantrax.com/login'
DEFAULT_START = "2025-10-21"  # Season start
DEFAULT_END = "2026-03-29"    # Regular season end

def kill_chromedriver_processes(also_chrome: bool = False) -> None:
	"""Best-effort kill of stray chromedriver (and optionally chrome) processes."""
	try:
		if os.name == 'nt':
			subprocess.run(['taskkill', '/F', '/IM', 'chromedriver.exe', '/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			if also_chrome:
				subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe', '/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		else:
			subprocess.run(['pkill', '-f', 'chromedriver'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			if also_chrome:
				subprocess.run(['pkill', '-f', 'chrome'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except Exception:
		pass

def get_chrome_driver(download_dir):
    """Initializes and returns a headless Chrome WebDriver."""
    chrome_options = Options()
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "safebrowsing.disable_download_protection": True,
        "profile.default_content_settings.popups": 0,
        "profile.default_content_setting_values.automatic_downloads": 1
    }
    chrome_options.add_experimental_option('prefs', prefs)
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)

    driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
    params = {
        'cmd': 'Page.setDownloadBehavior',
        'params': {
            'behavior': 'allow',
            'downloadPath': download_dir
        }
    }
    driver.execute("send_command", params)
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

def download_players_csv(driver, start_date=None, end_date=None, league_id=None):
    """
    Downloads player stats CSV for a given date range and league using proven logic.
    """
    if not start_date:
        start_date = DEFAULT_START
    if not end_date:
        end_date = DEFAULT_END
    if not league_id:
        raise ValueError('league_id must be provided')

    # Corrected logic from the original, working fantrax_downloader.py
    if start_date == DEFAULT_START and end_date == DEFAULT_END:
        season_proj = "SEASON_41l_YEAR_TO_DATE"
        timeframe = "YEAR_TO_DATE"
    else:
        season_proj = "SEASON_41l_BY_DATE"
        timeframe = "BY_DATE"

    csv_url = (
        f"https://www.fantrax.com/fxpa/downloadPlayerStats?"
        f"leagueId={league_id}&pageNumber=1&statusOrTeamFilter=ALL"
        f"&seasonOrProjection={season_proj}&timeframeTypeCode={timeframe}"
        f"&startDate={start_date}&endDate={end_date}"
        "&view=STATS&positionOrGroup=BASKETBALL_PLAYER"
        "&transactionPeriod=22&miscDisplayType=1&sortType=SCORE&maxResultsPerPage=500"
        "&scoringCategoryType=5&timeStartType=PERIOD_ONLY"
        "&schedulePageAdj=0&searchName=&datePlaying=ALL&"
    )

    # --- Filename Generation (using the improved logic) ---
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1
    except Exception:
        days = "NA"

    def get_league_name_map():
        ids = os.environ.get("FANTRAX_LEAGUE_IDS", "")
        names = os.environ.get("FANTRAX_LEAGUE_NAMES", "")
        return dict(zip([i.strip() for i in ids.split(",") if i.strip()], [n.strip() for n in names.split(",") if n.strip()]))

    def sanitize_name(name):
        import re
        return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().replace(" ", "_"))

    league_name = sanitize_name(get_league_name_map().get(league_id, league_id))
    range_name = 'YTD' if timeframe == 'YEAR_TO_DATE' else f"{days}"
    csv_filename = f"Fantrax-Players-{league_name}-({range_name}).csv"
    download_path = os.path.join(DOWNLOAD_DIR, csv_filename)

    # --- Download Logic ---
    try:
        # Wrap get_cookies in a timeout because a dead driver can hang for 120s (socket default)
        def get_cookies_safe():
            return driver.get_cookies()
            
        cookies_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = cookies_executor.submit(get_cookies_safe)
            cookies = future.result(timeout=5)
        except Exception as e:
            # If timeout or other error, re-raise to be caught below
            raise e
        finally:
            cookies_executor.shutdown(wait=False)
            
    except Exception as e:
        msg = f'{range_name}: Error: Chrome driver died (cookies) - {e}'
        print(msg)
        return False, msg, "", 0
    
    session = requests.Session()
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'])

    print(f'Downloading CSV for {start_date} to {end_date}...')
    
    # Use a thread with a hard wall-clock timeout to prevent slow-trickling responses
    # from blocking for minutes. The requests timeout only applies to connection/read gaps.
    HARD_TIMEOUT_SECONDS = 15
    
    def do_request():
        return session.get(csv_url, timeout=10)
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(do_request)
        resp = future.result(timeout=HARD_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        msg = f'{range_name}: Error: Download timed out after {HARD_TIMEOUT_SECONDS}s'
        print(msg)
        return False, msg, "", 0
    except requests.exceptions.RequestException as e:
        msg = f'{range_name}: Error: Exception while downloading CSV: {e}'
        print(msg)
        return False, msg, "", 0
    finally:
        # Important: wait=False prevents the main thread from blocking 
        # if the worker thread is still stuck on a slow request.
        executor.shutdown(wait=False)

    if resp.status_code == 200 and resp.content:
        with open(download_path, 'wb') as f:
            f.write(resp.content)
        print(f'Success: Downloaded {csv_filename} to {DOWNLOAD_DIR}')
        # Return path and actual day count (int or 'NA') for callers that need it
        return True, f'Success: Downloaded {csv_filename}', download_path, days if isinstance(days, int) else 0
    else:
        msg = f'Error: Failed to download CSV (HTTP {resp.status_code})'
        print(msg)
        return False, msg, "", 0

def download_all_ranges(league_id: str, progress_callback=None):
	"""Downloads player stats for all standard time ranges."""
	if not FANTRAX_USERNAME or not FANTRAX_PASSWORD:
		return ["Set FANTRAX_USERNAME and FANTRAX_PASSWORD in fantrax.env"]
	
	kill_chromedriver_processes(also_chrome=True)
	time.sleep(1)  # Give OS time to fully release killed processes
	messages = []
	driver = get_chrome_driver(DOWNLOAD_DIR)
	try:
		login_to_fantrax(driver, FANTRAX_USERNAME, FANTRAX_PASSWORD)
		today = datetime.now().date()
		season_start = datetime.strptime(DEFAULT_START, "%Y-%m-%d").date()
		end_date = today.strftime("%Y-%m-%d") if today < datetime.strptime(DEFAULT_END, "%Y-%m-%d").date() else DEFAULT_END

		# Calculate start dates, but never go before season start
		def get_start_date(days_back: int) -> str:
			calculated_start = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days_back - 1)
			return max(calculated_start.date(), season_start).strftime("%Y-%m-%d")

		ranges = [
			('YTD', (DEFAULT_START, DEFAULT_END, 0)),
			('60 days', (get_start_date(60), end_date, 60)),
			('30 days', (get_start_date(30), end_date, 30)),
			('14 days', (get_start_date(14), end_date, 14)),
			('7 days', (get_start_date(7), end_date, 7)),
		]

		# Helper to compute league name for constructing fallback filenames
		def get_league_name_map():
			ids = os.environ.get("FANTRAX_LEAGUE_IDS", "")
			names = os.environ.get("FANTRAX_LEAGUE_NAMES", "")
			return dict(zip([i.strip() for i in ids.split(",") if i.strip()], [n.strip() for n in names.split(",") if n.strip()]))

		def sanitize_name(name: str) -> str:
			import re
			return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().replace(" ", "_"))

		league_name = sanitize_name(get_league_name_map().get(league_id, league_id))

		import shutil
		best_non_ytd_days = 0
		best_non_ytd_path = ""

		total_ranges = len(ranges)
		max_attempts = 2
		for i, (name, (start, end, requested_days)) in enumerate(ranges):
			for attempt in range(1, max_attempts + 1):
				if progress_callback:
					progress_label = f"Downloading {name} (attempt {attempt}/{max_attempts})"
					progress_callback(i / total_ranges, progress_label)
				success, msg, path, actual_days = download_players_csv(driver, start, end, league_id)
				
				# Self-healing: If driver died, restart it and retry immediately
				if not success and "Chrome driver died" in msg:
					messages.append(f"{name} (attempt {attempt}): Driver died, restarting...")
					try:
						try:
							driver.quit()
						except:
							pass
						kill_chromedriver_processes(also_chrome=True)
						time.sleep(1)
						driver = get_chrome_driver(DOWNLOAD_DIR)
						login_to_fantrax(driver, FANTRAX_USERNAME, FANTRAX_PASSWORD)
					except Exception as e:
						messages.append(f"{name}: Restart failed ({e}). Skipping retry.")
						continue

					# Retry this attempt counter (decrement loop var? no, just let the loop continue to next attempt)
					# Ideally we'd retry this exact attempt, but moving to the next attempt count is fine.
					# If this was the last attempt, we might want one extra bonus attempt?
					if attempt == max_attempts:
						# Give it one bonus attempt since this wasn't a download failure but a crash
						success, msg, path, actual_days = download_players_csv(driver, start, end, league_id)
						messages.append(f"{name} (bonus attempt): {msg}")

				else:
					messages.append(f"{name} (attempt {attempt}): {msg}")
				
				if success:
					break
				if attempt < max_attempts:
					time.sleep(1)

			if not success:
				continue

			if name != 'YTD' and actual_days > best_non_ytd_days:
				best_non_ytd_days = actual_days
				best_non_ytd_path = path
		
			if name in ('60 days', '30 days', '14 days', '7 days') and isinstance(actual_days, int) and actual_days < requested_days:
				if best_non_ytd_path:
					fallback_filename = f"Fantrax-Players-{league_name}-({requested_days}).csv"
					fallback_path = os.path.join(DOWNLOAD_DIR, fallback_filename)
					try:
						shutil.copyfile(best_non_ytd_path, fallback_path)
						messages.append(f"Filled {requested_days}-day file using most complete range so far ({best_non_ytd_days} days)")
					except Exception as e:
						messages.append(f"Failed to create fallback file for {requested_days} days: {e}")

			time.sleep(1)  # Add a delay to avoid overwhelming the server
		
		if progress_callback:
			progress_callback(1.0, "All downloads complete!")

	except Exception as e:
		messages.append(f'Error: {e}')
	finally:
		driver.quit()
		kill_chromedriver_processes(also_chrome=True)
	
	return messages
