import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Always load env from fantrax.env in this script's directory
load_dotenv(os.path.join(os.path.dirname(__file__), 'fantrax.env'))
FANTRAX_USERNAME = os.getenv('FANTRAX_USERNAME')
FANTRAX_PASSWORD = os.getenv('FANTRAX_PASSWORD')
DOWNLOAD_DIR = os.getenv('FANTRAX_DOWNLOAD_DIR', os.getcwd())

FANTRAX_LOGIN_URL = 'https://www.fantrax.com/login'


def get_chrome_driver(download_dir):
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

	# Enable downloads in headless mode
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
	driver.get(FANTRAX_LOGIN_URL)
	time.sleep(2)
	username_box = driver.find_element(By.ID, 'mat-input-0')
	password_box = driver.find_element(By.ID, 'mat-input-1')
	username_box.send_keys(username)
	password_box.send_keys(password)
	password_box.send_keys(Keys.RETURN)
	time.sleep(5)  # Wait for login to complete
	if 'login' in driver.current_url:
		raise Exception('Login failed. Check credentials.')


from datetime import datetime

def download_players_csv(driver, start_date=None, end_date=None, league_id=None):
	"""
	Logs in with Selenium, then downloads the Fantrax CSV using requests and session cookies.
	start_date, end_date: strings in YYYY-MM-DD format
	league_id: Fantrax league ID string (required)
	"""
	DEFAULT_START = "2024-10-22"
	DEFAULT_END = "2025-04-13"
	if not start_date:
		start_date = DEFAULT_START
	if not end_date:
		end_date = DEFAULT_END
	if not league_id:
		raise ValueError('league_id must be provided to download_players_csv')

	if start_date == DEFAULT_START and end_date == DEFAULT_END:
		season_proj = "SEASON_41j_YEAR_TO_DATE"
		timeframe = "YEAR_TO_DATE"
	else:
		season_proj = "SEASON_41j_BY_DATE"
		timeframe = "BY_DATE"

	csv_url = (
		f"https://www.fantrax.com/fxpa/downloadPlayerStats?"
		f"leagueId={league_id}&pageNumber=1&statusOrTeamFilter=ALL"
		f"&seasonOrProjection={season_proj}&timeframeTypeCode={timeframe}"
		f"&startDate={start_date}&endDate={end_date}"
		"&view=STATS&positionOrGroup=BASKETBALL_PLAYER"
		"&transactionPeriod=22&miscDisplayType=1&sortType=SCORE&maxResultsPerPage=50"
		"&scoringCategoryType=5&timeStartType=PERIOD_ONLY"
		"&schedulePageAdj=0&searchName=&datePlaying=ALL&"
	)

	# Calculate days in range (inclusive)
	try:
		start_dt = datetime.strptime(start_date, "%Y-%m-%d")
		end_dt = datetime.strptime(end_date, "%Y-%m-%d")
		days = (end_dt - start_dt).days + 1
		if days < 1:
			days = 1
		datestr = f"{start_date}_to_{end_date}"
	except Exception:
		days = "NA"
		datestr = f"{start_date}_to_{end_date}"

	# League ID to name mapping from environment
	def get_league_name_map():
		ids = os.environ.get("FANTRAX_LEAGUE_IDS", "")
		names = os.environ.get("FANTRAX_LEAGUE_NAMES", "")
		id_list = [i.strip() for i in ids.split(",") if i.strip()]
		name_list = [n.strip() for n in names.split(",") if n.strip()]
		return dict(zip(id_list, name_list))

	def sanitize_name(name):
		import re
		return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().replace(" ", "_"))

	LEAGUE_NAME_MAP = get_league_name_map()
	league_name = LEAGUE_NAME_MAP.get(league_id, league_id)
	league_name = sanitize_name(league_name)

	# Determine file name
	if start_date == DEFAULT_START and end_date == DEFAULT_END:
		csv_filename = f"Fantrax-Players-{league_name}-(YTD).csv"
	else:
		csv_filename = f"Fantrax-Players-{league_name}-{datestr}-({days}).csv"
	download_path = os.path.join(DOWNLOAD_DIR, csv_filename)

	# Extract cookies from Selenium
	cookies = driver.get_cookies()
	session = requests.Session()
	for cookie in cookies:
		session.cookies.set(cookie['name'], cookie['value'])

	print(f'Downloading CSV for {start_date} to {end_date}...')
	resp = session.get(csv_url)
	if resp.status_code == 200 and resp.content:
		with open(download_path, 'wb') as f:
			f.write(resp.content)
		print(f'Success: Downloaded {csv_filename} to {DOWNLOAD_DIR}')
	else:
		print(f'Error: Failed to download CSV (HTTP {resp.status_code})')


def main():
	if not FANTRAX_USERNAME or not FANTRAX_PASSWORD:
		print('Set FANTRAX_USERNAME and FANTRAX_PASSWORD as environment variables.')
		return
	print('Starting Fantrax CSV downloader...')
	driver = get_chrome_driver(DOWNLOAD_DIR)
	try:
		login_to_fantrax(driver, FANTRAX_USERNAME, FANTRAX_PASSWORD)

		from datetime import datetime, timedelta
		DEFAULT_START = "2024-10-22"
		DEFAULT_END = "2025-04-13"
		league_id = os.environ.get("FANTRAX_DEFAULT_LEAGUE_ID")
		if not league_id:
			print('Set FANTRAX_DEFAULT_LEAGUE_ID in your environment.')
			return

		# Get today's date (for date math)
		today = datetime.now().date()
		end_date = DEFAULT_END
		if today < datetime.strptime(DEFAULT_END, "%Y-%m-%d").date():
			end_date = today.strftime("%Y-%m-%d")

		# YTD
		download_players_csv(driver, DEFAULT_START, DEFAULT_END, league_id)
		# 60 days
		start_60 = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=59)).strftime("%Y-%m-%d")
		download_players_csv(driver, start_60, end_date, league_id)
		# 30 days
		start_30 = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=29)).strftime("%Y-%m-%d")
		download_players_csv(driver, start_30, end_date, league_id)
		# 14 days
		start_14 = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=13)).strftime("%Y-%m-%d")
		download_players_csv(driver, start_14, end_date, league_id)
		# 7 days
		start_7 = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=6)).strftime("%Y-%m-%d")
		download_players_csv(driver, start_7, end_date, league_id)

		print('Download attempted. Check your download directory.')
	except Exception as e:
		print(f'Error: {e}')
	finally:
		driver.quit()


if __name__ == '__main__':
	main()
