import os
import time
import json
import pandas as pd
import concurrent.futures
import subprocess
from typing import Optional, List
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('fantrax.env'))

FANTRAX_USERNAME = os.getenv('FANTRAX_USERNAME')
FANTRAX_PASSWORD = os.getenv('FANTRAX_PASSWORD')
FANTRAX_LOGIN_URL = 'https://www.fantrax.com/login'
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'league_standings_cache')
STOP_FLAG_PATH = os.path.join(CACHE_DIR, '_STOP')
EXPECTED_OUTPUT_COLUMNS = ['league_id', 'league_name', 'team_name', 'W', 'L', 'T', 'FPtsF', 'FPtsA']
PLAYOFFS_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'league_playoffs_cache')
EXPECTED_PLAYOFFS_COLUMNS = [
    'league_id',
    'league_name',
    'round',
    'date_range',
    'away_team',
    'away_total',
    'home_team',
    'home_total',
]

def get_league_id_to_name_map() -> dict:
    try:
        from league_config import LEAGUE_ID_TO_NAME
        return dict(LEAGUE_ID_TO_NAME)
    except Exception:
        pass

    ids_raw = os.environ.get('FANTRAX_LEAGUE_IDS', '')
    names_raw = os.environ.get('FANTRAX_LEAGUE_NAMES', '')
    ids = [i.strip() for i in ids_raw.split(',') if i.strip()]
    names = [n.strip() for n in names_raw.split(',') if n.strip()]
    return dict(zip(ids, names))

def get_all_league_ids() -> List[str]:
    try:
        from league_config import FANTRAX_LEAGUE_IDS
        return list(FANTRAX_LEAGUE_IDS)
    except Exception:
        pass

    ids_raw = os.environ.get('FANTRAX_LEAGUE_IDS', '')
    return [i.strip() for i in ids_raw.split(',') if i.strip()]

def get_chrome_driver():
    options = webdriver.ChromeOptions()
    #options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--log-level=3')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.set_page_load_timeout(25)
    driver.set_script_timeout(25)

    return driver

def login_to_fantrax(driver, username, password):
    def get_login():
        driver.get(FANTRAX_LOGIN_URL)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(get_login).result(timeout=25)

    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, 'mat-input-0'))
    )

    username_box = driver.find_element(By.ID, 'mat-input-0')
    password_box = driver.find_element(By.ID, 'mat-input-1')
    username_box.send_keys(username)
    password_box.send_keys(password)
    password_box.send_keys(Keys.RETURN)

    time.sleep(3)
    if 'login' in driver.current_url:
        time.sleep(2)
        if 'login' in driver.current_url:
            raise Exception('Login failed. Check credentials.')

def _safe_quit(driver):
    if not driver:
        return
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(driver.quit).result(timeout=5)
    except Exception:
        pass

def _kill_chromedriver_processes():
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'chromedriver.exe', '/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe', '/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass

def _normalize_header(h: str) -> str:
    return ''.join(ch for ch in str(h).strip() if ch.isalnum()).lower()

def _extract_ultimate_table_rows(page_source: str) -> pd.DataFrame:
    soup = BeautifulSoup(page_source, 'html.parser')
    ultimate_table = soup.find('ultimate-table')
    if not ultimate_table:
        raise Exception('Could not find the standings table on the page.')

    aside = ultimate_table.find('aside', class_='_ut__aside')
    if not aside:
        raise Exception('Could not find the team information section.')

    teams_data = []
    team_cells = aside.find_all('td', recursive=False)
    for cell in team_cells:
        link_tag = cell.find('a')
        team_name = link_tag.text.strip() if link_tag else 'N/A'
        teams_data.append({'team_name': team_name})

    content_div = ultimate_table.find('div', class_='_ut__content')
    if not content_div:
        raise Exception('Could not find the stats section.')

    stats_rows = content_div.find('table').find_all('tr')
    header_row = ultimate_table.find('header').find('tr', class_='_ut__head')
    headers = [th.text.strip() for th in header_row.find_all('th')]

    all_stats = []
    for row in stats_rows:
        stats = [td.text.strip().replace(',', '') for td in row.find_all('td')]
        if stats:
            all_stats.append(stats)

    teams_data = [t for t in teams_data if t['team_name'] != 'N/A']
    if len(teams_data) != len(all_stats):
        raise Exception(f'Mismatch between number of teams ({len(teams_data)}) and stats rows ({len(all_stats)}).')

    for i, team in enumerate(teams_data):
        for j, header in enumerate(headers):
            val = all_stats[i][j] if i < len(all_stats) and j < len(all_stats[i]) else ''
            try:
                team[header] = float(val)
            except Exception:
                team[header] = val

    df = pd.DataFrame(teams_data)
    return df

def _select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    header_map = {_normalize_header(c): c for c in df.columns}
    required = {
        'team': 'team_name',
        'w': 'W',
        'l': 'L',
        't': 'T',
        'fptsf': 'FPtsF',
        'fptsa': 'FPtsA'
    }

    out = pd.DataFrame()
    out['team_name'] = df['team_name'].astype(str).str.strip() if 'team_name' in df.columns else ''

    for norm_key, out_col in required.items():
        if norm_key == 'team':
            continue
        src = header_map.get(norm_key)
        if src is None:
            out[out_col] = pd.NA
            continue
        out[out_col] = pd.to_numeric(df[src], errors='coerce')

    return out

def get_league_standings(league_id: str, username: str, password: str, force_refresh: bool = False):
    if not username or not password:
        raise Exception('Fantrax username or password not found. Please set them in fantrax.env')

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'league_standings_{league_id}.json')

    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            df = pd.DataFrame.from_records(cache_data.get('data', []))
            return df, True
        except Exception:
            pass

    driver = None
    max_retries = 2
    retry_count = 0

    while retry_count <= max_retries:
        try:
            if not driver:
                driver = get_chrome_driver()
                login_to_fantrax(driver, username, password)

            standings_url = f'https://www.fantrax.com/fantasy/league/{league_id}/standings'

            def get_standings():
                driver.get(standings_url)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(get_standings).result(timeout=25)

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, 'ultimate-table'))
            )
            time.sleep(1)

            def get_source():
                return driver.page_source

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                page_source = executor.submit(get_source).result(timeout=10)
            raw_df = _extract_ultimate_table_rows(page_source)
            df = _select_required_columns(raw_df)

            cache_data = {
                'metadata': {
                    'league_id': league_id,
                    'timestamp': time.time()
                },
                'data': df.to_dict('records')
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=4)

            _safe_quit(driver)
            return df, False

        except Exception as e:
            retry_count += 1
            if driver:
                _safe_quit(driver)
                driver = None
            if retry_count == max_retries:
                _kill_chromedriver_processes()
            if retry_count > max_retries:
                raise e
            time.sleep(2)


def request_stop() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(STOP_FLAG_PATH, 'w') as f:
            f.write(str(time.time()))
    except Exception:
        pass


def clear_stop() -> None:
    try:
        if os.path.exists(STOP_FLAG_PATH):
            os.remove(STOP_FLAG_PATH)
    except Exception:
        pass


def is_stop_requested() -> bool:
    return os.path.exists(STOP_FLAG_PATH)


def _empty_output_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_OUTPUT_COLUMNS)


def load_cached_league_standings(league_id: str) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'league_standings_{league_id}.json')
    if not os.path.exists(cache_file):
        return pd.DataFrame(columns=['team_name', 'W', 'L', 'T', 'FPtsF', 'FPtsA'])
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        return pd.DataFrame.from_records(cache_data.get('data', []))
    except Exception:
        return pd.DataFrame(columns=['team_name', 'W', 'L', 'T', 'FPtsF', 'FPtsA'])


def load_cached_all_league_standings(league_ids: Optional[List[str]] = None) -> pd.DataFrame:
    if league_ids is None:
        league_ids = get_all_league_ids()
    if not league_ids:
        return _empty_output_df()

    id_to_name = get_league_id_to_name_map()
    rows = []
    for lid in league_ids:
        df = load_cached_league_standings(lid)
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, 'league_id', lid)
        tmp.insert(1, 'league_name', id_to_name.get(lid, lid))
        rows.append(tmp)

    if not rows:
        return _empty_output_df()

    out = pd.concat(rows, ignore_index=True)
    for c in EXPECTED_OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[EXPECTED_OUTPUT_COLUMNS]


def scrape_all_league_standings(username: str, password: str, league_ids: Optional[List[str]] = None) -> pd.DataFrame:
    if not username or not password:
        return _empty_output_df()
    if league_ids is None:
        league_ids = get_all_league_ids()
    if not league_ids:
        return _empty_output_df()

    os.makedirs(CACHE_DIR, exist_ok=True)
    clear_stop()

    driver = None
    try:
        driver = get_chrome_driver()
        login_to_fantrax(driver, username, password)

        id_to_name = get_league_id_to_name_map()
        rows = []
        for lid in league_ids:
            if is_stop_requested():
                break

            standings_url = f'https://www.fantrax.com/fantasy/league/{lid}/standings'

            def get_standings():
                driver.get(standings_url)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(get_standings).result(timeout=25)

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, 'ultimate-table'))
            )
            time.sleep(1)

            if is_stop_requested():
                break

            raw_df = _extract_ultimate_table_rows(driver.page_source)
            df = _select_required_columns(raw_df)

            cache_file = os.path.join(CACHE_DIR, f'league_standings_{lid}.json')
            cache_data = {
                'metadata': {
                    'league_id': lid,
                    'timestamp': time.time(),
                },
                'data': df.to_dict('records')
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=4)

            if df is None or df.empty:
                continue
            tmp = df.copy()
            tmp.insert(0, 'league_id', lid)
            tmp.insert(1, 'league_name', id_to_name.get(lid, lid))
            rows.append(tmp)

        if not rows:
            return _empty_output_df()

        out = pd.concat(rows, ignore_index=True)
        for c in EXPECTED_OUTPUT_COLUMNS:
            if c not in out.columns:
                out[c] = pd.NA
        return out[EXPECTED_OUTPUT_COLUMNS]
    finally:
        _safe_quit(driver)
        if is_stop_requested():
            _kill_chromedriver_processes()


def get_all_league_standings(username: str, password: str, league_ids: Optional[List[str]] = None, force_refresh: bool = False) -> pd.DataFrame:
    if force_refresh:
        return scrape_all_league_standings(username=username, password=password, league_ids=league_ids)
    return load_cached_all_league_standings(league_ids=league_ids)


def _empty_playoffs_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_PLAYOFFS_COLUMNS)


def _parse_playoffs_sections(page_source: str) -> pd.DataFrame:
    soup = BeautifulSoup(page_source, 'html.parser')
    rows = []

    def norm_space(s: str) -> str:
        return ' '.join(str(s or '').split())

    for tag in soup.find_all(['h2', 'h3', 'h4', 'div', 'span']):
        txt = norm_space(tag.get_text(' ', strip=True))
        if not txt.lower().startswith('playoffs - round'):
            continue

        round_label = txt
        date_range = ''
        nxt = tag.find_next(string=True)
        if nxt:
            maybe = norm_space(nxt)
            if maybe.startswith('(') and maybe.endswith(')'):
                date_range = maybe.strip('()').strip()

        table = tag.find_next('table')
        if not table:
            continue

        header_cells = [norm_space(th.get_text(' ', strip=True)).lower() for th in table.find_all('th')]
        idx_away = header_cells.index('away') if 'away' in header_cells else -1
        idx_home = header_cells.index('home') if 'home' in header_cells else -1
        idx_total = header_cells.index('total') if 'total' in header_cells else -1

        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if not tds:
                continue
            cells = [norm_space(td.get_text(' ', strip=True)) for td in tds]
            if idx_away < 0 or idx_home < 0:
                if len(cells) >= 2:
                    away_team = cells[0]
                    home_team = cells[-1]
                else:
                    continue
                away_total = ''
                home_total = ''
            else:
                away_team = cells[idx_away] if idx_away < len(cells) else ''
                home_team = cells[idx_home] if idx_home < len(cells) else ''
                away_total = ''
                home_total = ''
                if idx_total >= 0:
                    if idx_total < idx_home and idx_total < len(cells):
                        away_total = cells[idx_total]
                    if (idx_total + 1) < len(cells):
                        home_total = cells[idx_total + 1]

            if not away_team and not home_team:
                continue

            rows.append({
                'round': round_label,
                'date_range': date_range,
                'away_team': away_team,
                'away_total': away_total,
                'home_team': home_team,
                'home_total': home_total,
            })

    if not rows:
        return pd.DataFrame(columns=['round', 'date_range', 'away_team', 'away_total', 'home_team', 'home_total'])
    return pd.DataFrame(rows)


def load_cached_league_playoffs(league_id: str) -> pd.DataFrame:
    os.makedirs(PLAYOFFS_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(PLAYOFFS_CACHE_DIR, f'league_playoffs_{league_id}.json')
    if not os.path.exists(cache_file):
        return pd.DataFrame(columns=['round', 'date_range', 'away_team', 'away_total', 'home_team', 'home_total'])
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        return pd.DataFrame.from_records(cache_data.get('data', []))
    except Exception:
        return pd.DataFrame(columns=['round', 'date_range', 'away_team', 'away_total', 'home_team', 'home_total'])


def load_cached_all_league_playoffs(league_ids: Optional[List[str]] = None) -> pd.DataFrame:
    if league_ids is None:
        league_ids = get_all_league_ids()
    if not league_ids:
        return _empty_playoffs_df()

    id_to_name = get_league_id_to_name_map()
    rows = []
    for lid in league_ids:
        df = load_cached_league_playoffs(lid)
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, 'league_id', lid)
        tmp.insert(1, 'league_name', id_to_name.get(lid, lid))
        rows.append(tmp)
    if not rows:
        return _empty_playoffs_df()
    out = pd.concat(rows, ignore_index=True)
    for c in EXPECTED_PLAYOFFS_COLUMNS:
        if c not in out.columns:
            out[c] = ''
    return out[EXPECTED_PLAYOFFS_COLUMNS]


def scrape_all_league_playoffs(username: str, password: str, league_ids: Optional[List[str]] = None) -> pd.DataFrame:
    if not username or not password:
        return _empty_playoffs_df()
    if league_ids is None:
        league_ids = get_all_league_ids()
    if not league_ids:
        return _empty_playoffs_df()

    os.makedirs(PLAYOFFS_CACHE_DIR, exist_ok=True)
    clear_stop()

    driver = None
    try:
        driver = get_chrome_driver()
        login_to_fantrax(driver, username, password)

        id_to_name = get_league_id_to_name_map()
        rows = []
        for lid in league_ids:
            if is_stop_requested():
                break

            playoffs_url = f'https://www.fantrax.com/fantasy/league/{lid}/standings;view=PLAYOFFS'

            def get_playoffs():
                driver.get(playoffs_url)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(get_playoffs).result(timeout=25)

            time.sleep(1)
            if is_stop_requested():
                break

            df = _parse_playoffs_sections(driver.page_source)
            cache_file = os.path.join(PLAYOFFS_CACHE_DIR, f'league_playoffs_{lid}.json')
            cache_data = {
                'metadata': {
                    'league_id': lid,
                    'timestamp': time.time(),
                },
                'data': df.to_dict('records')
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=4)

            if df is None or df.empty:
                continue
            tmp = df.copy()
            tmp.insert(0, 'league_id', lid)
            tmp.insert(1, 'league_name', id_to_name.get(lid, lid))
            rows.append(tmp)

        if not rows:
            return _empty_playoffs_df()
        out = pd.concat(rows, ignore_index=True)
        for c in EXPECTED_PLAYOFFS_COLUMNS:
            if c not in out.columns:
                out[c] = ''
        return out[EXPECTED_PLAYOFFS_COLUMNS]
    finally:
        _safe_quit(driver)
        if is_stop_requested():
            _kill_chromedriver_processes()
