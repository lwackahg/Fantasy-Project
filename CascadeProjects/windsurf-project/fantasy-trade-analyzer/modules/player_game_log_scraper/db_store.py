import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "player_game_log_cache.db"


def _get_connection() -> sqlite3.Connection:
	"""Return a new SQLite connection to the player game log cache DB."""
	DB_PATH.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(DB_PATH)
	return conn


def init_schema() -> None:
	"""Initialize DB schema for player game log cache (idempotent)."""
	with _get_connection() as conn:
		conn.execute("PRAGMA foreign_keys = ON;")
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS player_seasons (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				league_id TEXT NOT NULL,
				player_code TEXT NOT NULL,
				player_name TEXT NOT NULL,
				season TEXT NOT NULL,
				status TEXT NOT NULL,
				games INTEGER NOT NULL,
				created_at TEXT NOT NULL,
				updated_at TEXT NOT NULL,
				UNIQUE(league_id, player_code, season)
			);
			"""
		)
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS player_game_logs (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				player_season_id INTEGER NOT NULL,
				game_index INTEGER NOT NULL,
				game_date TEXT,
				opponent TEXT,
				home_away TEXT,
				fpts REAL,
				raw_json TEXT NOT NULL,
				FOREIGN KEY(player_season_id) REFERENCES player_seasons(id) ON DELETE CASCADE
			);
			"""
		)
		
		conn.execute(
			"""
			CREATE INDEX IF NOT EXISTS idx_player_seasons_player
			ON player_seasons(league_id, player_code);
			"""
		)
		conn.execute(
			"""
			CREATE INDEX IF NOT EXISTS idx_player_seasons_season
			ON player_seasons(league_id, season);
			"""
		)
		conn.execute(
			"""
			CREATE INDEX IF NOT EXISTS idx_player_game_logs_season
			ON player_game_logs(player_season_id);
			"""
		)


def store_player_season(
	player_code: str,
	player_name: str,
	league_id: str,
	season: str,
	status: str,
	game_log_records: List[dict],
) -> None:
	"""Insert or update a player-season and its game logs in the DB."""
	init_schema()
	games = len(game_log_records or [])
	now = datetime.utcnow().isoformat()

	with _get_connection() as conn:
		conn.execute("PRAGMA foreign_keys = ON;")
		cur = conn.cursor()
		cur.execute(
			"""
			INSERT INTO player_seasons (
				league_id, player_code, player_name, season, status, games, created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(league_id, player_code, season) DO UPDATE SET
				player_name=excluded.player_name,
				status=excluded.status,
				games=excluded.games,
				updated_at=excluded.updated_at;
			""",
			(league_id, player_code, player_name, season, status, games, now, now),
		)

		cur.execute(
			"""
			SELECT id FROM player_seasons
			WHERE league_id = ? AND player_code = ? AND season = ?;
			""",
			(league_id, player_code, season),
		)
		row = cur.fetchone()
		if row is None:
			return
		player_season_id = row[0]

		cur.execute(
			"DELETE FROM player_game_logs WHERE player_season_id = ?;",
			(player_season_id,),
		)

		for idx, rec in enumerate(game_log_records or []):
			try:
				fpts_val = rec.get("FPts")
			except AttributeError:
				fpts_val = None

			try:
				fpts = float(fpts_val) if fpts_val is not None else None
			except (TypeError, ValueError):
				fpts = None

			raw_json = json.dumps(rec, ensure_ascii=False)

			cur.execute(
				"""
				INSERT INTO player_game_logs (
					player_season_id, game_index, game_date, opponent, home_away, fpts, raw_json
				) VALUES (?, ?, ?, ?, ?, ?, ?);
				""",
				(player_season_id, idx, None, None, None, fpts, raw_json),
			)


def load_player_season(
	player_code: str,
	league_id: str,
	season: str,
) -> Optional[Tuple[List[dict], str, str]]:
	"""Load game logs for a player-season from DB.

	Returns (records, status, player_name) or None if not found.
	"""
	init_schema()
	with _get_connection() as conn:
		conn.execute("PRAGMA foreign_keys = ON;")
		cur = conn.cursor()
		cur.execute(
			"""
			SELECT id, player_name, status
			FROM player_seasons
			WHERE league_id = ? AND player_code = ? AND season = ?;
			""",
			(league_id, player_code, season),
		)
		row = cur.fetchone()
		if row is None:
			return None

		player_season_id, player_name, status = row
		cur.execute(
			"""
			SELECT raw_json
			FROM player_game_logs
			WHERE player_season_id = ?
			ORDER BY game_index ASC;
			""",
			(player_season_id,),
		)
		records: List[dict] = []
		for (raw_json,) in cur.fetchall():
			try:
				records.append(json.loads(raw_json))
			except Exception:
				continue

		return records, status, player_name


def import_json_cache_directory(cache_dir: Path) -> int:
	"""Import existing JSON cache files into the DB.

	Returns number of player-season records successfully imported.
	"""
	if not cache_dir.exists():
		return 0

	imported = 0
	for cache_file in cache_dir.glob("player_game_log_full_*.json"):
		try:
			with cache_file.open("r") as f:
				cache_data = json.load(f)
		except Exception:
			continue

		player_code = cache_data.get("player_code")
		player_name = cache_data.get("player_name", "Unknown")
		league_id = cache_data.get("league_id")
		season = cache_data.get("season")
		if not player_code or not league_id or not season:
			continue

		data = cache_data.get("data", cache_data.get("game_log", [])) or []
		status = cache_data.get("status", "success")

		try:
			store_player_season(
				player_code=player_code,
				player_name=player_name,
				league_id=league_id,
				season=season,
				status=status,
				game_log_records=data,
			)
		except Exception:
			continue

		imported += 1

	return imported


def get_league_player_seasons(league_id: str) -> List[dict]:
	"""Return metadata for all player seasons in a league.

	Each dict contains: player_code, player_name, season, status, games.
	"""
	init_schema()
	with _get_connection() as conn:
		conn.execute("PRAGMA foreign_keys = ON;")
		cur = conn.cursor()
		cur.execute(
			"""
			SELECT player_code, player_name, season, status, games
			FROM player_seasons
			WHERE league_id = ?;
			""",
			(league_id,),
		)
		rows = cur.fetchall()

	result: List[dict] = []
	for player_code, player_name, season, status, games in rows:
		result.append(
			{
				"player_code": player_code,
				"player_name": player_name,
				"season": season,
				"status": status,
				"games": int(games or 0),
			}
		)

	return result


def get_league_available_seasons(league_id: str) -> List[str]:
	"""Return all distinct seasons for a league, newest first."""
	init_schema()
	with _get_connection() as conn:
		conn.execute("PRAGMA foreign_keys = ON;")
		cur = conn.cursor()
		cur.execute(
			"""
			SELECT DISTINCT season
			FROM player_seasons
			WHERE league_id = ?;
			""",
			(league_id,),
		)
		seasons = [row[0] for row in cur.fetchall() if row and row[0]]

	# Sort in descending order (assumes season strings like "2025-26")
	return sorted(seasons, reverse=True)


def get_league_season_player_logs(league_id: str, season: str) -> List[tuple]:
	"""Return all player logs for a league + season from the DB.

	Returns a list of tuples: (player_code, player_name, records[list[dict]]).
	Only seasons with status='success' are included.
	"""
	init_schema()
	with _get_connection() as conn:
		conn.execute("PRAGMA foreign_keys = ON;")
		cur = conn.cursor()
		cur.execute(
			"""
			SELECT ps.player_code, ps.player_name, pgl.raw_json
			FROM player_seasons ps
			JOIN player_game_logs pgl ON pgl.player_season_id = ps.id
			WHERE ps.league_id = ? AND ps.season = ? AND ps.status = 'success'
			ORDER BY ps.player_code, pgl.game_index;
			""",
			(league_id, season),
		)
		rows = cur.fetchall()

	by_player: dict = {}
	for player_code, player_name, raw_json in rows:
		entry = by_player.setdefault(
			player_code,
			{"player_name": player_name, "records": []},
		)
		try:
			rec = json.loads(raw_json)
		except Exception:
			continue
		entry["records"].append(rec)

	result: List[tuple] = []
	for player_code, data in by_player.items():
		result.append((player_code, data["player_name"], data["records"]))

	return result


def get_league_last_updated(league_id: str) -> Optional[str]:
	"""Return the most recent updated_at ISO timestamp for a league, or None."""
	init_schema()
	with _get_connection() as conn:
		conn.execute("PRAGMA foreign_keys = ON;")
		cur = conn.cursor()
		cur.execute(
			"""
			SELECT MAX(updated_at)
			FROM player_seasons
			WHERE league_id = ?;
			""",
			(league_id,),
		)
		row = cur.fetchone()
		if row and row[0]:
			return row[0]
	return None
