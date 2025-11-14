import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd

from modules.player_game_log_scraper.logic import (
    get_cache_directory,
    load_league_cache_index,
)


def _parse_game_dates_for_season(dates: pd.Series, season: str) -> pd.Series:
	"""Parse short-form dates like 'Apr 11' into datetimes using the season.

	For a season string like '2024-25', months >= October are mapped to the first
	year (2024) and months < October to the second year (2025). This makes
	comparisons against real trade dates meaningful while avoiding ambiguous
	parsing warnings.
	"""
	if dates.empty:
		return pd.to_datetime([])

	try:
		parts = str(season).split("-")
		start_year = int(parts[0])
		end_part = parts[1]
		if len(end_part) == 2:
			end_year = 2000 + int(end_part)
		else:
			end_year = int(end_part)
	except Exception:
		start_year = 2000
		end_year = 2001

	def _parse_one(value: Any) -> datetime | pd.Timestamp:
		text = str(value).strip()
		if not text:
			return pd.NaT
		try:
			md = datetime.strptime(text, "%b %d")
		except Exception:
			return pd.NaT
		month = md.month
		day = md.day
		year = start_year if month >= 10 else end_year
		return datetime(year, month, day)

	return dates.apply(_parse_one)


def _build_player_code_index(league_id: str) -> Tuple[Dict[str, str], Dict[str, Any], Path]:
    """Build a mapping from player name -> player_code using the league cache index.

    Returns (name_to_code, index, cache_dir).
    """
    index = load_league_cache_index(league_id, rebuild_if_missing=True)
    if not index or not index.get("players"):
        return {}, {}, Path(get_cache_directory())

    cache_dir = Path(get_cache_directory())
    name_to_code: Dict[str, str] = {}
    for player_code, pdata in index["players"].items():
        name = pdata.get("player_name")
        if not name:
            continue
        key = str(name).strip().lower()
        if not key:
            continue
        name_to_code[key] = player_code

    return name_to_code, index, cache_dir


def _load_player_games(
    player_name: str,
    league_id: str,
    season: str,
    name_to_code: Dict[str, str],
    index: Dict[str, Any],
    cache_dir: Path,
) -> pd.DataFrame:
    """Load a player's game log for a specific season from cache.

    Returns an empty DataFrame if not found.
    """
    key = str(player_name).strip().lower()
    if not key:
        return pd.DataFrame()
    code = name_to_code.get(key)
    if not code:
        return pd.DataFrame()

    season_meta = index.get("players", {}).get(code, {}).get("seasons", {})
    meta = season_meta.get(season)
    cache_path: Path

    if meta and meta.get("cache_file"):
        cache_path = cache_dir / meta["cache_file"]
    else:
        # Fallback to filename convention
        cache_path = cache_dir / f"player_game_log_full_{code}_{league_id}_{season.replace('-', '_')}.json"

    if not cache_path.exists():
        return pd.DataFrame()

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except Exception:
        return pd.DataFrame()

    game_log = cache_data.get("data", cache_data.get("game_log", []))
    if not game_log:
        return pd.DataFrame()

    df = pd.DataFrame(game_log)
    if "FPts" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame()

    # Parse dates and FPts
    df = df.copy()
    df["DateParsed"] = _parse_game_dates_for_season(df["Date"], season)
    df["FPts"] = pd.to_numeric(df["FPts"], errors="coerce")
    df = df.dropna(subset=["DateParsed", "FPts"])
    if df.empty:
        return pd.DataFrame()

    # Ensure most recent first
    df = df.sort_values("DateParsed", ascending=False)
    return df


def _window_stats(games: pd.DataFrame) -> Dict[str, float]:
    """Compute basic stats for a subset of games.

    Expects a DataFrame with a numeric 'FPts' column.
    """
    if games.empty or "FPts" not in games.columns:
        return {}

    fpts = games["FPts"].astype(float)
    gp = len(fpts)
    if gp == 0:
        return {}

    total_fpts = float(fpts.sum())
    mean_fpg = float(total_fpts / gp)
    return {
        "FP/G": mean_fpg,
        "FPts": total_fpts,
        "GP": float(gp),
        "Median": float(fpts.median()),
        "StdDev": float(fpts.std() if gp > 1 else 0.0),
    }


def build_historical_combined_data(
    trade_date: date,
    league_id: str,
    season: str,
    rosters_by_team: Dict[str, List[str]],
) -> pd.DataFrame:
    """Build a synthetic combined_data DataFrame as of trade_date using cached game logs.

    The resulting DataFrame is shaped like the normal combined_data used by TradeAnalyzer,
    with rows per (player, time range) and a Timestamp column in
    {"YTD", "60 Days", "30 Days", "14 Days", "7 Days"}.
    """
    name_to_code, index, cache_dir = _build_player_code_index(league_id)
    if not name_to_code or not index:
        return pd.DataFrame()

    trade_dt = trade_date if isinstance(trade_date, date) else trade_date.date()

    rows: List[Dict[str, Any]] = []
    for team_id, players in rosters_by_team.items():
        for player_name in players:
            player_name = player_name.strip()
            if not player_name:
                continue

            games = _load_player_games(player_name, league_id, season, name_to_code, index, cache_dir)
            if games.empty:
                continue

            # Filter to games on or before trade date
            games_filtered = games[games["DateParsed"] <= pd.Timestamp(trade_dt)].copy()
            if games_filtered.empty:
                continue

            # Build date-based windows
            windows = {
                "YTD": None,  # special case: all games up to date
                "60 Days": 60,
                "30 Days": 30,
                "14 Days": 14,
                "7 Days": 7,
            }

            for label, days in windows.items():
                if days is None:
                    window_df = games_filtered
                else:
                    cutoff = pd.Timestamp(trade_dt) - timedelta(days=days)
                    window_df = games_filtered[games_filtered["DateParsed"] >= cutoff]

                stats = _window_stats(window_df)
                if not stats:
                    continue

                rows.append(
                    {
                        "Player": player_name,
                        "Team": games_filtered.iloc[0].get("Team", ""),
                        "Status": team_id,
                        "FP/G": stats["FP/G"],
                        "FPts": stats["FPts"],
                        "GP": stats["GP"],
                        "Timestamp": label,
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Match the normal combined_data shape as closely as possible
    df = df[["Player", "Team", "Status", "FPts", "FP/G", "GP", "Timestamp"]]
    df = df.reset_index(drop=True)
    df.set_index("Player", inplace=True)
    return df
