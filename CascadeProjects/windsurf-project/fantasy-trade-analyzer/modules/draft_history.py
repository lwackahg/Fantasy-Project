from pathlib import Path
from typing import List

import pandas as pd

from modules.team_mappings import TEAM_ALIASES


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _normalize_team_name(raw: str) -> str:
	"""Normalize historical fantasy team names using TEAM_ALIASES.

	If no alias exists, returns the original trimmed name.
	"""
	if raw is None:
		return ""
	name = str(raw).strip()
	if not name:
		return ""
	return TEAM_ALIASES.get(name, name)


def _load_single_draft(csv_path: Path, season_key: str) -> pd.DataFrame:
	"""Load a single season's draft CSV into a normalized DataFrame.

	Expected minimal columns (case-insensitive):
	- Player / Player Name
	- Team or Fantasy Team (owner name)
	Optional columns are kept when present: Bid, Pick, Pos, Time (EST).
	"""
	if not csv_path.exists():
		return pd.DataFrame()

	df = pd.read_csv(csv_path)
	if df.empty:
		return pd.DataFrame()

	cols = {c.lower(): c for c in df.columns}

	player_col = cols.get("player") or cols.get("player name")
	team_col = cols.get("fantasy team") or cols.get("team")
	if not player_col or not team_col:
		return pd.DataFrame()

	out = pd.DataFrame()
	out["SeasonKey"] = [season_key] * len(df)
	out["Player"] = df[player_col].astype(str).str.strip()
	out["FantasyTeamRaw"] = df[team_col].astype(str).str.strip()

	# Optional numeric bid/price field
	bid_col = None
	for key in ("bid", "price", "salary", "winning bid"):
		if key in cols:
			bid_col = cols[key]
			break
	if bid_col:
		out["Bid"] = pd.to_numeric(df[bid_col], errors="coerce")
	else:
		out["Bid"] = pd.NA

	# Optional metadata columns we carry through if present
	for key, name in (
		("pick", "Pick"),
		("pos", "Pos"),
		("time (est)", "Time (EST)"),
	):
		if key in cols:
			out[name] = df[cols[key]]

	# Canonical owner name using aliases
	out["FantasyTeamCanonical"] = out["FantasyTeamRaw"].map(_normalize_team_name)

	return out


def load_draft_history(include_latest_fantrax: bool = True) -> pd.DataFrame:
	"""Load unified draft history across S1–S4 and the latest Fantrax draft (S5).

	Returns a DataFrame with at least:
	- SeasonKey: "S1".."S5"
	- Player: player display name
	- FantasyTeamRaw: original fantasy team name in that season
	- FantasyTeamCanonical: normalized owner name using TEAM_ALIASES
	- Bid: numeric bid where available (NaN otherwise)
	- Optional: Pick, Pos, Time (EST)
	"""
	seasons: List[tuple[str, Path]] = [
		("S1", DATA_DIR / "S1Draft.csv"),
		("S2", DATA_DIR / "S2Draft.csv"),
		("S3", DATA_DIR / "S3Draft.csv"),
		("S4", DATA_DIR / "S4Draft.csv"),
	]

	frames: List[pd.DataFrame] = []
	for key, path in seasons:
		frame = _load_single_draft(path, key)
		if not frame.empty:
			frames.append(frame)

	if include_latest_fantrax:
		# Treat the most recent Fantrax draft results for this repo as S5.
		candidates = sorted(
			DATA_DIR.glob("Fantrax-Draft-Results-*.csv"),
			key=lambda p: p.stat().st_mtime,
			reverse=True,
		)
		if candidates:
			latest = candidates[0]
			frame = _load_single_draft(latest, "S5")
			if not frame.empty:
				frames.append(frame)

	if not frames:
		return pd.DataFrame()

	combined = pd.concat(frames, ignore_index=True)
	return combined
