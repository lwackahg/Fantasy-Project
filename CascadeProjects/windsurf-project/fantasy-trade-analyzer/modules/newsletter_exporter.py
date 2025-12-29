from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

from data_loader import load_schedule_data
from logic.schedule_analysis import calculate_team_stats
from modules.draft_history import load_draft_history
from modules.player_game_log_scraper.logic import _parse_game_date_for_season
from modules.trade_analysis.logic import _get_trade_history_path

try:
    from league_config import FANTRAX_DEFAULT_LEAGUE_ID
except ImportError:
    FANTRAX_DEFAULT_LEAGUE_ID = ""


CURRENT_SEASON = "2025-26"


def _df_to_records(df: pd.DataFrame, limit_rows: Optional[int] = None) -> List[Dict[str, Any]]:
	if df is None or not isinstance(df, pd.DataFrame) or df.empty:
		return []
	try:
		out = df.copy()
		if limit_rows is not None:
			out = out.head(max(0, int(limit_rows)))
		return out.to_dict("records")
	except Exception:
		return []


def build_newsletter_export_json(
	data_dir: Path,
	docs_dir: Path,
	include_player_game_logs: bool = False,
	include_past_seasons_logs: bool = False,
	include_weekly_player_scoring: bool = False,
	max_players_for_weekly_scoring: Optional[int] = None,
	limit_rows_per_table: Optional[int] = None,
) -> Tuple[bytes, str, Dict[str, Any]]:
	generated_at = datetime.utcnow().isoformat() + "Z"
	league_id = str(st.session_state.get("league_id") or FANTRAX_DEFAULT_LEAGUE_ID or "").strip()
	loaded_league_name = str(st.session_state.get("loaded_league_name") or "").strip()
	label = loaded_league_name or league_id or "league"
	filename = f"newsletter_payload_{_safe_filename_piece(label)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

	schedule_df = _try_get_schedule_df()
	standings_df = pd.DataFrame()
	try:
		if schedule_df is not None and not schedule_df.empty:
			standings_df = calculate_team_stats(schedule_df).reset_index().rename(columns={"index": "Team"})
	except Exception:
		standings_df = pd.DataFrame()

	combined = st.session_state.get("combined_data")
	data_ranges = st.session_state.get("data_ranges")

	draft_history_df = pd.DataFrame()
	try:
		draft_history_df = load_draft_history(include_latest_fantrax=True)
	except Exception:
		draft_history_df = pd.DataFrame()

	trade_history_path = None
	try:
		trade_history_path = _get_trade_history_path()
	except Exception:
		trade_history_path = None

	injury_overrides_path = data_dir / "injured_players.json"
	rules_doc_path = docs_dir / "deep_dive" / "902_League_Advanced_Strategy_and_Modeling.md"

	weekly_cache_dir = data_dir / "weekly_standings_cache"
	player_log_cache_dir = data_dir / "player_game_log_cache"

	manifest: Dict[str, Any] = {
		"generated_at": generated_at,
		"league_id": league_id,
		"loaded_league_name": loaded_league_name,
		"missing_or_not_available": {
			"transactions_log": True,
			"faab_budget": True,
			"ir_usage": True,
			"league_chat": True,
		},
	}

	payload: Dict[str, Any] = {
		"meta": manifest,
		"tables": {},
		"texts": {},
	}

	if isinstance(combined, pd.DataFrame) and not combined.empty:
		try:
			payload["tables"]["combined_player_data"] = _df_to_records(combined.reset_index(), limit_rows_per_table)
		except Exception:
			payload["tables"]["combined_player_data"] = []

	if isinstance(data_ranges, dict) and data_ranges:
		out_ranges: Dict[str, Any] = {}
		for key, df in data_ranges.items():
			if not isinstance(df, pd.DataFrame) or df.empty:
				continue
			out_ranges[str(key)] = _df_to_records(df.reset_index(drop=True), limit_rows_per_table)
		payload["tables"]["player_data_ranges"] = out_ranges

	if schedule_df is not None and not schedule_df.empty:
		payload["tables"]["schedule_and_results"] = _df_to_records(schedule_df, limit_rows_per_table)
	if standings_df is not None and not standings_df.empty:
		payload["tables"]["standings_from_schedule"] = _df_to_records(standings_df, limit_rows_per_table)

	# Weekly standings cache as raw JSON blobs
	weekly_blobs = []
	if weekly_cache_dir.exists():
		for fp in sorted(weekly_cache_dir.glob("*.json")):
			try:
				weekly_blobs.append(json.loads(fp.read_text(encoding="utf-8")))
			except UnicodeDecodeError:
				try:
					weekly_blobs.append(json.loads(fp.read_text(encoding="cp1252")))
				except Exception:
					continue
			except Exception:
				continue
	if weekly_blobs:
		payload["tables"]["weekly_standings_cache"] = weekly_blobs

	if draft_history_df is not None and not draft_history_df.empty:
		payload["tables"]["draft_history_all_seasons"] = _df_to_records(draft_history_df, limit_rows_per_table)

	# Manager IDs
	try:
		mgr_path = data_dir / "ManagerIDs.csv"
		if mgr_path.exists():
			mgr_df = pd.read_csv(mgr_path, encoding="utf-8")
			payload["tables"]["manager_ids"] = _df_to_records(mgr_df, limit_rows_per_table)
	except UnicodeDecodeError:
		try:
			mgr_df = pd.read_csv(mgr_path, encoding="cp1252")
			payload["tables"]["manager_ids"] = _df_to_records(mgr_df, limit_rows_per_table)
		except Exception:
			pass
	except Exception:
		pass

	# Injuries overrides
	if injury_overrides_path.exists():
		try:
			payload["tables"]["injured_players_overrides"] = json.loads(injury_overrides_path.read_text(encoding="utf-8"))
		except UnicodeDecodeError:
			try:
				payload["tables"]["injured_players_overrides"] = json.loads(injury_overrides_path.read_text(encoding="cp1252"))
			except Exception:
				pass
		except Exception:
			pass

	# Trades
	if trade_history_path is not None and isinstance(trade_history_path, Path) and trade_history_path.exists():
		try:
			payload["tables"]["trade_history"] = json.loads(trade_history_path.read_text(encoding="utf-8"))
		except UnicodeDecodeError:
			try:
				payload["tables"]["trade_history"] = json.loads(trade_history_path.read_text(encoding="cp1252"))
			except Exception:
				pass
		except Exception:
			pass

	# Rules
	if rules_doc_path.exists():
		txt = _read_text_file(rules_doc_path)
		if txt is not None:
			payload["texts"]["902_League_Advanced_Strategy_and_Modeling_md"] = txt

	# Optional player logs + availability indexes
	if include_player_game_logs and player_log_cache_dir.exists() and league_id:
		log_blobs = []
		try:
			if include_past_seasons_logs:
				pattern = f"player_game_log_full_*_{league_id}_*.json"
			else:
				pattern = f"player_game_log_full_*_{league_id}_{CURRENT_SEASON.replace('-', '_')}.json"
			for fp in sorted(player_log_cache_dir.glob(pattern)):
				try:
					log_blobs.append(json.loads(fp.read_text(encoding="utf-8")))
				except UnicodeDecodeError:
					try:
						log_blobs.append(json.loads(fp.read_text(encoding="cp1252")))
					except Exception:
						continue
				except Exception:
					continue
		except Exception:
			log_blobs = []
		if log_blobs:
			payload["tables"]["player_game_logs"] = log_blobs

		avail_blobs = []
		avail_pattern = (
			f"availability_index_{league_id}_*.json"
			if include_past_seasons_logs
			else f"availability_index_{league_id}_{CURRENT_SEASON.replace('-', '_')}.json"
		)
		for fp in sorted(player_log_cache_dir.glob(avail_pattern)):
			try:
				avail_blobs.append(json.loads(fp.read_text(encoding="utf-8")))
			except UnicodeDecodeError:
				try:
					avail_blobs.append(json.loads(fp.read_text(encoding="cp1252")))
				except Exception:
					continue
			except Exception:
				continue
		if avail_blobs:
			payload["tables"]["player_availability"] = avail_blobs

	# Optional weekly player scoring (best-effort)
	if include_weekly_player_scoring and league_id and player_log_cache_dir.exists():
		period_ranges = _period_ranges_from_schedule(schedule_df)
		player_map = _build_player_to_team_map()
		weekly_df = _weekly_player_scoring_breakdown(
			league_id=league_id,
			season=CURRENT_SEASON,
			period_ranges=period_ranges,
			player_team_map=player_map,
			cache_dir=player_log_cache_dir,
			max_players=max_players_for_weekly_scoring,
		)
		if weekly_df is not None and not weekly_df.empty:
			payload["tables"]["weekly_player_scoring_breakdown"] = _df_to_records(weekly_df, limit_rows_per_table)

	json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
	return json_bytes, filename, manifest


def _json_dumps_bytes(obj: Any) -> bytes:
	return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _chunk_records_to_json_files(
	*,
	base_meta: Dict[str, Any],
	table_name: str,
	records: List[Any],
	max_bytes: int,
	filename_prefix: str,
) -> List[Tuple[str, bytes]]:
	"""Split a large list of records into multiple JSON blobs under a target byte size.

	Each part is a JSON object of shape:
	{
	  "meta": { ... },
	  "table": "<table_name>",
	  "part": <n>,
	  "records": [ ... ]
	}
	"""
	if not records:
		return []
	try:
		limit = max(10_000, int(max_bytes))
	except Exception:
		limit = 250_000

	files: List[Tuple[str, bytes]] = []
	part = 1
	cur: List[Any] = []

	def _make_blob(part_num: int, rows: List[Any]) -> bytes:
		obj = {
			"meta": base_meta,
			"table": table_name,
			"part": int(part_num),
			"records": rows,
		}
		return _json_dumps_bytes(obj)

	# Greedy pack until size would exceed limit
	for rec in records:
		trial = cur + [rec]
		b = _make_blob(part, trial)
		if len(b) <= limit or not cur:
			cur = trial
			continue
		# flush current
		blob = _make_blob(part, cur)
		files.append((f"{filename_prefix}{table_name}_part_{part:03d}.json", blob))
		part += 1
		cur = [rec]

	if cur:
		blob = _make_blob(part, cur)
		files.append((f"{filename_prefix}{table_name}_part_{part:03d}.json", blob))

	return files


def build_newsletter_export_json_bundle(
	data_dir: Path,
	docs_dir: Path,
	include_player_game_logs: bool = False,
	include_past_seasons_logs: bool = False,
	include_weekly_player_scoring: bool = False,
	max_players_for_weekly_scoring: Optional[int] = None,
	max_bytes_per_file: int = 250_000,
	limit_rows_per_table: Optional[int] = None,
) -> Tuple[bytes, str, Dict[str, Any]]:
	"""Build a zip of JSON-only files, chunked into chat-friendly sizes."""
	generated_at = datetime.utcnow().isoformat() + "Z"
	league_id = str(st.session_state.get("league_id") or FANTRAX_DEFAULT_LEAGUE_ID or "").strip()
	loaded_league_name = str(st.session_state.get("loaded_league_name") or "").strip()
	label = loaded_league_name or league_id or "league"
	filename = f"newsletter_json_bundle_{_safe_filename_piece(label)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"

	base_meta: Dict[str, Any] = {
		"generated_at": generated_at,
		"league_id": league_id,
		"loaded_league_name": loaded_league_name,
		"season": CURRENT_SEASON,
	}

	manifest: Dict[str, Any] = {
		**base_meta,
		"format": "json_bundle",
		"max_bytes_per_file": int(max_bytes_per_file),
		"files": [],
	}

	schedule_df = _try_get_schedule_df()
	standings_df = pd.DataFrame()
	try:
		if schedule_df is not None and not schedule_df.empty:
			standings_df = calculate_team_stats(schedule_df).reset_index().rename(columns={"index": "Team"})
	except Exception:
		standings_df = pd.DataFrame()

	combined = st.session_state.get("combined_data")
	data_ranges = st.session_state.get("data_ranges")

	draft_history_df = pd.DataFrame()
	try:
		draft_history_df = load_draft_history(include_latest_fantrax=True)
	except Exception:
		draft_history_df = pd.DataFrame()

	trade_history_path = None
	try:
		trade_history_path = _get_trade_history_path()
	except Exception:
		trade_history_path = None

	injury_overrides_path = data_dir / "injured_players.json"
	rules_doc_path = docs_dir / "deep_dive" / "902_League_Advanced_Strategy_and_Modeling.md"
	weekly_cache_dir = data_dir / "weekly_standings_cache"
	player_log_cache_dir = data_dir / "player_game_log_cache"

	files_to_write: List[Tuple[str, bytes]] = []
	files_to_write.append(("meta.json", _json_dumps_bytes({"meta": base_meta})))

	# Rules
	if rules_doc_path.exists():
		txt = _read_text_file(rules_doc_path)
		if txt is not None:
			files_to_write.append(("rules_902.json", _json_dumps_bytes({"meta": base_meta, "text": txt})))

	def _add_table(name: str, recs: List[Any]) -> None:
		if not recs:
			return
		files_to_write.extend(
			_chunk_records_to_json_files(
				base_meta=base_meta,
				table_name=name,
				records=recs,
				max_bytes=max_bytes_per_file,
				filename_prefix="table_",
			)
		)

	# Tables
	if isinstance(combined, pd.DataFrame) and not combined.empty:
		_add_table("combined_player_data", _df_to_records(combined.reset_index(), limit_rows_per_table))
	if schedule_df is not None and not schedule_df.empty:
		_add_table("schedule_and_results", _df_to_records(schedule_df, limit_rows_per_table))
	if standings_df is not None and not standings_df.empty:
		_add_table("standings_from_schedule", _df_to_records(standings_df, limit_rows_per_table))
	if draft_history_df is not None and not draft_history_df.empty:
		_add_table("draft_history_all_seasons", _df_to_records(draft_history_df, limit_rows_per_table))

	# Per-range tables
	if isinstance(data_ranges, dict) and data_ranges:
		for key, df in data_ranges.items():
			if not isinstance(df, pd.DataFrame) or df.empty:
				continue
			name = f"player_data_{_safe_filename_piece(str(key)).lower()}"
			_add_table(name, _df_to_records(df.reset_index(drop=True), limit_rows_per_table))

	# Weekly standings cache
	weekly_blobs: List[Any] = []
	if weekly_cache_dir.exists():
		for fp in sorted(weekly_cache_dir.glob("*.json")):
			try:
				weekly_blobs.append(json.loads(fp.read_text(encoding="utf-8")))
			except UnicodeDecodeError:
				try:
					weekly_blobs.append(json.loads(fp.read_text(encoding="cp1252")))
				except Exception:
					continue
			except Exception:
				continue
	if weekly_blobs:
		_add_table("weekly_standings_cache", weekly_blobs)

	# Injuries overrides
	if injury_overrides_path.exists():
		try:
			inj_obj = json.loads(injury_overrides_path.read_text(encoding="utf-8"))
		except UnicodeDecodeError:
			try:
				inj_obj = json.loads(injury_overrides_path.read_text(encoding="cp1252"))
			except Exception:
				inj_obj = None
		except Exception:
			inj_obj = None
		if inj_obj is not None:
			files_to_write.append(("injured_players_overrides.json", _json_dumps_bytes({"meta": base_meta, "data": inj_obj})))

	# Trades
	if trade_history_path is not None and isinstance(trade_history_path, Path) and trade_history_path.exists():
		try:
			trade_obj = json.loads(trade_history_path.read_text(encoding="utf-8"))
		except UnicodeDecodeError:
			try:
				trade_obj = json.loads(trade_history_path.read_text(encoding="cp1252"))
			except Exception:
				trade_obj = None
		except Exception:
			trade_obj = None
		if trade_obj is not None:
			if isinstance(trade_obj, list):
				_add_table("trade_history", trade_obj)
			else:
				files_to_write.append(("trade_history.json", _json_dumps_bytes({"meta": base_meta, "data": trade_obj})))

	# Optional weekly player scoring
	if include_weekly_player_scoring and league_id and player_log_cache_dir.exists():
		period_ranges = _period_ranges_from_schedule(schedule_df)
		player_map = _build_player_to_team_map()
		weekly_df = _weekly_player_scoring_breakdown(
			league_id=league_id,
			season=CURRENT_SEASON,
			period_ranges=period_ranges,
			player_team_map=player_map,
			cache_dir=player_log_cache_dir,
			max_players=max_players_for_weekly_scoring,
		)
		if weekly_df is not None and not weekly_df.empty:
			_add_table("weekly_player_scoring_breakdown", _df_to_records(weekly_df, limit_rows_per_table))

	# Optional player logs: add as raw json files, not chunked
	if include_player_game_logs and player_log_cache_dir.exists() and league_id:
		pattern = (
			f"player_game_log_full_*_{league_id}_*.json"
			if include_past_seasons_logs
			else f"player_game_log_full_*_{league_id}_{CURRENT_SEASON.replace('-', '_')}.json"
		)
		for fp in sorted(player_log_cache_dir.glob(pattern)):
			try:
				files_to_write.append((f"playerlog_{fp.stem}.json", fp.read_bytes()))
			except Exception:
				continue
		avail_pattern = (
			f"availability_index_{league_id}_*.json"
			if include_past_seasons_logs
			else f"availability_index_{league_id}_{CURRENT_SEASON.replace('-', '_')}.json"
		)
		for fp in sorted(player_log_cache_dir.glob(avail_pattern)):
			try:
				files_to_write.append((f"availability_{fp.stem}.json", fp.read_bytes()))
			except Exception:
				continue

	# Finalize manifest
	manifest["files"] = [name for name, _ in files_to_write] + ["manifest.json"]
	files_to_write.append(("manifest.json", _json_dumps_bytes(manifest)))

	buf = io.BytesIO()
	with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
		for name, blob in files_to_write:
			zf.writestr(name, blob)

	return buf.getvalue(), filename, manifest


def _chunk_object_to_json_files(
	*,
	base_meta: Dict[str, Any],
	filename_base: str,
	obj: Dict[str, Any],
	max_bytes: int,
) -> List[Tuple[str, bytes]]:
	"""Chunk a JSON-serializable object into one or more files.

	If the object is larger than max_bytes, it attempts to chunk the top-level
	"data" entries that are lists (records) into parts.
	"""
	try:
		limit = max(50_000, int(max_bytes))
	except Exception:
		limit = 250_000

	blob = _json_dumps_bytes(obj)
	if len(blob) <= limit:
		return [(f"{filename_base}.json", blob)]

	data = obj.get("data") if isinstance(obj, dict) else None
	if not isinstance(data, dict):
		# Can't intelligently chunk; just return one big blob
		return [(f"{filename_base}.json", blob)]

	# Find list-valued entries to chunk
	list_keys = [k for k, v in data.items() if isinstance(v, list) and v]
	if not list_keys:
		return [(f"{filename_base}.json", blob)]

	# Greedy chunk by splitting the largest list entries first
	files: List[Tuple[str, bytes]] = []
	remaining = dict(data)
	part = 1

	# Base wrapper without data
	base_obj = {k: v for k, v in obj.items() if k != "data"}

	def _fits(trial_data: Dict[str, Any], part_num: int) -> bool:
		try:
			b = _json_dumps_bytes({**base_obj, "data": trial_data, "part": part_num})
			return len(b) <= limit
		except Exception:
			return False

	def _max_prefix_count_that_fits(
		*,
		vals: List[Any],
		chunk_data_base: Dict[str, Any],
		list_key: str,
		part_num: int,
	) -> int:
		"""Return max n (0..len(vals)) such that vals[:n] fits in the JSON blob."""
		if not vals:
			return 0
		# Fast path: full list fits
		trial_full = dict(chunk_data_base)
		trial_full[list_key] = vals
		if _fits(trial_full, part_num):
			return len(vals)

		lo = 0
		hi = len(vals)
		while lo + 1 < hi:
			mid = (lo + hi) // 2
			trial_mid = dict(chunk_data_base)
			trial_mid[list_key] = vals[:mid]
			if _fits(trial_mid, part_num):
				lo = mid
			else:
				hi = mid
		return lo

	max_parts = 10_000
	while True:
		# Build a chunk object including non-list entries and an empty placeholder
		chunk_data: Dict[str, Any] = {k: v for k, v in remaining.items() if k not in list_keys}

		# Try to pack each list key in full, else pack partial
		packed_any = False
		for lk in list_keys:
			vals = remaining.get(lk)
			if not isinstance(vals, list) or not vals:
				continue

			# Determine max prefix that fits (binary search; much faster than per-record growth)
			n = _max_prefix_count_that_fits(
				vals=vals,
				chunk_data_base=chunk_data,
				list_key=lk,
				part_num=part,
			)
			if n <= 0:
				# Ensure progress: include at least one element even if it exceeds limit
				n = 1
			chunk_data[lk] = vals[:n]
			remaining[lk] = vals[n:]
			packed_any = True

		# Emit this part
		out_obj = {**base_obj, "data": chunk_data, "part": part}
		files.append((f"{filename_base}_part_{part:03d}.json", _json_dumps_bytes(out_obj)))
		part += 1
		if part > max_parts:
			break

		# Stop when all list keys are drained
		all_empty = True
		for lk in list_keys:
			vals = remaining.get(lk)
			if isinstance(vals, list) and vals:
				all_empty = False
				break
			remaining.pop(lk, None)
		if all_empty:
			break

		# continue, but keep list_keys for keys that still exist
		list_keys = [k for k in list_keys if isinstance(remaining.get(k), list) and remaining.get(k)]
		if not list_keys:
			break
		if not packed_any:
			break

	return files


def build_newsletter_export_message_bundle(
	data_dir: Path,
	docs_dir: Path,
	include_player_game_logs: bool = False,
	include_past_seasons_logs: bool = False,
	include_weekly_player_scoring: bool = False,
	max_players_for_weekly_scoring: Optional[int] = None,
	max_bytes_per_file: int = 250_000,
	limit_rows_per_table: Optional[int] = None,
) -> Tuple[bytes, str, Dict[str, Any]]:
	"""Create 5-10 chat-ready JSON files grouped by topic.

	Returns a ZIP containing only JSON files with filenames like:
	- msg_01_standings_and_results.json
	- msg_02_rosters_and_player_stats.json
	... (with _part_### if needed)
	"""
	generated_at = datetime.utcnow().isoformat() + "Z"
	league_id = str(st.session_state.get("league_id") or FANTRAX_DEFAULT_LEAGUE_ID or "").strip()
	loaded_league_name = str(st.session_state.get("loaded_league_name") or "").strip()
	label = loaded_league_name or league_id or "league"
	filename = f"newsletter_message_bundle_{_safe_filename_piece(label)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"

	base_meta: Dict[str, Any] = {
		"generated_at": generated_at,
		"league_id": league_id,
		"loaded_league_name": loaded_league_name,
		"season": CURRENT_SEASON,
	}

	schedule_df = _try_get_schedule_df()
	standings_df = pd.DataFrame()
	try:
		if schedule_df is not None and not schedule_df.empty:
			standings_df = calculate_team_stats(schedule_df).reset_index().rename(columns={"index": "Team"})
	except Exception:
		standings_df = pd.DataFrame()

	combined = st.session_state.get("combined_data")
	data_ranges = st.session_state.get("data_ranges")

	draft_history_df = pd.DataFrame()
	try:
		draft_history_df = load_draft_history(include_latest_fantrax=True)
	except Exception:
		draft_history_df = pd.DataFrame()

	trade_history_path = None
	try:
		trade_history_path = _get_trade_history_path()
	except Exception:
		trade_history_path = None

	injury_overrides_path = data_dir / "injured_players.json"
	rules_doc_path = docs_dir / "deep_dive" / "902_League_Advanced_Strategy_and_Modeling.md"
	weekly_cache_dir = data_dir / "weekly_standings_cache"
	player_log_cache_dir = data_dir / "player_game_log_cache"

	# Assemble message objects
	messages: List[Tuple[str, Dict[str, Any]]] = []

	# Message 1: Standings + Matchup Results
	msg1_data: Dict[str, Any] = {
		"standings_from_schedule": _df_to_records(standings_df, limit_rows_per_table),
		"schedule_and_results": _df_to_records(schedule_df, limit_rows_per_table),
		"weekly_standings_cache": [],
	}
	if weekly_cache_dir.exists():
		blobs: List[Any] = []
		for fp in sorted(weekly_cache_dir.glob("*.json")):
			try:
				blobs.append(json.loads(fp.read_text(encoding="utf-8")))
			except UnicodeDecodeError:
				try:
					blobs.append(json.loads(fp.read_text(encoding="cp1252")))
				except Exception:
					continue
			except Exception:
				continue
		msg1_data["weekly_standings_cache"] = blobs
	msg1 = {"meta": base_meta, "title": "Standings + Matchup Results", "data": msg1_data}
	messages.append(("msg_01_standings_and_results", msg1))

	# Message 2: Rosters + Player Stats
	msg2_data: Dict[str, Any] = {
		"combined_player_data": _df_to_records(combined.reset_index(), limit_rows_per_table)
		if isinstance(combined, pd.DataFrame) and not combined.empty
		else [],
		"player_data_ranges": {},
	}
	if isinstance(data_ranges, dict) and data_ranges:
		for key, df in data_ranges.items():
			if not isinstance(df, pd.DataFrame) or df.empty:
				continue
			msg2_data["player_data_ranges"][str(key)] = _df_to_records(df.reset_index(drop=True), limit_rows_per_table)
	msg2 = {"meta": base_meta, "title": "Rosters + Player Stats", "data": msg2_data}
	messages.append(("msg_02_rosters_and_player_stats", msg2))

	# Message 3: Draft + Manager History (+ placeholder Transactions)
	msg3_data: Dict[str, Any] = {
		"draft_history_all_seasons": _df_to_records(draft_history_df, limit_rows_per_table),
		"manager_ids": [],
		"transactions": [],
		"transactions_note": "Transactions/FAAB/IR not yet sourced by the app.",
	}
	try:
		mgr_path = data_dir / "ManagerIDs.csv"
		if mgr_path.exists():
			try:
				mgr_df = pd.read_csv(mgr_path, encoding="utf-8")
			except UnicodeDecodeError:
				mgr_df = pd.read_csv(mgr_path, encoding="cp1252")
			msg3_data["manager_ids"] = _df_to_records(mgr_df, limit_rows_per_table)
	except Exception:
		pass
	msg3 = {"meta": base_meta, "title": "Transactions + Draft", "data": msg3_data}
	messages.append(("msg_03_transactions_and_draft", msg3))

	# Message 4: Trades
	trade_obj = None
	if trade_history_path is not None and isinstance(trade_history_path, Path) and trade_history_path.exists():
		try:
			trade_obj = json.loads(trade_history_path.read_text(encoding="utf-8"))
		except UnicodeDecodeError:
			try:
				trade_obj = json.loads(trade_history_path.read_text(encoding="cp1252"))
			except Exception:
				trade_obj = None
		except Exception:
			trade_obj = None
	msg4 = {
		"meta": base_meta,
		"title": "Trade History",
		"data": {"trade_history": trade_obj if trade_obj is not None else []},
	}
	messages.append(("msg_04_trade_history", msg4))

	# Message 5: Rules + Injuries
	inj_obj = None
	if injury_overrides_path.exists():
		try:
			inj_obj = json.loads(injury_overrides_path.read_text(encoding="utf-8"))
		except UnicodeDecodeError:
			try:
				inj_obj = json.loads(injury_overrides_path.read_text(encoding="cp1252"))
			except Exception:
				inj_obj = None
		except Exception:
			inj_obj = None
		
	rules_txt = _read_text_file(rules_doc_path) if rules_doc_path.exists() else None
	msg5 = {
		"meta": base_meta,
		"title": "League Rules + Injury Overrides",
		"data": {
			"rules_902_md": rules_txt or "",
			"injured_players_overrides": inj_obj or {},
		},
	}
	messages.append(("msg_05_rules_and_injuries", msg5))

	# Message 6: Weekly scoring breakdown (optional)
	if include_weekly_player_scoring and league_id and player_log_cache_dir.exists():
		period_ranges = _period_ranges_from_schedule(schedule_df)
		player_map = _build_player_to_team_map()
		weekly_df = _weekly_player_scoring_breakdown(
			league_id=league_id,
			season=CURRENT_SEASON,
			period_ranges=period_ranges,
			player_team_map=player_map,
			cache_dir=player_log_cache_dir,
			max_players=max_players_for_weekly_scoring,
		)
		msg6 = {
			"meta": base_meta,
			"title": "Weekly Player Scoring Breakdown",
			"data": {"weekly_player_scoring_breakdown": _df_to_records(weekly_df, limit_rows_per_table)},
		}
		messages.append(("msg_06_weekly_player_scoring", msg6))

	# Message 7: Player game logs (current season only by default)
	if include_player_game_logs and league_id and player_log_cache_dir.exists():
		pattern = (
			f"player_game_log_full_*_{league_id}_*.json"
			if include_past_seasons_logs
			else f"player_game_log_full_*_{league_id}_{CURRENT_SEASON.replace('-', '_')}.json"
		)
		# Keep as list of file blobs (not records) so we don't explode memory with huge dict merges
		log_files = []
		for fp in sorted(player_log_cache_dir.glob(pattern)):
			try:
				log_files.append({"file": fp.name, "json": json.loads(fp.read_text(encoding="utf-8"))})
			except UnicodeDecodeError:
				try:
					log_files.append({"file": fp.name, "json": json.loads(fp.read_text(encoding="cp1252"))})
				except Exception:
					continue
			except Exception:
				continue
		msg7 = {
			"meta": base_meta,
			"title": "Player Game Logs (JSON)",
			"data": {"player_game_logs": log_files},
		}
		messages.append(("msg_07_player_game_logs", msg7))

	# Write zip with chunking
	files_to_write: List[Tuple[str, bytes]] = []
	files_to_write.append(("meta.json", _json_dumps_bytes({"meta": base_meta, "note": "Paste msg_*.json files into chat in order."})))

	for fname_base, obj in messages:
		files_to_write.extend(
			_chunk_object_to_json_files(
				base_meta=base_meta,
				filename_base=fname_base,
				obj=obj,
				max_bytes=max_bytes_per_file,
			)
		)

	manifest: Dict[str, Any] = {
		**base_meta,
		"format": "message_bundle",
		"max_bytes_per_file": int(max_bytes_per_file),
		"files": [name for name, _ in files_to_write],
	}
	files_to_write.append(("manifest.json", _json_dumps_bytes(manifest)))

	buf = io.BytesIO()
	with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
		for name, blob in files_to_write:
			zf.writestr(name, blob)

	return buf.getvalue(), filename, manifest


def _safe_filename_piece(val: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", " ") else "_" for ch in str(val or "")).strip()
    out = out.replace(" ", "_")
    return out or "export"


def _zip_write_bytes(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    zf.writestr(arcname, data)


def _zip_write_text(zf: zipfile.ZipFile, arcname: str, text: str) -> None:
    zf.writestr(arcname, (text or "").encode("utf-8"))


def _zip_write_df_csv(zf: zipfile.ZipFile, arcname: str, df: pd.DataFrame, index: bool = False) -> None:
    csv_bytes = df.to_csv(index=index).encode("utf-8")
    _zip_write_bytes(zf, arcname, csv_bytes)


def _read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="cp1252")
        except Exception:
            return None
    except Exception:
        return None


def _add_existing_file(zf: zipfile.ZipFile, root: Path, path: Path, arc_prefix: str) -> Optional[str]:
    try:
        rel = path.relative_to(root)
    except Exception:
        rel = path.name
    arcname = str(Path(arc_prefix) / rel).replace("\\", "/")
    try:
        data = path.read_bytes()
    except Exception:
        return None
    _zip_write_bytes(zf, arcname, data)
    return arcname


def _try_get_schedule_df() -> pd.DataFrame:
    sched = st.session_state.get("schedule_data")
    if isinstance(sched, pd.DataFrame) and not sched.empty:
        return sched
    sched2 = load_schedule_data()
    if isinstance(sched2, pd.DataFrame) and not sched2.empty:
        return sched2
    return pd.DataFrame()


def _period_ranges_from_schedule(schedule_df: pd.DataFrame) -> Dict[int, Tuple[Any, Any]]:
    out: Dict[int, Tuple[Any, Any]] = {}
    if schedule_df is None or schedule_df.empty:
        return out
    if "Period Number" not in schedule_df.columns or "Date Range" not in schedule_df.columns:
        return out
    for period, grp in schedule_df.groupby("Period Number"):
        try:
            pnum = int(period)
        except Exception:
            continue
        if grp is None or grp.empty:
            continue
        date_range = str(grp["Date Range"].iloc[0] or "").strip()
        if not date_range:
            continue
        parts = [p.strip() for p in date_range.split("-")]
        if len(parts) < 2:
            continue
        try:
            start_dt = datetime.strptime(parts[0], "%a %b %d, %Y").date()
            end_dt = datetime.strptime(parts[1], "%a %b %d, %Y").date()
        except Exception:
            continue
        out[pnum] = (start_dt, end_dt)
    return out


def _build_player_to_team_map() -> Dict[str, Dict[str, str]]:
    combined = st.session_state.get("combined_data")
    if combined is None or not isinstance(combined, pd.DataFrame) or combined.empty:
        return {}
    try:
        flat = combined.reset_index() if "Player" not in combined.columns else combined.copy()
    except Exception:
        return {}
    if "Player" not in flat.columns:
        return {}
    if "Timestamp" in flat.columns and (flat["Timestamp"] == "YTD").any():
        flat = flat[flat["Timestamp"] == "YTD"].copy()
    flat = flat.drop_duplicates(subset=["Player"], keep="first")
    cols = [c for c in ["Status", "Fantasy_Manager"] if c in flat.columns]
    if not cols:
        return {}
    out = {}
    for _, row in flat.iterrows():
        player = str(row.get("Player") or "").strip()
        if not player:
            continue
        out[player] = {c: str(row.get(c) or "") for c in cols}
    return out


def _weekly_player_scoring_breakdown(
    league_id: str,
    season: str,
    period_ranges: Dict[int, Tuple[Any, Any]],
    player_team_map: Dict[str, Dict[str, str]],
    cache_dir: Path,
    max_players: Optional[int] = None,
) -> pd.DataFrame:
    pattern = f"player_game_log_full_*_{league_id}_{season.replace('-', '_')}.json"
    cache_files = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if max_players is not None:
        cache_files = cache_files[: max(0, int(max_players))]
    rows: List[Dict[str, Any]] = []
    for fp in cache_files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            try:
                payload = json.loads(fp.read_text(encoding="cp1252"))
            except Exception:
                continue
        except Exception:
            continue
        player_name = str(payload.get("player_name") or "").strip()
        data = payload.get("data") or []
        if not player_name or not isinstance(data, list) or not data:
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            raw_date = rec.get("Date")
            game_dt = _parse_game_date_for_season(str(raw_date or ""), season)
            if game_dt is None:
                continue
            period_num = None
            for p, (start_dt, end_dt) in period_ranges.items():
                try:
                    if start_dt <= game_dt <= end_dt:
                        period_num = int(p)
                        break
                except Exception:
                    continue
            if period_num is None:
                continue
            try:
                fpts = float(rec.get("FPts"))
            except Exception:
                continue
            team_info = player_team_map.get(player_name, {})
            rows.append({
                "Week": period_num,
                "TeamCode": team_info.get("Status", ""),
                "TeamName": team_info.get("Fantasy_Manager", ""),
                "Player": player_name,
                "GameDate": game_dt.isoformat(),
                "FPts": fpts,
            })
    if not rows:
        return pd.DataFrame()
    games_df = pd.DataFrame(rows)
    grouped = (
        games_df
        .groupby(["Week", "TeamCode", "TeamName", "Player"], dropna=False)
        .agg(FPts=("FPts", "sum"), Games=("FPts", "count"))
        .reset_index()
    )
    return grouped.sort_values(["Week", "FPts"], ascending=[True, False]).reset_index(drop=True)


def build_newsletter_export_zip(
    data_dir: Path,
    docs_dir: Path,
    include_player_game_logs: bool = False,
    include_past_seasons_logs: bool = False,
    include_weekly_player_scoring: bool = False,
    max_players_for_weekly_scoring: Optional[int] = None,
) -> Tuple[bytes, str, Dict[str, Any]]:
	generated_at = datetime.utcnow().isoformat() + "Z"
	league_id = str(st.session_state.get("league_id") or FANTRAX_DEFAULT_LEAGUE_ID or "").strip()
	loaded_league_name = str(st.session_state.get("loaded_league_name") or "").strip()
	label = loaded_league_name or league_id or "league"
	filename = f"newsletter_export_{_safe_filename_piece(label)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"

	schedule_df = _try_get_schedule_df()
	standings_df = pd.DataFrame()
	try:
		if schedule_df is not None and not schedule_df.empty:
			standings_df = calculate_team_stats(schedule_df).reset_index().rename(columns={"index": "Team"})
	except Exception:
		standings_df = pd.DataFrame()

	combined = st.session_state.get("combined_data")
	data_ranges = st.session_state.get("data_ranges")

	draft_history_df = pd.DataFrame()
	try:
		draft_history_df = load_draft_history(include_latest_fantrax=True)
	except Exception:
		draft_history_df = pd.DataFrame()

	trade_history_path = None
	try:
		trade_history_path = _get_trade_history_path()
	except Exception:
		trade_history_path = None

	injury_overrides_path = data_dir / "injured_players.json"
	rules_doc_path = docs_dir / "deep_dive" / "902_League_Advanced_Strategy_and_Modeling.md"

	weekly_cache_dir = data_dir / "weekly_standings_cache"
	trade_history_dir = data_dir / "trade_history"
	player_log_cache_dir = data_dir / "player_game_log_cache"

	manifest: Dict[str, Any] = {
		"generated_at": generated_at,
		"league_id": league_id,
		"loaded_league_name": loaded_league_name,
		"includes": [],
		"missing_or_not_available": {
			"transactions_log": True,
			"faab_budget": True,
			"ir_usage": True,
			"league_chat": True,
		},
	}

	buf = io.BytesIO()
	with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
		if isinstance(combined, pd.DataFrame) and not combined.empty:
			try:
				_zip_write_df_csv(zf, "league/combined_player_data.csv", combined.reset_index(), index=False)
				manifest["includes"].append("league/combined_player_data.csv")
			except Exception:
				pass

		if isinstance(data_ranges, dict) and data_ranges:
			for key, df in data_ranges.items():
				if not isinstance(df, pd.DataFrame) or df.empty:
					continue
				try:
					arc = f"league/player_data_ranges/{_safe_filename_piece(str(key))}.csv"
					_zip_write_df_csv(zf, arc, df.reset_index(drop=True), index=False)
					manifest["includes"].append(arc)
				except Exception:
					continue

		if schedule_df is not None and not schedule_df.empty:
			try:
				_zip_write_df_csv(zf, "standings_and_results/schedule_and_results.csv", schedule_df, index=False)
				manifest["includes"].append("standings_and_results/schedule_and_results.csv")
			except Exception:
				pass

		if standings_df is not None and not standings_df.empty:
			try:
				_zip_write_df_csv(zf, "standings_and_results/standings_from_schedule.csv", standings_df, index=False)
				manifest["includes"].append("standings_and_results/standings_from_schedule.csv")
			except Exception:
				pass

		if weekly_cache_dir.exists():
			for fp in sorted(weekly_cache_dir.glob("*.json")):
				arcname = _add_existing_file(zf, data_dir, fp, "weekly_standings_cache")
				if arcname:
					manifest["includes"].append(arcname)

		if not draft_history_df.empty:
			try:
				_zip_write_df_csv(zf, "draft/draft_history_all_seasons.csv", draft_history_df, index=False)
				manifest["includes"].append("draft/draft_history_all_seasons.csv")
			except Exception:
				pass

		for fp in sorted(data_dir.glob("Fantrax-Draft-Results-*.csv")):
			arcname = _add_existing_file(zf, data_dir, fp, "draft/raw")
			if arcname:
				manifest["includes"].append(arcname)

		for fp in sorted(data_dir.glob("S*Draft.csv")):
			arcname = _add_existing_file(zf, data_dir, fp, "draft/raw")
			if arcname:
				manifest["includes"].append(arcname)

		for fp in sorted(data_dir.glob("ManagerIDs.csv")):
			arcname = _add_existing_file(zf, data_dir, fp, "managers")
			if arcname:
				manifest["includes"].append(arcname)

		if injury_overrides_path.exists():
			arcname = _add_existing_file(zf, data_dir, injury_overrides_path, "injuries")
			if arcname:
				manifest["includes"].append(arcname)

		if trade_history_path is not None and isinstance(trade_history_path, Path) and trade_history_path.exists():
			arcname = _add_existing_file(zf, trade_history_path.parent, trade_history_path, "trades")
			if arcname:
				manifest["includes"].append(arcname)
		elif trade_history_dir.exists():
			for fp in sorted(trade_history_dir.glob("*.json")):
				arcname = _add_existing_file(zf, data_dir, fp, "trades")
				if arcname:
					manifest["includes"].append(arcname)

		if rules_doc_path.exists():
			txt = _read_text_file(rules_doc_path)
			if txt is not None:
				_zip_write_text(zf, "league_rules/902_League_Advanced_Strategy_and_Modeling.md", txt)
				manifest["includes"].append("league_rules/902_League_Advanced_Strategy_and_Modeling.md")

		if include_player_game_logs and player_log_cache_dir.exists() and league_id:
			pattern = (
				f"player_game_log_full_*_{league_id}_*.json"
				if include_past_seasons_logs
				else f"player_game_log_full_*_{league_id}_{CURRENT_SEASON.replace('-', '_')}.json"
			)
			for fp in sorted(player_log_cache_dir.glob(pattern)):
				arcname = _add_existing_file(zf, data_dir, fp, "player_game_logs")
				if arcname:
					manifest["includes"].append(arcname)

		if include_player_game_logs and player_log_cache_dir.exists() and league_id:
			avail_pattern = (
				f"availability_index_{league_id}_*.json"
				if include_past_seasons_logs
				else f"availability_index_{league_id}_{CURRENT_SEASON.replace('-', '_')}.json"
			)
			for fp in sorted(player_log_cache_dir.glob(avail_pattern)):
				arcname = _add_existing_file(zf, data_dir, fp, "player_availability")
				if arcname:
					manifest["includes"].append(arcname)

		if include_weekly_player_scoring and league_id and player_log_cache_dir.exists():
			period_ranges = _period_ranges_from_schedule(schedule_df)
			player_map = _build_player_to_team_map()
			weekly_df = _weekly_player_scoring_breakdown(
				league_id=league_id,
				season=CURRENT_SEASON,
				period_ranges=period_ranges,
				player_team_map=player_map,
				cache_dir=player_log_cache_dir,
				max_players=max_players_for_weekly_scoring,
			)
			if weekly_df is not None and not weekly_df.empty:
				_zip_write_df_csv(zf, "weekly_scoring/weekly_player_scoring_breakdown.csv", weekly_df, index=False)
				manifest["includes"].append("weekly_scoring/weekly_player_scoring_breakdown.csv")

		_zip_write_text(zf, "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

	return buf.getvalue(), filename, manifest
