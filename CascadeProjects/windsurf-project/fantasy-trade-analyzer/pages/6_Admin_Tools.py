import streamlit as st
from modules.auth.ui import check_password
from modules.fantrax_downloader.ui import display_downloader_ui
from modules.player_game_log_scraper.ui import show_player_game_log_scraper
from modules.standings_adjuster.ui import show_standings_adjuster
from modules.weekly_standings_analyzer.ui import show_weekly_standings_analyzer
from modules.historical_ytd_downloader.ui import display_historical_ytd_ui
from modules.historical_trade_analyzer.ui import show_historical_trade_analyzer
from modules.legacy.data_loader_ui.ui import display_data_loader_ui
import pandas as pd
from streamlit_compat import dataframe
from modules.sidebar.ui import display_global_sidebar
import json
from pathlib import Path
from datetime import date, timedelta
from modules.newsletter_exporter import (
	build_newsletter_export_json,
	build_newsletter_export_json_bundle,
	build_newsletter_export_message_bundle,
	build_newsletter_export_zip,
)

st.set_page_config(page_title="Admin Tools", page_icon="🔐", layout="wide")

display_global_sidebar()

# Password protection
if not check_password():
	st.stop()

st.title("🔐 Admin Tools")
st.write("Commissioner-only tools for league management and data scraping.")

# Create tabs for different admin tools
tab_league, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
	"📂 League Data / CSV Loader",
	"📥 Downloader for Current Season (Trading Files)",
	"📅 Historical YTD (When Significant Data Available)",
	"📊 Player Game Logs (when Signifcant Data Available)", 
	"📈 Weekly Standings",
	"⚙️ Weekly Standings Adjuster",
	"📜 Historical Trade Analyzer",
	"🩺 Injury & Availability Overrides",
	"📰 Newsletter Export",
])

with tab_league:
	st.subheader("League Data")
	st.caption("Select and load a league dataset from available Fantrax player CSVs.")
	data_dir = Path(__file__).resolve().parent.parent / "data"
	display_data_loader_ui(data_dir)

with tab1:
	display_downloader_ui()

with tab2:
	display_historical_ytd_ui()

with tab3:
	show_player_game_log_scraper()

with tab4:
	show_weekly_standings_analyzer()

with tab5:
	show_standings_adjuster()

with tab6:
	show_historical_trade_analyzer()

with tab7:
	st.subheader("🩺 Injury & Availability Overrides")
	st.caption("Commissioner-only overrides for players with medium/long-term injury risk. These feed into availability and MinGames cushion modeling.")

	data_dir = Path(__file__).resolve().parent.parent / "data"
	data_dir.mkdir(parents=True, exist_ok=True)
	inj_path = data_dir / "injured_players.json"
	try:
		if inj_path.exists():
			with inj_path.open("r", encoding="utf-8") as f:
				inj_data = json.load(f) or {}
		else:
			inj_data = {}
	except Exception:
		inj_data = {}

	if not isinstance(inj_data, dict):
		inj_data = {}

	st.markdown("#### Current Injuries / Overrides")
	col_a, col_b, col_c = st.columns([1.2, 1, 1])
	with col_a:
		filter_text = st.text_input("Filter player", value="", key="inj_override_filter")
	with col_b:
		show_expired = st.checkbox("Show expired", value=False, key="inj_override_show_expired")
	with col_c:
		only_overrides = st.checkbox("Only overrides", value=False, key="inj_override_only_overrides")

	rows = []
	for pname, val in sorted(inj_data.items()):
		if isinstance(val, str):
			rows.append({"Player": pname, "Tag": val, "Duration": "", "Added": "", "Expires": "", "Active": True, "Source": "Override"})
		elif isinstance(val, dict):
			added_str = str(val.get("added") or "").strip()
			added_dt = None
			try:
				parts = [int(p) for p in added_str.split("-")]
				if len(parts) == 3:
					added_dt = date(parts[0], parts[1], parts[2])
			except Exception:
				added_dt = None

			dv = val.get("duration_value")
			du = str(val.get("duration_unit") or "").strip().lower()
			days = None
			try:
				if dv is not None and str(dv).strip() != "":
					dv_int = int(dv)
					if dv_int > 0:
						if du == "days":
							days = dv_int
						elif du == "weeks":
							days = dv_int * 7
						elif du == "months":
							days = dv_int * 30
			except Exception:
				days = None

			expires = ""
			active = True
			if added_dt is not None and days is not None:
				exp_dt = added_dt + timedelta(days=int(days))
				expires = exp_dt.strftime("%Y-%m-%d")
				active = exp_dt >= date.today()

			rows.append({
				"Player": pname,
				"Tag": str(val.get("tag") or ""),
				"Duration": f"{val.get('duration_value', '')} {val.get('duration_unit', '')}".strip(),
				"Added": added_str,
				"Expires": expires,
				"Active": active,
				"Source": "Override",
			})

	inj_df = pd.DataFrame(rows)

	player_df = st.session_state.get("combined_data")
	if not only_overrides and player_df is not None and not getattr(player_df, "empty", True):
		try:
			flat = player_df.reset_index() if "Player" not in player_df.columns else player_df.copy()
			non_override_cols = [c for c in flat.columns if str(c).lower() not in {"player", "timestamp"}]
			candidate_cols = [
				c for c in non_override_cols
				if any(k in str(c).lower() for k in ("inj", "availability", "health", "il", "ir"))
			]
			if candidate_cols:
				inj_col = candidate_cols[0]
				inj_vals = flat[["Player", inj_col]].copy()
				inj_vals[inj_col] = inj_vals[inj_col].astype(str).str.strip()
				inj_vals = inj_vals[inj_vals[inj_col].ne("") & ~inj_vals[inj_col].str.lower().isin(["nan", "none"])]
				if not inj_vals.empty:
					inj_vals = inj_vals.rename(columns={inj_col: "Tag"})
					inj_vals["Duration"] = ""
					inj_vals["Added"] = ""
					inj_vals["Expires"] = ""
					inj_vals["Active"] = True
					inj_vals["Source"] = "Data"
					inj_df = pd.concat([inj_df, inj_vals[["Player", "Tag", "Duration", "Added", "Expires", "Active", "Source"]]], ignore_index=True)
		except Exception:
			pass

	if not inj_df.empty:
		inj_df = inj_df.copy()
		if filter_text.strip():
			q = filter_text.strip().lower()
			inj_df = inj_df[inj_df["Player"].astype(str).str.lower().str.contains(q, na=False)]
		if not show_expired and "Active" in inj_df.columns:
			inj_df = inj_df[inj_df["Active"] == True]
		inj_df = inj_df.sort_values(["Source", "Active", "Player"], ascending=[True, False, True])
		dataframe(inj_df, width="stretch", hide_index=True)
	else:
		st.caption("No overrides currently set.")

	players_existing = sorted(list(inj_data.keys()))
	players_options = ["<Add new player>"] + players_existing

	col_sel, col_new = st.columns([1, 1])
	with col_sel:
		selected = st.selectbox("Select player override", options=players_options, index=0)
	with col_new:
		new_name = st.text_input("Or type a new player name", value="")

	picked_loaded = ""
	player_df = st.session_state.get("combined_data")
	if player_df is not None and not getattr(player_df, "empty", True):
		try:
			flat = player_df.reset_index() if "Player" not in player_df.columns else player_df.copy()
			player_names = sorted(flat["Player"].dropna().astype(str).str.strip().unique().tolist())
			picked_loaded = st.selectbox("Or pick from loaded players", options=[""] + player_names, index=0)
		except Exception:
			picked_loaded = ""

	if new_name.strip():
		player_name = new_name.strip()
	elif picked_loaded.strip():
		player_name = picked_loaded.strip()
	elif selected != "<Add new player>":
		player_name = selected
	else:
		player_name = ""

	if not player_name:
		st.info("Select or type a player name to edit their injury override.")
		player_name = None

	if not player_name:
		pass
	else:

		entry = inj_data.get(player_name)
		if isinstance(entry, str):
			current_tag = entry
			current_duration_value = 0
			current_duration_unit = "weeks"
			current_added = None
		elif isinstance(entry, dict):
			current_tag = entry.get("tag") or ""
			current_duration_value = int(entry.get("duration_value") or 0)
			current_duration_unit = str(entry.get("duration_unit") or "weeks")
			current_added = entry.get("added")
		else:
			current_tag = ""
			current_duration_value = 0
			current_duration_unit = "weeks"
			current_added = None

		st.markdown(f"### Editing override for **{player_name}**")

		col_tag, col_duration, col_unit = st.columns([1, 1, 1])
		with col_tag:
			inj_tag = st.selectbox(
				"Injury tag",
				options=["", "Full Season", "Half Season", "1/4 Season"],
				index=["", "Full Season", "Half Season", "1/4 Season"].index(current_tag) if current_tag in ["Full Season", "Half Season", "1/4 Season"] else 0,
				help="Leave blank to clear any override for this player.",
			)
		with col_duration:
			duration_value = st.number_input("Duration", min_value=0, max_value=365, value=current_duration_value, step=1, help="Rough length of this injury window.")
		with col_unit:
			unit = st.selectbox("Unit", options=["days", "weeks", "months"], index=["days", "weeks", "months"].index(current_duration_unit) if current_duration_unit in ["days", "weeks", "months"] else 1)

		default_added = None
		try:
			if current_added:
				parts = [int(p) for p in str(current_added).split("-")]
				if len(parts) == 3:
					default_added = date(parts[0], parts[1], parts[2])
		except Exception:
			default_added = None

		added_date = st.date_input("Date override added", value=default_added or date.today(), help="Used with duration to auto-expire this tag.")

		col_save, col_clear = st.columns([1, 1])
		with col_save:
			if st.button("💾 Save Override"):
				if inj_tag:
					inj_data[player_name] = {
						"tag": inj_tag,
						"duration_value": int(duration_value),
						"duration_unit": unit,
						"added": added_date.strftime("%Y-%m-%d"),
					}
					with inj_path.open("w", encoding="utf-8") as f:
						json.dump(inj_data, f, indent=4, ensure_ascii=False)
					st.success("Override saved.")
				else:
					if player_name in inj_data:
						inj_data.pop(player_name, None)
						with inj_path.open("w", encoding="utf-8") as f:
							json.dump(inj_data, f, indent=4, ensure_ascii=False)
						st.success("Override cleared.")

		with col_clear:
			if st.button("🗑️ Remove Player Override"):
				if player_name in inj_data:
					inj_data.pop(player_name, None)
					with inj_path.open("w", encoding="utf-8") as f:
						json.dump(inj_data, f, indent=4, ensure_ascii=False)
					st.success("Override removed.")

	st.markdown("---")
	st.markdown("#### Current Overrides")
	if inj_data:
		rows = []
		for pname, val in sorted(inj_data.items()):
			if isinstance(val, str):
				rows.append({"Player": pname, "Tag": val, "Duration": "", "Added": ""})
			elif isinstance(val, dict):
				rows.append({
					"Player": pname,
					"Tag": val.get("tag", ""),
					"Duration": f"{val.get('duration_value', '')} {val.get('duration_unit', '')}",
					"Added": val.get("added", ""),
				})
		dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
	else:
		st.caption("No overrides currently set.")


with tab8:
	st.subheader("📰 Newsletter Export")
	st.caption("Generate a one-shot export bundle for AI newsletter generation.")

	data_dir = Path(__file__).resolve().parent.parent / "data"
	docs_dir = Path(__file__).resolve().parent.parent / "docs"

	export_format = st.radio(
		"Export format",
		options=[
			"Single JSON (recommended)",
			"Message bundle (chat-ready 5–10 JSONs)",
			"JSON bundle (multiple files)",
			"ZIP (multi-file)",
		],
		horizontal=True,
		key="newsletter_export_format",
	)

	col_a, col_b = st.columns([1, 1])
	with col_a:
		include_logs = st.checkbox(
			"Include player game logs (can be large)",
			value=False,
			key="newsletter_export_include_logs",
		)
	with col_b:
		include_weekly_scoring = st.checkbox(
			"Include weekly player scoring breakdown (best-effort)",
			value=False,
			key="newsletter_export_include_weekly_scoring",
		)

	include_past_logs = False
	if include_logs:
		include_past_logs = st.checkbox(
			"Include past seasons game logs too (very large)",
			value=False,
			key="newsletter_export_include_past_logs",
		)

	max_players = None
	if include_weekly_scoring:
		max_players = st.number_input(
			"Max players to scan for weekly scoring (leave high for completeness)",
			min_value=10,
			max_value=5000,
			value=400,
			step=50,
			key="newsletter_export_max_players",
		)

	row_limit = None
	max_bytes_per_file = None
	if export_format in {"Single JSON (recommended)", "JSON bundle (multiple files)", "Message bundle (chat-ready 5–10 JSONs)"}:
		row_limit = st.number_input(
			"Optional: limit rows per table (0 = no limit)",
			min_value=0,
			max_value=50000,
			value=0,
			step=500,
			key="newsletter_export_row_limit",
		)
	if export_format in {"JSON bundle (multiple files)", "Message bundle (chat-ready 5–10 JSONs)"}:
		max_bytes_per_file = st.number_input(
			"Max size per JSON file (bytes)",
			min_value=50_000,
			max_value=2_000_000,
			value=250_000,
			step=50_000,
			key="newsletter_export_max_bytes_per_file",
		)

	if st.button("🧰 Build Newsletter Export", type="primary", key="newsletter_export_build"):
		try:
			with st.spinner("Building export bundle..."):
				if export_format == "ZIP (multi-file)":
					export_bytes, file_name, _manifest = build_newsletter_export_zip(
						data_dir=data_dir,
						docs_dir=docs_dir,
						include_player_game_logs=bool(include_logs),
						include_past_seasons_logs=bool(include_past_logs),
						include_weekly_player_scoring=bool(include_weekly_scoring),
						max_players_for_weekly_scoring=int(max_players) if max_players is not None else None,
					)
					st.session_state["newsletter_export_bytes"] = export_bytes
					st.session_state["newsletter_export_name"] = file_name
					st.session_state["newsletter_export_mime"] = "application/zip"
				elif export_format == "Message bundle (chat-ready 5–10 JSONs)":
					limit_rows = int(row_limit) if row_limit is not None else 0
					max_bytes = int(max_bytes_per_file) if max_bytes_per_file is not None else 250_000
					export_bytes, file_name, _manifest = build_newsletter_export_message_bundle(
						data_dir=data_dir,
						docs_dir=docs_dir,
						include_player_game_logs=bool(include_logs),
						include_past_seasons_logs=bool(include_past_logs),
						include_weekly_player_scoring=bool(include_weekly_scoring),
						max_players_for_weekly_scoring=int(max_players) if max_players is not None else None,
						max_bytes_per_file=max_bytes,
						limit_rows_per_table=(limit_rows if limit_rows > 0 else None),
					)
					st.session_state["newsletter_export_bytes"] = export_bytes
					st.session_state["newsletter_export_name"] = file_name
					st.session_state["newsletter_export_mime"] = "application/zip"
				elif export_format == "JSON bundle (multiple files)":
					limit_rows = int(row_limit) if row_limit is not None else 0
					max_bytes = int(max_bytes_per_file) if max_bytes_per_file is not None else 250_000
					export_bytes, file_name, _manifest = build_newsletter_export_json_bundle(
						data_dir=data_dir,
						docs_dir=docs_dir,
						include_player_game_logs=bool(include_logs),
						include_past_seasons_logs=bool(include_past_logs),
						include_weekly_player_scoring=bool(include_weekly_scoring),
						max_players_for_weekly_scoring=int(max_players) if max_players is not None else None,
						max_bytes_per_file=max_bytes,
						limit_rows_per_table=(limit_rows if limit_rows > 0 else None),
					)
					st.session_state["newsletter_export_bytes"] = export_bytes
					st.session_state["newsletter_export_name"] = file_name
					st.session_state["newsletter_export_mime"] = "application/zip"
				else:
					limit_rows = int(row_limit) if row_limit is not None else 0
					export_bytes, file_name, _manifest = build_newsletter_export_json(
						data_dir=data_dir,
						docs_dir=docs_dir,
						include_player_game_logs=bool(include_logs),
						include_past_seasons_logs=bool(include_past_logs),
						include_weekly_player_scoring=bool(include_weekly_scoring),
						max_players_for_weekly_scoring=int(max_players) if max_players is not None else None,
						limit_rows_per_table=(limit_rows if limit_rows > 0 else None),
					)
					st.session_state["newsletter_export_bytes"] = export_bytes
					st.session_state["newsletter_export_name"] = file_name
					st.session_state["newsletter_export_mime"] = "application/json"
			st.success("Export bundle ready.")
		except Exception as e:
			st.error(f"Error building export: {e}")

	export_bytes = st.session_state.get("newsletter_export_bytes")
	export_name = st.session_state.get("newsletter_export_name")
	export_mime = st.session_state.get("newsletter_export_mime")
	if export_bytes and export_name and export_mime:
		st.download_button(
			label="⬇️ Download Newsletter Export",
			data=export_bytes,
			file_name=export_name,
			mime=export_mime,
			key="newsletter_export_download",
			width="stretch",
		)
