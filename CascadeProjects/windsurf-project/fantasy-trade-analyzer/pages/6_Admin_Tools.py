import streamlit as st
from modules.auth.ui import check_password
from modules.fantrax_downloader.ui import display_downloader_ui
from modules.player_game_log_scraper.ui import show_player_game_log_scraper
from modules.standings_adjuster.ui import show_standings_adjuster
from modules.weekly_standings_analyzer.ui import show_weekly_standings_analyzer
from modules.historical_ytd_downloader.ui import display_historical_ytd_ui
from modules.historical_trade_analyzer.ui import show_historical_trade_analyzer
from modules.legacy.data_loader_ui.ui import display_data_loader_ui
import json
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Admin Tools", page_icon="🔐", layout="wide")

# Password protection
if not check_password():
	st.stop()

st.title("🔐 Admin Tools")
st.write("Commissioner-only tools for league management and data scraping.")

# Create tabs for different admin tools
tab_league, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
	"📂 League Data / CSV Loader",
	"📥 Downloader for Current Season (Trading Files)",
	"📅 Historical YTD (When Significant Data Available)",
	"📊 Player Game Logs (when Signifcant Data Available)", 
	"📈 Weekly Standings",
	"⚙️ Weekly Standings Adjuster",
	"📜 Historical Trade Analyzer",
	"🩺 Injury & Availability Overrides",
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

	players_existing = sorted(list(inj_data.keys()))
	players_options = ["<Add new player>"] + players_existing

	col_sel, col_new = st.columns([1, 1])
	with col_sel:
		selected = st.selectbox("Select player override", options=players_options, index=0)
	with col_new:
		new_name = st.text_input("Or type a new player name", value="")

	if new_name.strip():
		player_name = new_name.strip()
	elif selected != "<Add new player>":
		player_name = selected
	else:
		player_name = ""

	if not player_name:
		st.info("Select or type a player name to edit their injury override.")
		st.stop()

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
		st.dataframe(rows, hide_index=True, use_container_width=True)
	else:
		st.caption("No overrides currently set.")
