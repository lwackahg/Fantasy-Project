import streamlit as st
import pandas as pd
from datetime import date, timedelta

from modules.lineup_optimizer.logic import compare_players_for_date, get_actual_fpts_for_date
from modules.player_game_log_scraper.logic import get_available_players_from_csv
from modules.team_mappings import TEAM_MAPPINGS


def show_tonight_decision_helper(default_league_id: str | None = None):
	"""Small UI for comparing 2–4 players for a single date.

	This is intended as Phase 1 of the lineup optimizer: a focused tool to help
	with nightly start/sit decisions (e.g., "CJ vs Kyshawn"). It assumes that
	cached game logs are already available for the selected players.
	"""
	st.header("🧮 Tonight's Lineup Decision Helper")
	st.write("Compare a few players for a specific date using recent form and simple matchup factors.")

	# Pre-load rostered players from the latest Fantrax-Players CSV so that the
	# Player column can be a searchable dropdown instead of free text.
	try:
		player_dict = get_available_players_from_csv()
		player_options = sorted(player_dict.keys())
	except Exception:
		player_options = []

	with st.expander("Step 1 – Basic Settings", expanded=True):
		col1, col2 = st.columns(2)
		with col1:
			league_id = st.text_input(
				"Fantrax League ID",
				value=default_league_id or "",
				help="Used to resolve player codes from the cache index.",
			)
		with col2:
			game_date = st.date_input("Game Date").isoformat()

		if not league_id:
			st.info("Enter a league ID to enable player resolution.")

	with st.expander("Step 1b – Weekly Context (Optional)", expanded=False):
		st.caption(
			"Provide current weekly context so the helper can estimate how each additional game "
			"impacts your team PPG under the league's minimum/penalty rules. You can copy these "
			"numbers from your Fantrax weekly matchup view mid-week."
		)
		c1, c2, c3 = st.columns(3)
		with c1:
			min_games_required = st.number_input(
				"Min games this week",
				min_value=0,
				value=25,
				step=1,
				key="tonight_min_games_this_week",
			)
		with c2:
			games_played_so_far = st.number_input(
				"Games started so far",
				min_value=0,
				value=0,
				step=1,
				key="tonight_games_started_so_far",
			)
		with c3:
			total_points_so_far = st.number_input(
				"Total points so far this week",
				min_value=0.0,
				value=0.0,
				step=1.0,
				key="tonight_total_points_so_far",
			)

		current_week_ppg: float | None
		if games_played_so_far > 0 and total_points_so_far > 0:
			current_week_ppg = float(total_points_so_far) / float(games_played_so_far)
			st.caption(f"Derived weekly PPG: {current_week_ppg:.2f} over {games_played_so_far} games.")
		else:
			current_week_ppg = None
			st.info(
				"If you leave these at zero, PPG impact metrics will be omitted. "
				"Once some games have been played, enter games and total points to see PPG impact."
			)

	st.markdown("---")
	st.subheader("Step 2 – Players, Opponents & Context")
	st.caption(
		"Pick players from your league and optionally supply tonight's opponent, a softness "
		"score, and whether key teammates are out (usage bump)."
	)

	# Editable player table: all inputs in one place so you don't have to repeat
	# opponent names between tables.
	default_rows = [
		{"Player": "", "Opponent": "", "SoftnessScore": 3, "TeammatesOut?": False},
		{"Player": "", "Opponent": "", "SoftnessScore": 3, "TeammatesOut?": False},
	]
	player_df = st.data_editor(
		pd.DataFrame(default_rows),
		key="tonight_players_editor",
		use_container_width=True,
		num_rows="dynamic",
		column_config={
			"Player": st.column_config.SelectboxColumn(
				"Player",
				options=player_options,
				help="Rostered players from the latest Fantrax-Players CSV. Start typing to search.",
				width="medium",
			),
			"Opponent": st.column_config.TextColumn(
				"Opponent",
				help="Short team code like BOS, CLE, WAS.",
				width="small",
			),
			"SoftnessScore": st.column_config.NumberColumn(
				"Softness (1=tough, 5=soft)",
				min_value=1,
				max_value=5,
				step=1,
				width="small",
			),
			"TeammatesOut?": st.column_config.CheckboxColumn(
				"Teammates out?",
				help="Check if multiple teammates are out / expected usage bump.",
			),
		},
	)

	# Build mapping inputs from the combined table
	valid_rows = player_df[player_df["Player"].astype(str).str.strip() != ""].copy()
	player_names = valid_rows["Player"].astype(str).str.strip().tolist()

	opponent_by_player: dict[str, str] = {}
	opponent_strength: dict[str, float] = {}
	for _, row in valid_rows.iterrows():
		name = str(row.get("Player", "")).strip()
		opp = str(row.get("Opponent", "")).strip()
		if opp:
			opponent_by_player[name] = opp
			try:
				soft = float(row.get("SoftnessScore", 3))
			except Exception:
				soft = 3.0
			# If multiple players face the same team we simply overwrite with the
			# latest softness score; they should normally match anyway.
			opponent_strength[opp] = soft

	# Keep teammates-out flags for potential future use; for now they are
	# informational only and can help annotate decisions manually.
	teammates_out_flags = {
		str(row.get("Player", "")).strip(): bool(row.get("TeammatesOut?", False))
		for _, row in valid_rows.iterrows()
	}

	st.markdown("---")

	col_run, col_info = st.columns([1, 2])
	with col_run:
		run = st.button("🔍 Compare Players", type="primary")
	with col_info:
		st.write("Fill in at least one player, then click **Compare Players**.")

	if not run:
		return

	if not league_id:
		st.error("Please enter a Fantrax League ID before running the comparison.")
		return

	if len(player_names) < 1:
		st.error("Enter at least one player to compare.")
		return

	with st.spinner("Analyzing players from cached game logs..."):
		result_df = compare_players_for_date(
			player_names=player_names,
			game_date=game_date,
			league_id=league_id,
			season=None,
			opponent_by_player=opponent_by_player if opponent_by_player else None,
			opponent_strength=opponent_strength if opponent_strength else None,
			injured_flags=None,
			min_games_required=min_games_required if current_week_ppg is not None else None,
			games_played_so_far=games_played_so_far if current_week_ppg is not None else None,
			current_week_ppg=current_week_ppg if current_week_ppg is not None else None,
		)

	if result_df is None or result_df.empty:
		st.warning("No results were produced. Ensure that cached logs exist for the selected players.")
		return

	st.subheader("Results")
	st.caption(
		"Players are ordered by decision tier, then by utility score. "
		"If weekly context was provided, PPG impact columns show how an extra game would move your team PPG."
	)

	# If PPG impact fields are available, show them alongside the core metrics.
	if {"ppg_before", "ppg_after_if_started", "ppg_delta", "ppg_tag"}.issubset(result_df.columns):
		display_cols = [
			"player_name",
			"decision",
			"expected_fpts",
			"confidence",
			"utility",
			"ppg_before",
			"ppg_after_if_started",
			"ppg_delta",
			"ppg_tag",
			"reason",
		]
		display_df = result_df[[c for c in display_cols if c in result_df.columns]].copy()
	else:
		display_df = result_df

	st.dataframe(
		display_df,
		use_container_width=True,
		hide_index=True,
		column_config={
			"expected_fpts": st.column_config.NumberColumn(
				"expected_fpts",
				help="Blended projection using Empirical Bayes shrinkage (recent form regressed toward baseline).",
				format="%.1f",
			),
			"confidence": st.column_config.NumberColumn(
				"confidence",
				help="0–1 score combining recent-form stability and matchup favorability. Informational only.",
				format="%.2f",
			),
			"utility": st.column_config.NumberColumn(
				"utility",
				help="Risk-adjusted score: E[FPts] - λ×σ. Penalizes variance proportionally, not multiplicatively.",
				format="%.1f",
			),
			"reason": st.column_config.TextColumn(
				"reason",
				width="large",
			),
		},
	)

	st.markdown("---")
	st.caption(
		"This helper uses cached game logs only. If a player is missing data, run Bulk Scrape in Admin Tools first. "
		"The 'Teammates out?' flag is currently informational only; future versions may factor it into expected FPts."
	)


def _get_default_week_start() -> date:
	"""Return Monday of the current week as a sensible default start date."""
	today = date.today()
	return today - timedelta(days=today.weekday())



def _parse_opponent_from_cell(raw: str, player_team: str | None = None) -> str | None:
	"""Best-effort parsing of an opponent code from a Fantrax schedule cell.

	Cells can look like "MIL 126<br/>@WAS 129 F" or "NY 117<br/>@BOS 123 F".
	We parse both team tokens and, if the player's team is known, return the
	*other* team as the opponent. If we can't determine that, fall back to the
	first token.
	"""
	raw = str(raw or "").strip()
	if not raw:
		return None

	parts = str(raw).split("<br/>")
	team_tokens: list[str] = []
	for part in parts:
		p = part.strip()
		if not p:
			continue
		first_token = p.split()[0]
		code = first_token.lstrip("@")
		if code:
			team_tokens.append(code)

	if not team_tokens:
		return None

	if player_team:
		player_team = player_team.strip()
		if player_team and len(team_tokens) >= 2:
			# If one of the tokens matches the player's team, treat the other as opp.
			if team_tokens[0] == player_team and team_tokens[1] != player_team:
				return team_tokens[1]
			if team_tokens[1] == player_team and team_tokens[0] != player_team:
				return team_tokens[0]

	# Fallback: use the first token.
	return team_tokens[0]



def show_weekly_planner(default_league_id: str | None = None):
	"""Weekly schedule grid for a manager's team.

	This tab lets you:
	- Load your roster automatically from the latest Fantrax-Players YTD CSV.
	- Build a weekly schedule grid (players × days with opponents).
	- Analyze any single day using the same comparison engine as the tonight helper,
	  with optional weekly PPG context.
	"""
	st.header("📅 Weekly Lineup Planner (Grid)")
	st.write(
		"Build your weekly schedule for your team, then analyze any single day using recent "
		"form, matchup softness, and optional weekly PPG impact."
	)

	# Load the latest Fantrax-Players YTD CSV so we can list teams and filter to a single
	# fantasy roster (using the Status column as the manager/team identifier).
	from pathlib import Path  # imported lazily to avoid circulars at module import time

	roster_players: list[str] = []
	team_options: list[str] = []
	player_df: pd.DataFrame | None = None
	data_dir = Path(__file__).resolve().parent.parent.parent / "data"
	player_files = sorted(
		data_dir.glob("Fantrax-Players-*-(YTD).csv"),
		key=lambda f: f.stat().st_mtime,
		reverse=True,
	)
	if player_files:
		csv_file = player_files[0]
		try:
			player_df = pd.read_csv(csv_file)
			if {"Player", "Status"}.issubset(player_df.columns):
				# Filter out free agents and any status codes that don't have a
				# friendly mapping (we only want real fantasy teams here).
				mask = player_df["Status"].notna() & player_df["Status"].isin(TEAM_MAPPINGS.keys())
				player_df = player_df[mask].copy()
				team_options = sorted(player_df["Status"].dropna().unique().tolist())
		except Exception:
			player_df = None
			team_options = []

	with st.expander("Step 1 – Week & League Settings", expanded=True):
		col1, col2, col3 = st.columns(3)
		with col1:
			league_id = st.text_input(
				"Fantrax League ID",
				value=default_league_id or "",
				help="Used to resolve player codes when analyzing a day.",
				key="weekly_league_id",
			)
		with col2:
			default_start = _get_default_week_start()
			week_start = st.date_input("Week start (Monday)", value=default_start, key="weekly_week_start")
		with col3:
			if team_options:
				selected_team = st.selectbox(
					"Your fantasy team",
					options=team_options,
					format_func=lambda code: TEAM_MAPPINGS.get(code, code),
					key="weekly_team_select",
				)
			else:
				selected_team = None
				st.warning(
					"Could not detect any teams from the latest Fantrax-Players YTD file. "
					"Ensure the CSV exists and includes a Status column."
				)

		week_dates = [week_start + timedelta(days=i) for i in range(7)]
		day_labels = [d.strftime("%a %m/%d") for d in week_dates]

		c1, c2, c3 = st.columns(3)
		with c1:
			min_games_required = st.number_input(
				"Min games this week",
				min_value=0,
				value=25,
				step=1,
				key="weekly_min_games_this_week",
			)
		with c2:
			games_played_so_far = st.number_input(
				"Games started so far",
				min_value=0,
				value=0,
				step=1,
				key="weekly_games_started_so_far",
			)
		with c3:
			total_points_so_far = st.number_input(
				"Total points so far this week",
				min_value=0.0,
				value=0.0,
				step=1.0,
				key="weekly_total_points_so_far",
			)

		current_week_ppg: float | None
		if games_played_so_far > 0 and total_points_so_far > 0:
			current_week_ppg = float(total_points_so_far) / float(games_played_so_far)
			st.caption(
				f"Derived weekly PPG: {current_week_ppg:.2f} over {games_played_so_far} games. "
				"PPG impact will be factored into daily analyses."
			)
		else:
			current_week_ppg = None
			st.info(
				"If you leave games and total points at zero, PPG impact metrics will be omitted "
				"for the weekly analyses."
			)

		current_day = st.date_input(
			"Today's date (used to lock completed games)",
			value=date.today(),
			key="weekly_current_day",
		)

	st.markdown("---")
	st.subheader("Step 2 – Weekly Schedule Grid")
	st.caption(
		"Roster is pre-populated from the latest Fantrax-Players YTD export for the team you selected "
		"above. Enter opponents for each day in the format BOS, CLE, @IND, etc. Optionally upload a "
		"Fantrax team schedule CSV to auto-fill this grid."
	)

	if player_df is None or not team_options or not selected_team:
		st.warning("Unable to load a team roster from the Fantrax-Players YTD file.")
		return

	team_df = player_df[player_df["Status"] == selected_team].copy()
	if team_df.empty:
		st.warning(f"No players found for team '{selected_team}' in the latest YTD export.")
		return

	# Optional upload of a Fantrax schedule CSV (like ExampleSched.csv) to pre-fill the
	# weekly grid. This file is typically a per-team export, so we can trust that the
	# players listed belong to the selected fantasy team.
	sched_upload = st.file_uploader(
		"Optional: Upload Fantrax team schedule CSV",
		type=["csv"],
		key="weekly_schedule_upload",
	)

	parsed_schedule_df: pd.DataFrame | None = None
	parsed_day_labels: list[str] | None = None
	if sched_upload is not None:
		try:
			# Fantrax schedule exports often have a first header row with a blank column and
			# a second header row with real column names (ID, Pos, Player, Team, Eligible, ...).
			sched_df = pd.read_csv(sched_upload, header=1)
			if "Player" in sched_df.columns:
				# Drop FAs from the schedule view.
				if "Status" in sched_df.columns:
					sched_df = sched_df[sched_df["Status"] != "FA"].copy()
				# Detect day columns by header prefix.
				day_candidates: list[str] = [
					c
					for c in sched_df.columns
					if isinstance(c, str)
					and c.startswith(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
				]
				if day_candidates:
					parsed_day_labels = day_candidates[: len(day_labels)]
					day_labels = parsed_day_labels
					base_cols: list[str] = ["Player"]
					if "Team" in sched_df.columns:
						base_cols.append("Team")
					parsed_schedule_df = sched_df[base_cols + parsed_day_labels].copy()
		except Exception as e:
			st.error(f"Could not parse uploaded schedule CSV: {e}")
	
	# Build the base schedule data (from upload or empty roster)
	if parsed_schedule_df is not None and parsed_day_labels is not None:
		base_schedule_df = parsed_schedule_df.copy()
	else:
		# Fallback: build an empty grid from the YTD roster for the selected team.
		roster_players = sorted(team_df["Player"].astype(str).tolist())
		base_data = {"Player": roster_players}
		for label in day_labels:
			base_data[label] = ""
		base_schedule_df = pd.DataFrame(base_data)

	# Build a combined schedule + started grid: each day has opponent cell + started checkbox
	# Interleave columns: Day1 (opp), Day1 ✓, Day2 (opp), Day2 ✓, ...
	combined_base = {"Player": base_schedule_df["Player"].astype(str).tolist()}
	if "Team" in base_schedule_df.columns:
		combined_base["Team"] = base_schedule_df["Team"].astype(str).tolist()
	for label in day_labels:
		# Copy opponent data from base schedule
		if label in base_schedule_df.columns:
			combined_base[label] = base_schedule_df[label].astype(str).tolist()
		else:
			combined_base[label] = [""] * len(base_schedule_df)
		combined_base[f"{label} ✓"] = [False] * len(base_schedule_df)

	combined_df = st.data_editor(
		pd.DataFrame(combined_base),
		key="weekly_combined_editor",
		use_container_width=True,
		num_rows="fixed",
		column_config={
			**{f"{label} ✓": st.column_config.CheckboxColumn(
				f"✓",
				help=f"Check if you started this player on {label}",
				width="small",
			) for label in day_labels},
		},
	)

	# Extract schedule data back from combined grid for downstream use
	schedule_df = combined_df[["Player"] + (["Team"] if "Team" in combined_df.columns else []) + day_labels].copy()

	st.markdown("---")
	st.subheader("Injury Flags")
	st.caption(
		"Mark players who were injured on specific days or the entire week. "
		"Injured-all-week players are excluded from weekly recommendations."
	)

	injury_base = {"Player": schedule_df["Player"].astype(str).tolist()}
	for label in day_labels:
		injury_base[f"{label} injured?"] = False
	injury_base["Injured all week"] = False

	injury_df = st.data_editor(
		pd.DataFrame(injury_base),
		key="weekly_injury_editor",
		use_container_width=True,
		num_rows="fixed",
	)

	# Build a flags_df that merges started from combined_df and injured from injury_df
	# This keeps downstream code compatible
	flags_base = {"Player": schedule_df["Player"].astype(str).tolist()}
	for label in day_labels:
		started_col = f"{label} ✓"
		if started_col in combined_df.columns:
			flags_base[f"{label} started?"] = combined_df[started_col].tolist()
		else:
			flags_base[f"{label} started?"] = [False] * len(schedule_df)
		injured_col = f"{label} injured?"
		if injured_col in injury_df.columns:
			flags_base[f"{label} injured?"] = injury_df[injured_col].tolist()
		else:
			flags_base[f"{label} injured?"] = [False] * len(schedule_df)
	if "Injured all week" in injury_df.columns:
		flags_base["Injured all week"] = injury_df["Injured all week"].tolist()
	else:
		flags_base["Injured all week"] = [False] * len(schedule_df)
	flags_df = pd.DataFrame(flags_base)

	st.markdown("---")
	st.subheader("Played Games Summary (optional)")
	if not league_id:
		st.info("Enter a Fantrax League ID above to backfill actual scores from cached logs.")
	else:
		played_rows: list[dict[str, object]] = []
		flags_lookup = None
		if "Player" in flags_df.columns:
			try:
				flags_lookup = flags_df.set_index("Player")
			except Exception:
				flags_lookup = None
		for _, row in schedule_df.iterrows():
			player_name = str(row.get("Player", "")).strip()
			if not player_name:
				continue
			flags_row = None
			if flags_lookup is not None and player_name in flags_lookup.index:
				flags_row = flags_lookup.loc[player_name]
			for idx, label in enumerate(day_labels):
				day_date = week_dates[idx]
				if day_date >= current_day:
					continue
				cell_value = str(row.get(label, "")).strip()
				if not cell_value:
					continue
				player_team = (
					str(row.get("Team", "")).strip() if "Team" in schedule_df.columns else None
				)
				opp = _parse_opponent_from_cell(cell_value, player_team=player_team)
				if not opp:
					continue
				started_col = f"{label} started?"
				injured_col = f"{label} injured?"
				started_val = False
				injured_val = False
				if flags_row is not None:
					if started_col in flags_row.index:
						started_val = bool(flags_row.get(started_col, False))
					if injured_col in flags_row.index:
						injured_val = injured_val or bool(flags_row.get(injured_col, False))
					if "Injured all week" in flags_row.index:
						injured_val = injured_val or bool(flags_row.get("Injured all week", False))
				if not started_val:
					continue
				actual_fpts = get_actual_fpts_for_date(
					player_name=player_name,
					game_date=day_date,
					league_id=league_id,
					season=None,
					opponent=opp,
				)
				data_source = "found" if actual_fpts is not None else "missing_log"
				played_rows.append(
					{
						"Player": player_name,
						"Date": label,
						"Opponent": opp,
						"Started": started_val,
						"Injured": injured_val,
						"Actual_FPts": actual_fpts,
						"Data_source": data_source,
					}
				)
		if played_rows:
			played_df = pd.DataFrame(played_rows)
			st.dataframe(played_df, use_container_width=True, hide_index=True)
			# Also show a grid in the same shape as the schedule, but with FPts in
			# each cell instead of opponent strings so it is easy to see which
			# games have already contributed to your weekly total.
			fpts_grid = pd.DataFrame({"Player": schedule_df["Player"].astype(str).tolist()})
			for label in day_labels:
				fpts_grid[label] = ""
			for row in played_rows:
				p_name = str(row.get("Player", "")).strip()
				day_label = str(row.get("Date", "")).strip()
				fpts = row.get("Actual_FPts")
				if day_label not in fpts_grid.columns or fpts is None:
					continue
				mask = fpts_grid["Player"].astype(str).str.strip() == p_name
				if not mask.any():
					continue
				try:
					fpts_str = f"{float(fpts):.0f}"
				except Exception:
					fpts_str = ""
				fpts_grid.loc[mask, day_label] = fpts_str
			st.dataframe(
				fpts_grid,
				use_container_width=True,
				hide_index=True,
			)
		else:
			st.info("No completed started games found before today's date.")

	# ─────────────────────────────────────────────────────────────────────────────
	# Step 3A – Analyze a Day
	# ─────────────────────────────────────────────────────────────────────────────
	st.markdown("---")
	st.subheader("Step 3A – Analyze a Day")
	st.caption(
		"Pick a day from the grid and we'll analyze only the players with a non-empty opponent "
		"cell on that day using the same engine as Tonight's Helper."
	)

	if not league_id:
		st.error("Please enter a Fantrax League ID before analyzing a day or the week.")
		return

	day_index = st.selectbox(
		"Day to analyze",
		options=list(range(len(day_labels))),
		format_func=lambda idx: day_labels[idx],
	)

	col_run, col_hint = st.columns([1, 2])
	with col_run:
		run_day = st.button("🔍 Analyze Selected Day", type="primary")
	with col_hint:
		st.write(
			"Only players with an opponent listed for the selected day will be included in the analysis."
		)

	if run_day:
		day_label = day_labels[day_index]
		day_date = week_dates[day_index]

		base_cols = ["Player", day_label]
		if "Team" in schedule_df.columns:
			base_cols.append("Team")
		valid_rows = schedule_df[base_cols].copy()
		valid_rows[day_label] = valid_rows[day_label].astype(str).str.strip()
		no_game_values = {"", "none", "nan", "na", "-"}
		mask_has_game = ~valid_rows[day_label].str.lower().isin(no_game_values)
		day_rows = valid_rows[mask_has_game]
		if day_rows.empty:
			st.warning("No players have a scheduled game on this day in the grid.")
		else:
			# Look up flags for this day and full-week injuries from the separate flags grid.
			flags_lookup = None
			if "Player" in flags_df.columns:
				try:
					flags_lookup = flags_df.set_index("Player")
				except Exception:
					flags_lookup = None

			player_names: list[str] = []
			opponent_by_player: dict[str, str] = {}
			injured_flags: dict[str, bool] = {}
			day_injured_col = f"{day_label} injured?"
			for _, row in day_rows.iterrows():
				name = str(row["Player"]).strip()
				if not name:
					continue
				flags_row = None
				if flags_lookup is not None and name in flags_lookup.index:
					flags_row = flags_lookup.loc[name]
				# Skip players marked injured for the full week entirely.
				if flags_row is not None and "Injured all week" in flags_row.index and bool(flags_row.get("Injured all week", False)):
					continue
				raw = str(row[day_label]).strip()
				if not raw:
					continue
				player_team = (
					str(row.get("Team", "")).strip() if "Team" in day_rows.columns else None
				)
				opp = _parse_opponent_from_cell(raw, player_team=player_team)
				if not opp:
					continue
				opponent_by_player[name] = opp
				is_injured = False
				if flags_row is not None:
					if day_injured_col in flags_row.index and bool(flags_row.get(day_injured_col, False)):
						is_injured = True
					if "Injured all week" in flags_row.index and bool(flags_row.get("Injured all week", False)):
						is_injured = True
				if is_injured:
					injured_flags[name] = True
				player_names.append(name)

			with st.spinner("Analyzing selected day from cached game logs..."):
				result_df = compare_players_for_date(
					player_names=player_names,
					game_date=day_date.isoformat(),
					league_id=league_id,
					season=None,
					opponent_by_player=opponent_by_player if opponent_by_player else None,
					opponent_strength=None,
					injured_flags=injured_flags if injured_flags else None,
					min_games_required=min_games_required if current_week_ppg is not None else None,
					games_played_so_far=games_played_so_far if current_week_ppg is not None else None,
					current_week_ppg=current_week_ppg if current_week_ppg is not None else None,
				)

			if result_df is None or result_df.empty:
				st.warning("No results were produced for this day. Ensure cached logs exist for the selected players.")
			else:
				st.subheader(f"Results for {day_label}")
				st.caption(
					"Players are ordered by decision tier, then utility. "
					"`expected_fpts` is a blended projection from recent form and baseline stats. "
					"`utility` is a risk-adjusted score (expected_fpts minus a penalty for volatility). "
					"If weekly context was provided, PPG impact columns show how starting this game would move your team PPG."
				)

				if {"ppg_before", "ppg_after_if_started", "ppg_delta", "ppg_tag"}.issubset(result_df.columns):
					display_cols = [
						"player_name",
						"decision",
						"expected_fpts",
						"confidence",
						"utility",
						"ppg_before",
						"ppg_after_if_started",
						"ppg_delta",
						"ppg_tag",
						"reason",
					]
					display_df = result_df[[c for c in display_cols if c in result_df.columns]].copy()
				else:
					display_df = result_df

				st.dataframe(
					display_df,
					use_container_width=True,
					hide_index=True,
					column_config={
						"expected_fpts": st.column_config.NumberColumn(
							"expected_fpts",
							help="Blended projection using Empirical Bayes shrinkage (recent form regressed toward baseline).",
							format="%.1f",
						),
						"confidence": st.column_config.NumberColumn(
							"confidence",
							help="0–1 score combining recent-form stability and matchup favorability. Informational only.",
							format="%.2f",
						),
						"utility": st.column_config.NumberColumn(
							"utility",
							help="Risk-adjusted score: E[FPts] - λ×σ. Penalizes variance proportionally, not multiplicatively.",
							format="%.1f",
						),
						"reason": st.column_config.TextColumn(
							"reason",
							width="large",
						),
					},
				)

	# ─────────────────────────────────────────────────────────────────────────────
	# Step 3B – Weekly Start Recommendations
	# ─────────────────────────────────────────────────────────────────────────────
	st.markdown("---")
	st.subheader("Step 3B – Weekly Start Recommendations")
	st.caption(
		"Generate optimal start recommendations for the remaining week. It uses the same "
		"expected_fpts and utility model as Step 3A, builds a pool of all remaining games, and "
		"chooses the number of starts that maximizes your final projected weekly PPG while "
		"respecting your min games target and excluding injured-all-week players."
	)

	col_opt, col_hint = st.columns([1, 2])
	with col_opt:
		run_optimizer = st.button("🎯 Recommend Week", type="primary")
	with col_hint:
		st.write(
			"This will analyze all remaining days and suggest optimal starts to hit your min games target."
		)

	if not run_optimizer:
		return

	# Build injured-all-week set
	injured_all_week_set: set[str] = set()
	if "Injured all week" in flags_df.columns:
		for _, row in flags_df.iterrows():
			if bool(row.get("Injured all week", False)):
				injured_all_week_set.add(str(row.get("Player", "")).strip())

	# Count games already started (before today)
	games_already_started = 0
	for _, row in flags_df.iterrows():
		for idx_day, label_day in enumerate(day_labels):
			day_date_check = week_dates[idx_day]
			if day_date_check >= current_day:
				continue
			started_col = f"{label_day} started?"
			if started_col in flags_df.columns and bool(row.get(started_col, False)):
				games_already_started += 1

	# Use the user-provided games_played_so_far and total_points_so_far if available.
	# We treat these as the fixed baseline for global optimization, then simulate
	# added starts on top of this baseline when building the projections.
	base_games = games_played_so_far if games_played_so_far > 0 else games_already_started
	base_points = total_points_so_far if total_points_so_far > 0 else 0.0

	running_games = base_games
	running_points = base_points

	# Calculate remaining games needed relative to the baseline only.
	games_remaining_needed = max(0, min_games_required - base_games)

	# Count total available game slots for remaining days (excluding injured-all-week)
	total_available_slots = 0
	for idx_slot, label_slot in enumerate(day_labels):
		day_date_check = week_dates[idx_slot]
		if day_date_check < current_day:
			continue
		for _, row in schedule_df.iterrows():
			player_name = str(row.get("Player", "")).strip()
			if player_name in injured_all_week_set:
				continue
			cell_value = str(row.get(label_slot, "")).strip()
			if cell_value and cell_value.lower() not in {"", "none", "nan", "na", "-"}:
				total_available_slots += 1

	st.info(
		f"📊 **Week Status**: {base_games} games started, {games_remaining_needed} more needed to hit {min_games_required}. "
		f"{total_available_slots} game slots available for remaining days."
	)

	# Walk through each remaining day once to collect all candidate starts into a
	# single pool. We'll then choose the optimal subset globally, and finally map
	# them back to days to build the summaries.
	all_candidates: list[dict] = []
	day_available_counts: dict[str, int] = {}
	day_indices: dict[str, int] = {}

	for idx_rec, label_rec in enumerate(day_labels):
		day_date_iter = week_dates[idx_rec]
		if day_date_iter < current_day:
			continue
		day_indices[label_rec] = idx_rec

		# Get players with games on this day (excluding injured-all-week)
		day_players: list[str] = []
		day_opponents: dict[str, str] = {}
		day_injured: dict[str, bool] = {}

		flags_lookup_opt = None
		if "Player" in flags_df.columns:
			try:
				flags_lookup_opt = flags_df.set_index("Player")
			except Exception:
				flags_lookup_opt = None

		for _, row in schedule_df.iterrows():
			player_name = str(row.get("Player", "")).strip()
			if not player_name or player_name in injured_all_week_set:
				continue
			cell_value = str(row.get(label_rec, "")).strip()
			if not cell_value or cell_value.lower() in {"", "none", "nan", "na", "-"}:
				continue
			player_team = str(row.get("Team", "")).strip() if "Team" in schedule_df.columns else None
			opp = _parse_opponent_from_cell(cell_value, player_team=player_team)
			if not opp:
				continue
			day_players.append(player_name)
			day_opponents[player_name] = opp

			# Check day-specific injury
			if flags_lookup_opt is not None and player_name in flags_lookup_opt.index:
				flags_row_opt = flags_lookup_opt.loc[player_name]
				day_injured_col_opt = f"{label_rec} injured?"
				if day_injured_col_opt in flags_row_opt.index and bool(flags_row_opt.get(day_injured_col_opt, False)):
					day_injured[player_name] = True

		if not day_players:
			day_available_counts[label_rec] = 0
			continue

		# Analyze this day but *do not* commit to starts yet
		day_result_df = compare_players_for_date(
			player_names=day_players,
			game_date=day_date_iter.isoformat(),
			league_id=league_id,
			season=None,
			opponent_by_player=day_opponents,
			opponent_strength=None,
			injured_flags=day_injured if day_injured else None,
			min_games_required=min_games_required,
			games_played_so_far=base_games,
			current_week_ppg=base_points / float(base_games) if base_games > 0 else None,
		)

		if day_result_df is None or day_result_df.empty:
			day_available_counts[label_rec] = len(day_players)
			continue

		# Sort by utility descending, filter out injured; all remaining rows are
		# candidates for the global optimizer.
		day_result_df = day_result_df.sort_values("utility", ascending=False)
		available_players = day_result_df[day_result_df["decision"] != "INJURED"]
		day_available_counts[label_rec] = len(available_players)
		if available_players.empty:
			continue

		for _, rec_row in available_players.iterrows():
			exp_fpts = float(rec_row.get("expected_fpts", 0) or 0.0)
			all_candidates.append(
				{
					"DayIndex": idx_rec,
					"Day": label_rec,
					"Player": rec_row["player_name"],
					"Opponent": day_opponents.get(rec_row["player_name"], ""),
					"Expected FPts": exp_fpts,
					"Utility": float(rec_row.get("utility", 0) or 0.0),
					"Decision": rec_row.get("decision", ""),
				}
			)

	# If there are no candidates at all, we can't recommend anything.
	if not all_candidates:
		st.warning("No recommendations generated. Check that players have scheduled games for remaining days.")
		return

	# Global optimization across all remaining games: choose how many starts to
	# take to maximize final PPG, subject to meeting the minimum if enough slots
	# exist.
	cand_df = pd.DataFrame(all_candidates)
	cand_df = cand_df.sort_values(["Expected FPts", "Utility"], ascending=[False, False]).reset_index(drop=True)

	expected_vals = [float(x or 0.0) for x in cand_df["Expected FPts"].tolist()]
	prefix_sums: list[float] = []
	_running = 0.0
	for val in expected_vals:
		_running += val
		prefix_sums.append(_running)

	total_candidates = len(expected_vals)
	needed_games = max(0, min_games_required - base_games)

	# If even playing every remaining game does not reach the minimum, we must
	# simply start everyone.
	if base_games + total_candidates <= min_games_required:
		chosen_n = total_candidates
	else:
		best_n = 0
		best_ppg: float | None
		if needed_games == 0 and base_games > 0:
			best_ppg = base_points / float(base_games)
		else:
			best_ppg = None

		start_n = max(1, needed_games) if total_candidates > 0 else 0
		for n in range(start_n, total_candidates + 1):
			total_pts_n = base_points + prefix_sums[n - 1]
			total_games_n = base_games + n
			ppg_n = total_pts_n / float(total_games_n)
			if best_ppg is None or ppg_n > best_ppg + 1e-9:
				best_ppg = ppg_n
				best_n = n
		chosen_n = best_n

	cand_df["chosen"] = False
	if chosen_n > 0:
		cand_df.loc[: chosen_n - 1, "chosen"] = True

	chosen_df = cand_df[cand_df["chosen"]].copy()
	if chosen_df.empty:
		st.warning("No recommendations generated. Check that players have scheduled games for remaining days.")
		return

	# Build daily summaries and detailed recommendations from the chosen set.
	recommendations: list[dict] = []
	daily_summaries: list[dict] = []
	available_by_day = cand_df.groupby("Day").size().to_dict()
	chosen_by_day = {day: grp for day, grp in chosen_df.groupby("Day")}

	running_games = base_games
	running_points = base_points

	for idx_rec, label_rec in enumerate(day_labels):
		day_date_iter = week_dates[idx_rec]
		if day_date_iter < current_day:
			continue

		available_count = int(available_by_day.get(label_rec, 0))
		day_chosen = chosen_by_day.get(label_rec)
		if day_chosen is None or day_chosen.empty:
			projected_ppg_after_day = running_points / float(running_games) if running_games > 0 else None
			daily_summaries.append(
				{
					"Day": label_rec,
					"Available": available_count,
					"Recommended": 0,
					"Players": "-",
					"Projected games": running_games if running_games > 0 else None,
					"Projected PPG": round(projected_ppg_after_day, 2) if projected_ppg_after_day is not None else None,
				}
			)
			continue

		day_chosen = day_chosen.sort_values(["Utility", "Expected FPts"], ascending=[False, False])
		rec_names: list[str] = []
		for _, row in day_chosen.iterrows():
			name = str(row.get("Player", "")).strip()
			if not name:
				continue
			rec_names.append(name)
			exp_fpts = float(row.get("Expected FPts", 0.0) or 0.0)
			recommendations.append(
				{
					"Day": label_rec,
					"Player": name,
					"Opponent": str(row.get("Opponent", "")),
					"Expected FPts": round(exp_fpts, 1),
					"Utility": round(float(row.get("Utility", 0.0) or 0.0), 2),
					"Decision": row.get("Decision", ""),
				}
			)
			# Update running totals for projections
			running_games += 1
			running_points += exp_fpts

		projected_ppg_after_day = running_points / float(running_games) if running_games > 0 else None
		daily_summaries.append(
			{
				"Day": label_rec,
				"Available": available_count,
				"Recommended": len(rec_names),
				"Players": ", ".join(rec_names) if rec_names else "-",
				"Projected games": running_games if running_games > 0 else None,
				"Projected PPG": round(projected_ppg_after_day, 2) if projected_ppg_after_day is not None else None,
			}
		)

	# Display results
	if recommendations:
		st.success(f"✅ Recommended {len(recommendations)} starts across remaining days to optimize your week.")

		# Daily summary table
		st.markdown("#### Daily Summary")
		summary_df = pd.DataFrame(daily_summaries)
		st.dataframe(summary_df, use_container_width=True, hide_index=True)

		# Recommendations views
		st.markdown("#### Recommendations")
		rec_df = pd.DataFrame(recommendations)
		tab_by_game, tab_by_player = st.tabs(["By game", "By player"])
		with tab_by_game:
			st.dataframe(rec_df, use_container_width=True, hide_index=True)
		with tab_by_player:
			if rec_df.empty:
				st.info("No recommended starts to summarize by player.")
			else:
				grouped = rec_df.groupby("Player", sort=False)

				def _format_days(group: pd.DataFrame) -> str:
					parts = []
					for _, r in group.sort_values("Day").iterrows():
						day_label = str(r.get("Day", "")).strip()
						opp = str(r.get("Opponent", "")).strip()
						if opp:
							parts.append(f"{day_label} ({opp})")
						else:
							parts.append(day_label)
					return ", ".join(parts)

				_days_series = grouped.apply(_format_days)
				_stats = grouped.agg(
					Starts=("Day", "count"),
					AvgExpected=("Expected FPts", "mean"),
					TotalUtility=("Utility", "sum"),
				)

				per_player_df = _stats.join(_days_series.rename("Days / Opponents"))
				per_player_df["AvgExpected"] = per_player_df["AvgExpected"].round(1)
				per_player_df["TotalUtility"] = per_player_df["TotalUtility"].round(2)
				per_player_df = per_player_df.reset_index().rename(
					columns={
						"Player": "Player",
						"Starts": "Starts",
						"AvgExpected": "Expected FPts / start",
					},
				)
				per_player_df = per_player_df[
					["Player", "Starts", "Days / Opponents", "Expected FPts / start", "TotalUtility"]
				]

				st.dataframe(per_player_df, use_container_width=True, hide_index=True)
	else:
		st.warning("No recommendations generated. Check that players have scheduled games for remaining days.")

	# Final projection
	final_games = running_games
	final_ppg = running_points / final_games if final_games > 0 else 0
	st.info(
		f"📈 **Projected Week**: {final_games} games, {running_points:.0f} total points, "
		f"{final_ppg:.2f} PPG"
	)


def show_stat_line_calculator() -> None:
	"""Interactive tool to convert a raw box score into fantasy points.

	This uses the league's scoring rules from the constitution:

	- 3D (0 points)
	- 2D (4 points)
	- 3PTM (3 points)
	- AST (3 points)
	- BLK (4 points)
	- EJ (-12 points)
	- FG- (-2 points)
	- FT- (-1 point)
	- OREB (1 point)
	- PTS (2 points)
	- REB (2 points)
	- ST (6 points)
	- TF (-6 points)
	- TO (-2 points)

	We infer 2D/3D automatically from how many of PTS/REB/AST/ST/BLK are >= 10.
	"""
	st.header("📊 Stat Line Fantasy Score Calculator")
	st.caption("Enter a box score and see the equivalent fantasy score using your league's scoring.")

	with st.expander("Step 1 – Box Score", expanded=True):
		c1, c2, c3, c4 = st.columns(4)
		with c1:
			pts = st.number_input("Points (PTS)", min_value=0, step=1)
			reb = st.number_input("Rebounds (REB)", min_value=0, step=1)
			ast = st.number_input("Assists (AST)", min_value=0, step=1)
		with c2:
			stl = st.number_input("Steals (ST)", min_value=0, step=1)
			blk = st.number_input("Blocks (BLK)", min_value=0, step=1)
			oreb = st.number_input("Off. Rebounds (OREB)", min_value=0, step=1)
		with c3:
			fgm = st.number_input("FG Made (FGM)", min_value=0, step=1)
			fga = st.number_input("FG Attempted (FGA)", min_value=0, step=1)
			threes = st.number_input("3PM (3PTM)", min_value=0, step=1)

		with c4:
			ftm = st.number_input("FT Made (FTM)", min_value=0, step=1)
			fta = st.number_input("FT Attempted (FTA)", min_value=0, step=1)
			turnovers = st.number_input("Turnovers (TO)", min_value=0, step=1)

	with st.expander("Step 2 – Misc. Events", expanded=False):
		c5, c6, c7 = st.columns(3)
		with c5:
			techs = st.number_input("Tech fouls (TF)", min_value=0, step=1)
		with c6:
			ejections = st.number_input("Ejections (EJ)", min_value=0, step=1)
		with c7:
			st.caption("2D/3D bonuses are derived automatically from PTS/REB/AST/ST/BLK.")

	# Derived stats
	fg_missed = max(0, fga - fgm)
	ft_missed = max(0, fta - ftm)

	double_stats = [pts, reb, ast, stl, blk]
	n_double_categories = sum(1 for v in double_stats if v >= 10)
	d2 = 1 if n_double_categories >= 2 else 0
	d3 = 1 if n_double_categories >= 3 else 0

	# Scoring weights
	weights = {
		"PTS": 2.0,
		"REB": 2.0,
		"AST": 3.0,
		"ST": 6.0,
		"BLK": 4.0,
		"OREB": 1.0,
		"3PTM": 3.0,
		"FG-": -2.0,
		"FT-": -1.0,
		"TO": -2.0,
		"TF": -6.0,
		"EJ": -12.0,
		"2D": 4.0,
		"3D": 0.0,
	}

	components = {
		"PTS": pts * weights["PTS"],
		"REB": reb * weights["REB"],
		"AST": ast * weights["AST"],
		"ST": stl * weights["ST"],
		"BLK": blk * weights["BLK"],
		"OREB": oreb * weights["OREB"],
		"3PTM": threes * weights["3PTM"],
		"FG-": fg_missed * weights["FG-"],
		"FT-": ft_missed * weights["FT-"],
		"TO": turnovers * weights["TO"],
		"TF": techs * weights["TF"],
		"EJ": ejections * weights["EJ"],
		"2D": d2 * weights["2D"],
		"3D": d3 * weights["3D"],
	}

	total_fpts = sum(components.values())

	st.markdown("---")
	st.subheader("Result")
	st.metric("Fantasy points", f"{total_fpts:.1f}")

	ordered_cats = [
		"FG-",
		"3PTM",
		"FT-",
		"PTS",
		"OREB",
		"REB",
		"AST",
		"ST",
		"BLK",
		"TO",
		"TF",
		"EJ",
		"3D",
		"2D",
	]

	breakdown_rows = []
	for cat in ordered_cats:
		contrib = components.get(cat, 0.0)
		if cat == "FG-":
			count = fg_missed
		elif cat == "FT-":
			count = ft_missed
		elif cat == "2D":
			count = d2
		elif cat == "3D":
			count = d3
		elif cat == "PTS":
			count = pts
		elif cat == "REB":
			count = reb
		elif cat == "AST":
			count = ast
		elif cat == "ST":
			count = stl
		elif cat == "BLK":
			count = blk
		elif cat == "OREB":
			count = oreb
		elif cat == "3PTM":
			count = threes
		elif cat == "TO":
			count = turnovers
		elif cat == "TF":
			count = techs
		elif cat == "EJ":
			count = ejections
		else:
			count = 0
		breakdown_rows.append(
			{
				"Category": cat,
				"Stat": count,
				"Weight": weights[cat],
				"Contribution": round(contrib, 1),
			}
		)

	breakdown_df = pd.DataFrame(breakdown_rows)
	st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
