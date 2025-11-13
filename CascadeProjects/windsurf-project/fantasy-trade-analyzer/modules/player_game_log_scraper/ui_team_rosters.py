import streamlit as st
import pandas as pd
from pathlib import Path
import json
from modules.player_game_log_scraper.logic import (
	calculate_variability_stats,
	get_cache_directory
)

def _find_latest_draft_results():
	data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
	candidates = sorted(list(data_dir.glob('Fantrax-Draft-Results-*.csv')), key=lambda p: p.stat().st_mtime, reverse=True)
	if candidates:
		return candidates[0]
	# fallbacks by season files if present
	for name in ['S4Draft.csv', 'S3Draft.csv', 'S2Draft.csv', 'S1Draft.csv']:
		p = data_dir / name
		if p.exists():
			return p
	return None

def _load_team_rosters_from_draft():
	csv_path = _find_latest_draft_results()
	if not csv_path or not csv_path.exists():
		return {}
	try:
		df = pd.read_csv(csv_path)
		cols = {c.lower(): c for c in df.columns}
		if 'team' not in cols or 'player' not in cols:
			return {}
		team_col = cols['team']
		player_col = cols['player']
		rosters = {}
		for _, row in df[[team_col, player_col]].dropna().iterrows():
			team = str(row[team_col]).strip()
			player = str(row[player_col]).strip()
			if not team or not player:
				continue
			rosters.setdefault(team, []).append(player)
		return rosters
	except Exception:
		return {}

def _cache_index_by_player_name(cache_files):
	index = {}
	for p in cache_files:
		try:
			with open(p, 'r') as f:
				data = json.load(f)
			name = data.get('player_name') or 'Unknown'
			index[name] = {
				'code': data.get('player_code', ''),
				'path': p,
				'games': pd.DataFrame(data.get('data', data.get('game_log', [])))
			}
		except Exception:
			continue
	return index

def _build_team_view(league_id, cache_files):
	rosters = _load_team_rosters_from_draft()
	cache_index = _cache_index_by_player_name(cache_files)
	teams = {}
	for team, players in rosters.items():
		rows = []
		for name in players:
			entry = cache_index.get(name)
			if not entry or entry['games'].empty or 'FPts' not in entry['games']:
				continue
			stats = calculate_variability_stats(entry['games'])
			if not stats:
				continue
			rows.append({
				'Player': name,
				'GP': stats['games_played'],
				'Mean FPts': round(stats['mean_fpts'], 1),
				'CV %': round(stats['coefficient_of_variation'], 1),
				'Boom %': round(stats['boom_rate'], 1),
				'Bust %': round(stats['bust_rate'], 1),
				'code': entry['code']
			})
		if rows:
			teams[team] = pd.DataFrame(rows).sort_values(['Mean FPts'], ascending=False).reset_index(drop=True)
	return teams

def _display_teams_overview(rosters_by_team):
	"""Display summary table of all teams with aggregate metrics."""
	summary_rows = []
	for team, df in rosters_by_team.items():
		if df.empty:
			continue
		summary_rows.append({
			'Team': team,
			'Players': len(df),
			'Avg FPts': round(df['Mean FPts'].mean(), 1),
			'Avg CV%': round(df['CV %'].mean(), 1),
			'🟢 Consistent': len(df[df['CV %'] < 20]),
			'🟡 Moderate': len(df[(df['CV %'] >= 20) & (df['CV %'] <= 30)]),
			'🔴 Volatile': len(df[df['CV %'] > 30]),
			'Total GP': int(df['GP'].sum())
		})
	
	if not summary_rows:
		st.info("No team summary data available.")
		return
	
	summary_df = pd.DataFrame(summary_rows).sort_values('Avg FPts', ascending=False)
	
	st.markdown("### Team Performance Summary")
	st.dataframe(
		summary_df,
		use_container_width=True,
		height=400,
		column_config={
			'Team': st.column_config.TextColumn('Team', width='medium'),
			'Players': st.column_config.NumberColumn('Players'),
			'Avg FPts': st.column_config.NumberColumn('Avg FPts', format='%.1f'),
			'Avg CV%': st.column_config.NumberColumn('Avg CV%', format='%.1f', help='Lower = more consistent'),
			'🟢 Consistent': st.column_config.NumberColumn('🟢 Consistent', help='CV% < 20%'),
			'🟡 Moderate': st.column_config.NumberColumn('🟡 Moderate', help='CV% 20-30%'),
			'🔴 Volatile': st.column_config.NumberColumn('🔴 Volatile', help='CV% > 30%'),
			'Total GP': st.column_config.NumberColumn('Total GP')
		}
	)
	
	# Download button
	csv = summary_df.to_csv(index=False)
	st.download_button(
		label="📥 Download Team Summary",
		data=csv,
		file_name="team_summary.csv",
		mime="text/csv"
	)

def show_team_rosters_viewer(league_id, cache_files, selected_season):
	st.subheader(f"🏟️ Team Rosters & Consistency - {selected_season}")
	rosters_by_team = _build_team_view(league_id, cache_files)
	if not rosters_by_team:
		st.info("No team roster data found from draft results or no matching cached players.")
		return
	
	# Add overview section
	with st.expander("📊 All Teams Overview", expanded=True):
		_display_teams_overview(rosters_by_team)
	
	st.markdown("---")
	st.subheader("🔍 Team Detail View")
	
	team_names = sorted(rosters_by_team.keys())
	col1, col2 = st.columns([2, 1])
	with col1:
		team = st.selectbox("Select Fantasy Team", options=team_names)
	with col2:
		min_gp = st.number_input("Min GP", 0, 82, 0)
	df = rosters_by_team.get(team)
	if df is None or df.empty:
		st.warning("No players available for this team.")
		return
	filtered = df[df['GP'] >= min_gp].copy()
	def _consistency(cv):
		return '🟢 Very Consistent' if cv < 20 else ('🟡 Moderate' if cv <= 30 else '🔴 Volatile')
	filtered.insert(3, 'Consistency', filtered['CV %'].apply(_consistency))
	st.dataframe(
		filtered.drop(columns=['code']),
		use_container_width=True,
		height=500,
		column_config={
			'Player': st.column_config.TextColumn('Player', width='medium'),
			'GP': st.column_config.NumberColumn('GP'),
			'Mean FPts': st.column_config.NumberColumn('Mean FPts', format='%.1f'),
			'CV %': st.column_config.NumberColumn('CV %', format='%.1f'),
			'Boom %': st.column_config.NumberColumn('Boom %', format='%.1f'),
			'Bust %': st.column_config.NumberColumn('Bust %', format='%.1f'),
			'Consistency': st.column_config.TextColumn('Consistency')
		}
	)
	st.markdown("---")
	player = st.selectbox("View Player Details", options=list(filtered['Player']))
	cache_dir = get_cache_directory()
	code = df.loc[df['Player'] == player, 'code'].values[0]
	# Look for new format files for this player
	cache_files = list(cache_dir.glob(f"player_game_log_full_{code}_{league_id}_*.json"))
	if not cache_files:
		st.error("Player data not found in cache.")
		return
	
	# Use the most recent season file
	cache_file = max(cache_files, key=lambda f: f.stat().st_mtime)
	if not cache_file.exists():
		st.error("Player data not found in cache.")
		return
	with open(cache_file, 'r') as f:
		data = json.load(f)
	name = data.get('player_name', player)
	games = pd.DataFrame(data.get('data', data.get('game_log', [])))
	if games.empty:
		st.warning("No game log data for this player.")
		return
	st.subheader(f"🔍 {name} - Details")
	stats = calculate_variability_stats(games)
	from modules.player_game_log_scraper.ui_components import (
		display_variability_metrics,
		display_boom_bust_analysis,
		display_fpts_trend_chart,
		display_distribution_chart,
		display_boom_bust_zones_chart,
		display_category_breakdown,
	)
	display_variability_metrics(stats, name)
	st.markdown("---")
	display_boom_bust_analysis(stats)
	st.markdown("---")
	viz1, viz2, viz3, viz4 = st.tabs(["FPts Trend", "Distribution", "Boom/Bust Zones", "Category Breakdown"])
	with viz1:
		display_fpts_trend_chart(games, stats, name)
	with viz2:
		display_distribution_chart(games, stats, name)
	with viz3:
		display_boom_bust_zones_chart(games, stats, name)
	with viz4:
		display_category_breakdown(games, name)
	st.markdown("---")
	priority_cols = ['Date', 'Team', 'Opp', 'Score', 'FPts', 'MIN', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
	other_cols = [c for c in games.columns if c not in priority_cols]
	display_cols = [c for c in priority_cols if c in games.columns] + other_cols
	st.dataframe(games[display_cols], use_container_width=True, height=350)
	csv = games.to_csv(index=False)
	st.download_button(
		label="📥 Download Player Game Log",
		data=csv,
		file_name=f"{name.replace(' ', '_')}_game_log.csv",
		mime="text/csv"
	)
