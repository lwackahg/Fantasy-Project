"""Read-only viewer for player consistency data - no scraping controls."""
import streamlit as st
import pandas as pd
from datetime import datetime
from modules.player_game_log_scraper.logic import (
	calculate_variability_stats,
	get_cache_directory,
	load_league_cache_index
)
from modules.player_game_log_scraper.ui_components import (
	display_variability_metrics,
	display_boom_bust_analysis,
	display_fpts_trend_chart,
	display_distribution_chart,
	display_boom_bust_zones_chart,
	display_category_breakdown
)
from modules.player_game_log_scraper.ui_league_overview import (
	_load_all_cached_data,
	_display_summary_metrics,
	_display_consistency_table,
	_display_visualizations
)
from modules.player_game_log_scraper.ui_team_rosters import (
	show_team_rosters_viewer
)
from modules.player_game_log_scraper.ui_fantasy_teams import (
	show_fantasy_teams_viewer
)
import json

# Import public league config (no sensitive data)
try:
	from league_config import FANTRAX_DEFAULT_LEAGUE_ID, LEAGUE_ID_TO_NAME
except ImportError:
	FANTRAX_DEFAULT_LEAGUE_ID = "ifa1anexmdgtlk9s"
	LEAGUE_ID_TO_NAME = {}

def get_cache_last_updated(league_id):
	"""Get the most recent cache file modification time."""
	cache_dir = get_cache_directory()
	cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
	
	if not cache_files:
		return None
	
	# Get the most recent modification time
	latest_time = max(f.stat().st_mtime for f in cache_files)
	return datetime.fromtimestamp(latest_time)

def get_available_seasons_for_player(player_code, league_id):
	"""Get all available seasons for a specific player."""
	cache_dir = get_cache_directory()
	
	# Look for new format cache files only
	new_format = cache_dir.glob(f"player_game_log_full_{player_code}_{league_id}_*.json")
	
	seasons = []
	
	# Check new format (extract season from filename)
	for file in new_format:
		# Extract season from filename: player_game_log_full_{code}_{league}_{season}.json
		# Example: player_game_log_full_03al0_ifa1anexmdgtlk9s_2025_26.json
		parts = file.stem.split('_')
		if len(parts) >= 6:
			# Find where the league_id ends and season starts
			# The season should be the last part after the league_id
			season_part = parts[-1]  # Get the last part (e.g., "2025" and "26")
			if len(parts) >= 7:
				# If we have year and season parts separately
				season = f"{parts[-2]}-{parts[-1]}"  # e.g., "2025-26"
			else:
				# Single season part, convert underscores to dashes
				season = season_part.replace('_', '-')
			seasons.append(season)
	
	return sorted(list(set(seasons)), reverse=True)  # Most recent first, no duplicates

def load_player_season_data(player_code, league_id, season):
	"""Load game log data for a specific player and season."""
	cache_dir = get_cache_directory()
	
	# Only use new format
	season_filename = season.replace('-', '_')
	cache_file = cache_dir / f"player_game_log_full_{player_code}_{league_id}_{season_filename}.json"
	
	if cache_file.exists():
		try:
			with open(cache_file, 'r') as f:
				cache_data = json.load(f)
			
			# Check if this is an empty season (player didn't play)
			status = cache_data.get('status', 'success')
			data = cache_data.get('data', [])
			player_name = cache_data.get('player_name', 'Unknown')
			
			# Return data even if empty (could be injury season, etc.)
			return data, player_name, True
		except:
			pass
	
	return [], 'Unknown', False

def calculate_cross_season_stats(player_code, league_id, seasons):
	"""Calculate summary stats across multiple seasons."""
	all_stats = {}
	total_games = 0
	total_fpts = 0
	
	for season in seasons:
		game_log, player_name, _ = load_player_season_data(player_code, league_id, season)
		if game_log:
			# Season with games
			df = pd.DataFrame(game_log)
			stats = calculate_variability_stats(df)
			if stats:
				all_stats[season] = stats
				total_games += stats['games_played']
				total_fpts += stats['mean_fpts'] * stats['games_played']
		else:
			# Empty season (injury, didn't play, etc.) - still include in analysis
			all_stats[season] = {
				'games_played': 0,
				'mean_fpts': 0,
				'median_fpts': 0,
				'std_dev': 0,
				'coefficient_of_variation': 0,
				'min_fpts': 0,
				'max_fpts': 0,
				'boom_rate': 0,
				'bust_rate': 0,
				'status': 'no_games_played'
			}
	
	if total_games > 0:
		overall_avg = total_fpts / total_games
		return {
			'seasons': all_stats,
			'total_games': total_games,
			'overall_avg_fpts': overall_avg,
			'seasons_played': len(all_stats),
			'player_name': player_name if 'player_name' in locals() else 'Unknown'
		}
	
	return None

def show_player_consistency_viewer():
	"""Public viewer for player consistency data (read-only)."""
	st.title("📊 Player Consistency Analysis")
	st.write("View game-by-game performance variability and consistency metrics for all rostered players.")
	
	# League ID input with name display
	league_id = st.text_input(
		"Fantrax League ID", 
		value=FANTRAX_DEFAULT_LEAGUE_ID or "",
		help="Enter your Fantrax league ID"
	)
	
	# Show league name if available
	if league_id and league_id in LEAGUE_ID_TO_NAME:
		st.info(f"🏀 **{LEAGUE_ID_TO_NAME[league_id]}**")
	
	if not league_id:
		st.warning("Please enter a league ID to view cached data.")
		return
	
	# Load or build league cache index
	index = load_league_cache_index(league_id, rebuild_if_missing=True)
	if not index or not index.get("players"):
		st.info("📭 No data available yet. Contact your commissioner to run a data update.")
		return
	
	cache_dir = get_cache_directory()
	
	# Show last updated timestamp
	last_updated = get_cache_last_updated(league_id)
	if last_updated:
		time_ago = datetime.now() - last_updated
		days_ago = time_ago.days
		hours_ago = time_ago.seconds // 3600
		
		if days_ago > 0:
			update_text = f"{days_ago} day{'s' if days_ago != 1 else ''} ago"
		elif hours_ago > 0:
			update_text = f"{hours_ago} hour{'s' if hours_ago != 1 else ''} ago"
		else:
			update_text = "less than an hour ago"
		
		st.success(f"✅ Data last updated: {last_updated.strftime('%B %d, %Y at %I:%M %p')} ({update_text})")
	
	# Extract available seasons from index
	available_seasons = set()
	for player_data in index.get("players", {}).values():
		seasons = player_data.get("seasons", {})
		available_seasons.update(seasons.keys())
	
	available_seasons = sorted(list(available_seasons), reverse=True)  # Most recent first
	
	if not available_seasons:
		st.warning("No valid season data found in cache files.")
		return
	
	# Season selector for all tabs except Individual Player Analysis
	st.subheader("🗓️ Season Selection")
	selected_season = st.selectbox(
		"Select Season for Analysis",
		available_seasons,
		index=0,  # Default to most recent season
		key="main_season_selector",
		help="This affects League Overview, Fantasy Teams, and NBA Team Rosters tabs"
	)
	
	# Build list of cache files for selected season using the index (for league/fantasy/team views)
	season_cache_files = []
	for player_data in index.get("players", {}).values():
		seasons = player_data.get("seasons", {})
		season_info = seasons.get(selected_season)
		if not season_info:
			continue
		cache_file_name = season_info.get("cache_file")
		if not cache_file_name:
			continue
		cache_file_path = cache_dir / cache_file_name
		if cache_file_path.exists():
			season_cache_files.append(cache_file_path)
	
	st.success(f"Found {len(season_cache_files)} players with cached data for {selected_season}")
	
	# For Individual Player Analysis we still need the full set of cache files
	cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
	
	# Create main tabs
	main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
		"🔍 Individual Player Analysis", 
		"📊 League Overview", 
		"🏆 Fantasy Teams",
		"🏟️ NBA Team Rosters"
	])
	
	with main_tab2:
		show_league_overview_viewer(league_id, season_cache_files, selected_season)

	with main_tab3:
		show_fantasy_teams_viewer(league_id, season_cache_files, selected_season)
	
	with main_tab4:
		show_team_rosters_viewer(league_id, season_cache_files, selected_season)
	
	with main_tab1:
		# Individual player analysis gets all cache files for multi-season view
		show_individual_player_viewer(league_id, cache_files)

def show_league_overview_viewer(league_id, cache_files, selected_season):
	"""Display league overview (read-only)."""
	st.subheader(f"📊 League-Wide Consistency Analysis - {selected_season}")
	
	# Load all cached data
	overview_df = _load_all_cached_data(cache_files)
	
	if overview_df is None or overview_df.empty:
		st.warning("No valid player data found in cache.")
		return
	
	# Display summary metrics
	_display_summary_metrics(overview_df)
	
	# Display table (without download buttons for now, we'll add them back)
	_display_consistency_table(overview_df, league_id)
	
	# Display visualizations
	_display_visualizations(overview_df)

def show_individual_player_viewer(league_id, cache_files):
	"""Display individual player analysis with multi-season support."""
	st.subheader("🔍 Individual Player Analysis")
	
	# Load player list from all cache files (new format only)
	cache_dir = get_cache_directory()
	all_cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
	
	player_dict = {}
	for cache_file in all_cache_files:
		try:
			with open(cache_file, 'r') as f:
				cache_data = json.load(f)
			player_name = cache_data.get('player_name', 'Unknown')
			player_code = cache_data.get('player_code', '')
			if player_name and player_code:
				player_dict[player_name] = player_code
		except:
			continue
	
	if not player_dict:
		st.warning("No player data available.")
		return
	
	# Player selection
	st.markdown("---")
	col1, col2 = st.columns([3, 1])
	
	with col1:
		player_names = sorted(player_dict.keys())
		selected_player = st.selectbox(
			"Choose a player",
			options=player_names,
			help=f"Total: {len(player_dict)} players with cached data"
		)
		player_code = player_dict[selected_player]
	
	with col2:
		st.write("")
		st.write("")
		st.info(f"**Code:** {player_code}")
	
	# Get available seasons for this player
	available_seasons = get_available_seasons_for_player(player_code, league_id)
	
	if not available_seasons:
		st.error("No season data found for this player.")
		return
	
	# Season selection and analysis type
	col1, col2 = st.columns(2)
	
	with col1:
		analysis_type = st.radio(
			"Analysis Type",
			options=["Single Season", "Multi-Season Overview"],
			help="Choose between analyzing one season or comparing across seasons"
		)
	
	with col2:
		if analysis_type == "Single Season":
			selected_season = st.selectbox(
				"Select Season",
				options=available_seasons,
				help="Choose which season to analyze in detail"
			)
		else:
			st.info(f"**{len(available_seasons)} seasons** available: {', '.join(available_seasons)}")
	
	st.markdown("---")
	
	if analysis_type == "Single Season":
		# Single season analysis (existing functionality)
		show_single_season_analysis(player_code, league_id, selected_season, selected_player)
	else:
		# Multi-season overview
		show_multi_season_overview(player_code, league_id, available_seasons, selected_player)

def show_single_season_analysis(player_code, league_id, season, selected_player):
	"""Display detailed analysis for a single season."""
	game_log, player_name, is_full_format = load_player_season_data(player_code, league_id, season)
	
	if not game_log:
		st.warning(f"No game log data available for {season}.")
		return
	
	df = pd.DataFrame(game_log)
	format_indicator = "🆕 Full Games Tab" if is_full_format else "📊 Overview Tab"
	
	st.success(f"✅ Loaded {len(df)} games for **{player_name}** ({season}) - {format_indicator}")
	
	# Display analysis
	st.subheader(f"📊 Performance Analysis - {player_name} ({season})")
	
	stats = calculate_variability_stats(df)
	
	if stats:
		# Variability metrics
		display_variability_metrics(stats, f"{player_name} ({season})")
		
		st.markdown("---")
		
		# Boom/Bust analysis
		display_boom_bust_analysis(stats)
		
		# Visualizations
		st.markdown("---")
		st.subheader("📈 Visual Analysis")
		
		viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
			"FPts Trend", "Distribution", "Boom/Bust Zones", "Category Breakdown"
		])
		
		with viz_tab1:
			display_fpts_trend_chart(df, stats, f"{player_name} ({season})")
		
		with viz_tab2:
			display_distribution_chart(df, stats, f"{player_name} ({season})")
		
		with viz_tab3:
			display_boom_bust_zones_chart(df, stats, f"{player_name} ({season})")
		
		with viz_tab4:
			display_category_breakdown(df, f"{player_name} ({season})")
		
		st.markdown("---")
	
	# Display full game log
	st.subheader("📋 Complete Game Log")
	
	priority_cols = ['Date', 'Team', 'Opp', 'Score', 'FPts', 'MIN', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
	other_cols = [col for col in df.columns if col not in priority_cols]
	display_cols = [col for col in priority_cols if col in df.columns] + other_cols
	
	st.dataframe(
		df[display_cols],
		use_container_width=True,
		height=400
	)
	
	# Download button
	csv = df.to_csv(index=False)
	st.download_button(
		label=f"📥 Download {season} Game Log as CSV",
		data=csv,
		file_name=f"{player_name.replace(' ', '_')}_{season.replace('-', '_')}_game_log.csv",
		mime="text/csv"
	)

def show_multi_season_overview(player_code, league_id, seasons, selected_player):
	"""Display multi-season overview and comparison."""
	cross_season_stats = calculate_cross_season_stats(player_code, league_id, seasons)
	
	if not cross_season_stats:
		st.warning("No valid season data found for multi-season analysis.")
		return
	
	player_name = cross_season_stats['player_name']
	
	st.subheader(f"🏀 Multi-Season Overview - {player_name}")
	
	# Overall summary metrics
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		st.metric("Seasons Played", cross_season_stats['seasons_played'])
	with col2:
		st.metric("Total Games", cross_season_stats['total_games'])
	with col3:
		st.metric("Overall Avg FP/G", f"{cross_season_stats['overall_avg_fpts']:.1f}")
	with col4:
		st.metric("Avg GP/Season", f"{cross_season_stats['total_games'] / cross_season_stats['seasons_played']:.0f}")
	
	st.markdown("---")
	
	# Season-by-season breakdown
	st.subheader("📊 Season-by-Season Breakdown")
	
	season_data = []
	for season, stats in cross_season_stats['seasons'].items():
		season_data.append({
			'Season': season,
			'Games Played': stats['games_played'],
			'Avg FP/G': round(stats['mean_fpts'], 1),
			'Median FP/G': round(stats['median_fpts'], 1),
			'Std Dev': round(stats['std_dev'], 1),
			'CV%': round(stats['coefficient_of_variation'], 1),
			'Min FPts': round(stats['min_fpts'], 1),
			'Max FPts': round(stats['max_fpts'], 1),
			'Boom Rate%': round(stats['boom_rate'], 1),
			'Bust Rate%': round(stats['bust_rate'], 1)
		})
	
	season_df = pd.DataFrame(season_data)
	
	st.dataframe(
		season_df,
		use_container_width=True,
		height=300
	)
	
	# Download multi-season summary
	csv = season_df.to_csv(index=False)
	st.download_button(
		label="📥 Download Multi-Season Summary as CSV",
		data=csv,
		file_name=f"{player_name.replace(' ', '_')}_multi_season_summary.csv",
		mime="text/csv"
	)
	
	st.markdown("---")
	
	# Season comparison charts
	st.subheader("📈 Season Comparison Charts")
	
	chart_tab1, chart_tab2, chart_tab3 = st.tabs([
		"FP/G Trends", "Consistency Trends", "Volume Trends"
	])
	
	with chart_tab1:
		# FP/G comparison across seasons
		import plotly.graph_objects as go
		
		fig = go.Figure()
		fig.add_trace(go.Scatter(
			x=season_df['Season'],
			y=season_df['Avg FP/G'],
			mode='lines+markers',
			name='Avg FP/G',
			line=dict(color='blue', width=3),
			marker=dict(size=8)
		))
		fig.add_trace(go.Scatter(
			x=season_df['Season'],
			y=season_df['Median FP/G'],
			mode='lines+markers',
			name='Median FP/G',
			line=dict(color='orange', width=2, dash='dash'),
			marker=dict(size=6)
		))
		
		fig.update_layout(
			title=f"{player_name} - Fantasy Points Per Game Trends",
			xaxis_title="Season",
			yaxis_title="Fantasy Points Per Game",
			hovermode='x unified'
		)
		st.plotly_chart(fig, use_container_width=True)
	
	with chart_tab2:
		# Consistency metrics
		fig = go.Figure()
		fig.add_trace(go.Scatter(
			x=season_df['Season'],
			y=season_df['CV%'],
			mode='lines+markers',
			name='Coefficient of Variation %',
			line=dict(color='red', width=3),
			marker=dict(size=8)
		))
		
		fig.update_layout(
			title=f"{player_name} - Consistency Trends (Lower CV% = More Consistent)",
			xaxis_title="Season",
			yaxis_title="Coefficient of Variation %",
			hovermode='x unified'
		)
		st.plotly_chart(fig, use_container_width=True)
	
	with chart_tab3:
		# Games played trends
		fig = go.Figure()
		fig.add_trace(go.Bar(
			x=season_df['Season'],
			y=season_df['Games Played'],
			name='Games Played',
			marker_color='green'
		))
		
		fig.update_layout(
			title=f"{player_name} - Games Played by Season",
			xaxis_title="Season",
			yaxis_title="Games Played",
			showlegend=False
		)
		st.plotly_chart(fig, use_container_width=True)
