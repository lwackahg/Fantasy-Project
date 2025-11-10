"""Read-only viewer for player consistency data - no scraping controls."""
import streamlit as st
import pandas as pd
from datetime import datetime
from modules.player_game_log_scraper.logic import (
	calculate_variability_stats,
	get_cache_directory
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
	cache_files = list(cache_dir.glob(f"player_game_log_*_{league_id}.json"))
	
	if not cache_files:
		return None
	
	# Get the most recent modification time
	latest_time = max(f.stat().st_mtime for f in cache_files)
	return datetime.fromtimestamp(latest_time)

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
	
	# Check for cached data and show last updated
	cache_dir = get_cache_directory()
	cache_files = list(cache_dir.glob(f"player_game_log_*_{league_id}.json"))
	
	if not cache_files:
		st.info("📭 No data available yet. Contact your commissioner to run a data update.")
		return
	
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
	
	st.success(f"Found {len(cache_files)} players with cached data")
	
	# Create main tabs
	main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
		"🔍 Individual Player Analysis", 
		"📊 League Overview", 
		"🏆 Fantasy Teams",
		"🏟️ NBA Team Rosters"
	])
	
	with main_tab2:
		show_league_overview_viewer(league_id, cache_files)

	with main_tab3:
		show_fantasy_teams_viewer(league_id, cache_files)
	
	with main_tab4:
		show_team_rosters_viewer(league_id, cache_files)
	
	with main_tab1:
		show_individual_player_viewer(league_id, cache_files)

def show_league_overview_viewer(league_id, cache_files):
	"""Display league overview (read-only)."""
	st.subheader("📊 League-Wide Consistency Analysis")
	
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
	"""Display individual player analysis (read-only)."""
	st.subheader("🔍 Individual Player Analysis")
	
	# Load player list from cache
	player_dict = {}
	for cache_file in cache_files:
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
	
	# Load player data from cache
	cache_dir = get_cache_directory()
	cache_file = cache_dir / f"player_game_log_{player_code}_{league_id}.json"
	
	if not cache_file.exists():
		st.error("Player data not found in cache.")
		return
	
	try:
		with open(cache_file, 'r') as f:
			cache_data = json.load(f)
		
		player_name = cache_data.get('player_name', selected_player)
		game_log = cache_data.get('data', cache_data.get('game_log', []))
		
		if not game_log:
			st.warning("No game log data available for this player.")
			return
		
		df = pd.DataFrame(game_log)
		
		st.success(f"✅ Loaded {len(df)} games for **{player_name}**")
		
		# Display analysis
		st.markdown("---")
		st.subheader(f"📊 Performance Analysis - {player_name}")
		
		stats = calculate_variability_stats(df)
		
		if stats:
			# Variability metrics
			display_variability_metrics(stats, player_name)
			
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
				display_fpts_trend_chart(df, stats, player_name)
			
			with viz_tab2:
				display_distribution_chart(df, stats, player_name)
			
			with viz_tab3:
				display_boom_bust_zones_chart(df, stats, player_name)
			
			with viz_tab4:
				display_category_breakdown(df, player_name)
			
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
			label="📥 Download Game Log as CSV",
			data=csv,
			file_name=f"{player_name.replace(' ', '_')}_game_log.csv",
			mime="text/csv"
		)
	
	except Exception as e:
		st.error(f"Error loading player data: {e}")
		st.exception(e)
