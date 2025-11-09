import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from modules.player_game_log_scraper.logic import (
	get_player_game_log,
	calculate_variability_stats,
	get_available_players_from_csv,
	bulk_scrape_all_players,
	clear_all_cache,
	get_cache_directory
)
from modules.player_game_log_scraper.ui_league_overview import show_league_overview
import json

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent / 'fantrax.env'
load_dotenv(env_path)
FANTRAX_USERNAME = os.getenv('FANTRAX_USERNAME')
FANTRAX_PASSWORD = os.getenv('FANTRAX_PASSWORD')
FANTRAX_DEFAULT_LEAGUE_ID = os.getenv('FANTRAX_DEFAULT_LEAGUE_ID', '')

def clear_player_game_log_cache():
	"""Clear all cached player game logs."""
	try:
		count = clear_all_cache()
		return f"Cleared {count} cached player game logs", True
	except Exception as e:
		return f"Error clearing cache: {e}", False

def get_cache_last_updated(league_id):
	"""Get the most recent cache file modification time."""
	from datetime import datetime
	cache_dir = get_cache_directory()
	cache_files = list(cache_dir.glob(f"player_game_log_*_{league_id}.json"))
	
	if not cache_files:
		return None
	
	# Get the most recent modification time
	latest_time = max(f.stat().st_mtime for f in cache_files)
	return datetime.fromtimestamp(latest_time)

def show_player_game_log_scraper():
	"""Displays the UI for the Player Game Log Scraper tool."""
	st.title("Player Game Log Scraper")
	st.write("Scrape game-by-game statistics for any player to analyze performance variability and consistency.")
	
	# Show last updated info if available
	if FANTRAX_DEFAULT_LEAGUE_ID:
		last_updated = get_cache_last_updated(FANTRAX_DEFAULT_LEAGUE_ID)
		if last_updated:
			from datetime import datetime
			time_ago = datetime.now() - last_updated
			days_ago = time_ago.days
			hours_ago = time_ago.seconds // 3600
			
			if days_ago > 0:
				update_text = f"{days_ago} day{'s' if days_ago != 1 else ''} ago"
			elif hours_ago > 0:
				update_text = f"{hours_ago} hour{'s' if hours_ago != 1 else ''} ago"
			else:
				update_text = "less than an hour ago"
			
			st.info(f"📅 Data last updated: {last_updated.strftime('%B %d, %Y at %I:%M %p')} ({update_text})")
	
	# Create main tabs
	main_tab1, main_tab2 = st.tabs(["🔍 Individual Player Analysis", "📊 League Overview"])
	
	with main_tab2:
		show_league_overview()
	
	# Continue with individual player analysis in main_tab1
	with main_tab1:
		# League ID input
		league_id = st.text_input(
			"Fantrax League ID", 
			value=FANTRAX_DEFAULT_LEAGUE_ID or "",
			help="Enter your Fantrax league ID (e.g., ifa1anexmdgtlk9s)"
		)

		# Load available players from CSV
		player_dict = get_available_players_from_csv()
	
		if player_dict:
			st.markdown("---")
			st.subheader("Select Player")
		
			# Create two columns for player selection
			col1, col2 = st.columns([3, 1])
		
			with col1:
				# Dropdown for player selection
				player_names = sorted(player_dict.keys())
				selected_player = st.selectbox(
					"Choose a player from your roster",
					options=player_names,
					help=f"Rostered players only (Status != FA). Total: {len(player_dict)} players"
				)
				player_code = player_dict[selected_player]
		
			with col2:
				st.write("")  # Spacing
				st.write("")  # Spacing
				# Display player code
				st.info(f"**Code:** {player_code}")
		else:
			st.warning("Could not load players from season files. Please enter player code manually.")
			player_code = st.text_input(
				"Player Code",
				value="04ewe",
				help="Enter the player code from the Fantrax URL (e.g., 04ewe)"
			)

		st.markdown("---")
	
		# Options
		col1, col2, col3 = st.columns(3)
		with col1:
			force_refresh = st.checkbox(
				"Force Refresh (ignore cache)",
				key="player_game_log_force_refresh",
				help="Check this box to bypass the local cache and download the latest data from Fantrax."
			)
		with col2:
			if st.button("🗑️ Clear All Cache", help="Delete all cached player game logs"):
				message, success = clear_player_game_log_cache()
				if success:
					st.success(message)
				else:
					st.error(message)
		with col3:
			if st.button("📦 Bulk Scrape All", help="Scrape game logs for ALL players (one login, saves to cache)"):
				st.session_state['show_bulk_scrape'] = True

		# Scrape button
		if st.button("Get Player Game Log", type="primary"):
			if not league_id:
				st.error("Please enter a Fantrax League ID.")
				return
			if not player_code:
				st.error("Please enter a player code.")
				return
			if not FANTRAX_USERNAME or not FANTRAX_PASSWORD:
				st.error("Fantrax username or password not found. Please set them in your `fantrax.env` file.")
				return

			try:
				with st.spinner(f"Fetching game log for player {player_code}..."):
					df, from_cache, player_name = get_player_game_log(
						player_code, league_id, FANTRAX_USERNAME, FANTRAX_PASSWORD, force_refresh
					)
				
					cache_status = "✅ Loaded from cache" if from_cache else "🌐 Freshly scraped"
					st.success(f"{cache_status} - Found {len(df)} games for **{player_name}**")

					# Display variability stats
					st.markdown("---")
					st.subheader(f"📊 Performance Analysis - {player_name}")
				
					stats = calculate_variability_stats(df)
				
					if stats:
						# Create metrics display
						col1, col2, col3, col4 = st.columns(4)
					
						with col1:
							st.metric("Games Played", stats['games_played'])
							st.metric("Mean FPts", f"{stats['mean_fpts']:.1f}", help="Average fantasy points per game")
					
						with col2:
							st.metric("Median FPts", f"{stats['median_fpts']:.1f}", help="Middle value - less affected by outliers than mean")
							st.metric("Std Dev", f"{stats['std_dev']:.1f}", help="Standard deviation - measures spread of scores. Higher = more variable")
					
						with col3:
							st.metric("Min FPts", f"{stats['min_fpts']:.0f}", help="Lowest scoring game")
							st.metric("Max FPts", f"{stats['max_fpts']:.0f}", help="Highest scoring game")
					
						with col4:
							st.metric("Range", f"{stats['range']:.0f}", help="Difference between max and min. Shows total scoring spread")
							st.metric(
								"CV %", 
								f"{stats['coefficient_of_variation']:.1f}%",
								help="Coefficient of Variation = (Std Dev / Mean) × 100. Lower % = more consistent. <20% = very consistent, 20-30% = moderate, >30% = volatile"
							)
					
						st.markdown("---")
					
						# Context for high performers
						if stats['mean_fpts'] >= 80:
							st.info(f"⭐ **Elite Player Context:** With a {stats['mean_fpts']:.1f} FPts/G average, even 'low' games of {stats['min_fpts']:.0f} FPts are excellent. The variability metrics show consistency *relative to this player's elite production*, not absolute fantasy value.")
					
						# Boom/Bust Analysis
						st.subheader("💥 Boom/Bust Analysis")
						st.caption("Games beyond ±1 standard deviation from the player's mean")
						col1, col2 = st.columns(2)
					
						with col1:
							st.metric(
								"Boom Games", 
								stats['boom_games'],
								delta=f"{stats['boom_rate']:.1f}% of games",
								help="Games with FPts > Mean + 1 Std Dev"
							)
					
						with col2:
							st.metric(
								"Bust Games", 
								stats['bust_games'],
								delta=f"{stats['bust_rate']:.1f}% of games",
								delta_color="inverse",
								help="Games with FPts < Mean - 1 Std Dev"
							)
					
						# Visualizations
						st.markdown("---")
						st.subheader("📈 Visual Analysis")
					
						# Create tabs for different visualizations
						viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
							"FPts Trend", "Distribution", "Boom/Bust Zones", "Category Breakdown"
						])
					
						with viz_tab1:
							# Line chart showing FPts over time
							fig_trend = go.Figure()
						
							# Add FPts line
							fig_trend.add_trace(go.Scatter(
								x=list(range(len(df), 0, -1)),  # Reverse order (most recent first)
								y=df['FPts'].values,
								mode='lines+markers',
								name='FPts',
								line=dict(color='#1f77b4', width=2),
								marker=dict(size=6)
							))
						
							# Add mean line
							fig_trend.add_hline(
								y=stats['mean_fpts'], 
								line_dash="dash", 
								line_color="green",
								annotation_text=f"Mean: {stats['mean_fpts']:.1f}",
								annotation_position="right"
							)
						
							# Add boom/bust zones
							fig_trend.add_hrect(
								y0=stats['mean_fpts'] + stats['std_dev'],
								y1=df['FPts'].max() * 1.1,
								fillcolor="lightgreen",
								opacity=0.2,
								annotation_text="Boom Zone",
								annotation_position="top left"
							)
						
							fig_trend.add_hrect(
								y0=0,
								y1=stats['mean_fpts'] - stats['std_dev'],
								fillcolor="lightcoral",
								opacity=0.2,
								annotation_text="Bust Zone",
								annotation_position="bottom left"
							)
						
							fig_trend.update_layout(
								title=f"{player_name} - Fantasy Points Trend",
								xaxis_title="Games Ago",
								yaxis_title="Fantasy Points",
								hovermode='x unified',
								height=400
							)
						
							st.plotly_chart(fig_trend, use_container_width=True)
					
						with viz_tab2:
							# Histogram showing distribution
							fig_hist = go.Figure()
						
							fig_hist.add_trace(go.Histogram(
								x=df['FPts'],
								nbinsx=20,
								name='FPts Distribution',
								marker_color='#1f77b4'
							))
						
							# Add vertical lines for mean and median
							fig_hist.add_vline(
								x=stats['mean_fpts'],
								line_dash="dash",
								line_color="green",
								annotation_text=f"Mean: {stats['mean_fpts']:.1f}",
								annotation_position="top"
							)
						
							fig_hist.add_vline(
								x=stats['median_fpts'],
								line_dash="dot",
								line_color="orange",
								annotation_text=f"Median: {stats['median_fpts']:.1f}",
								annotation_position="top"
							)
						
							fig_hist.update_layout(
								title=f"{player_name} - FPts Distribution",
								xaxis_title="Fantasy Points",
								yaxis_title="Frequency",
								showlegend=False,
								height=400
							)
						
							st.plotly_chart(fig_hist, use_container_width=True)
						
							# Add distribution stats
							col1, col2, col3 = st.columns(3)
							with col1:
								st.metric("Skewness", f"{df['FPts'].skew():.2f}", help="Negative = left tail, Positive = right tail")
							with col2:
								st.metric("Kurtosis", f"{df['FPts'].kurtosis():.2f}", help="Higher = more extreme outliers")
							with col3:
								percentile_75 = df['FPts'].quantile(0.75)
								st.metric("75th Percentile", f"{percentile_75:.1f}", help="75% of games below this")
					
						with viz_tab3:
							# Categorize games into boom/normal/bust
							mean = stats['mean_fpts']
							std = stats['std_dev']
						
							df_viz = df.copy()
							df_viz['Category'] = 'Normal'
							df_viz.loc[df_viz['FPts'] > mean + std, 'Category'] = 'Boom'
							df_viz.loc[df_viz['FPts'] < mean - std, 'Category'] = 'Bust'
							df_viz['Game_Number'] = range(len(df_viz), 0, -1)
						
							# Color map
							color_map = {'Boom': 'green', 'Normal': 'gray', 'Bust': 'red'}
						
							fig_zones = px.scatter(
								df_viz,
								x='Game_Number',
								y='FPts',
								color='Category',
								color_discrete_map=color_map,
								hover_data=['Date', 'Opp', 'Score'],
								title=f"{player_name} - Boom/Bust Game Classification"
							)
						
							# Add mean line
							fig_zones.add_hline(
								y=mean,
								line_dash="dash",
								line_color="blue",
								annotation_text=f"Mean: {mean:.1f}"
							)
						
							# Add std dev lines
							fig_zones.add_hline(
								y=mean + std,
								line_dash="dot",
								line_color="green",
								annotation_text=f"+1 SD: {mean + std:.1f}"
							)
						
							fig_zones.add_hline(
								y=mean - std,
								line_dash="dot",
								line_color="red",
								annotation_text=f"-1 SD: {mean - std:.1f}"
							)
						
							fig_zones.update_layout(
								xaxis_title="Games Ago",
								yaxis_title="Fantasy Points",
								height=400
							)
						
							st.plotly_chart(fig_zones, use_container_width=True)
						
							# Summary table
							category_counts = df_viz['Category'].value_counts()
							st.write("**Game Classification Summary:**")
							summary_df = pd.DataFrame({
								'Category': category_counts.index,
								'Games': category_counts.values,
								'Percentage': (category_counts.values / len(df_viz) * 100).round(1)
							})
							st.dataframe(summary_df, hide_index=True)
					
						with viz_tab4:
							# Category breakdown (PTS, REB, AST, etc.)
							stat_cols = ['PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
							available_stats = [col for col in stat_cols if col in df.columns]
						
							if available_stats:
								# Calculate averages
								avg_stats = {col: df[col].mean() for col in available_stats}
							
								# Create bar chart
								fig_cats = go.Figure()
							
								fig_cats.add_trace(go.Bar(
									x=list(avg_stats.keys()),
									y=list(avg_stats.values()),
									marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(avg_stats)]
								))
							
								fig_cats.update_layout(
									title=f"{player_name} - Average Stats Per Game",
									xaxis_title="Category",
									yaxis_title="Average Per Game",
									showlegend=False,
									height=400
								)
							
								st.plotly_chart(fig_cats, use_container_width=True)
							
								# Show detailed stats table
								st.write("**Detailed Category Statistics:**")
								cat_stats = []
								for col in available_stats:
									cat_stats.append({
										'Category': col,
										'Mean': f"{df[col].mean():.1f}",
										'Median': f"{df[col].median():.1f}",
										'Std Dev': f"{df[col].std():.1f}",
										'Min': f"{df[col].min():.0f}",
										'Max': f"{df[col].max():.0f}"
									})
								st.dataframe(pd.DataFrame(cat_stats), hide_index=True)
							else:
								st.info("Category breakdown not available for this player's data.")
					
						st.markdown("---")

					# Display full game log
					st.subheader("📋 Complete Game Log")
				
					# Reorder columns to put most important ones first
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
				st.error(f"An error occurred: {e}")
				st.exception(e)
	
		# Bulk Scraper Section
		if st.session_state.get('show_bulk_scrape', False):
			st.markdown("---")
			st.subheader("📦 Bulk Scrape All Players")
			st.write("This will scrape game logs for **all players** in one session (single login). Perfect for weekly updates!")
		
			player_dict = get_available_players_from_csv()
			if not player_dict:
				st.error("No players found to scrape.")
				st.session_state['show_bulk_scrape'] = False
			else:
				st.info(f"Ready to scrape **{len(player_dict)} players**. This will take approximately {len(player_dict) * 2 / 60:.1f} minutes.")
			
				col1, col2 = st.columns(2)
				with col1:
					if st.button("▶️ Start Bulk Scrape", type="primary"):
						if not league_id:
							st.error("Please enter a Fantrax League ID.")
						elif not FANTRAX_USERNAME or not FANTRAX_PASSWORD:
							st.error("Fantrax credentials not found in fantrax.env")
						else:
							progress_bar = st.progress(0)
							status_text = st.empty()
						
							def progress_callback(current, total, player_name):
								progress = current / total
								progress_bar.progress(progress)
								status_text.text(f"Scraping {current}/{total}: {player_name}")
						
							try:
								with st.spinner("Starting bulk scrape..."):
									result = bulk_scrape_all_players(
										league_id, 
										FANTRAX_USERNAME, 
										FANTRAX_PASSWORD,
										player_dict,
										progress_callback
									)
							
								progress_bar.empty()
								status_text.empty()
							
								if "error" in result:
									st.error(result["error"])
								else:
									st.success(f"✅ Bulk scrape complete! Successfully scraped {result['success_count']}/{result['total']} players.")
								
									if result['fail_count'] > 0:
										with st.expander(f"⚠️ {result['fail_count']} players failed"):
											for player_name, error in result['failed_players']:
												st.write(f"- **{player_name}**: {error}")
								
									st.session_state['show_bulk_scrape'] = False
									st.rerun()
						
							except Exception as e:
								st.error(f"Bulk scrape failed: {e}")
								st.exception(e)
			
				with col2:
					if st.button("❌ Cancel"):
						st.session_state['show_bulk_scrape'] = False
						st.rerun()
