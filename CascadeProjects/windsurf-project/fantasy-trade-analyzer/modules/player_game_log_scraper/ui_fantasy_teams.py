"""Fantasy Teams roster viewer - shows rosters by fantasy manager."""
import streamlit as st
import pandas as pd
import re
from pathlib import Path
import json
from modules.player_game_log_scraper.logic import (
	calculate_variability_stats,
	get_cache_directory,
	clean_html_from_text
)
from modules.team_mappings import TEAM_MAPPINGS

def _load_fantasy_team_rosters():
	"""Load rosters organized by fantasy team (manager)."""
	# Find latest player data CSV with Status column
	data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
	candidates = sorted(
		list(data_dir.glob('Fantrax-Players-*-(YTD).csv')),
		key=lambda p: p.stat().st_mtime,
		reverse=True
	)
	
	if not candidates:
		# Fallback to any Fantrax-Players file
		candidates = sorted(
			list(data_dir.glob('Fantrax-Players-*.csv')),
			key=lambda p: p.stat().st_mtime,
			reverse=True
		)
	
	if not candidates:
		return {}
	
	try:
		df = pd.read_csv(candidates[0])
		if 'Player' not in df.columns or 'Status' not in df.columns:
			return {}
		
		# Clean HTML from Status column
		df['Status'] = df['Status'].apply(lambda x: clean_html_from_text(str(x)) if pd.notna(x) else x)
		
		# Filter to rostered players only (exclude FA and any HTML remnants)
		df = df[df['Status'] != 'FA'].copy()
		df = df[~df['Status'].str.contains('<', na=False)].copy()  # Extra safety
		
		# Group by Status (which contains team abbreviations)
		rosters = {}
		for status, group in df.groupby('Status'):
			# Skip if status is too short or empty
			if pd.isna(status) or len(str(status).strip()) < 2:
				continue
			
			# Map status to full team name
			team_name = TEAM_MAPPINGS.get(status, status)
			
			# Only include if it's a known team (either the code or the full name is in TEAM_MAPPINGS)
			if status in TEAM_MAPPINGS or team_name in TEAM_MAPPINGS.values():
				players = group['Player'].tolist()
				rosters[team_name] = players
		
		return rosters
	except Exception as e:
		print(f"Error loading fantasy team rosters: {e}")
		return {}

def _cache_index_by_player_name(cache_files):
	"""Build index of cached players by name."""
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

def _build_fantasy_team_view(league_id, cache_files):
	"""Build fantasy team rosters with consistency metrics."""
	rosters = _load_fantasy_team_rosters()
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

def _display_fantasy_teams_overview(rosters_by_team):
	"""Display summary table of all fantasy teams."""
	summary_rows = []
	for team, df in rosters_by_team.items():
		if df.empty:
			continue
		summary_rows.append({
			'Fantasy Team': team,
			'Players': len(df),
			'Avg FPts': round(df['Mean FPts'].mean(), 1),
			'Avg CV%': round(df['CV %'].mean(), 1),
			'🟢 Consistent': len(df[df['CV %'] < 20]),
			'🟡 Moderate': len(df[(df['CV %'] >= 20) & (df['CV %'] <= 30)]),
			'🔴 Volatile': len(df[df['CV %'] > 30]),
			'Total GP': int(df['GP'].sum())
		})
	
	if not summary_rows:
		st.info("No fantasy team data available.")
		return
	
	summary_df = pd.DataFrame(summary_rows).sort_values('Avg FPts', ascending=False)
	
	st.markdown("### Fantasy Team Performance Summary")
	st.dataframe(
		summary_df,
		use_container_width=True,
		height=400,
		column_config={
			'Fantasy Team': st.column_config.TextColumn('Fantasy Team', width='large'),
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
		label="📥 Download Fantasy Team Summary",
		data=csv,
		file_name="fantasy_team_summary.csv",
		mime="text/csv"
	)

def _display_team_deep_analysis(team_name, team_df, all_rosters):
	"""Display comprehensive analysis for a fantasy team."""
	import numpy as np
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
	import plotly.express as px
	
	st.markdown("### 📊 Team Performance Analysis")
	
	# Calculate team metrics
	total_players = len(team_df)
	avg_fpts = team_df['Mean FPts'].mean()
	avg_cv = team_df['CV %'].mean()
	total_gp = team_df['GP'].sum()
	
	# Tier breakdown
	elite_count = len(team_df[team_df['Mean FPts'] >= 80])
	star_count = len(team_df[(team_df['Mean FPts'] >= 60) & (team_df['Mean FPts'] < 80)])
	solid_count = len(team_df[(team_df['Mean FPts'] >= 40) & (team_df['Mean FPts'] < 60)])
	streamer_count = len(team_df[(team_df['Mean FPts'] >= 25) & (team_df['Mean FPts'] < 40)])
	bench_count = len(team_df[team_df['Mean FPts'] < 25])
	
	# Consistency breakdown
	consistent_count = len(team_df[team_df['CV %'] < 20])
	moderate_count = len(team_df[(team_df['CV %'] >= 20) & (team_df['CV %'] <= 30)])
	volatile_count = len(team_df[team_df['CV %'] > 30])
	
	# League ranking
	team_rankings = []
	for t_name, t_df in all_rosters.items():
		if not t_df.empty:
			team_rankings.append({
				'Team': t_name,
				'Avg FPts': t_df['Mean FPts'].mean(),
				'Avg CV%': t_df['CV %'].mean()
			})
	rankings_df = pd.DataFrame(team_rankings).sort_values('Avg FPts', ascending=False).reset_index(drop=True)
	team_rank = rankings_df[rankings_df['Team'] == team_name].index[0] + 1 if not rankings_df[rankings_df['Team'] == team_name].empty else None
	
	# Overall Assessment
	with st.expander("🎯 Overall Team Assessment", expanded=True):
		col1, col2, col3, col4 = st.columns(4)
		
		with col1:
			st.metric("League Rank", f"#{team_rank}/{len(rankings_df)}" if team_rank else "N/A", help="Based on average FPts")
			st.metric("Total Players", total_players)
		
		with col2:
			st.metric("Avg FPts", f"{avg_fpts:.1f}")
			percentile = ((len(rankings_df) - team_rank + 1) / len(rankings_df) * 100) if team_rank else 0
			st.metric("Percentile", f"{percentile:.0f}%", help="Higher = better")
		
		with col3:
			st.metric("Avg CV%", f"{avg_cv:.1f}%", help="Lower = more consistent")
			consistency_rank = rankings_df.sort_values('Avg CV%').reset_index(drop=True)
			cons_rank = consistency_rank[consistency_rank['Team'] == team_name].index[0] + 1 if not consistency_rank[consistency_rank['Team'] == team_name].empty else None
			st.metric("Consistency Rank", f"#{cons_rank}/{len(rankings_df)}" if cons_rank else "N/A")
		
		with col4:
			st.metric("Total GP", int(total_gp))
			sharpe = avg_fpts / avg_cv if avg_cv > 0 else 0
			st.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Risk-adjusted performance")
		
		# Team assessment
		st.markdown("---")
		if team_rank and team_rank <= 3:
			st.success(f"🏆 **Elite Team** - Top 3 in the league! Strong championship contender.")
		elif team_rank and team_rank <= len(rankings_df) // 3:
			st.success(f"💪 **Strong Team** - Top tier roster with playoff potential.")
		elif team_rank and team_rank <= 2 * len(rankings_df) // 3:
			st.info(f"📊 **Mid-Tier Team** - Solid roster, consider strategic trades to improve.")
		else:
			st.warning(f"⚠️ **Rebuilding Team** - Focus on acquiring consistent producers.")
	
	# Roster Composition Analysis
	with st.expander("🏀 Roster Composition & Depth", expanded=False):
		col1, col2 = st.columns(2)
		
		with col1:
			st.markdown("#### Production Tiers")
			tier_data = pd.DataFrame({
				'Tier': ['Elite (80+)', 'Star (60-80)', 'Solid (40-60)', 'Streamer (25-40)', 'Bench (<25)'],
				'Count': [elite_count, star_count, solid_count, streamer_count, bench_count]
			})
			
			fig_tiers = px.bar(
				tier_data,
				x='Tier',
				y='Count',
				color='Count',
				color_continuous_scale='Blues',
				title="Players by Production Tier"
			)
			fig_tiers.update_layout(showlegend=False, height=300)
			st.plotly_chart(fig_tiers, use_container_width=True)
			
			# Depth analysis
			if elite_count >= 2:
				st.success("✅ **Strong Top-End Talent** - Multiple elite players")
			elif elite_count == 1:
				st.info("🟡 **Single Star** - Consider adding another elite piece")
			else:
				st.warning("⚠️ **No Elite Players** - Target high-end talent in trades")
		
		with col2:
			st.markdown("#### Consistency Profile")
			cons_data = pd.DataFrame({
				'Type': ['🟢 Consistent', '🟡 Moderate', '🔴 Volatile'],
				'Count': [consistent_count, moderate_count, volatile_count]
			})
			
			fig_cons = px.pie(
				cons_data,
				values='Count',
				names='Type',
				title="Consistency Distribution",
				color_discrete_sequence=['#90ee90', '#ffd700', '#ff6b6b']
			)
			fig_cons.update_layout(height=300)
			st.plotly_chart(fig_cons, use_container_width=True)
			
			# Consistency analysis
			consistent_pct = (consistent_count / total_players * 100) if total_players > 0 else 0
			volatile_pct = (volatile_count / total_players * 100) if total_players > 0 else 0
			
			if consistent_pct > 50:
				st.success(f"✅ **Reliable Roster** - {consistent_pct:.0f}% of players are very consistent")
			elif volatile_pct > 50:
				st.warning(f"⚠️ **High Volatility** - {volatile_pct:.0f}% of players are volatile, risky for playoffs")
			else:
				st.info(f"🟡 **Balanced Risk** - Mix of consistent ({consistent_pct:.0f}%) and volatile ({volatile_pct:.0f}%) players")
	
	# Performance Distribution
	with st.expander("📈 Performance Distribution & Trends", expanded=False):
		col1, col2 = st.columns(2)
		
		with col1:
			# FPts distribution
			fig_dist = go.Figure()
			fig_dist.add_trace(go.Box(
				y=team_df['Mean FPts'],
				name="FPts",
				marker_color='lightblue',
				boxmean='sd'
			))
			fig_dist.update_layout(
				title="Team FPts Distribution",
				yaxis_title="Mean FPts",
				height=350
			)
			st.plotly_chart(fig_dist, use_container_width=True)
		
		with col2:
			# CV% distribution
			fig_cv = go.Figure()
			fig_cv.add_trace(go.Box(
				y=team_df['CV %'],
				name="CV%",
				marker_color='lightcoral',
				boxmean='sd'
			))
			fig_cv.update_layout(
				title="Team Consistency Distribution",
				yaxis_title="CV%",
				height=350
			)
			st.plotly_chart(fig_cv, use_container_width=True)
		
		# Scatter plot: Production vs Consistency
		fig_scatter = px.scatter(
			team_df,
			x='Mean FPts',
			y='CV %',
			size='GP',
			hover_data=['Player'],
			title="Production vs Consistency (bubble size = GP)",
			labels={'Mean FPts': 'Production (Mean FPts)', 'CV %': 'Risk (CV%)'}
		)
		fig_scatter.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Very Consistent")
		fig_scatter.add_hline(y=30, line_dash="dash", line_color="orange", opacity=0.5, annotation_text="Moderate")
		fig_scatter.update_layout(height=400)
		st.plotly_chart(fig_scatter, use_container_width=True)
	
	# League Comparison
	with st.expander("🏆 League Comparison", expanded=False):
		st.markdown("#### How Does This Team Stack Up?")
		
		# Create comparison dataframe
		comparison_data = []
		for t_name, t_df in all_rosters.items():
			if not t_df.empty:
				comparison_data.append({
					'Team': t_name,
					'Avg FPts': t_df['Mean FPts'].mean(),
					'Avg CV%': t_df['CV %'].mean(),
					'Elite Players': len(t_df[t_df['Mean FPts'] >= 80]),
					'Consistent Players': len(t_df[t_df['CV %'] < 20]),
					'Highlight': t_name == team_name
				})
		
		comp_df = pd.DataFrame(comparison_data).sort_values('Avg FPts', ascending=False)
		
		# Bar chart comparison
		fig_comp = go.Figure()
		
		colors = ['#ff6b6b' if row['Highlight'] else '#4a90e2' for _, row in comp_df.iterrows()]
		
		fig_comp.add_trace(go.Bar(
			x=comp_df['Team'],
			y=comp_df['Avg FPts'],
			marker_color=colors,
			text=comp_df['Avg FPts'].round(1),
			textposition='outside'
		))
		
		fig_comp.update_layout(
			title="League-Wide Average FPts Comparison",
			xaxis_title="Team",
			yaxis_title="Avg FPts",
			height=400,
			showlegend=False
		)
		st.plotly_chart(fig_comp, use_container_width=True)
		
		# Detailed comparison table
		st.markdown("#### Detailed League Standings")
		comp_df['Rank'] = range(1, len(comp_df) + 1)
		display_df = comp_df[['Rank', 'Team', 'Avg FPts', 'Avg CV%', 'Elite Players', 'Consistent Players']].copy()
		
		st.dataframe(
			display_df,
			use_container_width=True,
			height=400,
			column_config={
				'Rank': st.column_config.NumberColumn('Rank', width='small'),
				'Team': st.column_config.TextColumn('Team', width='large'),
				'Avg FPts': st.column_config.NumberColumn('Avg FPts', format='%.1f'),
				'Avg CV%': st.column_config.NumberColumn('Avg CV%', format='%.1f'),
				'Elite Players': st.column_config.NumberColumn('Elite (80+)'),
				'Consistent Players': st.column_config.NumberColumn('Consistent (<20% CV)')
			}
		)

def show_fantasy_teams_viewer(league_id, cache_files):
	"""Display fantasy teams roster viewer."""
	st.subheader("🏆 Fantasy Teams & Consistency")
	
	# Add explanatory info
	with st.expander("ℹ️ Understanding the Analysis", expanded=False):
		st.markdown("""
		### Key Metrics Explained
		
		**Mean FPts (Fantasy Points per Game)**
		- Average production across all games
		- Higher = better offensive output
		- Elite: 80+, Star: 60-80, Solid: 40-60
		
		**CV% (Coefficient of Variation)**
		- Measures consistency/volatility
		- Lower = more predictable performance
		- 🟢 Very Consistent: <20% (reliable floor)
		- 🟡 Moderate: 20-30% (balanced)
		- 🔴 Volatile: >30% (boom/bust)
		
		**Boom/Bust Rates**
		- Boom: Games >1 std dev above mean (ceiling games)
		- Bust: Games >1 std dev below mean (floor games)
		- High boom% = high upside potential
		- High bust% = injury/rest risk
		
		**Sharpe Ratio**
		- Risk-adjusted performance (Mean FPts / CV%)
		- Higher = better value (production per unit of risk)
		- Useful for comparing players with similar FPts
		
		### Team Assessment Categories
		
		**🏆 Elite Team (Top 3)**
		- Championship contender
		- Strong across all metrics
		
		**💪 Strong Team (Top 33%)**
		- Playoff-caliber roster
		- One or two moves from elite
		
		**📊 Mid-Tier Team (Middle 33%)**
		- Competitive but needs upgrades
		- Focus on strategic trades
		
		**⚠️ Rebuilding Team (Bottom 33%)**
		- Needs significant improvement
		- Target consistent producers
		
		### Roster Composition Tips
		
		**Elite Players (80+ FPts)**
		- League winners, irreplaceable
		- Target in trades even at premium cost
		
		**Depth vs Top-End**
		- 2-3 elite players > 5-6 solid players
		- Consolidate depth for stars
		
		**Consistency for Playoffs**
		- Reliable players win championships
		- High volatility = risky in elimination games
		""")
	
	rosters_by_team = _build_fantasy_team_view(league_id, cache_files)
	
	if not rosters_by_team:
		st.info("No fantasy team roster data found. Make sure player data is loaded with Status column.")
		return
	
	# Overview section
	with st.expander("📊 All Fantasy Teams Overview", expanded=True):
		_display_fantasy_teams_overview(rosters_by_team)
	
	st.markdown("---")
	st.subheader("🔍 Team Detail View")
	
	team_names = sorted(rosters_by_team.keys())
	col1, col2 = st.columns([2, 1])
	with col1:
		team = st.selectbox("Select Fantasy Team", options=team_names, key="fantasy_team_select")
	with col2:
		min_gp = st.number_input("Min GP", 0, 82, 0, key="fantasy_team_min_gp")
	
	df = rosters_by_team.get(team)
	if df is None or df.empty:
		st.warning("No players available for this team.")
		return
	
	filtered = df[df['GP'] >= min_gp].copy()
	
	def _consistency(cv):
		return '🟢 Very Consistent' if cv < 20 else ('🟡 Moderate' if cv <= 30 else '🔴 Volatile')
	
	filtered.insert(3, 'Consistency', filtered['CV %'].apply(_consistency))
	
	# Add comprehensive team analysis
	_display_team_deep_analysis(team, filtered, rosters_by_team)
	
	st.markdown("---")
	st.markdown("### 📋 Team Roster")
	
	st.dataframe(
		filtered.drop(columns=['code']),
		use_container_width=True,
		height=400,
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
	player = st.selectbox("View Player Details", options=list(filtered['Player']), key="fantasy_team_player_select")
	cache_dir = get_cache_directory()
	code = df.loc[df['Player'] == player, 'code'].values[0]
	cache_file = cache_dir / f"player_game_log_{code}_{league_id}.json"
	
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
		mime="text/csv",
		key="fantasy_team_player_download"
	)
