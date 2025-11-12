"""League-wide overview UI for player game log scraper."""
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from modules.player_game_log_scraper.logic import (
	calculate_variability_stats,
	get_cache_directory
)

# Import public league config (no sensitive data)
try:
	from league_config import FANTRAX_DEFAULT_LEAGUE_ID, LEAGUE_ID_TO_NAME
except ImportError:
	from dotenv import load_dotenv
	import os
	env_path = Path(__file__).resolve().parent.parent.parent / 'fantrax.env'
	load_dotenv(env_path)
	FANTRAX_DEFAULT_LEAGUE_ID = os.getenv('FANTRAX_DEFAULT_LEAGUE_ID', '')
	LEAGUE_ID_TO_NAME = {}
	
def show_league_overview():
	"""Displays a league-wide overview of all cached player data."""
	st.subheader("📊 League-Wide Consistency Analysis")
	st.write("View variability metrics for all rostered players with cached data.")
	
	league_id = st.text_input(
		"Fantrax League ID", 
		value=FANTRAX_DEFAULT_LEAGUE_ID or "",
		help="Enter your Fantrax league ID",
		key="overview_league_id"
	)
	
	# Show league name if available
	if league_id and league_id in LEAGUE_ID_TO_NAME:
		st.info(f"🏀 **{LEAGUE_ID_TO_NAME[league_id]}**")
	
	if not league_id:
		st.warning("Please enter a league ID to view cached data.")
		return
	
	# Get cache directory
	cache_dir = get_cache_directory()
	
	# Find all cache files for this league (new format only)
	cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
	
	if not cache_files:
		st.info("No cached player data found. Run a bulk scrape first to populate the cache.")
		return
	
	st.success(f"Found {len(cache_files)} players with cached data")
	
	# Load all cached data
	overview_df = _load_all_cached_data(cache_files)
	
	if overview_df is None or overview_df.empty:
		st.warning("No valid player data found in cache.")
		return
	
	# Display summary metrics
	_display_summary_metrics(overview_df)
	
	# Display table with built-in filtering
	_display_consistency_table(overview_df, league_id)
	
	# Display visualizations
	_display_visualizations(overview_df)

def _load_all_cached_data(cache_files):
	"""Load and process all cached player data."""
	all_player_stats = []
	
	with st.spinner("Loading cached data..."):
		for cache_file in cache_files:
			try:
				with open(cache_file, 'r') as f:
					cache_data = json.load(f)
				
				player_name = cache_data.get('player_name', 'Unknown')
				# Try both 'data' and 'game_log' keys for backwards compatibility
				game_log = cache_data.get('data', cache_data.get('game_log', []))
				
				if not game_log:
					continue
				
				# Convert to DataFrame and calculate stats
				df = pd.DataFrame(game_log)
				if 'FPts' not in df.columns:
					continue
				
				stats = calculate_variability_stats(df)
				if stats:
					stats['Player'] = player_name
					stats['player_code'] = cache_data.get('player_code', '')
					all_player_stats.append(stats)
			
			except Exception as e:
				continue
	
	if not all_player_stats:
		return None
	
	# Create DataFrame
	overview_df = pd.DataFrame(all_player_stats)
	
	# Reorder columns
	column_order = [
		'Player', 'games_played', 'mean_fpts', 'median_fpts', 'std_dev',
		'coefficient_of_variation', 'min_fpts', 'max_fpts', 'range',
		'boom_games', 'boom_rate', 'bust_games', 'bust_rate'
	]
	overview_df = overview_df[[col for col in column_order if col in overview_df.columns]]
	
	# Rename columns for display
	overview_df = overview_df.rename(columns={
		'games_played': 'GP',
		'mean_fpts': 'Mean FPts',
		'median_fpts': 'Median FPts',
		'std_dev': 'Std Dev',
		'coefficient_of_variation': 'CV %',
		'min_fpts': 'Min',
		'max_fpts': 'Max',
		'range': 'Range',
		'boom_games': 'Boom Games',
		'boom_rate': 'Boom %',
		'bust_games': 'Bust Games',
		'bust_rate': 'Bust %'
	})
	
	# Round numeric columns
	numeric_cols = overview_df.select_dtypes(include=['float64', 'int64']).columns
	for col in numeric_cols:
		if col in ['CV %', 'Boom %', 'Bust %']:
			overview_df[col] = overview_df[col].round(1)
		elif col != 'GP':
			overview_df[col] = overview_df[col].round(1)
	
	# Sort by Mean FPts descending
	overview_df = overview_df.sort_values('Mean FPts', ascending=False).reset_index(drop=True)
	
	return overview_df

def _display_summary_metrics(overview_df):
	"""Display summary metrics at the top."""
	st.markdown("---")
	st.subheader("📊 League-Wide Statistics")
	
	# First row - counts and averages
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		st.metric("Total Players", len(overview_df))
		avg_gp = overview_df['GP'].mean()
		st.metric("Avg Games Played", f"{avg_gp:.1f}")
	with col2:
		avg_fpts = overview_df['Mean FPts'].mean()
		st.metric("Avg Mean FPts", f"{avg_fpts:.1f}", help="Average of all players' mean FPts")
		median_fpts = overview_df['Median FPts'].mean()
		st.metric("Avg Median FPts", f"{median_fpts:.1f}")
	with col3:
		avg_std = overview_df['Std Dev'].mean()
		st.metric("Avg Std Dev", f"{avg_std:.1f}", help="Average standard deviation across league")
		avg_cv = overview_df['CV %'].mean()
		st.metric("Avg CV%", f"{avg_cv:.1f}%", help="Lower = more consistent league-wide")
	with col4:
		avg_min = overview_df['Min'].mean()
		st.metric("Avg Min FPts", f"{avg_min:.0f}")
		avg_max = overview_df['Max'].mean()
		st.metric("Avg Max FPts", f"{avg_max:.0f}")
	
	# Second row - consistency breakdown
	st.markdown("**Consistency Breakdown:**")
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		consistent_players = len(overview_df[overview_df['CV %'] < 20])
		st.metric("🟢 Very Consistent", consistent_players, help="CV% < 20%")
	with col2:
		moderate_players = len(overview_df[(overview_df['CV %'] >= 20) & (overview_df['CV %'] <= 30)])
		st.metric("🟡 Moderate", moderate_players, help="CV% 20-30%")
	with col3:
		volatile_players = len(overview_df[overview_df['CV %'] > 30])
		st.metric("🔴 Volatile", volatile_players, help="CV% > 30%")
	with col4:
		avg_range = overview_df['Range'].mean()
		st.metric("Avg Range", f"{avg_range:.0f}", help="Average scoring range")
	
	st.markdown("---")

def _display_consistency_table(overview_df, league_id):
	"""Display the main consistency table with filtering controls."""
	st.subheader(f"📋 Player Consistency Table")
	
	# Filtering controls - always visible
	st.markdown("**🔍 Filters:**")
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		min_gp = st.number_input("Min GP", 0, int(overview_df['GP'].max()), 0, key="overview_min_gp")
	
	with col2:
		min_std = st.number_input("Min Std Dev", 0.0, float(overview_df['Std Dev'].max()), 0.0, step=5.0, key="overview_min_std")
	
	with col3:
		max_cv = st.number_input("Max CV%", 0.0, 100.0, 100.0, step=5.0, key="overview_max_cv")
	
	with col4:
		min_fpts = st.number_input("Min FPts", 0.0, float(overview_df['Mean FPts'].max()), 0.0, step=5.0, key="overview_min_fpts")
	
	# Second row of filters
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		consistency_filter = st.selectbox(
			"Consistency",
			["All", "🟢 Very Consistent (<20%)", "🟡 Moderate (20-30%)", "🔴 Volatile (>30%)"],
			key="overview_consistency"
		)
	
	with col2:
		search_player = st.text_input("Search Player", "", key="overview_search")
	
	with col3:
		min_min = st.number_input("Min Floor", 0.0, float(overview_df['Min'].max()), 0.0, step=5.0, key="overview_min_min", help="Minimum of player's lowest game")
	
	with col4:
		if st.button("🔄 Reset Filters", key="reset_filters"):
			st.rerun()
	
	# Apply filters
	filtered_df = overview_df.copy()
	filtered_df = filtered_df[filtered_df['GP'] >= min_gp]
	filtered_df = filtered_df[filtered_df['Mean FPts'] >= min_fpts]
	filtered_df = filtered_df[filtered_df['Std Dev'] >= min_std]
	filtered_df = filtered_df[filtered_df['CV %'] <= max_cv]
	filtered_df = filtered_df[filtered_df['Min'] >= min_min]
	
	if consistency_filter == "🟢 Very Consistent (<20%)":
		filtered_df = filtered_df[filtered_df['CV %'] < 20]
	elif consistency_filter == "🟡 Moderate (20-30%)":
		filtered_df = filtered_df[(filtered_df['CV %'] >= 20) & (filtered_df['CV %'] <= 30)]
	elif consistency_filter == "🔴 Volatile (>30%)":
		filtered_df = filtered_df[filtered_df['CV %'] > 30]
	
	if search_player:
		filtered_df = filtered_df[filtered_df['Player'].str.contains(search_player, case=False, na=False)]
	
	# Show filter summary
	if len(filtered_df) < len(overview_df):
		st.info(f"📊 Showing {len(filtered_df)} of {len(overview_df)} players")
	
	# Add a color indicator column
	def get_consistency_indicator(cv):
		if cv < 20:
			return "🟢 Very Consistent"
		elif cv <= 30:
			return "🟡 Moderate"
		else:
			return "🔴 Volatile"
	
	display_df = filtered_df.copy()
	display_df.insert(6, 'Consistency', display_df['CV %'].apply(get_consistency_indicator))

	st.dataframe(
		display_df,
		use_container_width=True,
		height=600,
		column_config={
			"Player": st.column_config.TextColumn("Player", width="medium"),
			"GP": st.column_config.NumberColumn("GP", help="Games Played"),
			"Mean FPts": st.column_config.NumberColumn("Mean FPts", help="Average fantasy points per game", format="%.1f"),
			"Median FPts": st.column_config.NumberColumn("Median FPts", format="%.1f"),
			"Std Dev": st.column_config.NumberColumn("Std Dev", help="Standard deviation", format="%.1f"),
			"CV %": st.column_config.NumberColumn("CV %", help="Coefficient of Variation - lower is more consistent", format="%.1f"),
			"Consistency": st.column_config.TextColumn("Consistency", help="Consistency category based on CV%"),
			"Min": st.column_config.NumberColumn("Min", format="%.0f"),
			"Max": st.column_config.NumberColumn("Max", format="%.0f"),
			"Range": st.column_config.NumberColumn("Range", format="%.0f"),
			"Boom Games": st.column_config.NumberColumn("Boom Games"),
			"Boom %": st.column_config.NumberColumn("Boom %", format="%.1f"),
			"Bust Games": st.column_config.NumberColumn("Bust Games"),
			"Bust %": st.column_config.NumberColumn("Bust %", format="%.1f"),
		}
	)

	# Download buttons
	col1, col2 = st.columns(2)
	with col1:
		csv_filtered = display_df.to_csv(index=False)
		st.download_button(
			label=f"📥 Download Filtered ({len(display_df)} players)",
			data=csv_filtered,
			file_name=f"league_consistency_filtered_{league_id}.csv",
			mime="text/csv",
			key="download_filtered"
		)
	with col2:
		csv_all = overview_df.to_csv(index=False)
		st.download_button(
			label=f"📥 Download All ({len(overview_df)} players)",
			data=csv_all,
			file_name=f"league_consistency_all_{league_id}.csv",
			mime="text/csv",
			key="download_all"
		)

def _display_visualizations(overview_df):
	"""Display league-wide visualizations."""
	st.markdown("---")
	st.subheader("📊 League-Wide Visualizations")
	
	viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5, viz_tab6 = st.tabs([
		"CV% Distribution", 
		"Consistency vs Production", 
		"Boom/Bust Analysis",
		"Performance Ranges",
		"Tier Breakdown",
		"Advanced Metrics"
	])
	
	with viz_tab1:
		_display_cv_distribution(overview_df)
	
	with viz_tab2:
		_display_consistency_vs_production(overview_df)
	
	with viz_tab3:
		_display_boom_bust_scatter(overview_df)
	
	with viz_tab4:
		_display_performance_ranges(overview_df)
	
	with viz_tab5:
		_display_tier_breakdown(overview_df)
	
	with viz_tab6:
		_display_advanced_metrics(overview_df)

def _display_cv_distribution(overview_df):
	"""Display CV% distribution histogram."""
	fig_cv_dist = px.histogram(
		overview_df,
		x='CV %',
		nbins=30,
		title="Distribution of Player Consistency (CV%)",
		labels={'CV %': 'Coefficient of Variation (%)'},
		color_discrete_sequence=['#1f77b4']
	)
	
	# Add vertical lines for thresholds
	fig_cv_dist.add_vline(x=20, line_dash="dash", line_color="green", annotation_text="Very Consistent")
	fig_cv_dist.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Volatile")
	
	fig_cv_dist.update_layout(height=400)
	st.plotly_chart(fig_cv_dist, use_container_width=True)

def _display_consistency_vs_production(overview_df):
	"""Display consistency vs production scatter plot."""
	fig_scatter = px.scatter(
		overview_df,
		x='Mean FPts',
		y='CV %',
		hover_data=['Player', 'GP'],
		title="Player Consistency vs Production",
		labels={'Mean FPts': 'Average FPts/Game', 'CV %': 'Coefficient of Variation (%)'},
		color='CV %',
		color_continuous_scale='RdYlGn_r'
	)
	
	# Add quadrant lines
	median_fpts = overview_df['Mean FPts'].median()
	fig_scatter.add_vline(x=median_fpts, line_dash="dot", line_color="gray")
	fig_scatter.add_hline(y=25, line_dash="dot", line_color="gray")
	
	fig_scatter.update_layout(height=500)
	st.plotly_chart(fig_scatter, use_container_width=True)
	
	st.caption("**Top Right:** High production, high variance | **Top Left:** Low production, high variance")
	st.caption("**Bottom Right:** High production, consistent | **Bottom Left:** Low production, consistent")

def _display_boom_bust_scatter(overview_df):
	"""Display boom vs bust rates scatter plot."""
	fig_boom_bust = go.Figure()
	
	fig_boom_bust.add_trace(go.Scatter(
		x=overview_df['Boom %'],
		y=overview_df['Bust %'],
		mode='markers',
		marker=dict(
			size=overview_df['Mean FPts'] / 5,  # Size by production
			color=overview_df['CV %'],
			colorscale='RdYlGn_r',
			showscale=True,
			colorbar=dict(title="CV%")
		),
		text=overview_df['Player'],
		hovertemplate='<b>%{text}</b><br>Boom: %{x:.1f}%<br>Bust: %{y:.1f}%<extra></extra>'
	))
	
	fig_boom_bust.update_layout(
		title="Boom Rate vs Bust Rate",
		xaxis_title="Boom Rate (%)",
		yaxis_title="Bust Rate (%)",
		height=500
	)
	
	st.plotly_chart(fig_boom_bust, use_container_width=True)
	st.caption("**Bubble size** = Average FPts/Game | **Color** = CV% (red = volatile, green = consistent)")

def _display_performance_ranges(overview_df):
	"""Display performance range analysis."""
	st.markdown("### Performance Range Analysis")
	st.caption("Shows the scoring range (Max - Min) for each player to identify ceiling/floor gaps")
	
	# Create range chart
	fig_range = go.Figure()
	
	# Sort by Mean FPts for better visualization
	df_sorted = overview_df.sort_values('Mean FPts', ascending=True)
	
	fig_range.add_trace(go.Bar(
		y=df_sorted['Player'],
		x=df_sorted['Range'],
		orientation='h',
		marker=dict(
			color=df_sorted['CV %'],
			colorscale='RdYlGn_r',
			showscale=True,
			colorbar=dict(title="CV%")
		),
		hovertemplate='<b>%{y}</b><br>Range: %{x:.1f}<br>Min: %{customdata[0]:.0f}<br>Max: %{customdata[1]:.0f}<extra></extra>',
		customdata=df_sorted[['Min', 'Max']].values
	))
	
	fig_range.update_layout(
		title="Player Performance Ranges (Sorted by Mean FPts)",
		xaxis_title="Range (Max - Min FPts)",
		yaxis_title="Player",
		height=max(400, len(df_sorted) * 15),
		showlegend=False
	)
	
	st.plotly_chart(fig_range, use_container_width=True)
	
	# Range statistics
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Avg Range", f"{overview_df['Range'].mean():.1f}")
	with col2:
		st.metric("Largest Range", f"{overview_df['Range'].max():.1f}")
		st.caption(overview_df.loc[overview_df['Range'].idxmax(), 'Player'])
	with col3:
		st.metric("Smallest Range", f"{overview_df['Range'].min():.1f}")
		st.caption(overview_df.loc[overview_df['Range'].idxmin(), 'Player'])
	with col4:
		high_range_count = len(overview_df[overview_df['Range'] > overview_df['Range'].quantile(0.75)])
		st.metric("High Range Players", high_range_count, help="Players in top 25% of range")

def _display_tier_breakdown(overview_df):
	"""Display tier-based breakdown of players."""
	st.markdown("### Player Tier Breakdown")
	
	# Define tiers based on Mean FPts
	def get_tier(fpts):
		if fpts >= 80:
			return "Elite (80+ FPts)"
		elif fpts >= 60:
			return "Star (60-80 FPts)"
		elif fpts >= 40:
			return "Solid (40-60 FPts)"
		elif fpts >= 25:
			return "Streamer (25-40 FPts)"
		else:
			return "Bench (<25 FPts)"
	
	overview_df['Tier'] = overview_df['Mean FPts'].apply(get_tier)
	
	# Tier distribution
	tier_counts = overview_df['Tier'].value_counts()
	tier_order = ["Elite (80+ FPts)", "Star (60-80 FPts)", "Solid (40-60 FPts)", "Streamer (25-40 FPts)", "Bench (<25 FPts)"]
	tier_counts = tier_counts.reindex([t for t in tier_order if t in tier_counts.index])
	
	col1, col2 = st.columns([1, 1])
	
	with col1:
		# Pie chart
		fig_pie = px.pie(
			values=tier_counts.values,
			names=tier_counts.index,
			title="Player Distribution by Tier",
			color_discrete_sequence=px.colors.sequential.RdBu_r
		)
		st.plotly_chart(fig_pie, use_container_width=True)
	
	with col2:
		# Consistency by tier
		tier_cv = overview_df.groupby('Tier')['CV %'].agg(['mean', 'min', 'max']).reindex([t for t in tier_order if t in overview_df['Tier'].unique()])
		
		fig_tier_cv = go.Figure()
		fig_tier_cv.add_trace(go.Bar(
			x=tier_cv.index,
			y=tier_cv['mean'],
			name='Avg CV%',
			marker_color='lightblue',
			error_y=dict(
				type='data',
				symmetric=False,
				array=tier_cv['max'] - tier_cv['mean'],
				arrayminus=tier_cv['mean'] - tier_cv['min']
			)
		))
		
		fig_tier_cv.update_layout(
			title="Average CV% by Tier (with range)",
			xaxis_title="Tier",
			yaxis_title="CV%",
			showlegend=False
		)
		st.plotly_chart(fig_tier_cv, use_container_width=True)
	
	# Detailed tier table
	st.markdown("#### Tier Details")
	tier_summary = overview_df.groupby('Tier').agg({
		'Player': 'count',
		'Mean FPts': ['mean', 'min', 'max'],
		'CV %': 'mean',
		'Boom %': 'mean',
		'Bust %': 'mean'
	}).round(1)
	tier_summary.columns = ['Count', 'Avg FPts', 'Min FPts', 'Max FPts', 'Avg CV%', 'Avg Boom%', 'Avg Bust%']
	tier_summary = tier_summary.reindex([t for t in tier_order if t in tier_summary.index])
	
	st.dataframe(tier_summary, use_container_width=True)

def _display_advanced_metrics(overview_df):
	"""Display advanced statistical analysis."""
	st.markdown("### Advanced Statistical Analysis")
	
	# Correlation heatmap
	st.markdown("#### Metric Correlations")
	corr_cols = ['Mean FPts', 'Median FPts', 'Std Dev', 'CV %', 'Range', 'Boom %', 'Bust %', 'GP']
	corr_matrix = overview_df[corr_cols].corr()
	
	fig_corr = px.imshow(
		corr_matrix,
		text_auto='.2f',
		aspect='auto',
		color_continuous_scale='RdBu_r',
		title="Correlation Matrix of Performance Metrics",
		labels=dict(color="Correlation")
	)
	st.plotly_chart(fig_corr, use_container_width=True)
	
	st.caption("**Interpretation:** Values close to 1 or -1 indicate strong correlation. Look for unexpected relationships!")
	
	# Top performers by different metrics
	st.markdown("---")
	st.markdown("#### Top 10 Players by Different Metrics")
	
	metric_col1, metric_col2, metric_col3 = st.columns(3)
	
	with metric_col1:
		st.markdown("**🏆 Highest Production**")
		top_fpts = overview_df.nlargest(10, 'Mean FPts')[['Player', 'Mean FPts', 'CV %']]
		st.dataframe(top_fpts, hide_index=True, use_container_width=True)
	
	with metric_col2:
		st.markdown("**🟢 Most Consistent**")
		top_consistent = overview_df.nsmallest(10, 'CV %')[['Player', 'CV %', 'Mean FPts']]
		st.dataframe(top_consistent, hide_index=True, use_container_width=True)
	
	with metric_col3:
		st.markdown("**💥 Highest Boom Rate**")
		top_boom = overview_df.nlargest(10, 'Boom %')[['Player', 'Boom %', 'Mean FPts']]
		st.dataframe(top_boom, hide_index=True, use_container_width=True)
	
	# Statistical outliers
	st.markdown("---")
	st.markdown("#### Statistical Outliers")
	
	# Calculate z-scores for CV%
	mean_cv = overview_df['CV %'].mean()
	std_cv = overview_df['CV %'].std()
	overview_df['CV_zscore'] = (overview_df['CV %'] - mean_cv) / std_cv
	
	outlier_col1, outlier_col2 = st.columns(2)
	
	with outlier_col1:
		st.markdown("**🔴 Extremely Volatile Players** (CV% > 2 std dev)")
		volatile_outliers = overview_df[overview_df['CV_zscore'] > 2][['Player', 'CV %', 'Mean FPts', 'Boom %', 'Bust %']].sort_values('CV %', ascending=False)
		if not volatile_outliers.empty:
			st.dataframe(volatile_outliers, hide_index=True, use_container_width=True)
		else:
			st.info("No extreme outliers found")
	
	with outlier_col2:
		st.markdown("**🟢 Extremely Consistent Players** (CV% < -2 std dev)")
		consistent_outliers = overview_df[overview_df['CV_zscore'] < -2][['Player', 'CV %', 'Mean FPts', 'Min', 'Max']].sort_values('CV %')
		if not consistent_outliers.empty:
			st.dataframe(consistent_outliers, hide_index=True, use_container_width=True)
		else:
			st.info("No extreme outliers found")
	
	# Risk-Reward Analysis
	st.markdown("---")
	st.markdown("#### Risk-Reward Analysis")
	st.caption("Players in the top-right quadrant offer high production with acceptable consistency")
	
	# Create quadrant chart
	median_fpts = overview_df['Mean FPts'].median()
	median_cv = overview_df['CV %'].median()
	
	fig_quad = px.scatter(
		overview_df,
		x='Mean FPts',
		y='CV %',
		size='GP',
		color='Boom %',
		hover_data=['Player', 'Bust %'],
		title=f"Risk-Reward Quadrants (Median FPts: {median_fpts:.1f}, Median CV%: {median_cv:.1f})",
		labels={'Mean FPts': 'Production (Mean FPts)', 'CV %': 'Risk (CV%)'},
		color_continuous_scale='Viridis'
	)
	
	# Add quadrant lines
	fig_quad.add_hline(y=median_cv, line_dash="dash", line_color="gray", opacity=0.5)
	fig_quad.add_vline(x=median_fpts, line_dash="dash", line_color="gray", opacity=0.5)
	
	# Add quadrant labels
	fig_quad.add_annotation(x=overview_df['Mean FPts'].max() * 0.9, y=overview_df['CV %'].min() * 1.1,
							text="High Prod, Low Risk", showarrow=False, font=dict(size=10, color="green"))
	fig_quad.add_annotation(x=overview_df['Mean FPts'].max() * 0.9, y=overview_df['CV %'].max() * 0.9,
							text="High Prod, High Risk", showarrow=False, font=dict(size=10, color="orange"))
	fig_quad.add_annotation(x=overview_df['Mean FPts'].min() * 1.1, y=overview_df['CV %'].min() * 1.1,
							text="Low Prod, Low Risk", showarrow=False, font=dict(size=10, color="blue"))
	fig_quad.add_annotation(x=overview_df['Mean FPts'].min() * 1.1, y=overview_df['CV %'].max() * 0.9,
							text="Low Prod, High Risk", showarrow=False, font=dict(size=10, color="red"))
	
	fig_quad.update_layout(height=600)
	st.plotly_chart(fig_quad, use_container_width=True)
