"""UI components for player game log scraper - visualization and display helpers."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def display_variability_metrics(stats, player_name):
	"""Display variability statistics in a clean metric layout."""
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
			f"{stats['coefficient_of_variation']:.2f}%",
			help="Coefficient of Variation = (Std Dev / Mean) × 100. Lower % = more consistent. <25% = very consistent, 25–40% = solid/moderate, >40% = volatile/boom-bust"
		)
	
	# Context for high performers
	if stats['mean_fpts'] >= 80:
		st.info(f"⭐ **Elite Player Context:** With a {stats['mean_fpts']:.1f} FPts/G average, even 'low' games of {stats['min_fpts']:.0f} FPts are excellent. The variability metrics show consistency *relative to this player's elite production*, not absolute fantasy value.")

def display_boom_bust_analysis(stats):
	"""Display boom/bust analysis metrics."""
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

def display_fpts_trend_chart(df, stats, player_name):
	"""Display FPts trend line chart with boom/bust zones."""
	fig_trend = go.Figure()
	
	# Add FPts line
	fig_trend.add_trace(go.Scatter(
		x=list(range(len(df), 0, -1)),
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
	
	st.plotly_chart(fig_trend, width="stretch")

def display_distribution_chart(df, stats, player_name):
	"""Display FPts distribution histogram."""
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
	
	st.plotly_chart(fig_hist, width="stretch")
	
	# Add distribution stats
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Skewness", f"{df['FPts'].skew():.2f}", help="Negative = left tail, Positive = right tail")
	with col2:
		st.metric("Kurtosis", f"{df['FPts'].kurtosis():.2f}", help="Higher = more extreme outliers")
	with col3:
		percentile_75 = df['FPts'].quantile(0.75)
		st.metric("75th Percentile", f"{percentile_75:.1f}", help="75% of games below this")

def display_boom_bust_zones_chart(df, stats, player_name):
	"""Display boom/bust zones scatter plot."""
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
	
	st.plotly_chart(fig_zones, width="stretch")
	
	# Summary table
	category_counts = df_viz['Category'].value_counts()
	st.write("**Game Classification Summary:**")
	summary_df = pd.DataFrame({
		'Category': category_counts.index,
		'Games': category_counts.values,
		'Percentage': (category_counts.values / len(df_viz) * 100).round(1)
	})
	st.dataframe(summary_df, hide_index=True)

def display_category_breakdown(df, player_name):
	"""Display category breakdown bar chart."""
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
		
		st.plotly_chart(fig_cats, width="stretch")
		
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
