"""Trade Suggestions Page - AI-powered trade recommendations."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.trade_suggestions import find_trade_suggestions, calculate_exponential_value
from modules.player_game_log_scraper.logic import get_cache_directory
from modules.player_game_log_scraper.ui_fantasy_teams import _load_fantasy_team_rosters, _build_fantasy_team_view
from modules.trade_analysis.consistency_integration import (
	CONSISTENCY_VERY_MAX_CV,
	CONSISTENCY_MODERATE_MAX_CV,
)
from pathlib import Path
import json

st.set_page_config(page_title="Trade Suggestions", page_icon="🤝", layout="wide")

st.title("🤝 AI-Powered Trade Suggestions")
st.markdown("Get intelligent trade recommendations based on exponential value calculations and consistency analysis.")

# Load league data
try:
	from league_config import FANTRAX_DEFAULT_LEAGUE_ID
except ImportError:
	FANTRAX_DEFAULT_LEAGUE_ID = ""

league_id = st.text_input("League ID", value=FANTRAX_DEFAULT_LEAGUE_ID, key="trade_suggest_league_id")

if not league_id:
	st.warning("Please enter a league ID to get trade suggestions.")
	st.stop()

# Load cached player data (new full-format files)
cache_dir = get_cache_directory()
cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))

if not cache_files:
	st.error(f"No cached player data found for league {league_id}. Please run Bulk Scrape in Admin Tools first.")
	st.stop()

# Build team rosters
rosters_by_team = _build_fantasy_team_view(league_id, cache_files)

if not rosters_by_team:
	st.error("Could not load team rosters. Make sure player data is properly formatted.")
	st.stop()

st.success(f"✅ Loaded {len(rosters_by_team)} teams with player data")

# Configuration Section
st.markdown("---")
st.markdown("## ⚙️ Configuration")

col1, col2, col3 = st.columns(3)

with col1:
	your_team_name = st.selectbox(
		"Select Your Team",
		options=sorted(rosters_by_team.keys()),
		key="your_team_select"
	)

with col2:
	trade_patterns = st.multiselect(
		"Trade Patterns",
		options=['1-for-1', '2-for-1', '1-for-2', '2-for-2', '3-for-1', '1-for-3', '3-for-2', '2-for-3', '3-for-3'],
		default=['1-for-1', '2-for-1', '1-for-2', '2-for-2'],
		help="Select which trade patterns to consider"
	)

with col3:
	min_value_gain = st.slider(
		"Min Value Gain",
		min_value=0.0,
		max_value=50.0,
		value=10.0,
		step=5.0,
		help="Minimum value improvement to suggest a trade"
	)

# Advanced Filters
with st.expander("🔧 Advanced Filters", expanded=False):
	col1, col2 = st.columns(2)
	
	with col1:
		available_teams = [t for t in sorted(rosters_by_team.keys()) if t != your_team_name]
		target_teams = st.multiselect(
			"Target Specific Teams (optional)",
			options=available_teams,
			help="Leave empty to consider all teams"
		)
		
		exclude_teams = st.multiselect(
			"Exclude Teams (optional)",
			options=available_teams,
			help="Teams you do NOT want to trade with"
		)
		
		max_suggestions = st.number_input(
			"Max Suggestions",
			min_value=5,
			max_value=100,
			value=20,
			step=5
		)
	
	with col2:
		your_team_df = rosters_by_team[your_team_name]
		exclude_players = st.multiselect(
			"Exclude Your Players (untouchables)",
			options=sorted(your_team_df['Player'].tolist()),
			help="Players you don't want to trade away"
		)
		
		include_players = st.multiselect(
			"Must-Include From Your Team (trade bait)",
			options=sorted(your_team_df['Player'].tolist()),
			help="Only show trades where at least one of these players is included on your side"
		)
		
		# Opposing player filters
		other_players = sorted(
			p
			for team, df in rosters_by_team.items()
			if team != your_team_name
			for p in df['Player'].tolist()
		)
		target_opposing_players = st.multiselect(
			"Target Opposing Players",
			options=other_players,
			help="Only show trades where at least one of these players is included on the other side"
		)
		exclude_opposing_players = st.multiselect(
			"Exclude Opposing Players",
			options=other_players,
			help="Opposing players you do NOT want to receive in trades"
		)

# Helper function for displaying suggestions
def display_trade_suggestion(suggestion, rank):
	"""Display a single trade suggestion with details."""
	# Calculate core FP/G metrics (value_gain is weekly core FP change)
	# Assume MIN_GAMES = 25, core_size ≈ 7-8
	core_size_approx = 7.14  # 25 / 3.5
	min_games = 25
	
	# value_gain is already scaled to weekly FP, so derive core PPG change
	weekly_core_fp_change = suggestion['value_gain']
	core_ppg_change = weekly_core_fp_change / min_games
	
	# Opponent perspective
	opp_weekly_core_fp_change = suggestion.get('opp_core_gain', 0)
	opp_core_ppg_change = opp_weekly_core_fp_change / min_games
	
	# Header with core impact
	st.markdown(f"### 🔄 Trade Impact: **+{weekly_core_fp_change:.1f} weekly core FP** for you")
	st.caption(f"Your core FP/G improves by ~{core_ppg_change:.2f} across your top {int(core_size_approx)} players")
	
	col1, col2 = st.columns(2)
	
	with col1:
		st.markdown("### 📤 You Give")
		give_df = pd.DataFrame({
			'Player': suggestion['you_give'],
			'FP/G': suggestion['your_fpts'],
			'CV%': suggestion['your_cv']
		})
		st.dataframe(give_df, hide_index=True, use_container_width=True)
		your_avg_fpts = sum(suggestion['your_fpts']) / len(suggestion['your_fpts'])
		st.caption(f"Package avg: {your_avg_fpts:.1f} FP/G")
	
	with col2:
		st.markdown("### 📥 You Get")
		get_df = pd.DataFrame({
			'Player': suggestion['you_get'],
			'FP/G': suggestion['their_fpts'],
			'CV%': suggestion['their_cv']
		})
		st.dataframe(get_df, hide_index=True, use_container_width=True)
		their_avg_fpts = sum(suggestion['their_fpts']) / len(suggestion['their_fpts'])
		st.caption(f"Package avg: {their_avg_fpts:.1f} FP/G")
	
	# Core impact visualization
	st.markdown("---")
	st.markdown("#### 📊 Weekly Core FP Impact")
	
	fig = go.Figure()
	
	fig.add_trace(go.Bar(
		name='Your Gain',
		x=['Weekly Core FP'],
		y=[weekly_core_fp_change],
		marker_color='#4CAF50',
		text=[f"+{weekly_core_fp_change:.1f}"],
		textposition='outside'
	))
	
	fig.add_trace(go.Bar(
		name='Opponent Change',
		x=['Weekly Core FP'],
		y=[opp_weekly_core_fp_change],
		marker_color='#FF9800' if opp_weekly_core_fp_change < 0 else '#2196F3',
		text=[f"{opp_weekly_core_fp_change:+.1f}"],
		textposition='outside'
	))
	
	fig.update_layout(
		barmode='group',
		height=300,
		showlegend=True,
		yaxis_title="Weekly Core FP Change"
	)
	
	st.plotly_chart(fig, use_container_width=True, key=f"trade_value_chart_{rank}")
	
	# Trade assessment based on core impact
	if weekly_core_fp_change > 30:
		st.success("🟢 **Excellent Trade** - Major weekly core FP upgrade!")
	elif weekly_core_fp_change > 15:
		st.success("🟢 **Strong Trade** - Solid weekly core FP gain")
	elif weekly_core_fp_change > 5:
		st.info("🟡 **Decent Trade** - Modest weekly core FP improvement")
	else:
		st.info("🟡 **Marginal Trade** - Small weekly core FP gain")
	
	# Opponent fairness check
	if opp_weekly_core_fp_change < -15:
		st.warning("⚠️ **Opponent loses significant core FP** - they may not accept")
	elif opp_weekly_core_fp_change < -5:
		st.info("ℹ️ **Opponent loses some core FP** - negotiate carefully")
	elif opp_weekly_core_fp_change > 0:
		st.success("✅ **Win-win trade** - opponent also gains core FP")
	
	# Why this trade works (901/902 aligned)
	st.markdown("#### 💡 Why This Trade Works (League Philosophy)")
	
	fpts_diff = their_avg_fpts - your_avg_fpts
	
	your_avg_cv = sum(suggestion['your_cv']) / len(suggestion['your_cv'])
	their_avg_cv = sum(suggestion['their_cv']) / len(suggestion['their_cv'])
	cv_diff = their_avg_cv - your_avg_cv
	
	reasons = []
	
	# Core FP/G is king
	reasons.append(f"📈 **Core Roster Upgrade:** Your top ~{int(core_size_approx)} players improve by {core_ppg_change:.2f} FP/G on average")
	reasons.append(f"🎯 **Weekly Impact:** Estimated +{weekly_core_fp_change:.1f} FP per week at the 25-game minimum")
	
	# Consolidation strategy (from 901)
	if suggestion['pattern'] in ['2-for-1', '3-for-1']:
		reasons.append(f"✅ **Consolidation Strategy:** Trading depth for elite FP/G aligns with 25-game minimum format")
		reasons.append(f"💡 **Freed Roster Spot:** Opens space for smart streaming or waiver pickups")
	
	# Expansion strategy (depth play)
	if suggestion['pattern'] in ['1-for-2', '1-for-3']:
		reasons.append(f"✅ **Depth Strategy:** Trading star for solid depth helps hit 25 games without risky streamers")
		reasons.append(f"📊 **Floor Improvement:** Replaces weak roster spots (30-35 FP/G) with reliable players (40+ FP/G)")
		reasons.append(f"💡 **Streaming Risk Reduction:** Less reliance on volatile waiver pickups")
	
	# Package context (but emphasize it's not the main story)
	if fpts_diff > 5:
		reasons.append(f"📊 **Package FP/G:** +{fpts_diff:.1f} FP/G (but core impact matters more)")
	elif fpts_diff < -5:
		reasons.append(f"⚖️ **Trading Down in Package FP/G:** −{abs(fpts_diff):.1f} FP/G, but your **core** still improves")
	
	if cv_diff < -5:
		reasons.append(f"✅ **Consistency Upgrade:** {abs(cv_diff):.1f}% less volatile")
	
	for reason in reasons:
		st.markdown(reason)
	
	if not reasons:
		st.markdown("📊 **Value-based improvement** through optimized core roster construction")
	
	# Deep Dive Analysis
	with st.expander("🔬 Deep Dive Analysis", expanded=False):
		st.markdown("#### Statistical Breakdown")
		
		col1, col2 = st.columns(2)
		
		with col1:
			st.markdown("**Your Side:**")
			your_total_fpts = sum(suggestion['your_fpts'])
			your_total_cv = sum(suggestion['your_cv'])
			st.metric("Total FPts", f"{your_total_fpts:.1f}")
			st.metric("Avg CV%", f"{your_avg_cv:.1f}%")
			st.metric("Players", len(suggestion['you_give']))
			
			# Risk assessment based on global CV% bands
			if your_avg_cv < CONSISTENCY_VERY_MAX_CV:
				risk = "🟢 Low Risk"
			elif your_avg_cv <= CONSISTENCY_MODERATE_MAX_CV:
				risk = "🟡 Moderate Risk"
			else:
				risk = "🔴 High Risk"
			st.caption(f"Risk Level: {risk}")
		
		with col2:
			st.markdown("**Their Side:**")
			their_total_fpts = sum(suggestion['their_fpts'])
			their_total_cv = sum(suggestion['their_cv'])
			st.metric("Total FPts", f"{their_total_fpts:.1f}")
			st.metric("Avg CV%", f"{their_avg_cv:.1f}%")
			st.metric("Players", len(suggestion['you_get']))
			
			# Risk assessment
			if their_avg_cv < CONSISTENCY_VERY_MAX_CV:
				risk = "🟢 Low Risk"
			elif their_avg_cv <= CONSISTENCY_MODERATE_MAX_CV:
				risk = "🟡 Moderate Risk"
			else:
				risk = "🔴 High Risk"
			st.caption(f"Risk Level: {risk}")
		
		st.markdown("---")
		st.markdown("#### Trade Impact Analysis")
		
		# FPts change
		fpts_change = their_total_fpts - your_total_fpts
		fpts_pct = (fpts_change / your_total_fpts * 100) if your_total_fpts > 0 else 0
		
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("FPts Change", f"{fpts_change:+.1f}", f"{fpts_pct:+.1f}%")
		with col2:
			cv_change = their_avg_cv - your_avg_cv
			st.metric("CV% Change", f"{cv_change:+.1f}%", help="Negative = more consistent")
		with col3:
			roster_change = len(suggestion['you_get']) - len(suggestion['you_give'])
			st.metric("Roster Spots", f"{roster_change:+d}", help="Negative = consolidation")
		
		# Tier analysis
		st.markdown("---")
		st.markdown("#### Player Tier Analysis")
		
		def get_tier(fpts):
			if fpts >= 80:
				return "Elite (80+)"
			elif fpts >= 60:
				return "Star (60-80)"
			elif fpts >= 40:
				return "Solid (40-60)"
			elif fpts >= 25:
				return "Streamer (25-40)"
			else:
				return "Bench (<25)"
		
		your_tiers = [get_tier(f) for f in suggestion['your_fpts']]
		their_tiers = [get_tier(f) for f in suggestion['their_fpts']]
		
		col1, col2 = st.columns(2)
		with col1:
			st.markdown("**You Give:**")
			for player, fpts, tier in zip(suggestion['you_give'], suggestion['your_fpts'], your_tiers):
				st.caption(f"• {player}: {fpts:.1f} FPts ({tier})")
		
		with col2:
			st.markdown("**You Get:**")
			for player, fpts, tier in zip(suggestion['you_get'], suggestion['their_fpts'], their_tiers):
				st.caption(f"• {player}: {fpts:.1f} FPts ({tier})")
		
		# Strategic considerations
		st.markdown("---")
		st.markdown("#### Strategic Considerations")
		
		considerations = []
		
		# Consolidation vs expansion
		if roster_change < 0:
			considerations.append("📦 **Roster Consolidation:** Freeing up roster spots for waiver pickups or streaming")
		elif roster_change > 0:
			considerations.append("📈 **Roster Expansion:** Adding depth, useful if you have injuries or need flexibility")
		
		# Risk profile change
		if cv_change < -5:
			considerations.append("🛡️ **Risk Reduction:** Significantly more consistent roster, better for playoffs")
		elif cv_change > 5:
			considerations.append("🎲 **Risk Increase:** More volatile roster, higher ceiling but lower floor")
		
		# Production change
		if fpts_pct > 10:
			considerations.append("🚀 **Major Upgrade:** Significant production boost, worth pursuing aggressively")
		elif fpts_pct > 5:
			considerations.append("📊 **Solid Upgrade:** Meaningful production improvement")
		elif fpts_pct < -5:
			considerations.append("⚠️ **Production Loss:** Losing total FPts, only worth it if consolidating for elite talent")
		
		# Value-based
		if suggestion['value_gain'] > 30:
			considerations.append("💎 **Elite Value Play:** Exponential value heavily favors this trade")
		
		for consideration in considerations:
			st.markdown(consideration)
		
		if not considerations:
			st.markdown("📊 Balanced trade with marginal improvements")

# Generate Suggestions Button
if st.button("🔍 Find Trade Suggestions", type="primary"):
	with st.spinner("Analyzing trade opportunities..."):
		your_team_df = rosters_by_team[your_team_name]
		other_teams = {k: v for k, v in rosters_by_team.items() if k != your_team_name}
		
		suggestions = find_trade_suggestions(
			your_team=your_team_df,
			other_teams=other_teams,
			trade_patterns=trade_patterns,
			min_value_gain=min_value_gain,
			max_suggestions=max_suggestions,
			target_teams=target_teams if target_teams else None,
			exclude_players=exclude_players if exclude_players else None,
			include_players=include_players if include_players else None,
			exclude_teams=exclude_teams if exclude_teams else None,
			target_opposing_players=target_opposing_players if target_opposing_players else None,
			exclude_opposing_players=exclude_opposing_players if exclude_opposing_players else None,
		)
		
		if not suggestions:
			st.warning("No beneficial trades found with current filters. Try adjusting your criteria.")
			st.stop()
		
		st.success(f"✅ Found {len(suggestions)} trade suggestions!")
		
		# Display suggestions
		st.markdown("---")
		st.markdown("## 📊 Trade Suggestions")
		
		# Summary metrics
		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.metric("Total Suggestions", len(suggestions))
		with col2:
			best_gain = suggestions[0]['value_gain']
			st.metric("Best Value Gain", f"{best_gain:.1f}")
		with col3:
			avg_gain = sum(s['value_gain'] for s in suggestions) / len(suggestions)
			st.metric("Avg Value Gain", f"{avg_gain:.1f}")
		with col4:
			pattern_counts = {}
			for s in suggestions:
				pattern_counts[s['pattern']] = pattern_counts.get(s['pattern'], 0) + 1
			most_common = max(pattern_counts, key=pattern_counts.get)
			st.metric("Most Common Pattern", most_common)
		
		# Display each suggestion
		for i, suggestion in enumerate(suggestions, 1):
			with st.expander(f"#{i} - {suggestion['pattern']} with {suggestion['team']} (Value Gain: +{suggestion['value_gain']:.1f})", expanded=(i <= 3)):
				display_trade_suggestion(suggestion, i)

# Show value calculation explanation
with st.expander("ℹ️ How Value is Calculated", expanded=False):
	st.markdown("""
	### Exponential Value System
	
	This tool uses an **exponential value calculation** to account for the fact that elite players are worth 
	significantly more than their raw FPts suggest.
	
	#### Why Exponential?
	- A 95 FPts player is worth **WAY MORE** than two 47.5 FPts players
	- A 70 FPts player is worth **MORE** than a 45 FPts + 25 FPts combo
	- Elite talent is scarce and irreplaceable
	
	#### Value Formula
	```
	Base Value = (FPts ^ 1.8) * 0.35
	
	Elite Tier Bonuses (Scarcity Premium):
	- Top 5 (90+ FPts): +30% bonus (Jokic, Giannis tier)
	- Top 10 (80-90 FPts): +15% bonus
	- Top 20 (70-80 FPts): +8% bonus
	
	Consistency Adjustment:
	- Very Consistent (CV < 25%): +15% value
	- Solid / Moderate (CV 25-40%): No adjustment  
	- Volatile / Boom-Bust (CV > 40%): -15% value
	
	Games Played Penalty:
	- < 10 games: -30% value (small sample)
	- 10-20 games: -15% value
	- 20+ games: No penalty
	```
	
	#### Example Values
	- 45 FPts player ≈ 200 value
	- 70 FPts player ≈ 550 value (1.5x FPts, 2.75x value!)
	- 95 FPts player ≈ 1200 value (2.1x FPts, 6x value!)
	
	This ensures trades favor **quality over quantity** and properly value elite talent.
	""")

# Show exponential curve visualization
with st.expander("📈 Exponential Value Curve", expanded=False):
	fpts_range = list(range(20, 101, 5))
	values = [calculate_exponential_value(f) for f in fpts_range]
	
	fig = go.Figure()
	
	fig.add_trace(go.Scatter(
		x=fpts_range,
		y=values,
		mode='lines+markers',
		name='Exponential Value',
		line=dict(color='#4a90e2', width=3),
		marker=dict(size=8)
	))
	
	# Add linear comparison
	linear_values = [f * 5 for f in fpts_range]  # Simple linear scaling
	fig.add_trace(go.Scatter(
		x=fpts_range,
		y=linear_values,
		mode='lines',
		name='Linear Value (for comparison)',
		line=dict(color='gray', width=2, dash='dash'),
		opacity=0.5
	))
	
	fig.update_layout(
		title="Exponential vs Linear Value Scaling",
		xaxis_title="Fantasy Points per Game",
		yaxis_title="Player Value",
		height=400,
		hovermode='x unified'
	)
	
	st.plotly_chart(fig, use_container_width=True, key="exponential_curve_chart")
	
	st.caption("Notice how the exponential curve (blue) grows much faster than linear (gray) at higher FPts levels. This is why elite players are so valuable!")
