import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_compat import plotly_chart
from modules.trade_suggestions import find_trade_suggestions, calculate_exponential_value, set_trade_balance_preset
from modules.trade_suggestions.trade_suggestions_config import MIN_TRADE_FP_G
from modules.player_game_log_scraper.logic import get_cache_directory
from modules.player_game_log_scraper.ui_fantasy_teams import _build_fantasy_team_view
from modules.trade_analysis.consistency_integration import (
	CONSISTENCY_VERY_MAX_CV,
	CONSISTENCY_MODERATE_MAX_CV,
)
from modules.player_game_log_scraper import db_store


def display_trade_suggestions_tab():
	"""Embedded Trade Suggestions UI for use inside other pages (tabs)."""
	st.subheader("🤝 AI-Powered Trade Suggestions")
	st.markdown("Get intelligent trade recommendations based on exponential value calculations and consistency analysis.")

	try:
		from league_config import FANTRAX_DEFAULT_LEAGUE_ID
	except ImportError:
		FANTRAX_DEFAULT_LEAGUE_ID = ""

	league_id = st.text_input("League ID", value=FANTRAX_DEFAULT_LEAGUE_ID, key="tab_trade_suggest_league_id")
	if not league_id:
		st.warning("Please enter a league ID to get trade suggestions.")
		return

	cache_dir = get_cache_directory()
	cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
	if not cache_files:
		st.error(f"No cached player data found for league {league_id}. Please run Bulk Scrape in Admin Tools first.")
		return

	selected_season = None
	try:
		seasons = db_store.get_league_available_seasons(league_id)
		if seasons:
			selected_season = seasons[0]
	except Exception:
		selected_season = None

	if not selected_season:
		season_set = set()
		for cf in cache_files:
			parts = cf.stem.split('_')
			if len(parts) >= 2:
				season_part = '_'.join(parts[-2:])
				season = season_part.replace('_', '-')
				season_set.add(season)
		if season_set:
			selected_season = sorted(list(season_set), reverse=True)[0]

	if not selected_season:
		st.error("Could not determine a season to analyze for trade suggestions.")
		return

	rosters_by_team = _build_fantasy_team_view(league_id, cache_files, selected_season)
	if not rosters_by_team:
		st.error("Could not load team rosters. Make sure player data is properly formatted.")
		return

	st.success(f"✅ Loaded {len(rosters_by_team)} teams with player data for {selected_season}")

	st.markdown("---")
	st.markdown("## ⚙️ Configuration")

	col1, col2, col3, col4 = st.columns(4)

	with col1:
		your_team_name = st.selectbox(
			"Select Your Team",
			options=sorted(rosters_by_team.keys()),
			key="tab_your_team_select"
		)

	with col2:
		if "tab_trade_patterns" not in st.session_state:
			st.session_state["tab_trade_patterns"] = ['1-for-1', '2-for-1', '1-for-2', '2-for-2']
		trade_patterns = st.multiselect(
			"Trade Patterns",
			options=[
				'1-for-1',
				'2-for-1', '1-for-2',
				'2-for-2',
				'3-for-1', '1-for-3',
				'3-for-2', '2-for-3',
				'3-for-3',
				'4-for-1', '1-for-4',
				'4-for-2', '2-for-4',
				'4-for-3', '3-for-4',
				'4-for-4',
			],
			help="Select which trade patterns to consider",
			key="tab_trade_patterns",
		)

	with col3:
		min_value_gain = st.slider(
			"Min Value Gain",
			min_value=0.0,
			max_value=50.0,
			value=10.0,
			step=5.0,
			help="Minimum value improvement to suggest a trade",
			key="tab_min_value_gain",
		)

	with col4:
		trade_balance_level = st.slider(
			"Trade Balance (1=super strict, 10=super loose)",
			min_value=1,
			max_value=50,
			value=5,
			help="Controls how strict equal-count realism filters are. 5 = standard.",
			key="tab_trade_balance_level",
		)

	with st.expander("🔧 Advanced Filters", expanded=False):
		col1, col2 = st.columns(2)

		with col1:
			available_teams = [t for t in sorted(rosters_by_team.keys()) if t != your_team_name]
			target_teams = st.multiselect(
				"Target Specific Teams (optional)",
				options=available_teams,
				help="Leave empty to consider all teams",
				key="tab_target_teams",
			)

			exclude_teams = st.multiselect(
				"Exclude Teams (optional)",
				options=available_teams,
				help="Teams you do NOT want to trade with",
				key="tab_exclude_teams",
			)

			max_suggestions = st.number_input(
				"Max Suggestions",
				min_value=5,
				max_value=100,
				value=20,
				step=5,
				key="tab_max_suggestions",
			)

		with col2:
			your_team_df = rosters_by_team[your_team_name]
			all_nba_teams = set()
			try:
				for df in rosters_by_team.values():
					if df is None or df.empty:
						continue
					if 'NBA Team' in df.columns:
						all_nba_teams.update([t for t in df['NBA Team'].dropna().tolist() if str(t).strip()])
			except Exception:
				all_nba_teams = set()
			exclude_nba_teams = st.multiselect(
				"Exclude NBA Teams",
				options=sorted(list(all_nba_teams)),
				help="Exclude all players from these NBA teams (on both sides of a trade)",
				key="tab_exclude_nba_teams",
			)
			exclude_players = st.multiselect(
				"Exclude Your Players (untouchables)",
				options=sorted(your_team_df['Player'].tolist()),
				help="Players you don't want to trade away",
				key="tab_exclude_players",
			)

			include_players = st.multiselect(
				"Must-Include From Your Team (trade bait)",
				options=sorted(your_team_df['Player'].tolist()),
				help="Only show trades where at least one of these players is included on your side",
				key="tab_include_players",
			)
			require_all_include_players = st.checkbox(
				"Require ALL selected players in trade",
				value=False,
				help="If checked, ALL players selected above must be in the trade package (not just one)",
				key="tab_require_all_include",
			)

			other_players = sorted(
				p
				for team, df in rosters_by_team.items()
				if team != your_team_name
				for p in df['Player'].tolist()
			)
			target_opposing_players = st.multiselect(
				"Target Opposing Players",
				options=other_players,
				help="Only show trades where at least one of these players is included on the other side",
				key="tab_target_opposing_players",
			)
			exclude_opposing_players = st.multiselect(
				"Exclude Opposing Players",
				options=other_players,
				help="Opposing players you do NOT want to receive in trades",
				key="tab_exclude_opposing_players",
			)
			min_incoming_fp_g = st.number_input(
				"Min FP/G for incoming players",
				min_value=0.0,
				max_value=150.0,
				value=float(MIN_TRADE_FP_G),
				step=1.0,
				key="tab_min_incoming_fp_g",
			)

	if st.button("🔍 Find Trade Suggestions (Tab)", type="primary", key="tab_find_trade_suggestions"):
		with st.spinner("Analyzing trade opportunities..."):
			set_trade_balance_preset(trade_balance_level)

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
				exclude_nba_teams=exclude_nba_teams if exclude_nba_teams else None,
				require_all_include_players=require_all_include_players,
				min_incoming_fp_g=min_incoming_fp_g,
			)

			if not suggestions:
				st.warning("No beneficial trades found with current filters. Try adjusting your criteria.")
				return

			st.success(f"✅ Found {len(suggestions)} trade suggestions!")

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

			st.markdown("---")
			st.markdown("### 📊 Trade Suggestions (Top 5)")

			for i, suggestion in enumerate(suggestions[:5], 1):
				st.markdown(f"**#{i} - {suggestion['pattern']} with {suggestion['team']} (Value Gain: +{suggestion['value_gain']:.1f})**")
				give = ", ".join(suggestion['you_give'])
				get = ", ".join(suggestion['you_get'])
				st.caption(f"You give: {give} | You get: {get}")
				st.progress(min(1.0, max(0.0, suggestion['value_gain'] / 50.0)))

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

		linear_values = [f * 5 for f in fpts_range]
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

		plotly_chart(fig, width="stretch", key="tab_exponential_curve_chart")
