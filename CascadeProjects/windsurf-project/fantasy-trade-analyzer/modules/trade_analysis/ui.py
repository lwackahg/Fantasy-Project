"""
UI components for the trade analysis feature.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from typing import Dict, Any, List, Tuple
from streamlit_compat import plotly_chart
from streamlit_compat import dataframe
from modules.trade_analysis.logic import TradeAnalyzer, get_team_name, run_trade_analysis, get_all_teams
from data_loader import load_schedule_data
from logic.schedule_analysis import get_team_weekly_points_summary


def _friend_dollar_value(fpg):
	if fpg is None:
		return 0.0
	try:
		v = float(fpg)
	except Exception:
		return 0.0
	if v >= 125:
		return 22.0
	elif v >= 120:
		return 20.0
	elif v >= 115:
		return 18.0
	elif v >= 110:
		return 16.0
	elif v >= 105:
		return 15.0
	elif v >= 100:
		return 14.0
	elif v >= 95:
		return 12.0
	elif v >= 90:
		return 10.0
	elif v >= 85:
		return 9.0
	elif v >= 80:
		return 8.0
	elif v >= 75:
		return 6.0
	elif v >= 70:
		return 4.0
	elif v >= 65:
		return 3.0
	elif v >= 60:
		return 2.0
	elif v >= 55:
		return 1.0
	elif v >= 50:
		return 0.5
	elif v >= 40:
		return 0.0
	return 0.0

def display_trade_analysis_page():
    """Display the trade analysis page."""
    # Add custom CSS for trade analysis
    st.markdown("""
        <style>
        .highlight-trade {
            background-color: rgba(255, 215, 0, 0.2);
            padding: 0.5rem;
            border-radius: 0.3rem;
            border: 1px solid rgba(255, 215, 0, 0.3);
        }
        .metric-card {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #4a4a4a;
            margin: 0.5rem 0;
        }
        .positive-change { color: #00ff00; }
        .negative-change { color: #ff0000; }
        .neutral-change { color: #808080; }
        </style>
    """, unsafe_allow_html=True)
    
    st.write("## Trade Analysis")
    
    # Update trade analyzer with all data ranges if the data is stale
    if st.session_state.get('trade_analyzer') and st.session_state.get('trade_analyzer_data_is_stale', False):
        with st.spinner("Updating trade analyzer with new data..."):
            for range_key, data in st.session_state.data_ranges.items():
                st.session_state.trade_analyzer.update_data(data)
            st.session_state.trade_analyzer_data_is_stale = False
    
    # Setup trade interface
    trade_setup()
    
    # Display trade history in a collapsible section
    with st.expander("Trade Analysis History", expanded=False):
        if st.session_state.trade_analyzer:
            history = st.session_state.trade_analyzer.get_trade_history()
            if not history:
                st.info("No trades recorded yet.")
            else:
                selected_entry = None

                for idx, entry in enumerate(reversed(history)):
                    label = entry.get("label") or "Unlabeled trade"
                    date = entry.get("date") or ""
                    source = entry.get("source") or ""
                    badge = " (historical snapshot)" if source == "historical" else ""
                    header = f"{date} — {label}{badge}" if date else f"{label}{badge}"
                    st.markdown(f"**{header}**")

                    summary = entry.get("summary")
                    if summary:
                        st.text(summary)

                    # Narrow column just holds the button
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        entry_id = entry.get("id")
                        if not entry_id:
                            entry_id = f"{date}_{label}_{idx}".replace(" ", "_")
                        if st.button("View details", key=f"history_view_{entry_id}"):
                            selected_entry = entry
                    st.write("---")

                # Render the replay OUTSIDE the narrow column → full width
                if selected_entry is not None:
                    _replay_trade_from_history(selected_entry)

def _friend_dollar_value(fpg):
	if fpg is None:
		return 0.0
	try:
		v = float(fpg)
	except Exception:
		return 0.0
	if v >= 125:
		return 22.0
	elif v >= 120:
		return 20.0
	elif v >= 115:
		return 18.0
	elif v >= 110:
		return 16.0
	elif v >= 105:
		return 15.0
	elif v >= 100:
		return 14.0
	elif v >= 95:
		return 12.0
	elif v >= 90:
		return 10.0
	elif v >= 85:
		return 9.0
	elif v >= 80:
		return 8.0
	elif v >= 75:
		return 6.0
	elif v >= 70:
		return 4.0
	elif v >= 65:
		return 3.0
	elif v >= 60:
		return 2.0
	elif v >= 55:
		return 1.0
	elif v >= 50:
		return 0.5
	elif v >= 40:
		return 0.0
	return 0.0

def _display_player_selection_interface(selected_teams: List[str], key_suffix: str = "") -> Tuple[Dict[str, Dict[str, str]], Dict[str, float]]:
	"""Displays the UI for selecting players and their destinations."""
	st.write("### Select Players for Each Team")
	trade_teams: Dict[str, Dict[str, str]] = {}
	assumed_fpg_overrides: Dict[str, float] = {}

	if st.session_state.combined_data is None:
		st.error("No data available for analysis")
		return {}, assumed_fpg_overrides

	num_cols = min(len(selected_teams), 3)
	cols = st.columns(num_cols)

	for i, team in enumerate(selected_teams):
		with cols[i % num_cols]:
			st.write(f"#### {get_team_name(team)}")
			team_data = st.session_state.combined_data.reset_index()
			team_players = team_data[team_data['Status'] == team]['Player'].unique().tolist()

			selected_players = st.multiselect(
				f"Select players from {get_team_name(team)}",
				team_players,
				key=f"players_{team}{key_suffix}"
			)

			if selected_players:
				trade_teams[team] = {}
				for player in selected_players:
					other_teams = [t for t in selected_teams if t != team]
					dest = st.selectbox(
						f"Select destination team for {player}",
						other_teams,
						format_func=get_team_name,
						key=f"dest_{team}_{player}{key_suffix}"
					)
					trade_teams[team][player] = dest

					override_str = st.text_input(
						f"Assumed FP/G for {player} (optional)",
						key=f"assumed_fpg_{team}_{player}{key_suffix}",
					)
					if override_str.strip():
						try:
							assumed_val = float(override_str)
						except ValueError:
							assumed_val = None
						if assumed_val is not None:
							assumed_fpg_overrides[player] = assumed_val
	return trade_teams, assumed_fpg_overrides

def _replay_trade_from_history(entry: Dict[str, Any]) -> None:
    """Recompute and display a read-only view of a cached trade history entry.

    For historical entries with full context, this rebuilds the original
    game-log snapshot. For other entries, it recomputes the analysis using
    the current combined_data.
    """
    from modules.historical_trade_analyzer.logic import build_historical_combined_data

    trade_teams = entry.get("trade_teams") or {}
    if not trade_teams:
        st.info("This history entry does not have enough information to replay.")
        return

    num_players = int(entry.get("num_players") or 10)
    source = entry.get("source") or ""
    label = entry.get("label") or "Unlabeled trade"

    st.markdown("### Trade Replay (Read-Only)")
    st.caption(f"Replaying stored trade: {label}")

    # Historical snapshot replay: rebuild from cached game logs
    if (
        source == "historical"
        and all(k in entry for k in ("season", "date", "rosters_by_team", "league_id"))
    ):
        season = entry["season"]
        trade_date_str = entry["date"]
        league_id = entry["league_id"]
        rosters_by_team = entry["rosters_by_team"]

        try:
            trade_dt = pd.to_datetime(trade_date_str, errors="coerce").date()
        except Exception:
            trade_dt = None

        if trade_dt is None:
            st.info("Could not parse trade date for this historical entry.")
            return

        with st.spinner("Rebuilding historical snapshot from game logs..."):
            snapshot_df = build_historical_combined_data(
                trade_dt,
                league_id,
                season,
                rosters_by_team,
            )
            if snapshot_df is None or snapshot_df.empty:
                st.error("Could not rebuild historical snapshot for this entry.")
                return

            analyzer = TradeAnalyzer(snapshot_df)
            results = analyzer.evaluate_trade_fairness(trade_teams, num_players)
            if not results:
                st.error("No results produced when replaying this historical trade.")
                return

            # Stamp context so logs/consistency use the right season & league
            for _, team_results in results.items():
                if isinstance(team_results, dict):
                    team_results["season"] = season
                    team_results["trade_date"] = trade_date_str
                    team_results["league_id"] = league_id

            display_trade_results(results)
        return

    # Non-historical or legacy entries: recompute using current combined_data
    if (
        not st.session_state.get("trade_analyzer")
        or st.session_state.get("combined_data") is None
    ):
        st.info("Load a league dataset before replaying non-historical trades.")
        return

    with st.spinner("Recomputing trade analysis with current data..."):
        st.session_state.trade_analyzer.update_data(st.session_state.combined_data)
        results = st.session_state.trade_analyzer.evaluate_trade_fairness(
            trade_teams,
            num_players,
        )
        if not results:
            st.error("No results produced when replaying this trade with current data.")
            return

        display_trade_results(results)


def trade_setup():
    """Setup the trade interface."""
    st.write("## Analysis Settings")

    num_players = st.number_input(
        "Number of Top Players to Analyze",
        min_value=1,
        max_value=12,
        value=10,
        help="Select the number of top players to analyze"
    )

    trade_label = st.text_input(
        "Trade label / description (optional)",
        value="",
        help="Give this trade a name so you can recognize it later in the history."
    )

    include_advanced_metrics = st.checkbox(
        "Include advanced metrics (consistency, Total ValueScore)",
        value=True,
        help="Turn this off for faster runs if you don't need consistency and value-score details."
    )

    st.write("### Select Teams to Trade Between (2 or more)")
    teams = get_all_teams()
    if not teams:
        st.warning("No teams available for trade analysis")
        return

    selected_teams = st.multiselect(
        "Choose teams involved in the trade",
        options=teams,
        format_func=get_team_name,
        help="Select two or more teams to trade between"
    )

    if len(selected_teams) < 2:
        st.warning("Please select at least 2 teams")
        return

    enable_comparison = st.checkbox(
        "Compare two trade scenarios", 
        value=False, 
        help="Define two different trade packages (Scenario A vs Scenario B) to see a side-by-side comparison of their impact."
    )

    trade_teams_1, overrides_1 = {}, {}
    trade_teams_2, overrides_2 = {}, {}

    if enable_comparison:
        tab1, tab2 = st.tabs(["Scenario A (Primary)", "Scenario B (Alternative)"])
        with tab1:
            trade_teams_1, overrides_1 = _display_player_selection_interface(selected_teams, key_suffix="_scen_a")
        with tab2:
            st.info("Define the alternative trade package below.")
            trade_teams_2, overrides_2 = _display_player_selection_interface(selected_teams, key_suffix="_scen_b")
    else:
        trade_teams_1, overrides_1 = _display_player_selection_interface(selected_teams, key_suffix="")

    last_duration = st.session_state.get("trade_analysis_last_duration_sec")
    if last_duration:
        est_seconds = int(round(last_duration)) or 1
        if enable_comparison:
            est_seconds *= 2
        st.caption(f"Last analysis took about {est_seconds} seconds. Future runs should be similar.")

    if st.button("Analyze Trade"):
        if not trade_teams_1:
            st.warning("Please select players for the primary scenario.")
            return
        
        if enable_comparison and not trade_teams_2:
            st.warning("Please select players for Scenario B or disable comparison mode.")
            return

        spinner_msg = "Running trade analysis..."
        if last_duration:
            est_seconds = int(round(last_duration)) or 1
            if enable_comparison: 
                est_seconds *= 2
            spinner_msg = f"Running trade analysis (expected ~{est_seconds} seconds)..."

        with st.spinner(spinner_msg):
            results1 = run_trade_analysis(
                trade_teams_1,
                num_players,
                trade_label=f"{trade_label} (Scenario A)" if trade_label else "Scenario A",
                include_advanced_metrics=include_advanced_metrics,
                assumed_fpg_overrides=overrides_1,
            )
            
            if enable_comparison:
                results2 = run_trade_analysis(
                    trade_teams_2,
                    num_players,
                    trade_label=f"{trade_label} (Scenario B)" if trade_label else "Scenario B",
                    include_advanced_metrics=include_advanced_metrics,
                    assumed_fpg_overrides=overrides_2,
                )
                if results1 and results2:
                    display_trade_comparison(results1, results2)
            elif results1:
                display_trade_results(results1)

def display_trade_comparison(results_a: Dict[str, Dict[str, Any]], results_b: Dict[str, Dict[str, Any]]):
    """Display a side-by-side comparison of two trade scenarios."""
    st.markdown("## ⚖️ Scenario Comparison")
    
    # Get list of teams present in both
    teams = list(set(results_a.keys()) & set(results_b.keys()))
    
    for team in teams:
        with st.container(border=True):
            st.subheader(f"{get_team_name(team)}")
            
            res_a = results_a[team]
            res_b = results_b[team]
            
            # Check time range availability (default to YTD or first available)
            ranges_a = list(res_a.get('pre_trade_metrics', {}).keys())
            if not ranges_a:
                continue
            time_range = 'YTD' if 'YTD' in ranges_a else ranges_a[0]
            
            pre_metrics = res_a.get('pre_trade_metrics', {}).get(time_range, {})
            metrics_a = res_a.get('post_trade_metrics', {}).get(time_range, {})
            metrics_b = res_b.get('post_trade_metrics', {}).get(time_range, {})
            
            if not metrics_a or not metrics_b or not pre_metrics:
                st.caption(f"Insufficient data for {team} comparison.")
                continue
                
            col1, col2, col3 = st.columns(3)
            
            base_fpg = pre_metrics.get('mean_fpg', 0.0)
            fpg_a = metrics_a.get('mean_fpg', 0.0)
            fpg_b = metrics_b.get('mean_fpg', 0.0)
            
            diff_a = fpg_a - base_fpg
            diff_b = fpg_b - base_fpg
            
            incoming_a = res_a.get('incoming_players', [])
            incoming_b = res_b.get('incoming_players', [])
            
            # Helper for truncated lists
            def _fmt_list(lst):
                if not lst: return "None"
                if len(lst) > 3: return f"{', '.join(lst[:3])}..."
                return ", ".join(lst)

            with col1:
                st.markdown("**Scenario A**")
                st.caption(f"In: {_fmt_list(incoming_a)}")
                st.metric("FP/G Change", f"{diff_a:+.1f}", delta_color="normal")
                
            with col2:
                st.markdown("**Scenario B**")
                st.caption(f"In: {_fmt_list(incoming_b)}")
                st.metric("FP/G Change", f"{diff_b:+.1f}", delta_color="normal")
                
            with col3:
                st.markdown("**Net Advantage (B vs A)**")
                net_diff = diff_b - diff_a
                if net_diff > 0:
                    st.success(f"Scenario B is **+{net_diff:.1f} FP/G** better")
                elif net_diff < 0:
                    st.error(f"Scenario A is **+{abs(net_diff):.1f} FP/G** better")
                else:
                    st.info("Both scenarios have equal impact")
    
    st.markdown("---")
    st.markdown("### Detailed Views")
    tab_a, tab_b = st.tabs(["Scenario A Details", "Scenario B Details"])
    with tab_a:
        display_trade_results(results_a, key_suffix="_scen_a")
    with tab_b:
        display_trade_results(results_b, key_suffix="_scen_b")

def display_trade_results(analysis_results: Dict[str, Dict[str, Any]], key_suffix: str = ""):
    """Display the trade analysis results."""
    time_ranges = list(next(iter(analysis_results.values()))['pre_trade_metrics'].keys())

    teams = sorted(list(analysis_results.keys()), key=get_team_name)
    team_tabs = st.tabs([f"Team: {get_team_name(team)}" for team in teams])

    for team_tab, team in zip(team_tabs, teams):
        results = analysis_results[team]
        with team_tab:
            _display_trade_overview(results, key_suffix)
            _display_trade_impact_section(team, results, time_ranges)

def _display_trade_impact_section(team_id: str, results: Dict[str, Any], time_ranges: List[str]):
    """Display the main trade impact analysis section for a single team tab."""
    
    # Calculate common metrics needed for insights
    ytd_pre = results.get('pre_trade_metrics', {}).get('YTD', {})
    ytd_post = results.get('post_trade_metrics', {}).get('YTD', {})
    ytd_pre_consistency = results.get('pre_trade_consistency', {}).get('YTD', {})
    ytd_post_consistency = results.get('post_trade_consistency', {}).get('YTD', {})
    
    has_ytd = bool(ytd_pre and ytd_post)

    # Create Tabs
    tabs = st.tabs([
        "💡 Insights",
        "📊 Deep Dive",
        "🏗️ Roster Impact",
        "🔬 Advanced Analysis", 
        "👥 Player Comparison"
    ])
    
    # Tab 1: Insights (Overall Assessment)
    with tabs[0]:
        if has_ytd:
             _render_overall_assessment(results, ytd_pre, ytd_post, ytd_pre_consistency, ytd_post_consistency)
        else:
            st.info("Insufficient data for insights.")

    # Tab 2: Deep Dive (Metrics Table, Trends, Charts)
    with tabs[1]:
        _display_trade_metrics_table(results, time_ranges)
        st.markdown("---")
        if has_ytd:
             _render_trend_analysis(results, time_ranges)
        st.markdown("---")
        _display_performance_visualizations(results, time_ranges)

    # Tab 3: Roster Impact
    with tabs[2]:
        _display_roster_details(results, time_ranges)

    # Tab 4: Advanced Analysis (Stats, Monte Carlo)
    with tabs[3]:
        if has_ytd:
            _display_statistical_analysis(results, time_ranges, ytd_pre, ytd_post)
            st.markdown("---")
            _display_monte_carlo_simulation(results, team_id=team_id)
        else:
             st.info("Insufficient data for advanced analysis.")

    # Tab 5: Player Comparison
    with tabs[4]:
        outgoing = results.get('outgoing_players', [])
        incoming = results.get('incoming_players', [])
        if outgoing and incoming:
            _display_player_comparison(outgoing, incoming, results)
        else:
            st.info("No player comparison data available")

def _display_trade_overview(results: Dict[str, Any], key_suffix: str = ""):
    st.title("Trade Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='metric-card'><h3>Players Receiving</h3></div>""", unsafe_allow_html=True)
        incoming = results.get('incoming_players', [])
        st.markdown(", ".join([f"<span class='highlight-trade'>{p}</span>" for p in incoming]) or "None", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'><h3>Players Trading Away</h3></div>""", unsafe_allow_html=True)
        outgoing = results.get('outgoing_players', [])
        st.markdown(", ".join([f"<span class='highlight-trade'>{p}</span>" for p in outgoing]) or "None", unsafe_allow_html=True)
    st.write("---")
    
    # Add player game logs viewer
    _display_traded_players_game_logs(results, key_suffix)

def _display_trade_metrics_table(results: Dict[str, Any], time_ranges: List[str]):
    st.markdown(
        """ℹ️ **Metrics Guide**: - **FP/G**: Fantasy Points per Game - **GP**: Games Played - **Std Dev**: Standard Deviation (consistency measure) - **CV%**: Coefficient of Variation (lower = more consistent) - **Total ValueScore**: Sum of each roster player's composite value score (production, consistency, availability) — higher is better.""",
        unsafe_allow_html=True,
    )
    combined_data = []
    for time_range in time_ranges:
        pre_metrics = results.get('pre_trade_metrics', {}).get(time_range, {})
        post_metrics = results.get('post_trade_metrics', {}).get(time_range, {})
        pre_consistency = results.get('pre_trade_consistency', {}).get(time_range, {})
        post_consistency = results.get('post_trade_consistency', {}).get(time_range, {})
        pre_value_scores = results.get('pre_trade_value_scores', {}).get(time_range, {})
        post_value_scores = results.get('post_trade_value_scores', {}).get(time_range, {})
        
        if pre_metrics and post_metrics:
            row = {
                'Time Range': time_range,
                'Mean FP/G': f"{pre_metrics['mean_fpg']:.1f} → <span style='color:{'green' if post_metrics['mean_fpg'] > pre_metrics['mean_fpg'] else 'red'}'>{post_metrics['mean_fpg']:.1f}</span>",
                'Median FP/G': f"{pre_metrics['median_fpg']:.1f} → <span style='color:{'green' if post_metrics['median_fpg'] > pre_metrics['median_fpg'] else 'red'}'>{post_metrics['median_fpg']:.1f}</span>",
                'Std Dev': f"{pre_metrics['std_dev']:.1f} → <span style='color:{'green' if post_metrics['std_dev'] < pre_metrics['std_dev'] else 'red'}'>{post_metrics['std_dev']:.1f}</span>",
                'Total FPs': f"{pre_metrics['total_fpts']:.0f} → <span style='color:{'green' if post_metrics['total_fpts'] > pre_metrics['total_fpts'] else 'red'}'>{post_metrics['total_fpts']:.0f}</span>",
                'Avg GP': f"{pre_metrics['avg_gp']:.1f} → <span style='color:{'green' if post_metrics['avg_gp'] >= pre_metrics['avg_gp'] else 'red'}'>{post_metrics['avg_gp']:.1f}</span>"
            }
            
            # Add consistency metrics if available
            if pre_consistency and post_consistency:
                pre_cv = pre_consistency.get('avg_cv', 0)
                post_cv = post_consistency.get('avg_cv', 0)
                # Lower CV is better (more consistent)
                row['Avg CV%'] = f"{pre_cv:.1f}% → <span style='color:{'green' if post_cv < pre_cv else 'red'}'>{post_cv:.1f}%</span>"
            
            # Add value score aggregates if available
            if pre_value_scores and post_value_scores:
                pre_total_vs = pre_value_scores.get('total_value_score')
                post_total_vs = post_value_scores.get('total_value_score')
                if pre_total_vs is not None and post_total_vs is not None:
                    row['Total ValueScore'] = (
                        f"{pre_total_vs:.1f} → "
                        f"<span style='color:{'green' if post_total_vs >= pre_total_vs else 'red'}'>{post_total_vs:.1f}</span>"
                    )
            
            combined_data.append(row)
    
    combined_df = pd.DataFrame(combined_data)
    st.markdown("### Trade Metrics (Before → After)")
    st.markdown(combined_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Add consistency summary if available
    _display_consistency_summary(results, time_ranges)

def _create_performance_chart(metric_data: pd.DataFrame, display_name: str):
    """Creates a line chart to visualize performance metrics."""
    fig = px.line(
        metric_data,
        x='Time Range',
        y=display_name,
        color='Type',
        markers=True,
        labels={display_name: f"{display_name} Value"},
        title=f"{display_name} Trend (Before vs. After Trade)"
    )
    return fig

def _display_performance_visualizations(results: Dict[str, Any], time_ranges: List[str]):
    """Generates and displays performance charts for key metrics."""
    st.subheader("Performance Visualization")
    metrics_to_plot = [('FP/G', 'mean_fpg'), ('Median FP/G', 'median_fpg'), ('Std Dev', 'std_dev')]

    for display_name, metric_key in metrics_to_plot:
        pre_metrics = [results.get('pre_trade_metrics', {}).get(tr, {}).get(metric_key) for tr in time_ranges]
        post_metrics = [results.get('post_trade_metrics', {}).get(tr, {}).get(metric_key) for tr in time_ranges]

        if any(v is None for v in pre_metrics) or any(v is None for v in post_metrics):
            continue

        metric_data = pd.DataFrame({
            'Time Range': time_ranges * 2,
            display_name: pre_metrics + post_metrics,
            'Type': ['Before'] * len(time_ranges) + ['After'] * len(time_ranges)
        })

        fig = _create_performance_chart(metric_data, display_name)
        plotly_chart(fig, width="stretch")
    

def _display_styled_roster(title: str, roster_data: List[Dict[str, Any]], players_to_highlight: List[str], highlight_color: str):
    """Displays a styled roster dataframe, highlighting specific players."""
    st.write(f"#### {title}")
    if roster_data:
        roster_df = pd.DataFrame(roster_data)

        def highlight_players(row):
            return [f'background-color: {highlight_color}' if row['Player'] in players_to_highlight else '' for _ in row]

        styled_roster = roster_df.style.apply(highlight_players, axis=1)
        dataframe(styled_roster, hide_index=True, width="stretch")
    else:
        st.write("No data available.")

def _display_traded_players_game_logs(results: Dict[str, Any], key_suffix: str = ""):
    """Display game logs for all players involved in the trade."""
    from modules.trade_analysis.consistency_integration import load_player_consistency, get_player_game_log_df
    import json
    
    league_id = results.get('league_id', '')
    if not league_id:
        return
    
    all_players = results.get('outgoing_players', []) + results.get('incoming_players', [])
    if not all_players:
        return
    
    with st.expander("📋 View Traded Players' Game Logs", expanded=False):
        selected_player = st.selectbox(
            "Player",
            options=all_players,
            key=f"trade_game_logs_player_select{key_suffix}",
        )

        player_name = selected_player

        # Find and load player's game log using centralized loader
        preferred_season = results.get("season") if isinstance(results, dict) else None
        game_log_df = get_player_game_log_df(player_name, league_id, preferred_season)

        if game_log_df is not None and not game_log_df.empty:
            # For historical trades, filter logs to games on or before the trade date
            trade_date_str = results.get("trade_date") if isinstance(results, dict) else None
            if trade_date_str and preferred_season and "Date" in game_log_df.columns:
                try:
                    from modules.historical_trade_analyzer.logic import _parse_game_dates_for_season
                    trade_dt = pd.to_datetime(trade_date_str, errors="coerce")
                    if not pd.isna(trade_dt):
                        parsed_dates = _parse_game_dates_for_season(game_log_df["Date"], preferred_season)
                        game_log_df = game_log_df.copy()
                        game_log_df["DateParsed"] = parsed_dates
                        game_log_df = game_log_df[game_log_df["DateParsed"] <= trade_dt]
                        game_log_df = game_log_df.drop(columns=["DateParsed"])
                except Exception:
                    # If anything goes wrong, fall back to full log display
                    pass

        if game_log_df is not None and not game_log_df.empty:
            # Display consistency metrics
            consistency = load_player_consistency(player_name, league_id)
            if consistency:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Games", consistency['games_played'])
                    st.metric("Mean FPts", f"{consistency['mean_fpts']:.1f}")
                with col2:
                    st.metric("CV%", f"{consistency['cv_percent']:.1f}%", help="Lower = more consistent")
                    st.metric("Consistency", consistency['consistency_tier'])
                with col3:
                    st.metric("Boom Games", consistency['boom_games'], help=f"{consistency['boom_rate']:.1f}% of games")
                    st.metric("Bust Games", consistency['bust_games'], help=f"{consistency['bust_rate']:.1f}% of games")
                with col4:
                    st.metric("Min FPts", f"{consistency['min_fpts']:.0f}")
                    st.metric("Max FPts", f"{consistency['max_fpts']:.0f}")

            st.markdown("---")

            # Display game log table
            priority_cols = ['Date', 'Team', 'Opp', 'Score', 'FPts', 'MIN', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
            other_cols = [col for col in game_log_df.columns if col not in priority_cols]
            display_cols = [col for col in priority_cols if col in game_log_df.columns] + other_cols

            dataframe(
                game_log_df[display_cols],
                width="stretch",
                height=400
            )

            # Download button
            csv = game_log_df.to_csv(index=False)
            unique_key = f"trade_download_{player_name.replace(' ', '_')}{key_suffix}"
            st.download_button(
                label=f"📥 Download {player_name} Game Log",
                data=csv,
                file_name=f"{player_name.replace(' ', '_')}_game_log.csv",
                mime="text/csv",
                key=unique_key
            )
        else:
            st.info(f"No game log data available for {player_name}. Run Bulk Scrape in Admin Tools to populate cache.")

def _render_overall_assessment(results, ytd_pre, ytd_post, ytd_pre_consistency, ytd_post_consistency):
    """Render overall trade assessment and key metrics."""
    # Calculate changes
    fpg_change = ytd_post['mean_fpg'] - ytd_pre['mean_fpg']
    total_change = ytd_post['total_fpts'] - ytd_pre['total_fpts']
    std_change = ytd_post['std_dev'] - ytd_pre['std_dev']
    median_change = ytd_post['median_fpg'] - ytd_pre['median_fpg']
    
    # Consistency changes
    cv_change = 0
    if ytd_pre_consistency and ytd_post_consistency:
        cv_change = ytd_post_consistency.get('avg_cv', 0) - ytd_pre_consistency.get('avg_cv', 0)
    
    # Advanced metrics
    percent_change = (fpg_change / ytd_pre['mean_fpg'] * 100) if ytd_pre['mean_fpg'] > 0 else 0
    sharpe_ratio_pre = (ytd_pre['mean_fpg'] / ytd_pre['std_dev']) if ytd_pre['std_dev'] > 0 else 0
    sharpe_ratio_post = (ytd_post['mean_fpg'] / ytd_post['std_dev']) if ytd_post['std_dev'] > 0 else 0
    sharpe_change = sharpe_ratio_post - sharpe_ratio_pre
    
    st.markdown("### 📊 Overall Assessment & Key Metrics")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Determine trade verdict
        production_verdict = "✅ Gain" if fpg_change > 0 else "❌ Loss" if fpg_change < 0 else "➖ Neutral"
        consistency_verdict = "✅ More Consistent" if cv_change < 0 else "❌ Less Consistent" if cv_change > 0 else "➖ No Change"
        
        assessment_text = f"""
        **Production Impact:** {production_verdict} ({fpg_change:+.1f} FP/G, {percent_change:+.1f}%)
        
        **Consistency Impact:** {consistency_verdict} ({cv_change:+.1f}% CV)
        
        **Total Points Impact:** {total_change:+.0f} FPts over the season
        
        **Risk-Adjusted Return (Sharpe):** {sharpe_change:+.2f}
        """
        
        st.markdown(assessment_text)
        
        _render_friend_value_lens(results)
        
        # Trade recommendation
        small_production = abs(fpg_change) < 1.0 and abs(percent_change) < 5.0
        small_consistency = abs(cv_change) < 3.0

        if small_production and small_consistency:
            st.info("🟡 **Marginal Trade** - Minimal impact either way")
        elif fpg_change > 3 and cv_change < 5:
            st.success("🟢 **Strong Trade** - Significant production gain with acceptable consistency impact")
        elif fpg_change > 1.5 and cv_change < 0:
            st.success("🟢 **Excellent Trade** - Production gain AND improved consistency")
        elif fpg_change > 0 and cv_change < 10:
            st.info("🟡 **Decent Trade** - Production gain but slight consistency loss")
        elif fpg_change < -2:
            st.error("🔴 **Poor Trade** - Significant production loss")
        elif cv_change > 10:
            st.warning("🟠 **Risky Trade** - Major consistency loss, high volatility")
        else:
            st.info("🟡 **Marginal Trade** - Mixed signals or minimal net impact")
    
    with col2:
        st.markdown("**Production**")
        st.metric("FP/G Change", f"{fpg_change:+.1f}")
        st.metric("Median Change", f"{median_change:+.1f}")
        st.metric("% Change", f"{percent_change:+.1f}%")
    
    with col3:
        st.markdown("**Risk**")
        st.metric(
            "Core FP/G Spread",
            f"{std_change:+.1f}",
            delta_color="inverse",
            help="Std dev of FP/G across your top players; higher = a more top-heavy core.",
        )
        if cv_change != 0:
            st.metric(
                "Avg Player CV%",
                f"{cv_change:+.1f}%",
                delta_color="inverse",
                help="Average game-to-game CV% across your top players; lower = more stable scoring.",
            )
        st.metric("Sharpe", f"{sharpe_change:+.2f}")
        if std_change != 0 and cv_change != 0:
            st.caption(
                "Std Dev measures absolute swing in points; CV% measures volatility relative to your average. "
                "It's possible for Std Dev to increase while CV% improves if your new core scores more but stays relatively stable."
            )

def _render_trend_analysis(results, time_ranges):
    """Render trend analysis across time ranges."""
    st.markdown("### 📉 Trend Analysis Across Time Ranges")
    trend_data = []
    for time_range in time_ranges:
        pre = results.get('pre_trade_metrics', {}).get(time_range, {})
        post = results.get('post_trade_metrics', {}).get(time_range, {})
        if pre and post:
            trend_data.append({
                'Time Range': time_range,
                'FP/G Change': post['mean_fpg'] - pre['mean_fpg'],
                'Total FPts Change': post['total_fpts'] - pre['total_fpts'],
                'Core Spread Change': post['std_dev'] - pre['std_dev']
            })
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        
        # Create multi-metric trend visualization
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig_trend = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Production Change", "Risk Change (Core Spread)"),
            vertical_spacing=0.15
        )
        
        # Production trend
        fig_trend.add_trace(
            go.Scatter(
                x=trend_df['Time Range'],
                y=trend_df['FP/G Change'],
                mode='lines+markers',
                name='FP/G Change',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ),
            row=1, col=1
        )
        
        # Risk trend
        fig_trend.add_trace(
            go.Scatter(
                x=trend_df['Time Range'],
                y=trend_df['Core Spread Change'],
                mode='lines+markers',
                name='Core Spread Change',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(255, 127, 14, 0.2)'
            ),
            row=2, col=1
        )
        
        fig_trend.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig_trend.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        fig_trend.update_xaxes(title_text="Time Range", row=2, col=1)
        fig_trend.update_yaxes(title_text="FP/G Change", row=1, col=1)
        fig_trend.update_yaxes(title_text="Core Spread Change (Std Dev across players)", row=2, col=1)
        
        fig_trend.update_layout(height=500, showlegend=False)
        
        plotly_chart(fig_trend, width="stretch")
        
        # Trend interpretation
        recent_trend = trend_df.iloc[-1]['FP/G Change']
        long_term_trend = trend_df.iloc[0]['FP/G Change']
        
        # Calculate trend consistency (variance of changes)
        trend_variance = np.var(trend_df['FP/G Change'])
        trend_consistency = "High" if trend_variance < 1 else "Moderate" if trend_variance < 4 else "Low"
        
        col1, col2 = st.columns(2)
        with col1:
            if recent_trend > 0 and long_term_trend > 0:
                st.success("✅ **Consistent Positive Impact** - Trade improves production across all time ranges")
            elif recent_trend > 0 and long_term_trend < 0:
                st.warning("⚠️ **Recent Surge** - Players performing better recently, but long-term data suggests caution")
            elif recent_trend < 0 and long_term_trend > 0:
                st.warning("⚠️ **Recent Slump** - Players in recent downturn, but long-term data is positive")
            elif recent_trend < 0 and long_term_trend < 0:
                st.error("❌ **Consistent Negative Impact** - Trade hurts production across all time ranges")
        
        with col2:
            st.metric("Trend Consistency", trend_consistency, help="How consistent is the impact across time ranges")
            st.metric("Recent (7d) Impact", f"{recent_trend:+.1f} FP/G")
            st.metric("Long-term (YTD) Impact", f"{long_term_trend:+.1f} FP/G")


def _render_friend_value_lens(results: Dict[str, Any]) -> None:
    pre_rosters = results.get("pre_trade_rosters", {}) or {}
    post_rosters = results.get("post_trade_rosters", {}) or {}
    ytd_pre = pre_rosters.get("YTD") or []
    ytd_post = post_rosters.get("YTD") or []
    if not ytd_pre and not ytd_post:
        return

    pre_map: Dict[str, float] = {}
    for row in ytd_pre:
        name = row.get("Player")
        if not name:
            continue
        fpg = row.get("FP/G")
        if fpg is None:
            continue
        pre_map[str(name)] = fpg

    post_map: Dict[str, float] = {}
    for row in ytd_post:
        name = row.get("Player")
        if not name:
            continue
        fpg = row.get("FP/G")
        if fpg is None:
            continue
        post_map[str(name)] = fpg

    outgoing = results.get("outgoing_players", []) or []
    incoming = results.get("incoming_players", []) or []

    def _collect(names, primary, secondary):
        items = []
        total = 0.0
        for n in names:
            fp = primary.get(n)
            if fp is None:
                fp = secondary.get(n)
            if fp is None:
                continue
            dollars = _friend_dollar_value(fp)
            items.append((n, dollars))
            total += dollars
        return items, total

    outgoing_items, outgoing_total = _collect(outgoing, pre_map, post_map)
    incoming_items, incoming_total = _collect(incoming, post_map, pre_map)
    if not outgoing_items and not incoming_items:
        return

    star_player = None
    star_side = None
    star_dollar = 0.0

    for name, dollars in outgoing_items:
        if dollars > star_dollar:
            star_dollar = dollars
            star_player = name
            star_side = "outgoing"

    for name, dollars in incoming_items:
        if dollars > star_dollar:
            star_dollar = dollars
            star_player = name
            star_side = "incoming"

    ratio = None
    ratio_text = ""
    if star_player and star_dollar > 0:
        if star_side == "outgoing" and incoming_total > 0:
            ratio = incoming_total / star_dollar
            ratio_text = f"Incoming package comes to **{ratio*100:.0f}%** of `{star_player}`'s $ value on this scale."
        elif star_side == "incoming" and outgoing_total > 0:
            ratio = outgoing_total / star_dollar
            ratio_text = f"Outgoing package comes to **{ratio*100:.0f}%** of `{star_player}`'s $ value on this scale."

    verdict_text = ""
    if ratio is not None:
        veto_floor = None
        floor_label = "star trade"
        if star_dollar >= 20:
            veto_floor = 0.60
            floor_label = "apex-star (Jokic-tier) trade"
        elif star_dollar >= 14:
            veto_floor = 0.50
            floor_label = "top-star trade"
        if veto_floor is not None:
            if ratio < veto_floor:
                verdict_text = (
                    f"On this lens the package is **below** the {int(veto_floor*100)}% floor "
                    f"for a {floor_label} → **veto-leaning**."
                )
            else:
                verdict_text = (
                    f"On this lens the package is **above** the {int(veto_floor*100)}% floor "
                    f"for a {floor_label} → **not auto-veto** by that rule."
                )

    def _clean_name(n):
        return n.strip("'").strip("`")

    lines = []
    lines.append("**Friend dollar-value lens (YTD):**")
    if outgoing_items:
        parts = ", ".join(f"`{_clean_name(name)}` (\${dollars:.1f})" for name, dollars in outgoing_items)
        lines.append(f"Outgoing package: {parts} → **total \${outgoing_total:.1f}**")
    if incoming_items:
        parts = ", ".join(f"`{_clean_name(name)}` (\${dollars:.1f})" for name, dollars in incoming_items)
        lines.append(f"Incoming package: {parts} → **total \${incoming_total:.1f}**")
    if ratio_text:
        lines.append(ratio_text)
    if verdict_text:
        lines.append(verdict_text)

    text = "\n\n".join(lines)
    st.markdown(text)


def _display_statistical_analysis(results, time_ranges, ytd_pre, ytd_post):
    """Display statistical significance testing and advanced metrics."""
    import numpy as np
    from scipy import stats as scipy_stats
    
    st.markdown("#### Statistical Significance & Advanced Metrics")
    
    # Effect size calculation (Cohen's d)
    fpg_change = ytd_post['mean_fpg'] - ytd_pre['mean_fpg']
    pooled_std = np.sqrt((ytd_pre['std_dev']**2 + ytd_post['std_dev']**2) / 2)
    cohens_d = fpg_change / pooled_std if pooled_std > 0 else 0
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interpretation = "Negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "Small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Effect Size (Cohen's d)", f"{cohens_d:.2f}", help="Standardized measure of trade impact")
        st.caption(f"**{effect_interpretation}** effect")
    
    with col2:
        confidence_level = 95
        margin_of_error = 1.96 * (ytd_post['std_dev'] / np.sqrt(ytd_post.get('avg_gp', 10)))
        st.metric("95% Confidence Interval", f"±{margin_of_error:.1f} FP/G", help="Expected range of outcomes")
    
    with col3:
        # Calculate probability of improvement (simplified)
        z_score = fpg_change / (ytd_pre['std_dev'] / np.sqrt(ytd_pre.get('avg_gp', 10))) if ytd_pre['std_dev'] > 0 else 0
        prob_improvement = scipy_stats.norm.cdf(z_score) * 100
        st.metric("Prob. of Improvement", f"{prob_improvement:.1f}%", help="Statistical likelihood of positive impact")
    
    st.markdown("---")
    
    # Distribution comparison
    st.markdown("#### Performance Distribution Comparison")
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig_dist = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Before Trade", "After Trade"),
        specs=[[{"type": "box"}, {"type": "box"}]]
    )
    
    # Simulate distributions based on mean and std dev
    pre_dist = np.random.normal(ytd_pre['mean_fpg'], ytd_pre['std_dev'], 1000)
    post_dist = np.random.normal(ytd_post['mean_fpg'], ytd_post['std_dev'], 1000)
    
    fig_dist.add_trace(
        go.Box(y=pre_dist, name="Before", marker_color='lightblue', boxmean='sd'),
        row=1, col=1
    )
    
    fig_dist.add_trace(
        go.Box(y=post_dist, name="After", marker_color='lightgreen', boxmean='sd'),
        row=1, col=2
    )
    
    fig_dist.update_layout(height=400, showlegend=False)
    fig_dist.update_yaxes(title_text="Fantasy Points per Game")
    
    plotly_chart(fig_dist, width="stretch")
    
    st.caption("**Box plots** show the distribution of expected performance. The box represents the middle 50% of outcomes, with the line showing the median.")


def _display_monte_carlo_simulation(results: Dict[str, Any], team_id: str | None = None):
    """Display Monte Carlo simulation of weekly outcomes before/after trade.

    When schedule data is available, also compares the simulated weekly
    distribution to this team's actual season-to-date weekly scores and offers
    an optional calibration toggle that blends the simulation toward the
    historical distribution as more weeks are completed.
    """
    import plotly.graph_objects as go
    from scipy import stats as scipy_stats
    
    st.markdown("#### 🎲 Weekly Outcome Simulation")
    st.caption("Simulates 1000 weekly outcomes using player CV% to model game-to-game variance")
    
    # Get roster data
    pre_rosters = results.get("pre_trade_rosters", {}) or {}
    post_rosters = results.get("post_trade_rosters", {}) or {}
    ytd_pre = pre_rosters.get("YTD") or []
    ytd_post = post_rosters.get("YTD") or []
    
    if not ytd_pre or not ytd_post:
        st.info("Insufficient roster data for Monte Carlo simulation")
        return
    
    # Build DataFrames
    pre_df = pd.DataFrame(ytd_pre)
    post_df = pd.DataFrame(ytd_post)
    
    # Check required columns
    if 'FP/G' not in pre_df.columns or 'FP/G' not in post_df.columns:
        st.info("Missing FP/G data for simulation")
        return
    
    # Run simulations
    try:
        from modules.trade_suggestions.advanced_stats import simulate_weekly_outcome
        
        # Rename columns to match expected format
        pre_sim_df = pre_df.rename(columns={'FP/G': 'Mean FPts'})
        post_sim_df = post_df.rename(columns={'FP/G': 'Mean FPts'})
        
        # Add CV% if missing (default to 30%)
        if 'CV%' not in pre_sim_df.columns:
            pre_sim_df['CV%'] = 30.0
        if 'CV%' not in post_sim_df.columns:
            post_sim_df['CV%'] = 30.0
        
        with st.spinner("Running Monte Carlo simulation (1000 iterations)..."):
            before_stats = simulate_weekly_outcome(pre_sim_df, n_simulations=1000, games_target=25)
            after_stats = simulate_weekly_outcome(post_sim_df, n_simulations=1000, games_target=25)
        
        # Optional season-to-date diagnostics and calibration
        hist_summary = None
        if team_id is not None:
            try:
                schedule_df = load_schedule_data()
            except Exception:
                schedule_df = None

            if schedule_df is not None and not getattr(schedule_df, "empty", True):
                team_name = get_team_name(team_id)
                hist_summary = get_team_weekly_points_summary(schedule_df, team_name)

        use_calibration = False
        display_before = before_stats
        display_after = after_stats

        if hist_summary and isinstance(hist_summary, dict):
            weeks_played = int(hist_summary.get("weeks_played", 0) or 0)
            has_hist = weeks_played > 0

            if has_hist:
                default_calibrate = True
                use_calibration = st.checkbox(
                    "Use season-to-date calibration",
                    value=default_calibrate,
                    help=(
                        "Blend simulated weekly distribution with this team's actual "
                        "season scores. Calibration weight increases as more weeks "
                        "are completed."
                    ),
                    key=f"mc_calibration_{team_id or 'unknown'}",
                )

                if use_calibration:
                    w = min(1.0, weeks_played / 10.0)
                    hist_mean = float(hist_summary.get("mean_fpts", 0.0) or 0.0)
                    hist_std = float(hist_summary.get("std_fpts", 0.0) or 0.0)

                    # Calibrate the pre-trade distribution toward season-to-date
                    b_mean = float(before_stats.get("mean", 0.0) or 0.0)
                    b_std = float(before_stats.get("std", 0.0) or 0.0)

                    before_mean_cal = (1.0 - w) * b_mean + w * hist_mean
                    if b_std > 0 and hist_std > 0:
                        before_std_cal = (1.0 - w) * b_std + w * hist_std
                    else:
                        before_std_cal = b_std

                    # Preserve the trade's relative effect when calibrating "after"
                    a_mean = float(after_stats.get("mean", 0.0) or 0.0)
                    a_std = float(after_stats.get("std", 0.0) or 0.0)
                    mean_delta = a_mean - b_mean
                    std_delta = a_std - b_std

                    after_mean_cal = before_mean_cal + mean_delta
                    after_std_cal = max(1e-6, before_std_cal + std_delta)

                    def _rebuild_gaussian_stats(mu: float, sigma: float) -> Dict[str, float]:
                        if sigma <= 0:
                            return {
                                "mean": float(mu),
                                "std": float(0.0),
                                "p10": float(mu),
                                "p25": float(mu),
                                "p50": float(mu),
                                "p75": float(mu),
                                "p90": float(mu),
                            }
                        return {
                            "mean": float(mu),
                            "std": float(sigma),
                            "p10": float(scipy_stats.norm.ppf(0.10, loc=mu, scale=sigma)),
                            "p25": float(scipy_stats.norm.ppf(0.25, loc=mu, scale=sigma)),
                            "p50": float(scipy_stats.norm.ppf(0.50, loc=mu, scale=sigma)),
                            "p75": float(scipy_stats.norm.ppf(0.75, loc=mu, scale=sigma)),
                            "p90": float(scipy_stats.norm.ppf(0.90, loc=mu, scale=sigma)),
                        }

                    display_before = _rebuild_gaussian_stats(before_mean_cal, before_std_cal)
                    display_after = _rebuild_gaussian_stats(after_mean_cal, after_std_cal)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mean_change = display_after['mean'] - display_before['mean']
            st.metric(
                "Expected Weekly FP",
                f"{display_after['mean']:.0f}",
                f"{mean_change:+.0f}",
                help="Average weekly fantasy points across 1000 simulations"
            )
        
        with col2:
            floor_change = display_after['p10'] - display_before['p10']
            st.metric(
                "Floor (10th %ile)",
                f"{display_after['p10']:.0f}",
                f"{floor_change:+.0f}",
                help="Worst-case weekly outcome (10th percentile)"
            )
        
        with col3:
            ceiling_change = display_after['p90'] - display_before['p90']
            st.metric(
                "Ceiling (90th %ile)",
                f"{display_after['p90']:.0f}",
                f"{ceiling_change:+.0f}",
                help="Best-case weekly outcome (90th percentile)"
            )
        
        with col4:
            variance_change = display_after['std'] - display_before['std']
            variance_emoji = "📉" if variance_change < 0 else "📈"
            st.metric(
                "Volatility",
                f"{variance_emoji} {display_after['std']:.0f}",
                f"{variance_change:+.0f}",
                delta_color="inverse",
                help="Standard deviation of weekly outcomes (lower = more predictable)"
            )
        
        # Distribution visualization
        st.markdown("---")
        st.markdown("##### Weekly FP Distribution Comparison")
        
        # Create distribution curves
        x_min = min(display_before['p10'], display_after['p10']) - 100
        x_max = max(display_before['p90'], display_after['p90']) + 100
        x_range = np.linspace(x_min, x_max, 200)
        
        before_curve = scipy_stats.norm.pdf(x_range, display_before['mean'], max(display_before['std'], 1e-6))
        after_curve = scipy_stats.norm.pdf(x_range, display_after['mean'], max(display_after['std'], 1e-6))
        
        # Normalize for display
        before_curve = before_curve / before_curve.max()
        after_curve = after_curve / after_curve.max()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_range, y=before_curve,
            mode='lines', fill='tozeroy',
            name='Before Trade',
            line_color='#FF9800',
            fillcolor='rgba(255, 152, 0, 0.3)',
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range, y=after_curve,
            mode='lines', fill='tozeroy',
            name='After Trade',
            line_color='#4CAF50',
            fillcolor='rgba(76, 175, 80, 0.3)',
        ))
        
        # Add vertical lines for means
        fig.add_vline(x=display_before['mean'], line_dash='dash', line_color='#FF9800')
        fig.add_vline(x=display_after['mean'], line_dash='dash', line_color='#4CAF50')
        
        fig.update_layout(
            title='Projected Weekly FP Distribution',
            xaxis_title='Weekly Fantasy Points (25 games)',
            yaxis_title='Relative Probability',
            height=400,
            showlegend=True,
        )
        
        plotly_chart(fig, width="stretch")
        
        # Interpretation
        if mean_change > 50:
            st.success("🟢 **Strong improvement** - Significant increase in expected weekly output")
        elif mean_change > 20:
            st.success("🟢 **Solid improvement** - Meaningful increase in expected weekly output")
        elif mean_change > 0:
            st.info("🟡 **Modest improvement** - Small increase in expected weekly output")
        elif mean_change > -20:
            st.info("🟡 **Slight decrease** - Small reduction in expected weekly output")
        else:
            st.warning("🔴 **Notable decrease** - Significant reduction in expected weekly output")
        
        if floor_change > 30:
            st.success("✅ **Floor raised** - Your worst weeks should be better")
        elif floor_change < -30:
            st.warning("⚠️ **Floor lowered** - Your worst weeks could be worse")
        
        if variance_change < -20:
            st.success("✅ **More predictable** - Less week-to-week variance")
        elif variance_change > 20:
            st.info("ℹ️ **More volatile** - Higher week-to-week variance")
        
    except Exception as e:
        st.warning(f"Monte Carlo simulation unavailable: {str(e)}")
        st.caption("Ensure the advanced_stats module is properly installed")


def _display_player_comparison(outgoing_players, incoming_players, results):
    """Display detailed player-by-player comparison."""
    from modules.trade_analysis.consistency_integration import load_player_consistency
    
    league_id = results.get('league_id', '')
    if not league_id:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📤 Trading Away**")
        outgoing_data = []
        for player in outgoing_players:
            consistency = load_player_consistency(player, league_id)
            if consistency:
                outgoing_data.append({
                    'Player': player,
                    'Mean FPts': f"{consistency['mean_fpts']:.1f}",
                    'CV%': f"{consistency['cv_percent']:.1f}%",
                    'Tier': consistency['consistency_tier']
                })
        
        if outgoing_data:
            dataframe(pd.DataFrame(outgoing_data), hide_index=True, width="stretch")
            avg_fpg_out = sum([float(d['Mean FPts']) for d in outgoing_data]) / len(outgoing_data)
            st.caption(f"Average: {avg_fpg_out:.1f} FP/G")
        else:
            st.info("No consistency data available")
    
    with col2:
        st.markdown("**📥 Receiving**")
        incoming_data = []
        for player in incoming_players:
            consistency = load_player_consistency(player, league_id)
            if consistency:
                incoming_data.append({
                    'Player': player,
                    'Mean FPts': f"{consistency['mean_fpts']:.1f}",
                    'CV%': f"{consistency['cv_percent']:.1f}%",
                    'Tier': consistency['consistency_tier']
                })
        
        if incoming_data:
            dataframe(pd.DataFrame(incoming_data), hide_index=True, width="stretch")
            avg_fpg_in = sum([float(d['Mean FPts']) for d in incoming_data]) / len(incoming_data)
            st.caption(f"Average: {avg_fpg_in:.1f} FP/G")
        else:
            st.info("No consistency data available")
    
    # Net comparison
    if outgoing_data and incoming_data:
        avg_fpg_out = sum([float(d['Mean FPts']) for d in outgoing_data]) / len(outgoing_data)
        avg_fpg_in = sum([float(d['Mean FPts']) for d in incoming_data]) / len(incoming_data)
        net_change = avg_fpg_in - avg_fpg_out
        
        st.markdown("---")
        if net_change > 0:
            st.success(f"✅ **Net Gain:** Receiving players average {net_change:.1f} more FP/G")
        elif net_change < 0:
            st.error(f"❌ **Net Loss:** Receiving players average {abs(net_change):.1f} fewer FP/G")
        else:
            st.info("➖ **Even Trade:** Players have similar average production")

def _display_consistency_summary(results: Dict[str, Any], time_ranges: List[str]):
    """Display consistency change summary if data is available."""
    pre_consistency = results.get('pre_trade_consistency', {})
    post_consistency = results.get('post_trade_consistency', {})
    
    if not pre_consistency or not post_consistency:
        return
    
    # Use YTD as primary reference
    ytd_pre = pre_consistency.get('YTD', {})
    ytd_post = post_consistency.get('YTD', {})
    
    if not ytd_pre or not ytd_post:
        return
    
    st.markdown("---")
    st.markdown("### 📊 Consistency Impact (YTD)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pre_cv = ytd_pre.get('avg_cv', 0)
        post_cv = ytd_post.get('avg_cv', 0)
        delta = post_cv - pre_cv
        st.metric(
            "Avg CV%",
            f"{post_cv:.1f}%",
            delta=f"{delta:+.1f}%",
            delta_color="inverse",  # Lower is better
            help="Average Coefficient of Variation - lower = more consistent"
        )
    
    with col2:
        pre_consistent = ytd_pre.get('very_consistent', 0)
        post_consistent = ytd_post.get('very_consistent', 0)
        st.metric(
            "🟢 Very Consistent",
            post_consistent,
            delta=post_consistent - pre_consistent,
            help="Players with CV% below the very-consistent threshold (currently <25%)"
        )
    
    with col3:
        pre_moderate = ytd_pre.get('moderate', 0)
        post_moderate = ytd_post.get('moderate', 0)
        st.metric(
            "🟡 Moderate",
            post_moderate,
            delta=post_moderate - pre_moderate,
            help="Players with CV% in the solid/moderate band (currently ~25–40%)"
        )
    
    with col4:
        pre_volatile = ytd_pre.get('volatile', 0)
        post_volatile = ytd_post.get('volatile', 0)
        st.metric(
            "🔴 Volatile",
            post_volatile,
            delta=post_volatile - pre_volatile,
            delta_color="inverse",  # Fewer is better
            help="Players with CV% above the volatile threshold (currently >40%)"
        )

def _display_roster_details(results: Dict[str, Any], time_ranges: List[str]):
    """Displays detailed before and after roster data for each time range."""
    st.subheader("Roster Details (Before vs. After)")
    time_range_tabs = st.tabs(time_ranges)
    for time_tab, time_range in zip(time_range_tabs, time_ranges):
        with time_tab:
            outgoing_players = results.get('outgoing_players', [])
            incoming_players = results.get('incoming_players', [])

            col1, col2 = st.columns(2)
            with col1:
                _display_styled_roster(
                    "Before Trade",
                    results.get('pre_trade_rosters', {}).get(time_range, []),
                    outgoing_players,
                    '#ffcccb'  # Light red
                )
            with col2:
                _display_styled_roster(
                    "After Trade",
                    results.get('post_trade_rosters', {}).get(time_range, []),
                    incoming_players,
                    '#90ee90'  # Light green
                )
