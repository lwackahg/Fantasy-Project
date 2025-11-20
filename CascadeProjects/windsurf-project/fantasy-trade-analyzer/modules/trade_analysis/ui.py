"""
UI components for the trade analysis feature.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from typing import Dict, Any, List
from modules.trade_analysis.logic import TradeAnalyzer, get_team_name, run_trade_analysis, get_all_teams

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
                        if st.button("View details", key=f"history_view_{idx}"):
                            selected_entry = entry
                    st.write("---")

                # Render the replay OUTSIDE the narrow column → full width
                if selected_entry is not None:
                    _replay_trade_from_history(selected_entry)

def _display_player_selection_interface(selected_teams: List[str]) -> Dict[str, Dict[str, str]]:
    """Displays the UI for selecting players and their destinations."""
    st.write("### Select Players for Each Team")
    trade_teams = {}

    if st.session_state.combined_data is None:
        st.error("No data available for analysis")
        return {}

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
                key=f"players_{team}"
            )

            if selected_players:
                trade_teams[team] = {}
                for player in selected_players:
                    other_teams = [t for t in selected_teams if t != team]
                    dest = st.selectbox(
                        f"Select destination team for {player}",
                        other_teams,
                        format_func=get_team_name,
                        key=f"dest_{team}_{player}"
                    )
                    trade_teams[team][player] = dest
    return trade_teams

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

    trade_teams = _display_player_selection_interface(selected_teams)

    last_duration = st.session_state.get("trade_analysis_last_duration_sec")
    if last_duration:
        est_seconds = int(round(last_duration)) or 1
        st.caption(f"Last analysis took about {est_seconds} seconds. Future runs should be similar.")

    if trade_teams and st.button("Analyze Trade"):
        spinner_msg = "Running trade analysis..."
        if last_duration:
            est_seconds = int(round(last_duration)) or 1
            spinner_msg = f"Running trade analysis (expected ~{est_seconds} seconds)..."

        with st.spinner(spinner_msg):
            results = run_trade_analysis(
                trade_teams,
                num_players,
                trade_label=trade_label,
                include_advanced_metrics=include_advanced_metrics,
            )
        if results:
            display_trade_results(results)

def display_trade_results(analysis_results: Dict[str, Dict[str, Any]]):
    """Display the trade analysis results."""
    time_ranges = list(next(iter(analysis_results.values()))['pre_trade_metrics'].keys())
    
    team_tabs = st.tabs([f"Team: {get_team_name(team)}" for team in analysis_results.keys()])
    
    for team_tab, (team, results) in zip(team_tabs, analysis_results.items()):
        with team_tab:
            _display_trade_overview(results)
            _display_trade_impact_section(results, time_ranges)

def _display_trade_impact_section(results: Dict[str, Any], time_ranges: List[str]):
    """Display the main trade impact analysis section for a single team tab."""
    with st.expander("Trade Impact Analysis", expanded=True):
        _display_trade_metrics_table(results, time_ranges)
        st.write("---")
        _display_trade_insights(results, time_ranges)
        st.write("---")
        _display_performance_visualizations(results, time_ranges)
        _display_roster_details(results, time_ranges)

def _display_trade_overview(results: Dict[str, Any]):
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
    _display_traded_players_game_logs(results)

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
        st.plotly_chart(fig, width='stretch')
    

def _display_styled_roster(title: str, roster_data: List[Dict[str, Any]], players_to_highlight: List[str], highlight_color: str):
    """Displays a styled roster dataframe, highlighting specific players."""
    st.write(f"#### {title}")
    if roster_data:
        roster_df = pd.DataFrame(roster_data)

        def highlight_players(row):
            return [f'background-color: {highlight_color}' if row['Player'] in players_to_highlight else '' for _ in row]

        styled_roster = roster_df.style.apply(highlight_players, axis=1)
        st.dataframe(styled_roster, hide_index=True, width='stretch')
    else:
        st.write("No data available.")

def _display_traded_players_game_logs(results: Dict[str, Any]):
    """Display game logs for all players involved in the trade."""
    from modules.trade_analysis.consistency_integration import load_player_consistency, get_consistency_cache_directory
    import json
    
    league_id = results.get('league_id', '')
    if not league_id:
        return
    
    all_players = results.get('outgoing_players', []) + results.get('incoming_players', [])
    if not all_players:
        return
    
    with st.expander("📋 View Traded Players' Game Logs", expanded=False):
        player_tabs = st.tabs(all_players)
        
        cache_dir = get_consistency_cache_directory()
        
        for player_tab, player_name in zip(player_tabs, all_players):
            with player_tab:
                # Find and load player's game log
                cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
                game_log_df = None
                preferred_season = results.get("season") if isinstance(results, dict) else None
                best_candidate = None
                best_season = None
                
                for cache_file in cache_files:
                    try:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                        cached_name = cache_data.get('player_name', '')
                        if str(cached_name).strip().lower() != str(player_name).strip().lower():
                            continue
                        season_str = str(cache_data.get('season', '')).strip()
                        # If a specific season is attached to the results, prefer an exact match
                        if preferred_season and season_str == preferred_season:
                            best_candidate = cache_data
                            best_season = season_str
                            break
                        # Otherwise track the most recent season lexicographically
                        if not preferred_season:
                            if best_season is None or season_str > best_season:
                                best_candidate = cache_data
                                best_season = season_str
                    except Exception:
                        continue
                
                if best_candidate is not None:
                    game_log = best_candidate.get('data', best_candidate.get('game_log', []))
                    if game_log:
                        game_log_df = pd.DataFrame(game_log)
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
                    
                    st.dataframe(
                        game_log_df[display_cols],
                        width='stretch',
                        height=400
                    )
                    
                    # Download button
                    csv = game_log_df.to_csv(index=False)
                    # Create unique key by combining player name with index to avoid duplicates
                    unique_key = f"trade_download_{player_name.replace(' ', '_')}_{all_players.index(player_name)}"
                    st.download_button(
                        label=f"📥 Download {player_name} Game Log",
                        data=csv,
                        file_name=f"{player_name.replace(' ', '_')}_game_log.csv",
                        mime="text/csv",
                        key=unique_key
                    )
                else:
                    st.info(f"No game log data available for {player_name}. Run Bulk Scrape in Admin Tools to populate cache.")

def _display_trade_insights(results: Dict[str, Any], time_ranges: List[str]):
    """Display comprehensive trade insights and recommendations."""
    import numpy as np
    from scipy import stats as scipy_stats
    
    st.markdown("### 🎯 Trade Insights & Analysis")
    
    # Get YTD data for primary analysis
    ytd_pre = results.get('pre_trade_metrics', {}).get('YTD', {})
    ytd_post = results.get('post_trade_metrics', {}).get('YTD', {})
    ytd_pre_consistency = results.get('pre_trade_consistency', {}).get('YTD', {})
    ytd_post_consistency = results.get('post_trade_consistency', {}).get('YTD', {})
    
    if not ytd_pre or not ytd_post:
        st.info("Insufficient data for trade insights")
        return
    
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
    
    # Overall trade assessment with collapsible sections
    with st.expander("📊 Overall Assessment & Key Metrics", expanded=True):
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
    
    # Time range trend analysis - collapsible
    with st.expander("📉 Trend Analysis Across Time Ranges", expanded=False):
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
            
            st.plotly_chart(fig_trend, width='stretch')
            
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
    
    # Statistical significance testing
    with st.expander("🔬 Statistical Analysis & Significance", expanded=False):
        _display_statistical_analysis(results, time_ranges, ytd_pre, ytd_post)
    
    # Player-by-player comparison - collapsible
    with st.expander("👥 Player-by-Player Comparison", expanded=False):
        outgoing = results.get('outgoing_players', [])
        incoming = results.get('incoming_players', [])
        
        if outgoing and incoming:
            _display_player_comparison(outgoing, incoming, results)
        else:
            st.info("No player comparison data available")

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
    
    st.plotly_chart(fig_dist, width='stretch')
    
    st.caption("**Box plots** show the distribution of expected performance. The box represents the middle 50% of outcomes, with the line showing the median.")

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
            st.dataframe(pd.DataFrame(outgoing_data), hide_index=True, width='stretch')
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
            st.dataframe(pd.DataFrame(incoming_data), hide_index=True, use_container_width=True)
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
