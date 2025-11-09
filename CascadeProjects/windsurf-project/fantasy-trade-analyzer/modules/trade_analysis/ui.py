"""
UI components for the trade analysis feature.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List
from modules.trade_analysis.logic import get_team_name, run_trade_analysis, get_all_teams

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
            for trade, summary in history:
                st.text(summary)

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

    if trade_teams and st.button("Analyze Trade"):
        results = run_trade_analysis(trade_teams, num_players)
        if results:
            display_trade_results(results)

def display_trade_results(analysis_results: Dict[str, Dict[str, Any]]):
    """Display the trade analysis results."""
    time_ranges = list(next(iter(analysis_results.values()))['pre_trade_metrics'].keys())
    
    team_tabs = st.tabs([f"Team: {get_team_name(team)}" for team in analysis_results.keys()])
    
    for team_tab, (team, results) in zip(team_tabs, analysis_results.items()):
        with team_tab:
            _display_trade_overview(results)
            
            with st.expander("Trade Impact Analysis", expanded=True):
                _display_trade_metrics_table(results, time_ranges)
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

def _display_trade_metrics_table(results: Dict[str, Any], time_ranges: List[str]):
    st.markdown("""ℹ️ **Metrics Guide**: - **FP/G**: Fantasy Points per Game - **GP**: Games Played - **Std Dev**: Standard Deviation (consistency measure)""", unsafe_allow_html=True)
    combined_data = []
    for time_range in time_ranges:
        pre_metrics = results.get('pre_trade_metrics', {}).get(time_range, {})
        post_metrics = results.get('post_trade_metrics', {}).get(time_range, {})
        if pre_metrics and post_metrics:
            combined_data.append({
                'Time Range': time_range,
                'Mean FP/G': f"{pre_metrics['mean_fpg']:.1f} → <span style='color:{'green' if post_metrics['mean_fpg'] > pre_metrics['mean_fpg'] else 'red'}'>{post_metrics['mean_fpg']:.1f}</span>",
                'Median FP/G': f"{pre_metrics['median_fpg']:.1f} → <span style='color:{'green' if post_metrics['median_fpg'] > pre_metrics['median_fpg'] else 'red'}'>{post_metrics['median_fpg']:.1f}</span>",
                'Std Dev': f"{pre_metrics['std_dev']:.1f} → <span style='color:{'green' if post_metrics['std_dev'] < pre_metrics['std_dev'] else 'red'}'>{post_metrics['std_dev']:.1f}</span>",
                'Total FPs': f"{pre_metrics['total_fpts']} → <span style='color:{'green' if post_metrics['total_fpts'] > pre_metrics['total_fpts'] else 'red'}'>{post_metrics['total_fpts']}</span>",
                'Avg GP': f"{pre_metrics['avg_gp']:.1f} → <span style='color:{'green' if post_metrics['avg_gp'] >= pre_metrics['avg_gp'] else 'red'}'>{post_metrics['avg_gp']:.1f}</span>"
            })
    combined_df = pd.DataFrame(combined_data)
    st.markdown("### Trade Metrics (Before → After)")
    st.markdown(combined_df.to_html(escape=False, index=False), unsafe_allow_html=True)

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
        st.plotly_chart(fig, use_container_width=True)
    

def _display_styled_roster(title: str, roster_data: List[Dict[str, Any]], players_to_highlight: List[str], highlight_color: str):
    """Displays a styled roster dataframe, highlighting specific players."""
    st.write(f"#### {title}")
    if roster_data:
        roster_df = pd.DataFrame(roster_data)

        def highlight_players(row):
            return [f'background-color: {highlight_color}' if row['Player'] in players_to_highlight else '' for _ in row]

        styled_roster = roster_df.style.apply(highlight_players, axis=1)
        st.dataframe(styled_roster, hide_index=True)
    else:
        st.write("No data available.")

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
