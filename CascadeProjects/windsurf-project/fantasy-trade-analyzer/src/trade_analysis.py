"""Trade analysis module for Fantasy Basketball Trade Analyzer."""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from debug import debug_manager
from data_loader import TEAM_MAPPINGS

def get_team_name(team_id: str) -> str:
    """Get full team name from team ID."""
    return TEAM_MAPPINGS.get(team_id, team_id)

def get_all_teams() -> List[str]:
    """Get a list of all teams from the data."""
    if not st.session_state.data_ranges:
        return []
    
    # Use current range if available
    if st.session_state.current_range and st.session_state.current_range in st.session_state.data_ranges:
        return sorted(TEAM_MAPPINGS.keys())
    
    # Fallback to first available range
    return sorted(TEAM_MAPPINGS.keys())

def display_trade_analysis_page():
    """Display the trade analysis page."""
    st.write("## Trade Analysis")
    
    # Update trade analyzer with all data ranges
    if st.session_state.trade_analyzer:
        for range_key, data in st.session_state.data_ranges.items():
            st.session_state.trade_analyzer.update_data(data)
    
    # Setup trade interface
    trade_setup()
    
    # Display trade history
    st.write("## Trade Analysis History")
    if st.session_state.trade_analyzer:
        history = st.session_state.trade_analyzer.get_trade_history()
        for trade, summary in history:
            st.text(summary)

def trade_setup():
    """Setup the trade interface."""
    st.write("## Analysis Settings")
    
    # Number of top players to analyze
    num_players = st.number_input(
        "Number of Top Players to Analyze",
        min_value=1,
        max_value=14,
        value=9,
        help="Select the number of top players to analyze"
    )
    
    # Team selection
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
        
    # Player selection for each team
    st.write("### Select Players for Each Team")
    trade_teams = {}
    
    # Use combined data for player selection
    if st.session_state.combined_data is None:
        st.error("No data available for analysis")
        return
    
    # Define how many columns you want
    num_cols = 3
    num_rows = (len(selected_teams) + num_cols - 1) // num_cols  # Calculate number of rows needed
    cols = st.columns(num_cols)  # Create columns

    # Loop through teams and categorize players into the corresponding column
    for i, team in enumerate(selected_teams):
        with cols[i % num_cols]:  # Assign team to the corresponding column based on the index
            st.write(f"#### {get_team_name(team)}")
            # Reset index to access Player column, then filter by team
            team_data = st.session_state.combined_data.reset_index()
            team_players = team_data[team_data['Status'] == team]['Player'].unique().tolist()
            
            # Allow selection of players
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

    # Analyze trade button
    if trade_teams and st.button("Analyze Trade"):
        if st.session_state.trade_analyzer:
            # Update analyzer with latest data
            st.session_state.trade_analyzer.update_data(st.session_state.combined_data)
            results = st.session_state.trade_analyzer.evaluate_trade_fairness(trade_teams, num_players)
            display_trade_results(results)


def display_trade_results(analysis_results: Dict[str, Dict[str, Any]]): 
    """Display the trade analysis results.""" 
    # Get available time ranges from the data 
    time_ranges = list(next(iter(analysis_results.values()))['pre_trade_metrics'].keys()) 
    
    # Define a helper function for creating metric cards
    def create_metric_card(title, before_val, after_val, change, tooltip_text):
        change_class = "positive" if change > 0 else "negative" if change < 0 else "neutral"
        return f"**{title}**\nBefore: {before_val:.1f}\nAfter: {after_val:.1f}\nChange: {change:+.1f}"

    # Create tabs for each team
    team_tabs = st.tabs([f"Team: {get_team_name(team)}" for team in analysis_results.keys()])
    
    for team_tab, (team, results) in zip(team_tabs, analysis_results.items()): 
        with team_tab: 
            # Trade Overview Section 
            st.title("Trade Overview") 
            
            col1, col2 = st.columns(2) 
            with col1: 
                st.subheader("**Players Receiving**")
                incoming = results.get('incoming_players', [])
                if incoming:
                    st.write(", ".join(incoming))
                else:
                    st.write("None")

            with col2: 
                st.subheader("**Players Trading Away**") 
                outgoing = results.get('outgoing_players', [])
                if outgoing:
                    st.write(", ".join(outgoing))
                else:
                    st.write("None")
                    
            st.write("---")
             
            # Trade Impact Summary 
            st.markdown("## Trade Impact Analysis") 

            # Prepare data for before and after trade 
            combined_data = []
            for time_range in time_ranges:
                pre_metrics = results.get('pre_trade_metrics', {}).get(time_range, {})
                post_metrics = results.get('post_trade_metrics', {}).get(time_range, {})
                if pre_metrics and post_metrics:
                    combined_data.append({
                        'Time Range': time_range, 
                        'Mean FP/G': f"{pre_metrics['mean_fpg']:.1f}&nbsp;&nbsp;(<span style='color:{'green' if post_metrics['mean_fpg'] > pre_metrics['mean_fpg'] else 'red'}'>{post_metrics['mean_fpg']:.1f}</span>)", 
                        'Median FP/G': f"{pre_metrics['median_fpg']:.1f}&nbsp;&nbsp;(<span style='color:{'green' if post_metrics['median_fpg'] > pre_metrics['median_fpg'] else 'red'}'>{post_metrics['median_fpg']:.1f}</span>)", 
                        'Std Dev': f"{pre_metrics['std_dev']:.1f}&nbsp;&nbsp;(<span style='color:{'green' if post_metrics['std_dev'] < pre_metrics['std_dev'] else 'red'}'>{post_metrics['std_dev']:.1f}</span>)", 
                        'Total FPs': f"{pre_metrics['total_fpts']}&nbsp;&nbsp;(<span style='color:{'green' if post_metrics['total_fpts'] > pre_metrics['total_fpts'] else 'red'}'>{post_metrics['total_fpts']}</span>)", 
                        'Avg GP': f"{pre_metrics['avg_gp']:.1f}&nbsp;&nbsp;(<span style='color:{'green' if post_metrics['avg_gp'] >= pre_metrics['avg_gp'] else 'red'}'>{post_metrics['avg_gp']:.1f}</span>)"
                    })

            combined_df = pd.DataFrame(combined_data)

            # Update column names to indicate that after values are in brackets
            combined_df.columns = ['Time Range', 
                                    'Mean FP/G (Before - After)', 
                                    'Median FP/G (Before - After)', 
                                    'Std Dev (Before - After)', 
                                    'Total FPs (Before - After)', 
                                    'Avg GP (Before - After)']

            # Display the combined table with HTML rendering
            st.markdown("### Trade Metrics")
            st.markdown(combined_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.write("---")

            # Visualization Section 
            st.subheader("Performance Visualization") 
            metrics_to_plot = [('FP/G', 'mean_fpg'), ('Median', 'median_fpg'), ('Std Dev', 'std_dev')] 
            
            for display_name, metric_key in metrics_to_plot: 
                # Prepare metric data for plotting
                metric_data = pd.DataFrame({
                    'Time Range': time_ranges * 2,
                    display_name: [results['pre_trade_metrics'][tr][metric_key] for tr in time_ranges] + 
                                   [results['post_trade_metrics'][tr][metric_key] for tr in time_ranges],
                    'Type': ['Before'] * len(time_ranges) + ['After'] * len(time_ranges)
                })
                fig = px.line(metric_data, x='Time Range', y=display_name, color='Type', markers=True, 
                              labels={display_name: f"{display_name} Value"})
                st.plotly_chart(fig, use_container_width=True)

            # Time Range Analysis 
            time_range_tabs = st.tabs(time_ranges) 
            for time_tab, time_range in zip(time_range_tabs, time_ranges): 
                with time_tab: 
                    st.write("#### Trade Details") 
                    
                    # Create columns for Receiving and Trading Away players
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Receiving Players**")
                        st.write(", ".join(results.get('incoming_players', [])) or "None")
                    
                    with cols[1]:
                        st.write("**Trading Away Players**")
                        st.write(", ".join(results.get('outgoing_players', [])) or "None")
                    
                    st.write("---") 

                    # Create columns for Before Trade and After Trade Data
                    trade_cols = st.columns(2)
                    
                    with trade_cols[0]: 
                        st.write("#### Before Trade Data")
                        if time_range in results.get('pre_trade_rosters', {}):
                            roster_df_before = pd.DataFrame(results['pre_trade_rosters'][time_range]) 
                            st.dataframe(roster_df_before, hide_index=True)

                    with trade_cols[1]: 
                        st.write("#### After Trade Data")
                        if time_range in results.get('post_trade_rosters', {}):
                            roster_df_after = pd.DataFrame(results['post_trade_rosters'][time_range]) 
                            st.dataframe(roster_df_after, hide_index=True)


def format_change(value, inverse=False):
    """Format change values with color and sign"""
    if value == 0:
        return value
    return value  # Let the NumberColumn formatting handle the display
