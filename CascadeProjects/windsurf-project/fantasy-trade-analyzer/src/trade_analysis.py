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
        
    for team in selected_teams:
        st.write(f"#### {get_team_name(team)}")
        # Reset index to access Player column, then filter by team
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
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .metric-card {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #1E1E1E;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
        .metric-title {
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
            margin: 0.3rem 0;
        }
        .positive-change {
            color: #00CC00;
            font-weight: bold;
        }
        .negative-change {
            color: #FF4B4B;
            font-weight: bold;
        }
        .neutral-change {
            color: #808080;
            font-weight: bold;
        }
        .tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            background-color: #2C2C2C;
            color: #fff;
            text-align: center;
            padding: 8px 12px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            width: max-content;
            max-width: 250px;
            font-size: 0.9rem;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #333;
        }
        .player-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .player-item {
            display: flex;
            align-items: center;
            padding: 0.5rem;
            margin: 0.3rem 0;
            background: #2C2C2C;
            border-radius: 4px;
            transition: background 0.2s;
        }
        .player-item:hover {
            background: #383838;
        }
        .summary-box {
            background: linear-gradient(45deg, #1E1E1E, #2C2C2C);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid #333;
        }
        .trend-chart {
            background: #1E1E1E;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .stat-change {
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }
        .stat-change-icon {
            font-size: 1.2rem;
        }
        .divider {
            height: 1px;
            background: #333;
            margin: 1.5rem 0;
        }
        .metric-table {
            width: 20%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 1rem 0;
            background: #1E1E1E;
            border-radius: 8px;
            overflow: hidden;
            font-size: 0.9rem;
        }
        .metric-table th, .metric-table td {
            padding: 0.75rem 1rem;
            text-align: center;
            border: 1px solid #333;
            min-width: 20px;
        }
        .metric-table th {
            background: #2C2C2C;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 10;
            white-space: nowrap;
        }
        .metric-table tbody tr:hover {
            background: rgba(44, 44, 44, 0.6);
        }
        .metric-table td.positive-change {
            color: #00CC00;
            font-weight: bold;
        }
        .metric-table td.negative-change {
            color: #FF4B4B;
            font-weight: bold;
        }
        .metric-table td.neutral-change {
            color: #808080;
        }
        .metric-group {
            border-left: 2px solid #333;
        }
        .table-container {
            margin: 1rem 0;
            overflow-x: auto;
            border-radius: 8px;
            background: #1E1E1E;
            padding: 0.5rem;
            border: 1px solid #333;
        }
        .time-frame-col {
            background: #2C2C2C;
            font-weight: bold;
            position: sticky;
            left: 0;
            z-index: 5;
        }
        </style>
    """, unsafe_allow_html=True)

    def create_metric_card(title, before_val, after_val, change, tooltip_text, is_inverse=False):
        """Helper function to create a metric card with consistent styling"""
        if is_inverse:
            change_class = "negative-change" if change > 0 else "positive-change" if change < 0 else "neutral-change"
            change_icon = "" if change < 0 else "" if change > 0 else ""
        else:
            change_class = "positive-change" if change > 0 else "negative-change" if change < 0 else "neutral-change"
            change_icon = "" if change > 0 else "" if change < 0 else ""
        
        return f"""
            <div class="metric-card">
                <div class="metric-title">
                    <div class="tooltip">
                        {title} 
                        <span class="tooltiptext">{tooltip_text}</span>
                    </div>
                </div>
                <div class="metric-value">Before: {before_val:.1f}</div>
                <div class="metric-value">After: {after_val:.1f}</div>
                <div class="stat-change {change_class}">
                    <span class="stat-change-icon">{change_icon}</span>
                    Change: {change:+.1f}
                </div>
            </div>
        """

    team_tabs = st.tabs([f"Team: {get_team_name(team)}" for team in analysis_results.keys()])
    
    for team_tab, (team, results) in zip(team_tabs, analysis_results.items()):
        with team_tab:
            # Trade Overview Section
            st.markdown('<div class="section-header"> Trade Overview</div>', unsafe_allow_html=True)
            
            # Quick Summary Box
            st.markdown("""
                <div class="summary-box">
                    <div class="metric-title">Quick Impact Summary</div>
                    <div class="tooltip">
                        Click sections below for detailed analysis 
                        <span class="tooltiptext">Explore each section for in-depth statistics and visualizations</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-title"> Players Receiving</div>
                    </div>
                """, unsafe_allow_html=True)
                if results.get('incoming_players'):
                    st.markdown('<ul class="player-list">', unsafe_allow_html=True)
                    for player in results['incoming_players']:
                        st.markdown(f'<li class="player-item"> {player}</li>', unsafe_allow_html=True)
                    st.markdown('</ul>', unsafe_allow_html=True)
                else:
                    st.write("None")
            
            with col2:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-title"> Players Trading Away</div>
                    </div>
                """, unsafe_allow_html=True)
                if results.get('outgoing_players'):
                    st.markdown('<ul class="player-list">', unsafe_allow_html=True)
                    for player in results['outgoing_players']:
                        st.markdown(f'<li class="player-item"> {player}</li>', unsafe_allow_html=True)
                    st.markdown('</ul>', unsafe_allow_html=True)
                else:
                    st.write("None")
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Trade Impact Summary
            st.markdown("## Trade Impact Analysis")
            
            # Create DataFrames for before and after trade
            before_data = []
            after_data = []
            
            for time_range in time_ranges:
                if time_range in results.get('pre_trade_metrics', {}) and time_range in results.get('post_trade_metrics', {}):
                    pre_metrics = results['pre_trade_metrics'][time_range]
                    post_metrics = results['post_trade_metrics'][time_range]
                    
                    before_row = {
                        'Time Range': time_range,
                        'Mean FP/G': pre_metrics['mean_fpg'],
                        'Median FP/G': pre_metrics['median_fpg'],
                        'Std Dev': pre_metrics['std_dev'],
                        'Total FPs': pre_metrics['total_fpts'],
                        'Avg GP': pre_metrics['avg_gp']
                    }
                    before_data.append(before_row)
                    
                    after_row = {
                        'Time Range': time_range,
                        'Mean FP/G': post_metrics['mean_fpg'],
                        'Median FP/G': post_metrics['median_fpg'],
                        'Std Dev': post_metrics['std_dev'],
                        'Total FPs': post_metrics['total_fpts'],
                        'Avg GP': post_metrics['avg_gp']
                    }
                    after_data.append(after_row)
            
            if before_data and after_data:
                before_df = pd.DataFrame(before_data)
                after_df = pd.DataFrame(after_data)
                
                # Configure column styling
                column_config = {
                    'Time Range': st.column_config.TextColumn('Time Range'),
                    'Mean FP/G': st.column_config.NumberColumn('Mean FP/G', format='%.1f'),
                    'Median FP/G': st.column_config.NumberColumn('Median FP/G', format='%.1f'),
                    'Std Dev': st.column_config.NumberColumn('Std Dev', format='%.1f'),
                    'Total FPs': st.column_config.NumberColumn('Total FPs', format='%d'),  # No decimals
                    'Avg GP': st.column_config.NumberColumn('Avg GP', format='%.1f')
                }
                
                # Format the dataframes to round numbers appropriately
                before_df['Total FPs'] = before_df['Total FPs'].round(0).astype(int)
                after_df['Total FPs'] = after_df['Total FPs'].round(0).astype(int)
                
                for col in ['Mean FP/G', 'Median FP/G', 'Std Dev', 'Avg GP']:
                    before_df[col] = before_df[col].round(1)
                    after_df[col] = after_df[col].round(1)
                
                # Add some spacing
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Before Trade table
                st.markdown("### Before Trade")
                st.dataframe(
                    before_df,
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=False
                )
                
                # Add spacing between tables
                st.markdown("<br>", unsafe_allow_html=True)
                
                # After Trade table
                st.markdown("### After Trade")
                
                # Format after trade data
                after_data_formatted = []
                for after_row in after_data:
                    formatted_row = {'Time Range': after_row['Time Range']}
                    
                    for col in ['Mean FP/G', 'Median FP/G', 'Std Dev', 'Total FPs', 'Avg GP']:
                        val = after_row[col]
                        
                        # Format number
                        if col == 'Total FPs':
                            formatted_row[col] = f"{int(val)}"
                        else:
                            formatted_row[col] = f"{val:.1f}"
                    
                    after_data_formatted.append(formatted_row)
                
                after_df_formatted = pd.DataFrame(after_data_formatted)
                
                st.dataframe(
                    after_df_formatted,
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=False
                )
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Visualization Section
            st.markdown('<div class="section-header"> Performance Visualization</div>', unsafe_allow_html=True)
            
            # Create scatter plot with lines for key metrics
            metrics_to_plot = [
                ('FP/G', 'mean_fpg', 'Fantasy Points per Game - Team scoring potential'),
                ('Median', 'median_fpg', 'Median Fantasy Points - Consistent performance level'),
                ('Std Dev', 'std_dev', 'Standard Deviation - Team consistency (lower is better)')
            ]
            
            for display_name, metric_key, tooltip in metrics_to_plot:
                with st.container():
                    st.markdown(f"""
                        <div class="trend-chart">
                            <div class="metric-title">
                                <div class="tooltip">
                                    {display_name} Trend Analysis 
                                    <span class="tooltiptext">{tooltip}</span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    metric_data = pd.DataFrame({
                        'Time Range': time_ranges * 2,
                        display_name: [pre_metrics[metric_key] for pre_metrics in [results['pre_trade_metrics'].get(tr, {}) for tr in time_ranges]] +
                                    [post_metrics[metric_key] for post_metrics in [results['post_trade_metrics'].get(tr, {}) for tr in time_ranges]],
                        'Type': ['Before'] * len(time_ranges) + ['After'] * len(time_ranges)
                    })
                    
                    fig = px.line(metric_data, x='Time Range', y=display_name, color='Type', markers=True,
                                labels={display_name: tooltip},
                                color_discrete_map={'Before': '#FF4B4B', 'After': '#00CC00'})
                    
                    fig.update_traces(mode='lines+markers', marker=dict(size=10))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_size=20,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor='rgba(0,0,0,0.5)'
                        ),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Time Range Analysis
            time_range_tabs = st.tabs(time_ranges)
            for time_tab, time_range in zip(time_range_tabs, time_ranges):
                with time_tab:
                    st.write("#### Trade Details")
                    # Show receiving/trading away again for reference
                    trade_col1, trade_col2 = st.columns(2)
                    with trade_col1:
                        st.write("**Receiving**")
                        if results.get('incoming_players'):
                            for player in results['incoming_players']:
                                st.write(f"- {player}")
                        else:
                            st.write("None")
                    
                    with trade_col2:
                        st.write("**Trading Away**")
                        if results.get('outgoing_players'):
                            for player in results['outgoing_players']:
                                st.write(f"- {player}")
                        else:
                            st.write("None")
                    
                    st.write("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("#### Before Trade")
                        if time_range in results.get('pre_trade_rosters', {}):
                            roster_df = pd.DataFrame(results['pre_trade_rosters'][time_range])
                            st.dataframe(
                                roster_df,
                                hide_index=True,
                                column_config={
                                    'Player': st.column_config.TextColumn('Player'),
                                    'Team': st.column_config.TextColumn('Team'),
                                    'FPts': st.column_config.NumberColumn('FPts', format="%.1f"),
                                    'FP/G': st.column_config.NumberColumn('FP/G', format="%.1f"),
                                    'GP': st.column_config.NumberColumn('GP', format="%.1f")
                                }
                            )
                            
                            if time_range in results.get('pre_trade_metrics', {}):
                                metrics = results['pre_trade_metrics'][time_range]
                                st.write(f"Mean FP/G: {metrics['mean_fpg']:.1f}")
                                st.write(f"Median FP/G: {metrics['median_fpg']:.1f}")
                                st.write(f"Std Dev: {metrics['std_dev']:.1f}")
                                st.write(f"Avg GP: {metrics['avg_gp']:.1f}")
                    
                    with col2:
                        st.write("#### After Trade")
                        if time_range in results.get('post_trade_rosters', {}):
                            roster_df = pd.DataFrame(results['post_trade_rosters'][time_range])
                            st.dataframe(
                                roster_df,
                                hide_index=True,
                                column_config={
                                    'Player': st.column_config.TextColumn('Player'),
                                    'Team': st.column_config.TextColumn('Team'),
                                    'FPts': st.column_config.NumberColumn('FPts', format="%.1f"),
                                    'FP/G': st.column_config.NumberColumn('FP/G', format="%.1f"),
                                    'GP': st.column_config.NumberColumn('GP', format="%.1f")
                                }
                            )
                            
                            if time_range in results.get('post_trade_metrics', {}):
                                metrics = results['post_trade_metrics'][time_range]
                                st.write(f"Mean FP/G: {metrics['mean_fpg']:.1f}")
                                st.write(f"Median FP/G: {metrics['median_fpg']:.1f}")
                                st.write(f"Std Dev: {metrics['std_dev']:.1f}")
                                st.write(f"Avg GP: {metrics['avg_gp']:.1f}")
                    
                    # Show changes
                    if time_range in results.get('value_changes', {}):
                        changes = results['value_changes'][time_range]
                        st.write("#### Impact")
                        cols = st.columns(4)
                        
                        metrics = [
                            ('FP/G', changes['mean_fpg_change']),
                            ('Median', results['post_trade_metrics'][time_range]['median_fpg'] - results['pre_trade_metrics'][time_range]['median_fpg']),
                            ('Std Dev', results['post_trade_metrics'][time_range]['std_dev'] - results['pre_trade_metrics'][time_range]['std_dev']),
                            ('GP', changes['avg_gp_change'])
                        ]
                        
                        for col, (metric, change) in zip(cols, metrics):
                            with col:
                                color = "green" if change > 0 else "red"
                                st.markdown(f"{metric} Change: <span style='color:{color}'>{change:+.1f}</span>", unsafe_allow_html=True)

def format_change(value, inverse=False):
    """Format change values with color and sign"""
    if value == 0:
        return value
    return value  # Let the NumberColumn formatting handle the display
