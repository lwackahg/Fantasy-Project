import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from collections import Counter
from config.team_mappings import TEAM_MAPPINGS

# Import custom modules
from data_import import DataImporter
from trade_analysis import TradeAnalyzer
from statistical_analysis import StatisticalAnalyzer

# Set page config
st.set_page_config(
    page_title="Fantasy Basketball Trade Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_ranges' not in st.session_state:
    st.session_state.data_ranges = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Initialize analyzers
if 'trade_analyzer' not in st.session_state:
    st.session_state.trade_analyzer = None
if 'stats_analyzer' not in st.session_state:
    st.session_state.stats_analyzer = StatisticalAnalyzer()

def load_data():
    """Load and process player data from CSV files"""
    try:
        # Get the absolute path to the data directory
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data"
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            st.error("No CSV files found in the data directory")
            return None, None
        
        # Create a mapping of date ranges to DataFrames
        data_ranges = {
            '60 Days': None,
            '30 Days': None,
            '14 Days': None,
            '7 Days': None
        }
        
        # Combined data for all players
        all_player_data = {}
        
        numeric_columns = ['FPts', 'FP/G', 'PTS', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN']
        
        for file in csv_files:
            # Extract the date range from filename, handling both formats
            for days in ['60', '30', '14', '7']:
                if f"({days})" in file.name or f" ({days})" in file.name:
                    try:
                        # Read CSV with proper type conversion
                        df = pd.read_csv(file)
                        
                        # Convert numeric columns
                        for col in numeric_columns:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
                        
                        # Ensure required columns exist
                        required_columns = ['Player', 'Team', 'FP/G']
                        if not all(col in df.columns for col in required_columns):
                            st.error(f"Missing required columns in {file.name}. Required: {required_columns}")
                            continue
                        
                        # Clean up team names
                        if 'Team' in df.columns:
                            df['Team'] = df['Team'].fillna('FA')
                        
                        # Calculate GP (Games Played)
                        if 'FPts' in df.columns and 'FP/G' in df.columns:
                            df['GP'] = df['FPts'] / df['FP/G']
                        
                        # Ensure 'Player' is not the index
                        if 'Player' in df.columns:
                            df.reset_index(inplace=True, drop=True)
                        
                        # Store in data ranges
                        data_ranges[f'{days} Days'] = df
                        
                        # Update all_player_data
                        for player in df['Player']:
                            if player not in all_player_data:
                                all_player_data[player] = {}
                            all_player_data[player][f'{days} Days'] = df[df['Player'] == player]
                        
                        break
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {str(e)}")
        
        # Check if any data was loaded
        loaded_ranges = [k for k, v in data_ranges.items() if v is not None]
        if not loaded_ranges:
            st.error("No valid data files were loaded")
            return None, None
        
        st.success(f"Successfully loaded data for ranges: {', '.join(loaded_ranges)}")
        return data_ranges, all_player_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def handle_error(func):
    """Decorator to handle errors in the app"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error("An error occurred!")
            if st.session_state.debug_mode:
                st.error(f"Error details: {str(e)}")
                st.exception(e)
            else:
                st.error("Please try again or contact support if the issue persists.")
    return wrapper

@handle_error
def calculate_team_stats(team_data: pd.DataFrame, top_x: int = None) -> dict:
    """Calculate key statistics for a team's players"""
    try:
        # Ensure numeric columns
        team_data['FP/G'] = pd.to_numeric(team_data['FP/G'].astype(str).str.strip(), errors='coerce')
        team_data['FPts'] = pd.to_numeric(team_data['FPts'].astype(str).str.strip(), errors='coerce')
        
        # Remove any rows where FP/G is NaN
        team_data = team_data.dropna(subset=['FP/G'])
        
        # Sort by FP/G and get top X players if specified
        sorted_data = team_data.sort_values('FP/G', ascending=False)
        if top_x is not None and top_x > 0:
            sorted_data = sorted_data.head(top_x)
        
        stats = {
            'mean_fpg': float(sorted_data['FP/G'].mean()),
            'median_fpg': float(sorted_data['FP/G'].median()),
            'std_fpg': float(sorted_data['FP/G'].std()),
            'mean_fpts': float(sorted_data['FPts'].mean()),
            'median_fpts': float(sorted_data['FPts'].median()),
            'std_fpts': float(sorted_data['FPts'].std()),
            'num_players': len(sorted_data),
            'total_fpts': float(sorted_data['FPts'].sum()),
            'avg_gp': float(sorted_data['GP'].mean())
        }
    except (ValueError, TypeError) as e:
        st.error(f"Error calculating stats: {str(e)}")
        stats = {
            'mean_fpg': 0.0,
            'median_fpg': 0.0,
            'std_fpg': 0.0,
            'mean_fpts': 0.0,
            'median_fpts': 0.0,
            'std_fpts': 0.0,
            'num_players': 0,
            'total_fpts': 0.0,
            'avg_gp': 0.0
        }
    return stats

@handle_error
def calculate_player_value(player_data):
    """Calculate a player's value based on their stats across time ranges"""
    value = 0
    weights = {
        '7 Days': 0.4,
        '14 Days': 0.3,
        '30 Days': 0.2,
        '60 Days': 0.1
    }
    
    for time_range, weight in weights.items():
        if time_range not in player_data:
            continue
        stats = player_data[time_range]
        if len(stats) == 0:
            continue
        
        # Use mean FP/G as the primary value metric
        value += stats['FP/G'].mean() * weight
    
    return value

@handle_error
def calculate_trade_fairness(before_stats, after_stats, team_data):
    """Calculate trade fairness based on player values and statistical changes"""
    # Get incoming and outgoing players
    incoming = team_data['incoming_players']
    outgoing = team_data['outgoing_players']
    
    # Calculate individual player values
    incoming_values = []
    outgoing_values = []
    
    for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
        if time_range not in st.session_state.data_ranges:
            continue
            
        current_data = st.session_state.data_ranges[time_range]
        
        # Calculate values for this time range
        for player in incoming:
            player_data = current_data[current_data['Player'] == player]
            if not player_data.empty:
                value = calculate_player_value({time_range: player_data})
                if value > 0:
                    incoming_values.append(value)
        
        for player in outgoing:
            player_data = current_data[current_data['Player'] == player]
            if not player_data.empty:
                value = calculate_player_value({time_range: player_data})
                if value > 0:
                    outgoing_values.append(value)
    
    # Get average values
    incoming_avg = sum(incoming_values) / len(incoming_values) if incoming_values else 0
    outgoing_avg = sum(outgoing_values) / len(outgoing_values) if outgoing_values else 0
    
    # Calculate total value difference
    incoming_total = sum(incoming_values)
    outgoing_total = sum(outgoing_values)
    
    # Calculate FP/G differences across time ranges
    fpg_diffs = []
    for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
        if (time_range in before_stats and 
            time_range in after_stats):
            
            diff = after_stats[time_range]['mean_fpg'] - before_stats[time_range]['mean_fpg']
            fpg_diffs.append(diff)
    
    # A trade is unfair if:
    # 1. The total value difference is large
    # 2. Individual player values are very different
    # 3. Team loses significant FP/G across multiple time ranges
    value_ratio = min(incoming_total, outgoing_total) / max(incoming_total, outgoing_total)
    avg_ratio = min(incoming_avg, outgoing_avg) / max(incoming_avg, outgoing_avg)
    
    # Calculate FP/G penalty
    fpg_penalties = []
    for diff in fpg_diffs:
        if diff < -2:  # Losing more than 2 FP/G is bad
            penalty = min(1.0, abs(diff) / 10)  # Max 100% penalty at 10 FP/G loss
            fpg_penalties.append(penalty)
    
    fpg_penalty = max(fpg_penalties) if fpg_penalties else 0
    
    # Heavily weight both total value and average value differences
    fairness_score = (value_ratio * 0.4 + avg_ratio * 0.3) * (1 - fpg_penalty)
    
    # Additional penalty for multi-player trades where values are uneven
    if len(incoming) != len(outgoing):
        fairness_score *= 0.8  # 20% penalty for uneven player counts
    
    return fairness_score

@handle_error
def get_trend_color(value, is_positive_good=True):
    """Get color for trend visualization.
    
    Args:
        value (float): Change in value
        is_positive_good (bool): If True, positive changes are green
        
    Returns:
        str: Hex color code
    """
    if abs(value) < 0.1:  # Very small change
        return '#666666'  # Gray
    
    if (value > 0 and is_positive_good) or (value < 0 and not is_positive_good):
        return '#2ecc71'  # Green
    else:
        return '#e74c3c'  # Red

@handle_error
def display_team_comparison(team_data, team_name):
    """Display comparison of team stats before and after trade"""
    st.subheader(f"{team_name} Analysis")
    
    # Create two columns for the layout
    col1, col2 = st.columns(2)
    
    # Display players involved in trade
    with col1:
        st.write("Players Involved:")
        
        # Display outgoing players
        st.write("Outgoing:")
        if team_data['outgoing_players']:
            outgoing_df = pd.DataFrame([
                p for p in team_data['top_players_before']['season'].to_dict('records')
                if p['Player'] in team_data['outgoing_players']
            ])
            if not outgoing_df.empty:
                st.dataframe(outgoing_df[['Player', 'Team', 'FPts', 'FP/G', 'GP']])
        
        # Display incoming players
        st.write("Incoming:")
        if team_data['incoming_players']:
            incoming_df = pd.DataFrame([
                p for p in team_data['top_players_after']['season'].to_dict('records')
                if p['Player'] in team_data['incoming_players']
            ])
            if not incoming_df.empty:
                st.dataframe(incoming_df[['Player', 'Team', 'FPts', 'FP/G', 'GP']])
    
    # Display value change
    with col2:
        value_change = team_data['value_change']
        st.metric(
            "Trade Value Change",
            f"{value_change:.2f}",
            delta=value_change
        )
        
        # Display fairness score
        st.metric(
            "Trade Fairness Score",
            f"{team_data['fairness_score']:.2f}"
        )
    
    # Create performance trend visualization
    fig = go.Figure()
    
    # Add before trade line (turquoise, dotted)
    fig.add_trace(go.Scatter(
        x=list(team_data['before_stats'].keys()),
        y=[stats['mean_fpg'] for stats in team_data['before_stats'].values()],
        name="Before Trade",
        line=dict(color='#20c9bb', width=3, dash='dot'),
        mode='lines+markers+text',
        text=[f"{val:.1f}" for val in [stats['mean_fpg'] for stats in team_data['before_stats'].values()]],
        textposition="top center",
        textfont=dict(color='#20c9bb'),
        hovertemplate="%{x}<br>" +
                        "Mean FP/G: %{y:.2f}<br>" +
                        "<extra>Before Trade</extra>"
    ))
    
    # Add after trade line (green/red based on value change)
    line_color = '#2ecc71' if team_data['value_change'] >= 0 else '#e74c3c'
    fig.add_trace(go.Scatter(
        x=list(team_data['after_stats'].keys()),
        y=[stats['mean_fpg'] for stats in team_data['after_stats'].values()],
        name="After Trade",
        line=dict(color=line_color),
        mode='lines+markers+text',
        text=[f"{val:.1f}" for val in [stats['mean_fpg'] for stats in team_data['after_stats'].values()]],
        textposition="bottom center",
        textfont=dict(color=line_color),
        hovertemplate="%{x}<br>" +
                        "Mean FP/G: %{y:.2f}<br>" +
                        "<extra>After Trade</extra>"
    ))
    
    # Add change annotations
    for i, (x, y_before, y_after) in enumerate(zip(list(team_data['before_stats'].keys()), [stats['mean_fpg'] for stats in team_data['before_stats'].values()], [stats['mean_fpg'] for stats in team_data['after_stats'].values()])):
        change = y_after - y_before
        if i == len(list(team_data['before_stats'].keys())) - 1:  # Only for the last point
            fig.add_annotation(
                x=x,
                y=y_after,
                text=f"{change:+.1f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=line_color,
                font=dict(color='white'),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor=line_color,
                borderwidth=1
            )
    
    # Update layout
    fig.update_layout(
        title="Performance Trend",
        xaxis_title="Time Range",
        yaxis_title="Average Fantasy Points per Game",
        template="plotly_dark",
        showlegend=True
    )
    
    st.plotly_chart(fig)
    
    # Display detailed stats
    st.write("Detailed Statistics:")
    for time_range in team_data['before_stats'].keys():
        st.write(f"\n{time_range} Stats:")
        cols = st.columns(2)
        
        # Before trade stats
        with cols[0]:
            st.write("Before Trade:")
            stats = team_data['before_stats'][time_range]
            st.write(f"Mean FP/G: {stats['mean_fpg']:.2f}")
            st.write(f"Median FP/G: {stats['median_fpg']:.2f}")
            st.write(f"Std Dev FP/G: {stats['std_fpg']:.2f}")
            st.write(f"Total Fantasy Points: {stats['total_fpts']:.2f}")
        
        # After trade stats
        with cols[1]:
            st.write("After Trade:")
            stats = team_data['after_stats'][time_range]
            st.write(f"Mean FP/G: {stats['mean_fpg']:.2f}")
            st.write(f"Median FP/G: {stats['median_fpg']:.2f}")
            st.write(f"Std Dev FP/G: {stats['std_fpg']:.2f}")
            st.write(f"Total Fantasy Points: {stats['total_fpts']:.2f}")

@handle_error
def display_trade_analysis(analysis, teams):
    """Display the trade analysis results"""
    # Calculate overall trade fairness
    fairness_scores = [data['fairness_score'] for data in analysis.values()]
    overall_fairness = sum(fairness_scores) / len(fairness_scores)
    
    # Display overall trade fairness
    st.markdown(
        f"""
        <div style='background-color: rgb(17, 23, 29); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h3 style='margin: 0; color: white;'>Overall Trade Fairness</h3>
            <div style='display: flex; align-items: center; margin-top: 0.5rem;'>
                <div style='flex-grow: 1; background-color: rgb(38, 39, 48); height: 1rem; border-radius: 0.5rem; overflow: hidden;'>
                    <div style='width: {overall_fairness * 100}%; height: 100%; background-color: {get_fairness_color(overall_fairness)};'></div>
                </div>
                <span style='margin-left: 1rem; color: white; font-weight: bold;'>{overall_fairness:.2%}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for each team
    team_tabs = st.tabs([get_team_name(team) for team in teams])
    
    # Display trade details for each team
    for team, tab in zip(teams, team_tabs):
        with tab:
            team_data = analysis[team]
            fairness = team_data['fairness_score']
            
            # Team header with fairness score
            st.markdown(
                f"""
                <div style='background-color: rgb(17, 23, 29); padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='flex-grow: 1; background-color: rgb(38, 39, 48); height: 0.5rem; border-radius: 0.25rem; overflow: hidden;'>
                            <div style='width: {fairness * 100}%; height: 100%; background-color: {get_fairness_color(fairness)};'></div>
                        </div>
                        <span style='margin-left: 0.5rem; color: white;'>{fairness:.2%}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Create expanders for different sections
            # Display trade impact
            with st.expander("📈 Trade Impact", expanded=True):
                st.markdown("##### Receiving")
                for player in team_data.get('incoming_players', []):
                    st.write(f"- {player}")
                
                st.markdown("##### Trading Away")
                for player in team_data.get('outgoing_players', []):
                    st.write(f"- {player}")
                
                if 'value_change' in team_data:
                    value_change = team_data['value_change']
                    color = '#2ecc71' if value_change > 0 else '#e74c3c'
                    st.markdown(f"**Net Value Change:** <span style='color: {color}'>{value_change:+.1f}</span>", unsafe_allow_html=True)
                
                # Create before/after trade stats tables with trend indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Before Trade")
                    if team_data['before_stats']:
                        before_stats = []
                        for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
                            if time_range in team_data['before_stats']:
                                stats = team_data['before_stats'][time_range]
                                before_stats.append({
                                    'Time Range': time_range,
                                    'Mean FP/G': stats['mean_fpg'],
                                    'Median FP/G': stats['median_fpg'],
                                    'Std Dev': stats['std_fpg'],
                                    'Total FPts': stats['total_fpts'],
                                    'Avg GP': stats['avg_gp']
                                })
                        
                        df = pd.DataFrame(before_stats)
                        st.dataframe(
                            df.style.format({
                                'Mean FP/G': '{:.1f}',
                                'Median FP/G': '{:.1f}',
                                'Std Dev': '{:.1f}',
                                'Total FPts': '{:.1f}',
                                'Avg GP': '{:.1f}'
                            }).set_properties(**{
                                'background-color': 'rgb(17, 23, 29)',
                                'color': 'white'
                            }),
                            hide_index=True
                        )
                
                with col2:
                    st.markdown("##### After Trade")
                    if team_data['after_stats']:
                        after_stats = []
                        changes = {}
                        
                        for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
                            if time_range in team_data['after_stats'] and time_range in team_data['before_stats']:
                                after = team_data['after_stats'][time_range]
                                before = team_data['before_stats'][time_range]
                                
                                # Calculate changes
                                changes[time_range] = {
                                    'Mean FP/G': after['mean_fpg'] - before['mean_fpg'],
                                    'Median FP/G': after['median_fpg'] - before['median_fpg'],
                                    'Std Dev': after['std_fpg'] - before['std_fpg'],
                                    'Total FPts': after['total_fpts'] - before['total_fpts'],
                                    'Avg GP': after['avg_gp'] - before['avg_gp']
                                }
                                
                                after_stats.append({
                                    'Time Range': time_range,
                                    'Mean FP/G': after['mean_fpg'],
                                    'Median FP/G': after['median_fpg'],
                                    'Std Dev': after['std_fpg'],
                                    'Total FPts': after['total_fpts'],
                                    'Avg GP': after['avg_gp']
                                })
                        
                        df = pd.DataFrame(after_stats)
                        
                        # Create style functions for each column
                        def style_column(col_name):
                            def style_values(values):
                                if col_name == 'Time Range':
                                    return ['color: white'] * len(values)
                                
                                styles = []
                                for idx, _ in enumerate(values):
                                    time_range = df.iloc[idx]['Time Range']
                                    change = changes[time_range][col_name]
                                    
                                    # Reverse color logic for Std Dev
                                    if col_name == 'Std Dev':
                                        if change < 0:
                                            styles.append('color: #2ecc71')  # Green for decrease
                                        elif change > 0:
                                            styles.append('color: #e74c3c')  # Red for increase
                                        else:
                                            styles.append('color: white')  # White for no change
                                    else:
                                        if change > 0:
                                            styles.append('color: #2ecc71')  # Green for positive
                                        elif change < 0:
                                            styles.append('color: #e74c3c')  # Red for negative
                                        else:
                                            styles.append('color: white')  # White for no change
                                return styles
                            return style_values
                        
                        # Apply styles column by column
                        styler = df.style.format({
                            'Mean FP/G': '{:.1f}',
                            'Median FP/G': '{:.1f}',
                            'Std Dev': '{:.1f}',
                            'Total FPts': '{:.1f}',
                            'Avg GP': '{:.1f}'
                        }).set_properties(**{
                            'background-color': 'rgb(17, 23, 29)',
                        })
                        
                        for col in df.columns:
                            styler = styler.apply(style_column(col), axis=0, subset=[col])
                        
                        st.dataframe(styler, hide_index=True)
                
            
            with st.expander("📊 Team Statistics", expanded=True):
                
                # Add trend plots
                st.markdown("##### Performance Trends")
                for metric in ['mean_fpg', 'median_fpg', 'total_fpts', 'avg_gp']:
                    fig = go.Figure()
                    
                    # Get data for before and after
                    time_ranges = ['60 Days', '30 Days', '14 Days', '7 Days']
                    before_values = [team_data['before_stats'][tr][metric] for tr in time_ranges if tr in team_data['before_stats']]
                    after_values = [team_data['after_stats'][tr][metric] for tr in time_ranges if tr in team_data['after_stats']]
                    
                    # Calculate overall change
                    change = after_values[-1] - before_values[-1]
                    change_color = '#2ecc71' if change > 0 else '#e74c3c'
                    
                    # Add before trade line
                    fig.add_trace(go.Scatter(
                        x=time_ranges,
                        y=before_values,
                        name='Before Trade',
                        line=dict(color='#666666', dash='dot'),
                        mode='lines+markers+text',
                        text=[f"{val:.1f}" for val in before_values],
                        textposition="top center",
                        textfont=dict(color='#666666'),
                        hovertemplate="%{x}<br>" +
                                    f"{metric.replace('_', ' ').title()}: %{{y:.2f}}<br>" +
                                    "<extra>Before Trade</extra>"
                    ))
                    
                    # Add after trade line
                    fig.add_trace(go.Scatter(
                        x=time_ranges,
                        y=after_values,
                        name='After Trade',
                        line=dict(color=change_color),
                        mode='lines+markers+text',
                        text=[f"{val:.1f}" for val in after_values],
                        textposition="bottom center",
                        textfont=dict(color=change_color),
                        hovertemplate="%{x}<br>" +
                                    f"{metric.replace('_', ' ').title()}: %{{y:.2f}}<br>" +
                                    "<extra>After Trade</extra>"
                    ))
                    
                    # Add change annotations
                    for i, (x, y_before, y_after) in enumerate(zip(time_ranges, before_values, after_values)):
                        change = y_after - y_before
                        if i == len(time_ranges) - 1:  # Only for the last point
                            fig.add_annotation(
                                x=x,
                                y=y_after,
                                text=f"{change:+.1f}",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor=change_color,
                                font=dict(color='white'),
                                bgcolor='rgba(0,0,0,0.8)',
                                bordercolor=change_color,
                                borderwidth=1
                            )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis=dict(
                            ticktext=time_ranges,
                            tickvals=list(range(len(time_ranges))),
                            title="Time Range",
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)',
                            tickfont=dict(color='white')
                        ),
                        yaxis=dict(
                            title="Value",
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)',
                            tickfont=dict(color='white')
                        ),
                        title=metric,
                        hovermode='x unified',
                        showlegend=True,
                        height=250,
                        margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display players involved
            with st.expander("👥 Players Involved", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Players Before")
                    if 'before_stats' in team_data:
                        for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
                            if time_range in team_data['before_stats']:
                                st.write(f"**{time_range}**")
                                player_data = team_data['before_stats'][time_range].get('player_data', [])
                                if player_data:
                                    # Convert to DataFrame and select relevant columns
                                    df = pd.DataFrame(player_data)
                                    display_cols = ['Player', 'Team', 'FPts', 'FP/G', 'GP']
                                    df = df[display_cols]
                                    
                                    # Format and display the DataFrame
                                    st.dataframe(
                                        df.style.format({
                                            'FP/G': '{:.1f}',
                                            'FPts': '{:.1f}',
                                            'GP': '{:.1f}'
                                        }).set_properties(**{
                                            'background-color': 'rgb(17, 23, 29)',
                                            'color': 'white'
                                        }),
                                        hide_index=True
                                    )
                
                with col2:
                    st.markdown("##### Players After")
                    if 'after_stats' in team_data:
                        for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
                            if time_range in team_data['after_stats']:
                                st.write(f"**{time_range}**")
                                player_data = team_data['after_stats'][time_range].get('player_data', [])
                                if player_data:
                                    # Convert to DataFrame and select relevant columns
                                    df = pd.DataFrame(player_data)
                                    display_cols = ['Player', 'Team', 'FPts', 'FP/G', 'GP']
                                    df = df[display_cols]
                                    
                                    # Format and display the DataFrame
                                    st.dataframe(
                                        df.style.format({
                                            'FP/G': '{:.1f}',
                                            'FPts': '{:.1f}',
                                            'GP': '{:.1f}'
                                        }).set_properties(**{
                                            'background-color': 'rgb(17, 23, 29)',
                                            'color': 'white'
                                        }),
                                        hide_index=True
                                    )
            
            
@handle_error
def display_trade_analysis_page():
    """Display the trade analysis page"""
    st.write("## Trade Analysis")
    
    # Set current data based on selected range
    st.session_state.data = st.session_state.data_ranges['60 Days']
    
    # Setup and analyze trade
    trade_setup()

@handle_error
def trade_setup():
    """Setup the trade with drag and drop interface"""
    st.write("## Trade Setup")
    
    # Get all teams and sort them
    teams = get_all_teams()
    teams.sort()
    
    # Debug logging
    st.write("Debug: Available teams", len(teams))
    
    # Allow user to select number of top players to analyze
    st.write("### Analysis Settings")
    top_x = st.number_input("Number of top players to analyze", min_value=1, max_value=15, value=10)
    
    # Allow user to select teams involved in the trade first
    selected_teams = st.multiselect(
        "Select teams to trade between",
        options=teams,
        help="Choose teams involved in the trade"
    )

    if not selected_teams:
        st.warning("Please select teams to begin trading")
        return

    # Debug logging
    st.write("Debug: Selected teams", len(selected_teams))
    
    # Dictionary to store players involved in trade for each team
    trade_teams = {}
    
    # Create columns only for selected teams
    cols = st.columns(len(selected_teams))
    
    # Create drag and drop interface only for selected teams
    for i, team in enumerate(selected_teams):
        with cols[i]:
            team_name = get_team_name(team)
            st.write(f"### {team_name}")
            
            # Get available players for this team
            available_players = []
            for time_range, data in st.session_state.data_ranges.items():
                if data is not None:
                    data['Full Team Name'] = data['Status'].map(lambda x: get_team_name(x))
                    team_players = data[data['Full Team Name'] == team_name]['Player'].unique()
                    available_players.extend(team_players)
            
            available_players = list(set(available_players))
            available_players.sort()
            
            # Debug logging
            st.write(f"Debug: {team_name} available players", len(available_players))
            
            # Multi-select for players
            selected_players = st.multiselect(
                f"Select players from {team_name}",
                available_players,
                key=f"{team}_players"
            )
            
            trade_teams[team] = selected_players
            
            # Debug logging
            if selected_players:
                st.write(f"Debug: {team_name} selected players", len(selected_players))

    # Only show player assignment section if teams have selected players
    active_teams = {team: players for team, players in trade_teams.items() if players}
    
    # Debug logging
    st.write("Debug: Active teams", len(active_teams))
    
    if active_teams:
        st.write("### Assign Players to Teams")
        for team, players in active_teams.items():
            if players:  # Only show teams with selected players
                st.write(f"#### {get_team_name(team)}")
                for player in players:
                    destination_team = st.selectbox(
                        f"Select destination team for {player}",
                        options=[t for t in selected_teams if t != team],
                        key=f"{player}_destination"
                    )
                    # Store the destination team for each player
                    if isinstance(trade_teams[team], list):
                        trade_teams[team] = {}
                    trade_teams[team][player] = destination_team
        
        st.write("### Trade Summary")
        for team in selected_teams:
            players = trade_teams.get(team, {})
            if isinstance(players, dict) and players:  # Only show teams with assigned players
                st.write(f"**{get_team_name(team)}** will trade:")
                for player, dest in players.items():
                    st.write(f"- {player} to {get_team_name(dest)}")
        
        # Add analyze button with unique key
        if st.button("Analyze Trade", key="analyze_trade_button", help="Click to see detailed trade analysis"):
            # Debug logging
            st.write("Debug: Starting trade analysis")
            st.write("Debug: Trade teams data", trade_teams)
            
            analysis = st.session_state.trade_analyzer.evaluate_trade_fairness(trade_teams, top_x)
            
            # Debug logging
            st.write("Debug: Analysis complete")
            st.write("Debug: Analysis results", bool(analysis))
            
            display_trade_analysis(analysis, selected_teams)

@handle_error
def display_team_stats_analysis():
    """Display team statistics analysis page"""
    st.write("## Team Statistics Analysis")
    
    # Get available teams
    teams = get_all_teams()
    if not teams:
        st.error("No team data available. Please upload data files first.")
        return
    
    # Team selection
    team = st.selectbox("Select Team to Analyze", teams)
    
    # Number of top players to analyze
    n_top_players = st.number_input("Number of top players to analyze", min_value=1, max_value=20, value=10)
    
    if team and n_top_players:
        # Create two columns for metrics selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Per Game Metrics")
            show_mean = st.checkbox("Mean FP/G", value=True)
            show_median = st.checkbox("Median FP/G", value=True)
            show_std = st.checkbox("Std Dev", value=False)
            show_gp = st.checkbox("Avg GP", value=False)
        
        with col2:
            st.write("Total Points")
            show_total = st.checkbox("Total FPts", value=False)
        
        # Collect selected metrics
        selected_metrics = []
        if show_mean:
            selected_metrics.append('mean_fpg')
        if show_median:
            selected_metrics.append('median_fpg')
        if show_std:
            selected_metrics.append('std_fpg')
        if show_gp:
            selected_metrics.append('avg_gp')
        if show_total:
            selected_metrics.append('total_fpts')
        
        if not selected_metrics:
            st.warning("Please select at least one metric to display")
            return
        
        # Display team performance trends
        st.write("### Team Performance Trends")
        stats = {}
        
        for time_range, data in st.session_state.data_ranges.items():
            if data is not None:
                team_data = data[data['Status'].str.contains(team, case=False)]
                if not team_data.empty:
                    stats[time_range] = calculate_team_stats(team_data, n_top_players)
        
        if stats:
            fig = plot_performance_trends(stats, selected_metrics, f"{team} Performance Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed statistics table
            st.write("### Team Statistics")
            stats_df = pd.DataFrame([
                {
                    'Time Range': tr,
                    'Mean FP/G': s['mean_fpg'],
                    'Median FP/G': s['median_fpg'],
                    'Std Dev': s['std_fpg'],
                    'Total FPts': s['total_fpts'],
                    'Avg GP': s['avg_gp']
                }
                for tr, s in stats.items()
            ])
            
            st.dataframe(
                stats_df.style.format({
                    'Mean FP/G': '{:.1f}',
                    'Median FP/G': '{:.1f}',
                    'Std Dev': '{:.1f}',
                    'Total FPts': '{:.1f}',
                    'Avg GP': '{:.1f}'
                }).set_properties(**{
                    'background-color': 'rgb(17, 23, 29)',
                    'color': 'white'
                }),
                hide_index=True
            )
            
            # Display top players
            st.write(f"### Top {n_top_players} Players by FP/G")
            for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
                if time_range in stats:
                    with st.expander(f"{time_range}"):
                        team_data = st.session_state.data_ranges[time_range]
                        if team_data is not None:
                            players = team_data[team_data['Status'].str.contains(team, case=False)]
                            if not players.empty:
                                players = players.nlargest(n_top_players, 'FP/G')
                                st.dataframe(
                                    players[['Player', 'Status', 'FP/G', 'FPts', 'GP']].style.format({
                                        'FP/G': '{:.1f}',
                                        'FPts': '{:.1f}',
                                        'GP': '{:.0f}'
                                    }).set_properties(**{
                                        'background-color': 'rgb(17, 23, 29)',
                                        'color': 'white'
                                    }),
                                    hide_index=True
                                )

@handle_error
def display_player_performance(player_name):
    """Display detailed performance analysis for a player"""
    st.write(f"## Player Analysis: {player_name}")
    
    # Analyze player statistics
    st.session_state.stats_analyzer.analyze_player_stats(player_name)

@handle_error
def analyze_player_stats(player_name):
    """Analyze and display player statistics"""
    if not st.session_state.data_ranges:
        st.error("No data available for analysis")
        return
        
    # Get available metrics
    sample_data = next(iter(st.session_state.data_ranges.values()))
    numeric_cols = sample_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Pre-select important metrics
    default_metrics = ['FPts', 'FP/G', 'GP']
    available_metrics = sorted(list(set(numeric_cols) - set(default_metrics)))
    
    # Select metrics
    selected_metrics = st.multiselect(
        "Select metrics to analyze",
        options=default_metrics + available_metrics,
        default=default_metrics,
        key="player_metric_selector"
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric to analyze")
        return
        
    # Gather player data across time ranges
    player_data = {}
    for time_range, data in st.session_state.data_ranges.items():
        player_stats = data[data['Player'] == player_name]
        if not player_stats.empty:
            player_data[time_range] = {
                metric: player_stats[metric].iloc[0]
                for metric in selected_metrics
                if metric in player_stats.columns
            }
    
    # Create and display performance plot
    fig = plot_performance_trends(player_data, selected_metrics, f"{player_name}'s Performance Trends")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed stats table
    st.write("### Detailed Statistics")
    stats_data = []
    for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
        if time_range in player_data:
            row_data = {'Time Range': time_range}
            row_data.update(player_data[time_range])
            stats_data.append(row_data)
    
    if stats_data:
        df = pd.DataFrame(stats_data)
        st.dataframe(df.style.format({
            col: '{:.1f}' for col in df.columns if col != 'Time Range'
        }))

@handle_error
def get_team_name(team_id):
    """Get full team name from team ID"""
    return TEAM_MAPPINGS.get(team_id, team_id)

@handle_error
def plot_performance_trends(data, selected_metrics, title="Performance Trends"):
    """Create a performance trend plot with visible data points"""
    fig = go.Figure()
    
    # Define colors for different metric groups
    colors = {
        'mean_fpg': '#00ffff',      # Cyan
        'median_fpg': '#00ff99',    # Mint green
        'std_fpg': '#ff6666',       # Light red
        'avg_gp': '#cc99ff',        # Light purple
        'total_fpts': '#ffcc00'     # Gold
    }
    
    # Group metrics by scale
    fpg_metrics = ['mean_fpg', 'median_fpg']
    std_metrics = ['std_fpg']
    gp_metrics = ['avg_gp']
    total_metrics = ['total_fpts']
    
    # Get time ranges and sort them
    time_ranges = list(data.keys())
    time_ranges.sort(key=lambda x: int(x.split()[0]), reverse=True)
    
    # Create traces for each metric
    for metric in selected_metrics:
        values = [data[tr][metric] for tr in time_ranges]
        
        # Determine which y-axis to use
        if metric in fpg_metrics:
            yaxis = 'y'
            showlegend = True
        elif metric in std_metrics:
            yaxis = 'y2'
            showlegend = True
        elif metric in gp_metrics:
            yaxis = 'y3'
            showlegend = True
        else:  # total_fpts
            yaxis = 'y4'
            showlegend = True
            
        # Create the trace
        fig.add_trace(
            go.Scatter(
                x=time_ranges,
                y=values,
                name=metric.replace('_', ' ').title(),
                line=dict(color=colors[metric]),
                yaxis=yaxis,
                showlegend=showlegend,
                mode='lines+markers+text',
                text=[f'{v:.1f}' for v in values],
                textposition='top center',
                textfont=dict(color=colors[metric])
            )
        )
    
    # Update layout with multiple y-axes
    fig.update_layout(
        title=title,
        plot_bgcolor='rgb(17, 23, 29)',
        paper_bgcolor='rgb(17, 23, 29)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        yaxis=dict(
            title='Fantasy Points per Game',
            titlefont=dict(color='#00ffff'),
            tickfont=dict(color='#00ffff'),
            gridcolor='rgba(128, 128, 128, 0.2)',
            side='left',
            showgrid=True
        ),
        yaxis2=dict(
            title='Standard Deviation',
            titlefont=dict(color='#ff6666'),
            tickfont=dict(color='#ff6666'),
            anchor='free',
            overlaying='y',
            side='right',
            position=1.0,
            showgrid=False
        ),
        yaxis3=dict(
            title='Games Played',
            titlefont=dict(color='#cc99ff'),
            tickfont=dict(color='#cc99ff'),
            anchor='free',
            overlaying='y',
            side='right',
            position=0.85,
            showgrid=False
        ),
        yaxis4=dict(
            title='Total Fantasy Points',
            titlefont=dict(color='#ffcc00'),
            tickfont=dict(color='#ffcc00'),
            anchor='free',
            overlaying='y',
            side='right',
            position=0.70,
            showgrid=False
        ),
        margin=dict(r=150)  # Add right margin for multiple y-axes
    )
    
    return fig

@handle_error
def get_all_teams():
    """Get a list of all teams from the data"""
    if not st.session_state.data_ranges:
        return []
    
    # Use the 60-day data as reference for teams
    for days in ['60', '30', '14', '7']:
        key = f'{days} Days'
        if key in st.session_state.data_ranges and st.session_state.data_ranges[key] is not None:
            return sorted(st.session_state.data_ranges[key]['Status'].unique())
    return []

@handle_error
def get_fairness_color(fairness_score: float) -> str:
    """Get color for fairness score visualization.
    
    Args:
        fairness_score (float): Score between 0 and 1
        
    Returns:
        str: Hex color code
    """
    if fairness_score >= 0.8:
        return '#2ecc71'  # Green
    elif fairness_score >= 0.6:
        return '#f1c40f'  # Yellow
    elif fairness_score >= 0.4:
        return '#e67e22'  # Orange
    else:
        return '#e74c3c'  # Red

@handle_error
def display_league_statistics():
    """Display league statistics analysis page"""
    st.write("## League Statistics Analysis")
    
    # Check if data is available
    if 'data' not in st.session_state or st.session_state.data.empty:
        st.error("No league data available. Please upload data files first.")
        return
    
    # Display overall league metrics
    st.write("### Overall League Metrics")
    league_data = st.session_state.data
    
    # Calculate league-wide statistics
    league_stats = {
        'Total Players': len(league_data),
        'Average FP/G': league_data['FP/G'].mean(),
        'Median FP/G': league_data['FP/G'].median(),
        'Standard Deviation': league_data['FP/G'].std(),
        'Total Fantasy Points': league_data['FPts'].sum(),
        'Average Games Played': league_data['GP'].mean()
    }
    
    # Display statistics
    stats_df = pd.DataFrame.from_dict(league_stats, orient='index', columns=['Value'])
    st.dataframe(
        stats_df.style.format({'Value': '{:.1f}'}).set_properties(**{
            'background-color': 'rgb(17, 23, 29)',
            'color': 'white'
        }),
        hide_index=False
    )
    
    # Plot distribution of FP/G
    st.write("### Distribution of Fantasy Points per Game")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=league_data['FP/G'],
        nbinsx=30,
        marker_color='rgba(0, 204, 150, 0.8)'
    ))
    fig.update_layout(
        title="FP/G Distribution",
        xaxis_title="Fantasy Points per Game",
        yaxis_title="Frequency",
        plot_bgcolor='rgb(17, 23, 29)',
        paper_bgcolor='rgb(17, 23, 29)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

@handle_error
def main():
    """Main application function"""
    st.title("Fantasy Basketball Trade Analyzer")
    
    # Initialize or load data if not already loaded
    if ('data_ranges' not in st.session_state or 
        st.session_state.data_ranges is None or 
        'data' not in st.session_state or 
        st.session_state.data is None or
        'trade_analyzer' not in st.session_state or
        st.session_state.trade_analyzer is None):
        
        data_ranges, player_data = load_data()
        st.session_state.data_ranges = data_ranges
        st.session_state.data = player_data
        
        # Initialize analyzers with the loaded data
        if data_ranges is not None and player_data is not None:
            st.session_state.trade_analyzer = TradeAnalyzer(player_data)
            st.session_state.stats_analyzer = StatisticalAnalyzer()
    
    if st.session_state.data_ranges is None or st.session_state.data is None:
        st.error("Failed to load data. Please check the data files and try again.")
        st.info("""
        Please ensure you have CSV files in the data directory with the correct naming format:
        - Files should end with (X).csv where X is the number of days
        - Example: stats-(7).csv, data-(14).csv, export-(30).csv, any-name-(60).csv
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.title("Fantasy Trade Analyzer")
        
        # Add debug mode toggle
        st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)
        
        # Navigation
        pages = {
            "Trade Analysis": display_trade_analysis_page,
            "Team Statistics": display_team_stats_analysis,
            "League Statistics": display_league_statistics
        }
        page = st.selectbox(
            "Navigation",
            list(pages.keys())
        )
    
    # Display selected page
    pages[page]()

if __name__ == "__main__":
    main()
