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
            # Extract the date range from filename
            for days in ['60', '30', '14', '7']:
                if f"({days})" in file.name:
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
                            df.reset_index(inplace=True)
                        
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

def display_team_comparison(before_data, after_data, team_name, top_x):
    """Display before/after comparison for a team in a condensed format"""
    st.write(f"#### {team_name} Analysis (Top {top_x} Players)")
    
    # Calculate stats for all time ranges
    before_stats = {}
    after_stats = {}
    available_ranges = [k for k, v in st.session_state.data_ranges.items() if v is not None]
    
    for time_range in available_ranges:
        before_stats[time_range] = calculate_team_stats(before_data, top_x)
        after_stats[time_range] = calculate_team_stats(after_data, top_x)
    
    # Display stats in a compact format
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Trade**")
        stats_df = pd.DataFrame({
            'Range': available_ranges,
            'Mean FP/G': [before_stats[r]['mean_fpg'] for r in available_ranges],
            'Median FP/G': [before_stats[r]['median_fpg'] for r in available_ranges],
            'Std Dev': [before_stats[r]['std_fpg'] for r in available_ranges],
            'Total FPts': [before_stats[r]['total_fpts'] for r in available_ranges],
            'Avg GP': [before_stats[r]['avg_gp'] for r in available_ranges]
        })
        st.dataframe(stats_df.style.format({
            'Mean FP/G': '{:.1f}',
            'Median FP/G': '{:.1f}',
            'Std Dev': '{:.1f}',
            'Total FPts': '{:.1f}',
            'Avg GP': '{:.1f}'
        }), hide_index=True)
    
    with col2:
        st.write("**After Trade**")
        stats_df = pd.DataFrame({
            'Range': available_ranges,
            'Mean FP/G': [after_stats[r]['mean_fpg'] for r in available_ranges],
            'Median FP/G': [after_stats[r]['median_fpg'] for r in available_ranges],
            'Std Dev': [after_stats[r]['std_fpg'] for r in available_ranges],
            'Total FPts': [after_stats[r]['total_fpts'] for r in available_ranges],
            'Avg GP': [after_stats[r]['avg_gp'] for r in available_ranges],
            'Mean Δ': [(after_stats[r]['mean_fpg'] - before_stats[r]['mean_fpg']) for r in available_ranges],
            'Median Δ': [(after_stats[r]['median_fpg'] - before_stats[r]['median_fpg']) for r in available_ranges],
            'Std Dev Δ': [(after_stats[r]['std_fpg'] - before_stats[r]['std_fpg']) for r in available_ranges],
            'Total FPts Δ': [(after_stats[r]['total_fpts'] - before_stats[r]['total_fpts']) for r in available_ranges],
            'Avg GP Δ': [(after_stats[r]['avg_gp'] - before_stats[r]['avg_gp']) for r in available_ranges]
        })
        st.dataframe(stats_df.style.format({
            'Mean FP/G': '{:.1f}',
            'Median FP/G': '{:.1f}',
            'Std Dev': '{:.1f}',
            'Total FPts': '{:.1f}',
            'Avg GP': '{:.1f}',
            'Mean Δ': '{:+.1f}',
            'Median Δ': '{:+.1f}',
            'Std Dev Δ': '{:+.1f}',
            'Total FPts Δ': '{:+.1f}',
            'Avg GP Δ': '{:+.1f}'
        }).background_gradient(
            subset=['Mean Δ', 'Median Δ', 'Total FPts Δ'],
            cmap='RdYlGn'
        ), hide_index=True)
    
    # Display top players before and after trade side by side
    st.markdown(
        """
        <div style='background-color: rgb(13, 17, 23); padding: 0.5rem; border-radius: 0.5rem; margin: 1rem 0 0.5rem 0;'>
            <h5 style='margin: 0; color: white;'>Top Players by FP/G</h5>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    for time_range in ['60 Days', '30 Days']:
        if time_range in team_data['before_stats'] and time_range in team_data['after_stats']:
            st.markdown(f"**{time_range}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Before Trade**")
                if 'top_players' in team_data['before_stats'] and time_range in team_data['before_stats']['top_players']:
                    before_players = team_data['before_stats']['top_players'][time_range]
                    if not before_players.empty:
                        display_df = before_players[['Player', 'FP/G', 'FPts', 'GP']].copy()
                        st.dataframe(
                            display_df.style.format({
                                'FP/G': '{:.1f}',
                                'FPts': '{:.1f}',
                                'GP': '{:.0f}'
                            }).set_properties(**{
                                'background-color': 'rgb(17, 23, 29)',
                                'color': 'white',
                                'font-size': '14px'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [('background-color', 'rgb(13, 17, 23)'), ('color', 'white')]},
                                {'selector': 'td', 'props': [('border', '1px solid rgb(38, 39, 48)')]}
                            ]).bar(subset=['FP/G'], color='#1f77b4'),
                            hide_index=True,
                            height=400
                        )
            
            with col2:
                st.markdown("**After Trade**")
                if 'top_players' in team_data['after_stats'] and time_range in team_data['after_stats']['top_players']:
                    after_players = team_data['after_stats']['top_players'][time_range]
                    if not after_players.empty:
                        display_df = after_players[['Player', 'FP/G', 'FPts', 'GP']].copy()
                        st.dataframe(
                            display_df.style.format({
                                'FP/G': '{:.1f}',
                                'FPts': '{:.1f}',
                                'GP': '{:.0f}'
                            }).set_properties(**{
                                'background-color': 'rgb(17, 23, 29)',
                                'color': 'white',
                                'font-size': '14px'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [('background-color', 'rgb(13, 17, 23)'), ('color', 'white')]},
                                {'selector': 'td', 'props': [('border', '1px solid rgb(38, 39, 48)')]}
                            ]).bar(subset=['FP/G'], color='#1f77b4'),
                            hide_index=True,
                            height=400
                        )
    
    return before_stats, after_stats

def display_trade_analysis(analysis, teams):
    """Display the trade analysis results"""
    st.write("## Trade Analysis")
    
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
    
    # Create columns for each team
    cols = st.columns(len(teams))
    
    # Display trade details for each team
    for team, col in zip(teams, cols):
        with col:
            team_data = analysis[team]
            fairness = team_data['fairness_score']
            
            # Team header with fairness score
            st.markdown(
                f"""
                <div style='background-color: rgb(17, 23, 29); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                    <h3 style='margin: 0; color: white;'>{get_team_name(team)}</h3>
                    <div style='display: flex; align-items: center; margin-top: 0.5rem;'>
                        <div style='flex-grow: 1; background-color: rgb(38, 39, 48); height: 0.5rem; border-radius: 0.25rem; overflow: hidden;'>
                            <div style='width: {fairness * 100}%; height: 100%; background-color: {get_fairness_color(fairness)};'></div>
                        </div>
                        <span style='margin-left: 0.5rem; color: white;'>{fairness:.2%}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display detailed statistics tables
            st.markdown(
                """
                <div style='background-color: rgb(17, 23, 29); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                    <h4 style='margin: 0; color: white;'>Team Statistics</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Create before/after trade stats tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Before Trade")
                if team_data['before_stats']:
                    # Display before trade stats
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
                    
                    df_before = pd.DataFrame(before_stats)
                    st.dataframe(
                        df_before.style.format({
                            'Mean FP/G': '{:.1f}',
                            'Median FP/G': '{:.1f}',
                            'Std Dev': '{:.1f}',
                            'Total FPts': '{:.1f}',
                            'Avg GP': '{:.1f}'
                        }).set_properties(**{
                            'background-color': 'rgb(22, 27, 34)',
                            'color': 'white',
                            'font-size': '14px',
                            'border': '1px solid rgb(48, 54, 61)'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [
                                ('background-color', 'rgb(13, 17, 23)'),
                                ('color', 'white'),
                                ('font-weight', 'bold'),
                                ('text-align', 'left'),
                                ('padding', '8px')
                            ]},
                            {'selector': 'td', 'props': [
                                ('padding', '8px')
                            ]}
                        ]),
                        hide_index=True,
                        height=200
                    )
                    
                    # Display players included in before analysis
                    if '60 Days' in team_data['top_players_before']:
                        st.markdown("##### Players Included (Before)")
                        players_before = team_data['top_players_before']['60 Days']
                        df_players_before = players_before[['Player', 'FP/G', 'GP']].copy()
                        st.dataframe(
                            df_players_before.style.format({
                                'FP/G': '{:.1f}',
                                'GP': '{:.0f}'
                            }).set_properties(**{
                                'background-color': 'rgb(22, 27, 34)',
                                'color': 'white',
                                'font-size': '14px',
                                'border': '1px solid rgb(48, 54, 61)'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [
                                    ('background-color', 'rgb(13, 17, 23)'),
                                    ('color', 'white'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'left'),
                                    ('padding', '8px')
                                ]},
                                {'selector': 'td', 'props': [
                                    ('padding', '8px')
                                ]}
                            ]).bar(subset=['FP/G'], color='#1f77b4'),
                            hide_index=True
                        )
            
            with col2:
                st.markdown("##### After Trade")
                if team_data['after_stats']:
                    # Display after trade stats
                    after_stats = []
                    for time_range in ['60 Days', '30 Days', '14 Days', '7 Days']:
                        if time_range in team_data['after_stats']:
                            stats = team_data['after_stats'][time_range]
                            after_stats.append({
                                'Time Range': time_range,
                                'Mean FP/G': stats['mean_fpg'],
                                'Median FP/G': stats['median_fpg'],
                                'Std Dev': stats['std_fpg'],
                                'Total FPts': stats['total_fpts'],
                                'Avg GP': stats['avg_gp']
                            })
                    
                    df_after = pd.DataFrame(after_stats)
                    st.dataframe(
                        df_after.style.format({
                            'Mean FP/G': '{:.1f}',
                            'Median FP/G': '{:.1f}',
                            'Std Dev': '{:.1f}',
                            'Total FPts': '{:.1f}',
                            'Avg GP': '{:.1f}'
                        }).set_properties(**{
                            'background-color': 'rgb(22, 27, 34)',
                            'color': 'white',
                            'font-size': '14px',
                            'border': '1px solid rgb(48, 54, 61)'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [
                                ('background-color', 'rgb(13, 17, 23)'),
                                ('color', 'white'),
                                ('font-weight', 'bold'),
                                ('text-align', 'left'),
                                ('padding', '8px')
                            ]},
                            {'selector': 'td', 'props': [
                                ('padding', '8px')
                            ]}
                        ]),
                        hide_index=True,
                        height=200
                    )
                    
                    # Display players included in after analysis
                    if '60 Days' in team_data['top_players_after']:
                        st.markdown("##### Players Included (After)")
                        players_after = team_data['top_players_after']['60 Days']
                        df_players_after = players_after[['Player', 'FP/G', 'GP']].copy()
                        st.dataframe(
                            df_players_after.style.format({
                                'FP/G': '{:.1f}',
                                'GP': '{:.0f}'
                            }).set_properties(**{
                                'background-color': 'rgb(22, 27, 34)',
                                'color': 'white',
                                'font-size': '14px',
                                'border': '1px solid rgb(48, 54, 61)'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [
                                    ('background-color', 'rgb(13, 17, 23)'),
                                    ('color', 'white'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'left'),
                                    ('padding', '8px')
                                ]},
                                {'selector': 'td', 'props': [
                                    ('padding', '8px')
                                ]}
                            ]).bar(subset=['FP/G'], color='#1f77b4'),
                            hide_index=True
                        )
            
            # Display trade impact summary
            st.markdown(
                """
                <div style='background-color: rgb(17, 23, 29); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                    <h4 style='margin: 0; color: white;'>Trade Impact Summary</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display incoming/outgoing players
            incoming = team_data['incoming_players']
            outgoing = team_data['outgoing_players']
            
            if incoming:
                st.write("📥 **Receiving:**")
                for player in incoming:
                    value = st.session_state.trade_analyzer._calculate_player_value(player)
                    st.markdown(
                        f"""
                        <div style='background-color: rgb(22, 27, 34); padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.25rem;'>
                            <span style='color: #4CAF50;'>{player}</span>
                            <span style='float: right; color: white;'>{value:.1f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            if outgoing:
                st.write("📤 **Sending:**")
                for player in outgoing:
                    value = st.session_state.trade_analyzer._calculate_player_value(player)
                    st.markdown(
                        f"""
                        <div style='background-color: rgb(22, 27, 34); padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.25rem;'>
                            <span style='color: #F44336;'>{player}</span>
                            <span style='float: right; color: white;'>{value:.1f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Display value change
            value_change = float(team_data['value_change'])
            st.markdown(
                f"""
                <div style='background-color: rgb(17, 23, 29); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                    <h4 style='margin: 0; color: white;'>Net Value Change</h4>
                    <p style='margin: 0.5rem 0; color: {"#4CAF50" if value_change > 0 else "#F44336"}; font-size: 1.2rem;'>
                        {'+'if value_change > 0 else ''}{value_change:.1f} points
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display statistical comparison
            st.markdown(
                """
                <div style='background-color: rgb(17, 23, 29); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                    <h4 style='margin: 0; color: white;'>Statistical Impact</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Create DataFrame for before/after comparison
            stats_comparison = []
            for time_range in ['7 Days', '14 Days', '30 Days', '60 Days']:
                if (time_range in team_data['before_stats'] and 
                    time_range in team_data['after_stats']):
                    
                    before = team_data['before_stats'][time_range]
                    after = team_data['after_stats'][time_range]
                    
                    stats_comparison.append({
                        'Time Range': time_range,
                        'FP/G Before': before['mean_fpg'],
                        'FP/G After': after['mean_fpg'],
                        'FP/G Δ': after['mean_fpg'] - before['mean_fpg'],
                        'Total Before': before['total_fpts'],
                        'Total After': after['total_fpts'],
                        'Total Δ': after['total_fpts'] - before['total_fpts']
                    })
            
            if stats_comparison:
                df = pd.DataFrame(stats_comparison)
                st.dataframe(
                    df.style.format({
                        'FP/G Before': '{:.1f}',
                        'FP/G After': '{:.1f}',
                        'FP/G Δ': '{:+.1f}',
                        'Total Before': '{:.0f}',
                        'Total After': '{:.0f}',
                        'Total Δ': '{:+.0f}'
                    }).background_gradient(
                        subset=['FP/G Δ', 'Total Δ'],
                        cmap='RdYlGn'
                    ),
                    hide_index=True
                )

def get_fairness_color(score):
    """Get color for fairness score"""
    if score >= 0.8:
        return '#4CAF50'  # Green
    elif score >= 0.6:
        return '#8BC34A'  # Light Green
    elif score >= 0.4:
        return '#FFC107'  # Amber
    elif score >= 0.2:
        return '#FF9800'  # Orange
    else:
        return '#F44336'  # Red

def display_trade_analysis_page():
    """Display the trade analysis page"""
    st.write("## Trade Analysis")
    
    # Set current data based on selected range
    st.session_state.data = st.session_state.data_ranges['60 Days']
    
    # Setup and analyze trade
    trade_setup()

def trade_setup():
    """Setup the trade with drag and drop interface"""
    st.write("## Trade Setup")
    
    # Get all teams and sort them
    teams = get_all_teams()
    teams.sort()
    
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
            
            # Multi-select for players
            selected_players = st.multiselect(
                f"Select players from {team_name}",
                available_players,
                key=f"{team}_players"
            )
            
            trade_teams[team] = selected_players

    # Only show player assignment section if teams have selected players
    active_teams = {team: players for team, players in trade_teams.items() if players}
    
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
            analysis = st.session_state.trade_analyzer.evaluate_trade_fairness(trade_teams, top_x)
            display_trade_analysis(analysis, selected_teams)

def display_team_stats_analysis():
    """Display team statistics analysis page"""
    st.write("## Team Statistics Analysis")
    
    # Get available data ranges
    available_ranges = [k for k, v in st.session_state.data_ranges.items() if v is not None]
    
    # Initialize session state for persistent values if not exists
    if 'top_n_players' not in st.session_state:
        st.session_state.top_n_players = 10
    
    # Select team to analyze
    team = st.selectbox(
        "Select Team to Analyze",
        options=sorted(st.session_state.data['Status'].unique()),
        help="Choose a team to analyze their statistics"
    )
    
    # Number input for top players
    n_top_players = st.number_input(
        "Number of top players to analyze",
        min_value=1,
        max_value=20,
        value=st.session_state.top_n_players,
        help="Select how many top players to show in the analysis"
    )
    st.session_state.top_n_players = n_top_players
    
    # Analyze team statistics
    st.session_state.stats_analyzer.analyze_team_stats(team, n_top_players)

def display_player_performance(player_name):
    """Display detailed performance analysis for a player"""
    st.write(f"## Player Analysis: {player_name}")
    
    # Analyze player statistics
    st.session_state.stats_analyzer.analyze_player_stats(player_name)

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

def get_team_name(team_id):
    """Get full team name from team ID"""
    return TEAM_MAPPINGS.get(team_id, team_id)

def plot_performance_trends(data, selected_metrics, title="Performance Trends"):
    """Create a performance trend plot with visible data points"""
    fig = go.Figure()
    time_ranges = ['60 Days', '30 Days', '14 Days', '7 Days']
    x_positions = list(range(len(time_ranges)))
    
    for metric in selected_metrics:
        values = []
        hover_texts = []
        
        for time_range in time_ranges:
            if time_range in data:
                value = data[time_range].get(metric)
                values.append(value)
                hover_texts.append(f"{time_range}:<br>{metric}: {value:.1f}")
            else:
                values.append(None)
                hover_texts.append(f"No data for {time_range}")
        
        # Add both line and markers
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=values,
            name=metric,
            mode='lines+markers',  # Explicitly show both lines and markers
            line=dict(width=2),
            marker=dict(
                size=10,
                symbol='circle',
                line=dict(width=2, color='white')  # Add white border to markers
            ),
            hovertext=hover_texts,
            hoverinfo='text'
        ))
    
    fig.update_layout(
        xaxis=dict(
            ticktext=time_ranges,
            tickvals=x_positions,
            title="Time Range"
        ),
        yaxis=dict(title="Value"),
        title=title,
        hovermode='x unified',
        showlegend=True,
        height=400,  # Reduced height for more compact display
        margin=dict(l=50, r=50, t=50, b=50)  # Reduced margins
    )
    
    return fig

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

def main():
    """Main application function"""
    st.title("Fantasy Basketball Trade Analyzer")
    
    # Initialize or load data if not already loaded
    if ('data_ranges' not in st.session_state or 
        st.session_state.data_ranges is None or 
        'data' not in st.session_state or 
        st.session_state.data is None):
        
        data_ranges, player_data = load_data()
        st.session_state.data_ranges = data_ranges
        st.session_state.data = player_data
        
        # Initialize analyzers with the loaded data
        if player_data is not None:
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
    
    # Create navigation
    pages = {
        "Trade Analysis": display_trade_analysis_page,
        "Team Statistics": display_team_stats_analysis
    }
    
    page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page]()

if __name__ == "__main__":
    main()
