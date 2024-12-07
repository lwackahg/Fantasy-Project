"""Fantasy Basketball Trade Analyzer Application.

This module serves as the main entry point for the Fantasy Basketball Trade Analyzer.
It provides a comprehensive interface for analyzing fantasy basketball trades and team performance.

Key Features:
- Trade Analysis: Evaluate the fairness and impact of player trades
- Team Statistics: Analyze team performance across multiple metrics
- Player Performance: Track individual player trends and statistics
- League Analytics: View league-wide statistics and rankings

The application uses Streamlit for the user interface and integrates with various
data sources to provide up-to-date player statistics and analysis.

Dependencies:
    - streamlit: Web application framework
    - pandas: Data manipulation and analysis
    - plotly: Interactive data visualization
    - numpy: Numerical computations
"""

# Standard library imports
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from collections import Counter

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np

# Local application imports
from data_import import DataImporter
from trade_analysis import TradeAnalyzer
from statistical_analysis import StatisticalAnalyzer
from config.team_mappings import TEAM_MAPPINGS
from utils.visualization_utils import get_trend_color, get_fairness_color, plot_performance_trends, plot_player_trend
from utils.analysis_utils import calculate_team_stats, calculate_player_value, calculate_trade_fairness

# Application Configuration Constants
PAGE_TITLE: str = "Fantasy Basketball Trade Analyzer"
PAGE_ICON: str = "🏀"
LAYOUT: str = "wide"
INITIAL_SIDEBAR_STATE: str = "expanded"

# Data Processing Constants
REQUIRED_COLUMNS: List[str] = ['Player', 'Team', 'FP/G', 'Status']
NUMERIC_COLUMNS: List[str] = [
    'FPts', 'FP/G', 'PTS', 'OREB', 'DREB', 'REB', 
    'AST', 'STL', 'BLK', 'TOV'
]

# Analysis Weight Constants
TIME_RANGES: Dict[str, float] = {
    '7 Days': 0.4,   # Most recent data weighted highest
    '14 Days': 0.3,
    '30 Days': 0.2,
    '60 Days': 0.1   # Oldest data weighted lowest
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from config.team_mappings import TEAM_MAPPINGS
# Import custom modules
from data_import import DataImporter
from trade_analysis import TradeAnalyzer
from statistical_analysis import StatisticalAnalyzer
from visualization import DataVisualizer

# Initialize Streamlit session state variables for application state management.
def init_session_state():
    """
    Initialize session state variables for the application.
    
    This function sets up the initial state for the application, including:
    - Data storage for player statistics across different time ranges
    - Trade history tracking
    - Debug mode settings
    - Analysis tool instances
    
    The session state persists across reruns of the app, ensuring data consistency
    during user interactions.
    """
    if 'data_importer' not in st.session_state:
        st.session_state.data_importer = DataImporter()
    if 'data_ranges' not in st.session_state:
        st.session_state.data_ranges = {}
    if 'current_range' not in st.session_state:
        st.session_state.current_range = None
    if 'stats_analyzer' not in st.session_state:
        st.session_state.stats_analyzer = StatisticalAnalyzer()
    if 'trade_analyzer' not in st.session_state:
        st.session_state.trade_analyzer = TradeAnalyzer()  # Initialize TradeAnalyzer   
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []
    if 'DataVisualizer' not in st.session_state:
        st.session_state.DataVisualizer = DataVisualizer
def handle_file_upload():
    """Handle file upload and data processing"""
    uploaded_files = st.file_uploader(
        "Upload your player statistics CSV files",
        type='csv',
        accept_multiple_files=True,
        help="Upload CSV files with player statistics. Files should be named with the format: 'name-(days).csv'"
    )
    
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            # Save uploaded files temporarily
            file_paths = []
            for file in uploaded_files:
                temp_path = Path(file.name)
                temp_path.write_bytes(file.getvalue())
                file_paths.append(str(temp_path))
                
            try:
                # Import the files
                data_ranges = st.session_state.data_importer.import_multiple_files(file_paths)
                
                if data_ranges:
                    st.session_state.data_ranges = data_ranges
                    st.success(f"Successfully loaded data for {len(data_ranges)} time ranges!")
                    
                    # Initialize visualizer with the first available range
                    first_range = next(iter(data_ranges.values()))
                    st.session_state.visualizer = DataVisualizer(first_range)
                    
                    if st.session_state.debug_mode:
                        st.write("### Loaded Data Ranges")
                        for range_key, data in data_ranges.items():
                            st.write(f"- {range_key}: {len(data)} players")
                    
                else:
                    st.error("No valid data was found in the uploaded files.")
                    
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                logging.error(f"File processing error: {str(e)}")
                
            finally:
                # Clean up temporary files
                for file_path in file_paths:
                    try:
                        Path(file_path).unlink()
                    except Exception as e:
                        logging.error(f"Error cleaning up {file_path}: {str(e)}")

def select_time_range():
    """Allow user to select time range for analysis"""
    if st.session_state.data_ranges:
        available_ranges = st.session_state.data_importer.get_available_ranges()
        selected_range = st.selectbox(
            "Select Time Range for Analysis",
            available_ranges,
            index=0 if available_ranges else None,
            help="Choose the time range for player statistics analysis"
        )
        
        if selected_range and selected_range != st.session_state.current_range:
            st.session_state.current_range = selected_range
            data = st.session_state.data_ranges[selected_range]
            st.session_state.visualizer = DataVisualizer(data)
            
            # Display data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Players", len(data))
            with col2:
                st.metric("Teams", data['Team'].nunique())
            with col3:
                st.metric("Avg FP/G", f"{data['FP/G'].mean():.1f}")
            
            if st.session_state.debug_mode:
                with st.expander("View Data Details"):
                    st.dataframe(data.describe())

def load_csv_file(file: Path) -> Optional[pd.DataFrame]:
    """Load and process a single CSV file containing player statistics.
    
    Args:
        file (Path): Path to the CSV file containing player data
        
    Returns:
        Optional[pd.DataFrame]: Processed DataFrame if successful, None if an error occurs
        
    The function performs the following operations:
    1. Reads the CSV file into a pandas DataFrame
    2. Converts numeric columns to appropriate data types
    3. Validates required columns are present
    4. Handles missing values and data cleanup
    """
    try:
        df = pd.read_csv(file)
        
        # Convert numeric columns to appropriate types
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate required columns
        if not all(col in df.columns for col in REQUIRED_COLUMNS):
            logger.error(f"Missing required columns in {file}")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file}: {str(e)}")
        return None

def extract_days_from_filename(filename: str) -> Optional[str]:
    """Extract the time range information from a data file's name.
    
    Args:
        filename (str): Name of the file to parse
        
    Returns:
        Optional[str]: Time range (e.g., '7 Days', '30 Days') or None if not found
        
    Example:
        >>> extract_days_from_filename("player_stats_7days.csv")
        '7 Days'
    """
    import re
    
    # Extract number of days from filename using regex
    match = re.search(r'(\d+)(?:days|Days)', filename)
    if match:
        days = match.group(1)
        return f"{days} Days"
    return None

def load_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
    """Load and process player data from all available CSV files.
    
    Returns:
        Tuple containing:
        - Dict[str, pd.DataFrame]: Mapping of time ranges to player DataFrames
        - Dict[str, Dict[str, pd.DataFrame]]: Mapping of players to their data across time ranges
        
    This function:
    1. Scans the data directory for CSV files
    2. Loads and validates each file
    3. Organizes data by time range
    4. Creates player-specific data mappings
    5. Updates the session state with the loaded data
    
    Raises:
        FileNotFoundError: If the data directory doesn't exist
        ValueError: If no valid data files are found
    """
    try:
        data_dir = Path("data")
        if not data_dir.exists():
            raise FileNotFoundError("Data directory not found")
            
        data_ranges = {}
        player_data = {}
        
        # Load each CSV file and organize by time range
        for file in data_dir.glob("*.csv"):
            time_range = extract_days_from_filename(file.name)
            if not time_range:
                continue
                
            df = load_csv_file(file)
            if df is not None:
                data_ranges[time_range] = df
                
                # Organize data by player
                for player in df['Player'].unique():
                    if player not in player_data:
                        player_data[player] = {}
                    player_data[player][time_range] = df[df['Player'] == player]
        
        if not data_ranges:
            raise ValueError("No valid data files found")
            
        # Update session state
        st.session_state.data_ranges = data_ranges
        return data_ranges, player_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error("Failed to load player data. Please check the data directory and file format.")
        return {}, {}

def handle_error(func):
    """Decorator for consistent error handling across the application.
    
    This decorator wraps functions to provide uniform error handling by:
    1. Catching and logging exceptions
    2. Displaying user-friendly error messages
    3. Maintaining application state in case of errors
    
    Args:
        func: The function to be wrapped with error handling
        
    Returns:
        Wrapped function that includes error handling
        
    Example:
        @handle_error
        def my_function():
            # Function code here
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)
    return wrapper

@handle_error
def display_league_statistics() -> None:
    """Display league-wide statistics analysis."""
    st.title("League Statistics")
    
    if not st.session_state.data_ranges:
        st.error("Please load data first")
        return
    
    # Get the most recent data
    recent_data = st.session_state.data_ranges['7 Days']
    
    # Calculate league-wide statistics
    league_stats = calculate_team_stats(recent_data)
    
    st.markdown("### League Overview")
    st.write(f"Average FP/G: {league_stats['avg_fp']:.2f}")
    st.write(f"Median FP/G: {league_stats['median_fp']:.2f}")
    st.write(f"Standard Deviation: {league_stats['std_fp']:.2f}")
    
    # Display team rankings
    st.markdown("### Team Rankings")
    display_team_rankings(recent_data)

@handle_error
def display_team_rankings(data: pd.DataFrame) -> None:
    """Display team rankings based on various metrics.
    
    Args:
        data: DataFrame containing team data
    """
    team_stats = {}
    
    for team in data['Team'].unique():
        if team == 'FA':  # Skip free agents
            continue
        team_data = data[data['Team'] == team]
        team_stats[team] = calculate_team_stats(team_data)
    
    # Create rankings DataFrame
    rankings = pd.DataFrame.from_dict(team_stats, orient='index')
    rankings = rankings.sort_values('avg_fp', ascending=False)
    
    # Display rankings
    st.dataframe(
        rankings[['avg_fp', 'total_fp', 'depth']]
        .rename(columns={
            'avg_fp': 'Avg FP/G',
            'total_fp': 'Total Points',
            'depth': 'Roster Size'
        })
        .style.format({
            'Avg FP/G': '{:.2f}',
            'Total Points': '{:.0f}',
            'Roster Size': '{:.0f}'
        })
    )

@handle_error
def display_trade_analysis_page() -> None:
    """Display the trade analysis page"""
    st.write("## Trade Analysis")
    
    # Set current data based on selected range
    st.session_state.data = st.session_state.data_ranges['60 Days']
    
    # Setup and analyze trade
    trade_setup()
    
    # Display trade history
    st.write("## Trade Analysis History")
    if st.session_state.trade_analyzer:
        history = st.session_state.trade_analyzer.get_trade_history()
        for trade, blurb in history:
            st.write(blurb)
    else:
        st.write("No trade history available.")

@handle_error
def trade_setup() -> None:
    """Setup the trade with drag and drop interface"""
    st.write("## Trade Setup")
    
    # Display team legend with improved styling
    st.write("## Team Legend")
    team_legend = """
    <style>
    .team-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
    }
    .team-card {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 15px;
        text-align: center;
        width: 200px;
        transition: transform 0.2s, box-shadow 0.2s;
        color: #333;
    }
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    </style>
    <div class="team-legend">
    """
    for team_id, team_name in TEAM_MAPPINGS.items():
        team_legend += f"<div class='team-card'><strong>{team_id}</strong><br>{team_name}</div>"
    team_legend += "</div>"
    st.markdown(team_legend, unsafe_allow_html=True)

    # Adjust layout for better UX
    st.write("## Analysis Settings")
    st.write("### Number of Top Players to Analyze")
    top_x = st.number_input("Select the number of top players to analyze", min_value=1, max_value=15, value=10, help="Choose how many top players to include in the analysis.")

    st.write("### Select Teams to Trade Between")
    selected_teams = st.multiselect(
        "Choose teams involved in the trade",
        options=get_all_teams(),
        help="Select the teams that will participate in the trade.",
        format_func=lambda x: get_team_name(x)
    )

    if not selected_teams:
        st.warning("Please select teams to begin trading.")
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
            
            # Update trade history with formatted analysis data
            for team, team_analysis in analysis.items():
                incoming = ', '.join(team_analysis['incoming_players'])
                outgoing = ', '.join(team_analysis['outgoing_players'])
                net_value_change = team_analysis['value_change']
                blurb = (
                    f"Trade Impact for {team}:\n"
                    f"Receiving: {incoming}\n"
                    f"Trading Away: {outgoing}\n"
                    f"Net Value Change: {net_value_change}"
                )
                st.session_state.trade_analyzer.trade_history.append((team, blurb))
                if len(st.session_state.trade_analyzer.trade_history) > 25:
                    st.session_state.trade_analyzer.trade_history.pop(0)

@handle_error
def display_team_stats_analysis() -> None:
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
    n_top_players = st.number_input("Number of top players to analyze", min_value=1, max_value=14, value=9)
    
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
                team_data = data[data['Status'].str.contains(team, case=False)].copy()  # Create a copy explicitly
                if not team_data.empty:
                    stats[time_range] = calculate_team_stats(team_data, n_top_players)
        
        if stats:
            # Call the plot_performance_trends function with the properly structured stats
            fig = plot_performance_trends(stats, selected_metrics, f"{team} Performance Trends")
            st.plotly_chart(fig, use_container_width=False)
        
        # Optionally, you can add any additional display logic here if needed
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
        
       # Display trend lines for the top players
        st.write(f"### Performance Trends for Top {n_top_players} Players")
        for time_range in st.session_state.data_ranges.keys():  # Use keys from data_ranges dynamically
            if time_range in stats:
                if stats:
                    st.write("Stats data before plotting:", stats)
                with st.expander(f"{time_range} Performance Trends"):
                    team_data = st.session_state.data_ranges[time_range]
                    if team_data is not None:
                        players = team_data[team_data['Status'].str.contains(team, case=False)]
                        if not players.empty:
                            # Select the top n_top_players
                            top_players = players.nlargest(n_top_players, 'FP/G')['Player'].tolist()

                            # Iterate over the top players to display their data
                            for top_player in top_players:
                                player_data = team_data[team_data['Player'] == top_player]

                                st.write(f"Player Data for {top_player}:")
                                st.dataframe(player_data[['FP/G', 'FPts', 'GP']])

                                if not player_data.empty:
                                    # Prepare data for plotting
                                    player_metrics = player_data[['FP/G', 'FPts', 'GP']].copy()

                                    # Create a dictionary for plotting
                                    player_metrics_dict = {
                                    'mean_fpg': player_metrics['FP/G'].tolist(),  # You might want to check if it's the correct mapping
                                    'median_fpg': player_metrics['FP/G'].tolist(),  # Adjust as applicable
                                    'std_fpg': [],  # Placeholder if you don't have this data
                                    'avg_gp': player_metrics['GP'].tolist(),
                                    'total_fpts': player_metrics['FPts'].tolist()
                                }

                                    # Ensure metrics data is complete for plotting
                                    if all(len(player_metrics_dict[key]) > 0 for key in player_metrics_dict):
                                        try:
                                            # Now include selected metrics in the call
                                            fig = plot_performance_trends({top_player: player_metrics_dict}, selected_metrics, f"{top_player} Performance Trends")
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"An error occurred while plotting: {e}")
                                    else:
                                        st.warning("Metrics data is incomplete for plotting.")
                                else:
                                    st.warning("Metrics data is incomplete for plotting.")
                        else:
                            st.warning("No data available for the selected players.")
                    else:
                        st.error("No team data available for this time range.")
                
@handle_error
def display_player_performance(player_name: str) -> None:
    """Display detailed performance analysis for a player"""
    st.write(f"## Player Analysis: {player_name}")
    
    # Analyze player statistics
    st.session_state.stats_analyzer.analyze_player_stats(player_name)

@handle_error
def analyze_player_stats(player_name: str) -> None:
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
def get_team_name(team_id: str) -> str:
    """Get full team name from team ID"""
    return TEAM_MAPPINGS.get(team_id, team_id)

@handle_error
def get_all_teams() -> List[str]:
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
def main() -> None:
    """Main application entry point"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE
    )
    
    st.title(PAGE_TITLE)
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for data loading and configuration
    with st.sidebar:
        st.header("Data Configuration")
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox(
            "Debug Mode",
            value=st.session_state.debug_mode,
            help="Enable detailed logging and data inspection"
        )
        
        # File upload section
        st.subheader("Data Import")
        handle_file_upload()
        
        # Time range selection
        if st.session_state.data_ranges:
            st.subheader("Analysis Configuration")
            select_time_range()
            
            st.write("### Available Time Ranges")
            for range_key in st.session_state.data_importer.get_available_ranges():
                st.write(f"- {range_key}")
    
    # Main content area
    if not st.session_state.data_ranges:
        st.info("👈 Start by uploading your player statistics CSV files in the sidebar!")
        st.write("""
        ### File Format Requirements:
        1. Files should be named with the format: `name-(days).csv`
        2. Supported time ranges: 7, 14, 30, and 60 days
        3. Required columns: Player, Team, Position, Status, FPts, FP/G, MIN
        
        ### Example Files:
        - `stats-(7).csv`  - Last 7 days of player stats
        - `players-(14).csv` - Last 14 days of player stats
        - `fantasy-(30).csv` - Last 30 days of player stats
        - `data-(60).csv` - Last 60 days of player stats
        """)
    else:
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["Trade Analysis", "Team Analysis", "Player Analysis"])
        
        with tab1:
            display_trade_analysis_page()
            
        with tab2:
            display_team_stats_analysis()
            
        with tab3:
            st.subheader("Player Analysis")
            player_name = st.selectbox(
                "Select a player to analyze",
                sorted(st.session_state.data_ranges[st.session_state.current_range]['Player'].unique())
            )
            if player_name:
                analyze_player_stats(player_name)

if __name__ == "__main__":
    main()
