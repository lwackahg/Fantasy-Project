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
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Local application imports
from data_import import DataImporter
from trade_analysis import TradeAnalyzer
from statistical_analysis import StatisticalAnalyzer
from config.team_mappings import TEAM_MAPPINGS

# Application Configuration Constants
PAGE_TITLE: str = "Fantasy Basketball Trade Analyzer"
PAGE_ICON: str = "🏀"
LAYOUT: str = "wide"
INITIAL_SIDEBAR_STATE: str = "expanded"

# Data Processing Constants
REQUIRED_COLUMNS: List[str] = ['Player', 'Team', 'FP/G']
NUMERIC_COLUMNS: List[str] = [
    'FPts', 'FP/G', 'PTS', 'OREB', 'DREB', 'REB', 
    'AST', 'STL', 'BLK', 'TOV', 'MIN'
]

# Analysis Weight Constants
TIME_RANGES: Dict[str, float] = {
    '7 Days': 0.4,   # Most recent data weighted highest
    '14 Days': 0.3,
    '30 Days': 0.2,
    '60 Days': 0.1   # Oldest data weighted lowest
}

# UI Color Constants
ERROR_COLOR: str = "#e74c3c"     # Red for errors and negative trends
SUCCESS_COLOR: str = "#2ecc71"   # Green for success and positive trends
WARNING_COLOR: str = "#f39c12"   # Yellow for warnings and neutral trends
NEUTRAL_COLOR: str = "#95a5a6"   # Gray for neutral or inactive states

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

def init_session_state() -> None:
    """Initialize Streamlit session state variables for application state management.
    
    This function sets up the initial state for the application, including:
    - Data storage for player statistics across different time ranges
    - Trade history tracking
    - Debug mode settings
    - Analysis tool instances
    
    The session state persists across reruns of the app, ensuring data consistency
    during user interactions.
    """
    defaults = {
        'data_ranges': None,      # Stores player data across different time periods
        'data': None,             # Current active dataset
        'trade_history': [],      # List of previously analyzed trades
        'debug_mode': False,      # Toggle for additional debugging information
        'trade_analyzer': None,   # Instance of TradeAnalyzer class
        'stats_analyzer': StatisticalAnalyzer()  # Instance for statistical calculations
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

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
def calculate_team_stats(team_data: pd.DataFrame, top_x: Optional[int] = None) -> Dict[str, float]:
    """Calculate comprehensive statistics for a team's roster.
    
    This function analyzes team performance by calculating various statistical
    measures including scoring, efficiency, and consistency metrics.
    
    Args:
        team_data (pd.DataFrame): DataFrame containing team player statistics
        top_x (Optional[int]): Number of top players to consider (e.g., top 8)
        
    Returns:
        Dict[str, float]: Dictionary containing calculated statistics:
            - avg_fp: Average fantasy points per game
            - total_fp: Total fantasy points
            - consistency: Measure of scoring consistency
            - efficiency: Points per minute played
            - depth: Roster depth score
            
    Example:
        >>> team_df = pd.DataFrame(...)
        >>> stats = calculate_team_stats(team_df, top_x=8)
        >>> print(stats['avg_fp'])
    """
    if team_data.empty:
        return {
            'avg_fp': 0.0,
            'total_fp': 0.0,
            'consistency': 0.0,
            'efficiency': 0.0,
            'depth': 0.0
        }
    
    # Sort by fantasy points per game
    team_data = team_data.sort_values('FP/G', ascending=False)
    
    # Consider only top X players if specified
    if top_x is not None:
        team_data = team_data.head(top_x)
    
    # Calculate basic statistics
    stats = {
        'avg_fp': team_data['FP/G'].mean(),
        'total_fp': team_data['FPts'].sum(),
        'consistency': team_data['FP/G'].std() / team_data['FP/G'].mean(),
        'efficiency': (team_data['FPts'].sum() / team_data['MIN'].sum()) if team_data['MIN'].sum() > 0 else 0,
        'depth': len(team_data[team_data['FP/G'] > team_data['FP/G'].mean()])
    }
    
    return stats

@handle_error
def calculate_player_value(player_data: Dict[str, pd.DataFrame]) -> float:
    """Calculate a player's overall value based on performance across time ranges.
    
    This function computes a weighted value score that considers:
    1. Recent performance (higher weight for recent games)
    2. Consistency across time periods
    3. Minutes played and availability
    
    Args:
        player_data (Dict[str, pd.DataFrame]): Player statistics across different time ranges
        
    Returns:
        float: Calculated player value score
        
    The calculation weighs recent performance more heavily while also
    considering long-term consistency and reliability.
    """
    if not player_data:
        return 0.0
    
    value = 0.0
    total_weight = 0.0
    
    for time_range, data in player_data.items():
        if time_range not in TIME_RANGES or data.empty:
            continue
            
        weight = TIME_RANGES[time_range]
        avg_fp = data['FP/G'].mean()
        games_played = len(data)
        
        # Calculate weighted value considering:
        # 1. Fantasy points per game
        # 2. Games played (availability)
        # 3. Time range weight
        period_value = avg_fp * (games_played / 10) * weight
        
        value += period_value
        total_weight += weight
    
    # Normalize the value
    return value / total_weight if total_weight > 0 else 0.0

@handle_error
def calculate_trade_fairness(
    before_stats: Dict[str, Dict[str, float]],
    after_stats: Dict[str, Dict[str, float]],
    team_data: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, float]]:
    """Calculate the fairness and impact of a proposed trade.
    
    This function evaluates trade fairness by analyzing:
    1. Team strength before and after the trade
    2. Value differential between traded players
    3. Impact on team composition and depth
    
    Args:
        before_stats: Team statistics before the trade
        after_stats: Projected team statistics after the trade
        team_data: Current team rosters and player statistics
        
    Returns:
        Dict containing for each team:
            - fairness_score: Overall trade fairness (0-1)
            - value_change: Net change in team value
            - depth_impact: Impact on team depth
            - risk_score: Risk assessment of the trade
    
    The fairness calculation considers multiple factors including:
    - Short-term and long-term player value
    - Team needs and roster composition
    - Schedule and injury risk factors
    """
    results = {}
    
    for team in before_stats.keys():
        # Calculate value changes
        value_before = before_stats[team]['avg_fp'] * before_stats[team]['depth']
        value_after = after_stats[team]['avg_fp'] * after_stats[team]['depth']
        value_change = value_after - value_before
        
        # Calculate depth impact
        depth_before = before_stats[team]['depth']
        depth_after = after_stats[team]['depth']
        depth_impact = depth_after - depth_before
        
        # Calculate efficiency change
        efficiency_change = after_stats[team]['efficiency'] - before_stats[team]['efficiency']
        
        # Calculate risk score based on consistency changes
        consistency_change = after_stats[team]['consistency'] - before_stats[team]['consistency']
        risk_score = abs(consistency_change) * (1 + abs(depth_impact) / 10)
        
        # Calculate overall fairness score (0-1)
        fairness_score = 1.0 - min(1.0, abs(value_change) / (value_before + 1e-6))
        
        results[team] = {
            'fairness_score': fairness_score,
            'value_change': value_change,
            'depth_impact': depth_impact,
            'risk_score': risk_score
        }
    
    return results

@handle_error
def get_trend_color(value: float, is_positive_good: bool = True) -> str:
    """Determine the color for visualizing trend changes.
    
    Args:
        value: The change in value to visualize
        is_positive_good: If True, positive changes are green (default: True)
        
    Returns:
        Hex color code for the trend visualization
        
    Color Scheme:
        - Green: Positive/beneficial changes
        - Red: Negative/detrimental changes
        - Yellow: Neutral or minimal changes
    """
    if abs(value) < 0.05:  # Minimal change threshold
        return WARNING_COLOR
        
    if is_positive_good:
        return SUCCESS_COLOR if value > 0 else ERROR_COLOR
    else:
        return ERROR_COLOR if value > 0 else SUCCESS_COLOR

@handle_error
def get_fairness_color(fairness_score: float) -> str:
    """Get the color for visualizing trade fairness scores.
    
    Args:
        fairness_score: Trade fairness score (0-1)
        
    Returns:
        Hex color code based on the fairness level
        
    Color Ranges:
        - Green (>= 0.8): Very fair trade
        - Yellow (0.6-0.8): Moderately fair trade
        - Red (< 0.6): Potentially unfair trade
    """
    if fairness_score >= 0.8:
        return SUCCESS_COLOR
    elif fairness_score >= 0.6:
        return WARNING_COLOR
    else:
        return ERROR_COLOR

@handle_error
def display_team_comparison(team_data: Dict[str, pd.DataFrame], team_name: str) -> None:
    """Display a comprehensive comparison of team statistics.
    
    This function creates an interactive visualization showing:
    1. Key performance metrics
    2. Player distribution charts
    3. Statistical trends
    
    Args:
        team_data: Dictionary containing team statistics
        team_name: Name of the team to display
        
    The display includes:
    - Bar charts for key metrics
    - Distribution plots for player values
    - Trend lines for important statistics
    """
    st.subheader(f"Team Analysis: {team_name}")
    
    # Calculate and display key metrics
    stats = calculate_team_stats(team_data[team_name])
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Fantasy Points", f"{stats['avg_fp']:.1f}")
    with col2:
        st.metric("Team Depth", f"{stats['depth']}")
    with col3:
        st.metric("Efficiency", f"{stats['efficiency']:.2f}")
    
    # Create distribution plot
    fig = px.histogram(
        team_data[team_name],
        x='FP/G',
        title='Player Value Distribution',
        nbins=20
    )
    st.plotly_chart(fig)

@handle_error
def display_trade_analysis(analysis: Dict[str, Dict[str, float]], teams: List[str]) -> None:
    """Display the trade analysis results.
    
    Args:
        analysis: Dictionary containing analysis results for each team
        teams: List of team names involved in the trade
    """
    # Calculate overall trade fairness
    fairness_scores = [data['fairness_score'] for data in analysis.values()]
    min_fairness = min(fairness_scores)
    
    # Display overall fairness
    st.markdown("## Trade Analysis Results")
    st.markdown(f"### Overall Trade Fairness: {min_fairness:.1%}")
    
    # Display individual team fairness scores
    cols = st.columns(len(teams))
    for col, team in zip(cols, teams):
        with col:
            fairness = analysis[team]['fairness_score']
            color = get_fairness_color(fairness)
            st.markdown(
                f"**{team}**\n\n"
                f"<span style='color: {color}; font-size: 24px;'>{fairness:.1%}</span>",
                unsafe_allow_html=True
            )

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
def plot_performance_trends(data: Dict[str, Dict[str, float]], selected_metrics: List[str], title: str) -> go.Figure:
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
 

    # Ensure time_ranges only contains valid time strings
    valid_time_ranges = [tr for tr in time_ranges if tr in ['60 Days', '30 Days', '14 Days', '7 Days']]
    valid_time_ranges.sort(key=lambda x: int(x.split()[0]), reverse=True)
    
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
                textposition="top center",
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
def plot_player_trend(players_data: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create performance trend plots for multiple players."""
    fig = go.Figure()

    # Loop through each player's data
    for player, data in players_data.items():
        time_ranges = ['60 Days', '30 Days', '14 Days', '7 Days']
        if player in data.index:
            for time_range in time_ranges:
                if time_range in data:
                    fig.add_trace(
                        go.Scatter(
                            x=[time_range] * len(data[time_range]),  # X-axis is the time range
                            y=data[time_range]['FP/G'],  # Y-axis is the FP/G values
                            mode='lines+markers',
                            name=player
                        )
                    )

    # Update layout
    fig.update_layout(
        title="Player Performance Trends",
        xaxis_title="Time Range",
        yaxis_title="FP/G",
        plot_bgcolor='rgb(17, 23, 29)',
        paper_bgcolor='rgb(17, 23, 29)',
        font=dict(color='white'),
        showlegend=True
    )

    return fig

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
    """Main application function"""
    st.title("Fantasy Basketball Trade Analyzer")
    
    # Initialize session state
    init_session_state()
    
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
        
        # Add clear trade history checkbox
        clear_history = st.checkbox("Clear trade history before each run")
        
        # Clear trade history if checkbox is checked
        if clear_history and st.session_state.trade_analyzer:
            st.session_state.trade_analyzer.trade_history.clear()
        
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
