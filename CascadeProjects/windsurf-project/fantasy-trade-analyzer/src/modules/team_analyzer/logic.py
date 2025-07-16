"""
Logic for the Team Analyzer feature.
"""
import pandas as pd

# Define the core stat categories to be analyzed, using correct abbreviations
# Negative stats (TO, FGM, FTM) will be inverted so a higher rank is always better.
STAT_CATEGORIES = ['PTS', 'REB', 'AST', 'ST', 'BLK', '3PTM', 'TO']

def calculate_team_stats(combined_data: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """
    Calculates the total stats for each fantasy team for a given time range.

    Args:
        combined_data (pd.DataFrame): The main DataFrame containing all player data.
        time_range (str): The selected time range (e.g., 'YTD', '30 Days').

    Returns:
        pd.DataFrame: A DataFrame where each row is a fantasy team and each column is a stat category.
    """
    if combined_data is None or combined_data.empty:
        return pd.DataFrame()

    # Filter data for the selected time range
    time_range_df = combined_data[combined_data['Timestamp'] == time_range]

    # Filter stat categories to only those present in the dataframe to avoid KeyErrors
    existing_categories = [cat for cat in STAT_CATEGORIES if cat in time_range_df.columns]

    if not existing_categories:
        return pd.DataFrame()

    # Group by fantasy manager and sum the existing stat categories
    team_stats = time_range_df.reset_index().groupby('Fantasy_Manager')[existing_categories].sum()

    # Invert Turnovers (TO) so that a higher value is better for visualization
    if 'TO' in team_stats.columns:
        team_stats['TO'] = team_stats['TO'] * -1

    return team_stats

def calculate_league_ranks(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the rank of each team for each statistical category.

    Args:
        team_stats (pd.DataFrame): DataFrame of team statistics (output of calculate_team_stats).

    Returns:
        pd.DataFrame: A DataFrame with the same shape, but with rank values instead of stat totals.
    """
    if team_stats is None or team_stats.empty:
        return pd.DataFrame()

    # The 'rank' method in pandas assigns lower ranks to higher values by default (which is what we want)
    # method='min' gives teams with the same value the same rank
    league_ranks = team_stats.rank(method='min', ascending=False).astype(int)
    return league_ranks
