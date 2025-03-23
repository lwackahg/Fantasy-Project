import pandas as pd
import re
import streamlit as st
from pathlib import Path
import numpy as np
import os
import time
import csv
from functools import lru_cache

TEAM_MAPPINGS = {
    'Sar': 'Shaq\'s Anus Ripples',
    '15': '15 Dream Team',
    'DBD': 'Diddled By Diddy',
    'EE': 'Epstein Experience',
    'Fent': 'Give me the fentanyl',
    'TRUMP': 'Kamala\'s Gunz',
    '420': 'Kevin O\'Leary',
    'J&J': 'Tauras\' Torn Johnson',
    'BabyOil': 'P Diddy\'s Slip & Slide',
    'PRO': 'President of Retarded Opera',
    'ROMO': 'Rudy Homo',
    'Tribunal': 'Stevens Underaged Experien',
    'TRH': 'The Retirement Home',
    'MylS': 'Weinstein Wranglers'
}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by removing invalid entries and normalizing data."""
    numeric_columns = ['FPts', 'FP/G', 'PTS', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'TF', 'EJ', '3D', '2D', 'GP']
    
    # Check for the existence of numeric columns
    existing_numeric_columns = [col for col in numeric_columns if col in df.columns]

    # Convert numeric columns and drop rows with invalid entries
    df[existing_numeric_columns] = df[existing_numeric_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=existing_numeric_columns, inplace=True)

    # Ensure required columns exist
    required_columns = ['Player', 'Status', 'FP/G']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns. Required: {required_columns}")
        return pd.DataFrame()

    # Clean up team names and remove invalid player names
    df['Team'] = df.get('Team', pd.Series(['FA'] * len(df)))
    df = df[df['Player'].notna() & (df['Player'] != '')]

    # Drop unnecessary columns
    df.drop(columns=['RkOv', 'Opponent', 'Roster Status'], errors='ignore', inplace=True)

    # Calculate GP (Games Played) if columns exist
    if 'FPts' in df.columns and 'FP/G' in df.columns:
        df['GP'] = np.ceil(df['FPts'] / df['FP/G']).fillna(0).astype(int)

    # Map Fantasy Manager Names
    df['Fantasy_Manager'] = df['Status'].map(TEAM_MAPPINGS).fillna('Unknown Manager')

    return df.reset_index(drop=True)

def load_data():
    """Load data from CSV files."""
    try:
        data_dir = Path(__file__).parent.parent / "data"
        data_ranges = {}
        combined_data = pd.DataFrame()
        oldest_timestamp = None

        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            st.error("No CSV files found in the data directory")
            st.session_state.csv_timestamp = "No CSV files found"
            return data_ranges, combined_data

        for file in csv_files:
            # Get timestamp while we're already accessing the file
            timestamp = os.path.getmtime(str(file))
            if oldest_timestamp is None or timestamp < oldest_timestamp:
                oldest_timestamp = timestamp

            df = pd.read_csv(file)
            df = clean_data(df)  # Clean the DataFrame
            
            if not df.empty:
                # Extract timestamp from filename
                timestamp_value = None
                if "YTD" in file.name:
                    timestamp_value = 'YTD'
                    data_ranges['YTD'] = df
                else:
                    match = re.search(r'\((\d+)\)', file.name)
                    if match:
                        days = match.group(1)
                        timestamp_value = f'{days} Days'
                        data_ranges[f'{days} Days'] = df
                
                if timestamp_value:
                    df['Timestamp'] = timestamp_value
                    
                combined_data = pd.concat([combined_data, df], ignore_index=True)

        if combined_data.empty:
            st.error("No valid data files were loaded")
        else:
            combined_data.set_index('Player', inplace=True)

        # Store the timestamp in session state
        if oldest_timestamp is not None:
            # Check if running on localhost
            is_local = st.get_option('server.address') == 'localhost'
            
            # Only adjust timestamp if not running locally
            if not is_local:
                oldest_timestamp = oldest_timestamp - (5 * 3600)  # Subtract 5 hours in seconds
            
            st.session_state.csv_timestamp = time.strftime("%Y-%m-%d %I:%M:%S %p", 
                                                         time.localtime(oldest_timestamp))

        return data_ranges, combined_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.session_state.csv_timestamp = "Error loading CSV files"
        return {}, pd.DataFrame()

def csv_time():
    """Get the CSV timestamp from session state."""
    return st.session_state.get('csv_timestamp', "CSV timestamp not available")

@lru_cache(maxsize=1)
def load_schedule_data():
    """
    Load and parse the schedule data from CSV file.
    Cached to improve performance on repeated calls.
    
    Returns:
        pd.DataFrame: Processed schedule data with proper columns
    """
    try:
        # Get path to schedule.csv
        schedule_path = Path(__file__).parent.parent / "data" / "schedule" / "schedule.csv"
        
        if not os.path.exists(schedule_path):
            st.error("Schedule data file not found.")
            return None
        
        # Process the data using a more robust approach
        data = []
        current_period = None
        current_date_range = None
        
        # Read the file using csv module which handles quotes properly
        with open(schedule_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            
            for row in csv_reader:
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                
                # Check if this is a scoring period line
                if row[0].startswith("Scoring Period"):
                    current_period = row[0].strip()
                    continue
                
                # Check if this is a date range line
                if row[0].startswith("(") and ")" in row[0]:
                    current_date_range = row[0].strip('()"')
                    continue
                
                # Process matchup line - we expect 4 columns: Team1, Score1, Team2, Score2
                if len(row) >= 4:
                    team1 = row[0].strip()
                    score1_str = row[1].strip()
                    team2 = row[2].strip()
                    score2_str = row[3].strip()
                    
                    # Only add valid matchups (both teams have names)
                    if team1 and team2 and not (team1.startswith("(") or team2.startswith("(")):
                        # Convert scores to integers, handling commas in numbers
                        try:
                            score1 = int(score1_str.replace(",", ""))
                        except ValueError:
                            # If we can't convert to int, it might be a date or other text
                            # Just set to 0 and continue without warning
                            score1 = 0
                        
                        try:
                            score2 = int(score2_str.replace(",", ""))
                        except ValueError:
                            score2 = 0
                        
                        # Add to data
                        data.append({
                            "Scoring Period": current_period,
                            "Date Range": current_date_range,
                            "Team 1": team1,
                            "Score 1": score1,
                            "Score 1 Display": score1_str,
                            "Team 2": team2,
                            "Score 2": score2,
                            "Score 2 Display": score2_str,
                            "Matchup": f"{team1} - {score1_str} vs {team2} - {score2_str}"
                        })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            st.error("No data was parsed from the schedule file.")
            return None
        
        # Add winner column
        df["Winner"] = df.apply(
            lambda row: row["Team 1"] if row["Score 1"] > row["Score 2"] 
            else (row["Team 2"] if row["Score 2"] > row["Score 1"] else "Tie"),
            axis=1
        )
        
        # Extract period number for sorting
        df["Period Number"] = df["Scoring Period"].apply(
            lambda x: int(re.search(r"Scoring Period (\d+)", x).group(1)) if re.search(r"Scoring Period (\d+)", x) else 0
        )
        
        # Sort by period
        df = df.sort_values("Period Number")
        
        return df
    
    except Exception as e:
        st.error(f"Error loading schedule data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def calculate_team_stats(schedule_df):
    """
    Calculate performance statistics for each team.
    
    Args:
        schedule_df (pd.DataFrame): The schedule data
        
    Returns:
        pd.DataFrame: Team statistics
    """
    # Initialize stats dictionary
    team_stats = {}
    
    # Get all unique teams first
    all_teams = set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())
    
    # Initialize team entries
    for team in all_teams:
        team_stats[team] = {
            "Wins": 0,
            "Losses": 0,
            "Ties": 0,
            "Points For": 0,
            "Points Against": 0,
            "Total Matchups": 0
        }
    
    # Process each matchup - use vectorized operations where possible
    for _, row in schedule_df.iterrows():
        team1 = row["Team 1"]
        team2 = row["Team 2"]
        score1 = row["Score 1"]
        score2 = row["Score 2"]
        
        # Update team1 stats
        team_stats[team1]["Points For"] += score1
        team_stats[team1]["Points Against"] += score2
        team_stats[team1]["Total Matchups"] += 1
        
        # Update team2 stats
        team_stats[team2]["Points For"] += score2
        team_stats[team2]["Points Against"] += score1
        team_stats[team2]["Total Matchups"] += 1
        
        # Update win/loss/tie records
        if score1 > score2:
            team_stats[team1]["Wins"] += 1
            team_stats[team2]["Losses"] += 1
        elif score2 > score1:
            team_stats[team2]["Wins"] += 1
            team_stats[team1]["Losses"] += 1
        else:
            team_stats[team1]["Ties"] += 1
            team_stats[team2]["Ties"] += 1
    
    # Convert to DataFrame
    stats_df = pd.DataFrame.from_dict(team_stats, orient="index")
    
    # Calculate win percentage
    stats_df["Win %"] = stats_df.apply(
        lambda row: round(row["Wins"] / row["Total Matchups"] * 100, 1) if row["Total Matchups"] > 0 else 0,
        axis=1
    )
    
    # Calculate average points
    stats_df["Avg Points For"] = round(stats_df["Points For"] / stats_df["Total Matchups"], 1)
    stats_df["Avg Points Against"] = round(stats_df["Points Against"] / stats_df["Total Matchups"], 1)
    
    return stats_df

def load_draft_results(file_path: str) -> pd.DataFrame:
    """Load and process the draft results from a CSV file."""
    try:
        draft_df = pd.read_csv(file_path)
        # Ensure necessary columns are present
        required_columns = ['Player ID', 'Pick', 'Pos', 'Player', 'Team', 'Bid', 'Fantasy Team', 'Time (EDT)']
        if not all(col in draft_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns in draft results. Required: {required_columns}")
        
        # Process data as needed (e.g., convert types, handle missing values)
        draft_df['Bid'] = pd.to_numeric(draft_df['Bid'], errors='coerce').fillna(0)
        draft_df['Time (EDT)'] = pd.to_datetime(draft_df['Time (EDT)'], errors='coerce')

        # Map Fantasy Team Names
        draft_df['Fantasy Team Full Name'] = draft_df['Fantasy Team'].map(TEAM_MAPPINGS).fillna('Unknown Team')
        
        return draft_df
    except Exception as e:
        print(f"Error loading draft results: {e}")
        return pd.DataFrame()

# Example usage
# draft_results = load_draft_results('data/Fantrax-Draft-Results-Mr Squidward’s 69.csv')

# Display TEAM_MAPPINGS for reference
print("Fantasy Team Mappings:")
for key, value in TEAM_MAPPINGS.items():
    print(f"{key} -> {value}")
