import pandas as pd
import re
import streamlit as st
from pathlib import Path
import numpy as np
import os
import time
import csv
import logging
from modules.team_mappings import TEAM_MAPPINGS, TEAM_ALIASES

 

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by removing invalid entries and normalizing data."""
    # Comprehensive list of all possible numeric stat columns from Fantrax
    numeric_columns = [
        'FPts', 'FP/G', 'PTS', 'OREB', 'DREB', 'REB', 'AST', 'ST', 'BLK', 
        'TO', 'TF', 'EJ', '3D', '2D', 'GP', '3PTM', 'FGM', 'FTM'
    ]
    
    # Check for the existence of numeric columns in the current dataframe
    existing_numeric_columns = [col for col in numeric_columns if col in df.columns]

    # Convert numeric columns to numeric type, coercing errors to NaN
    df[existing_numeric_columns] = df[existing_numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaN values in stat columns with 0 instead of dropping the row
    df[existing_numeric_columns] = df[existing_numeric_columns].fillna(0)

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
    status_series = df['Status'].astype(str)
    status_resolved = status_series.map(TEAM_ALIASES).fillna(status_series)
    fantasy_manager = status_resolved.map(TEAM_MAPPINGS)
    unknown_mask = fantasy_manager.isna()
    if unknown_mask.any():
        unknowns = set(status_resolved[unknown_mask])
        if unknowns:
            st.session_state.setdefault('unknown_teams', set()).update(unknowns)
        fantasy_manager = fantasy_manager.fillna(status_resolved)
    df['Fantasy_Manager'] = fantasy_manager

    return df.reset_index(drop=True)

@st.cache_data
def _load_and_clean_csv_data(file_paths):
    """Loads and processes a tuple of CSV file paths, returning clean data."""
    data_ranges = {}
    combined_data = pd.DataFrame()

    for file_path in file_paths:
        file = Path(file_path)
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

    if not combined_data.empty:
        combined_data.set_index('Player', inplace=True)
    
    return data_ranges, combined_data

def load_data():
    """
    Finds CSV files, loads them using a cached function, and handles UI/session state.
    """
    try:
        project_root = Path(__file__).parent
        data_dir = project_root / "data"
        csv_files = [str(f) for f in data_dir.glob('*.csv') if f.is_file()]

        if not csv_files:
            st.error("No CSV files found in the data directory")
            st.session_state.csv_timestamp = "No CSV files found"
            return {}, pd.DataFrame()

        # Get the list of file paths as a tuple to make it hashable for caching
        file_paths_tuple = tuple(sorted([str(f) for f in csv_files]))
        data_ranges, combined_data = _load_and_clean_csv_data(file_paths_tuple)

        if combined_data.empty:
            st.error("No valid data files were loaded")
            return {}, pd.DataFrame()

        # Handle side-effects (timestamp calculation and session state) here
        oldest_timestamp = min(os.path.getmtime(file) for file in csv_files)
        
        is_local = st.get_option('server.address') == 'localhost'
        if not is_local:
            oldest_timestamp -= (5 * 3600)  # Subtract 5 hours
        
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

@st.cache_data
def load_schedule_data():
    """
    Load and parse the schedule data from CSV file.
    Cached to improve performance on repeated calls.
    
    Returns:
        pd.DataFrame: Processed schedule data with proper columns
    """
    try:
        # Construct an absolute path to the data directory from the project root
        project_root = Path(__file__).parent
        schedule_path = project_root / "data" / "schedule" / "schedule.csv"
        logging.info(f"Attempting to load schedule from: {schedule_path}")
        
        if not os.path.exists(schedule_path):
            logging.error(f"Schedule data file not found at: {schedule_path}")
            return None
        
        logging.info("Schedule file found. Proceeding to parse.")
        
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
            logging.warning("No data was parsed from the schedule file. The file might be empty or in an unexpected format.")
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
        
        logging.info(f"Successfully parsed {len(df)} rows from schedule file.")
        return df
    
    except Exception as e:
        logging.error(f"An exception occurred while loading schedule data: {e}", exc_info=True)
        return None
        return None


@st.cache_data
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

       
        
        return draft_df
    except Exception as e:
        print(f"Error loading draft results: {e}")
        return pd.DataFrame()

# Example usage
# draft_results = load_draft_results('data/Fantrax-Draft-Results-Mr Squidward’s 69.csv')
