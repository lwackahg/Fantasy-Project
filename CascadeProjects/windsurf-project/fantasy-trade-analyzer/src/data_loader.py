import pandas as pd
import re
import streamlit as st
from pathlib import Path
import numpy as np
import os
import time

TEAM_MAPPINGS = {
    'Sar': 'Shaq\'s Anus Ripples',
    '15': '15 Dream Team',
    'DBD': 'Diddled By Diddy',
    'EE': 'Epstein Experience',
    'Fent': 'Give me the fentanyl',
    'TRUMP': 'Kamala\'s Gunz',
    '420': 'Kevin O\'Leary',
    'J&J': 'Ligma Johnson',
    'BabyOil': 'P Diddy\'s Slip & Slide',
    'PRO': 'President of Retarded Opera',
    'ROMO': 'Rudy Homo',
    'Tribunal': 'Stevens Underaged Experien',
    'TRH': 'The Retirement Home',
    'MylS': 'Weinstein Wranglers'
}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by removing invalid entries and normalizing data."""
    numeric_columns = ['FPts', 'FP/G', 'PTS', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN']
    
    # Convert numeric columns and remove rows with invalid entries
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            df = df[df[col].notna()]  # Remove rows where conversion failed

    # Ensure that required columns exist
    required_columns = ['Player', 'Team', 'FP/G']
    if not all(col in df.columns for col in required_columns):
        print("Error: Missing required columns. Required: {}".format(required_columns))
        return pd.DataFrame()  # Return an empty DataFrame if required columns are missing

    # Clean up team names
    if 'Team' in df.columns:
        df['Team'] = df['Team'].fillna('FA')

    # Remove rows with missing or invalid player names
    df = df[df['Player'].notna() & (df['Player'] != '')]

    # Calculate GP (Games Played) based on provided logic
    if 'FPts' in df.columns and 'FP/G' in df.columns:
        df['GP'] = np.ceil(df['FPts'] / df['FP/G']).fillna(0).astype(int)  # Ceiling rounding and convert to int
        
    # Assign Fantasy Manager Names based on Status column
    if 'Status' in df.columns:
        df['Fantasy_Manager'] = df['Status'].map(TEAM_MAPPINGS).fillna('Unknown Manager')

    # Reset index if 'Player' is in DataFrame
    if 'Player' in df.columns:
        df.reset_index(inplace=True, drop=True)

    return df

def load_data():
    """Load data from CSV files."""
    try:
        data_dir = Path(__file__).parent.parent / "data"
        data_ranges = {}
        combined_data = pd.DataFrame()
        oldest_timestamp = None

        # Load data for each time range
        for file in data_dir.glob("*.csv"):
            if "Players" in file.name:
                # Get timestamp while we're already accessing the file
                timestamp = os.path.getmtime(str(file))
                if oldest_timestamp is None or timestamp < oldest_timestamp:
                    oldest_timestamp = timestamp

                # Rest of your existing data loading code...
                # [existing code continues...]

        # Store the timestamp in session state
        if oldest_timestamp is not None:
            st.session_state.csv_timestamp = time.strftime("%Y-%m-%d %I:%M:%S %p", 
                                                         time.localtime(oldest_timestamp))
        else:
            st.session_state.csv_timestamp = "No CSV files found"

        return data_ranges, combined_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.session_state.csv_timestamp = "Error loading CSV files"
        return {}, pd.DataFrame()

def csv_time():
    """Get the CSV timestamp from session state."""
    return st.session_state.get('csv_timestamp', "CSV timestamp not available")
