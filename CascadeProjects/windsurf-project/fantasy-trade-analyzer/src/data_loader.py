import pandas as pd
import re
import streamlit as st
from pathlib import Path
import numpy as np

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

def load_data(data_dir: Path) -> tuple:
    """Load and clean player data from CSV files in the data directory."""
    data_ranges = {}
    combined_data = pd.DataFrame()  # Combined DataFrame for all player data

    try:
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            st.error("No CSV files found in the data directory")
            return data_ranges, combined_data  # Return empty values if no CSVs
            
        for file in csv_files:
            df = pd.read_csv(file)
            df = clean_data(df)  # Clean the DataFrame
            
            if not df.empty:
                # Extract timestamp from filename
                timestamp_value = None
                if "YTD" in file.name:
                    # YTD data
                    timestamp_value = 'Year to Date'
                    data_ranges['YTD'] = df
                else:
                    match = re.search(r'\((\d+)\)', file.name)
                    if match:
                        days = match.group(1)
                        timestamp_value = f'{days} Days'
                        data_ranges[f'{days} Days'] = df
                
                # Add a column to the DataFrame with the extracted timestamp
                if timestamp_value:
                    df['Timestamp'] = timestamp_value  # New column for the timestamp
                    
                # Combine the data into one DataFrame
                combined_data = pd.concat([combined_data, df], ignore_index=True)

        if combined_data.empty:
            st.error("No valid data files were loaded")
        else:
            combined_data.set_index('Player', inplace=True)  # Setting Player as index for easier lookup

        return data_ranges, combined_data  # Return both the dictionary and DataFrame
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}, pd.DataFrame()  # Return empty values on exception