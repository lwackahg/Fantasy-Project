import pandas as pd
import re
import streamlit as st
from pathlib import Path

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by removing invalid entries and normalizing data."""
    # Define numeric columns
    numeric_columns = ['FPts', 'FP/G', 'PTS', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN']
    
    # Convert numeric columns and remove rows with invalid entries
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            df = df[df[col].notna()]  # Remove rows where conversion failed

    # Ensure that required columns exist
    required_columns = ['Player', 'Team', 'FP/G']
    if not all(col in df.columns for col in required_columns):
        st.error("Missing required columns. Required: {}".format(required_columns))
        return pd.DataFrame()  # Return an empty DataFrame if required columns are missing

    # Clean up team names
    if 'Team' in df.columns:
        df['Team'] = df['Team'].fillna('FA')

    # Remove rows with missing or invalid player names
    df = df[df['Player'].notna() & (df['Player'] != '')]

    # Calculate GP (Games Played) based on provided logic
    if 'FPts' in df.columns and 'FP/G' in df.columns:
        df['GP'] = df['FPts'] / df['FP/G']

    # Reset index if 'Player' is in DataFrame
    if 'Player' in df.columns:
        df.reset_index(inplace=True, drop=True)

    return df

def load_data(data_dir: Path) -> dict:
    """Load and clean player data from CSV files in the data directory."""
    data_ranges = {}
    
    try:
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            st.error("No CSV files found in the data directory")
            return data_ranges
            
        for file in csv_files:
            if "YTD" in file.name:
                df = pd.read_csv(file)
                df = clean_data(df)  # Clean the DataFrame
                if not df.empty:  # Only store if DataFrame is not empty
                    data_ranges['YTD'] = df
            else:
                match = re.search(r'\((\d+)\)', file.name)
                if match:
                    days = match.group(1)
                    df = pd.read_csv(file)
                    df = clean_data(df)  # Clean the DataFrame
                    if not df.empty:  # Only store if DataFrame is not empty
                        data_ranges[f'{days} Days'] = df
            
        if not data_ranges:
            st.error("No valid data files were loaded")
        else:
            # Sort ranges and prepare to return
            sorted_ranges = sorted(data_ranges.keys(), key=lambda x: (x != 'YTD', int(x.split()[0])))
            sorted_data_ranges = {key: data_ranges[key] for key in sorted_ranges}
            st.success(f"Successfully loaded data for ranges: {', '.join(sorted_data_ranges.keys())}")
            return sorted_data_ranges
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

    return data_ranges