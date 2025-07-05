import streamlit as st
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict
from typing import List, Dict
from data_loader import clean_data
from modules.trade_analysis.logic import TradeAnalyzer

def _get_league_name_from_filename(path: Path) -> str:
    """Extracts league name from a Fantrax CSV filename."""
    # Regex to capture the league name, which is between "Fantrax-Players-" and the date part.
    # This new regex handles cases where the league name is absent and is more specific.
    match = re.search(r'Fantrax-Players-(?:(.*?)-)?\((YTD|\d+)\)', path.name)
    if match and match.group(1):
        return match.group(1).replace('_', ' ').strip()
    return "Default League"  # Group files without a clear league name

def _group_files_by_league(data_dir: Path) -> Dict[str, List[Path]]:
    """Groups available player CSV files by league name."""
    csv_files = [f for f in data_dir.glob('Fantrax-Players-*.csv') if f.is_file()]
    grouped_files = defaultdict(list)
    for f in csv_files:
        league_name = _get_league_name_from_filename(f)
        grouped_files[league_name].append(f)
    for league in grouped_files:
        grouped_files[league].sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return grouped_files

from typing import List, Dict, Tuple

def find_default_files_to_load(data_dir: Path) -> Tuple[List[Path], str]:
    """Finds the most recent, complete set of CSVs for a single league and returns the files and league name."""
    grouped_files = _group_files_by_league(data_dir)
    if not grouped_files:
        return [], None

    # Find the league with the most complete and recent set of files
    best_league = None
    most_found = 0
    
    range_patterns = {
        'YTD': r'\(YTD\)',
        '60 Days': r'\(60|60-days\)',
        '30 Days': r'\(30|30-days\)',
        '14 Days': r'\(14|14-days\)',
        '7 Days': r'\(7|7-days\)'
    }

    for league, files in grouped_files.items():
        found_count = sum(1 for _, pattern in range_patterns.items() if any(re.search(pattern, f.name) for f in files))
        if found_count > most_found:
            most_found = found_count
            best_league = league

    if not best_league:
        return [], None

    league_files = grouped_files[best_league]
    default_files = []
    for _, pattern in range_patterns.items():
        # Find the most recent file for each pattern
        found_file = next((f for f in league_files if re.search(pattern, f.name)), None)
        if found_file:
            default_files.append(found_file)
            
    return default_files, best_league

def process_files_into_session_state(selected_files: List[Path]):
    """Loads and processes a list of CSV files into Streamlit's session state."""
    if not selected_files:
        st.warning("No files selected to load.")
        return

    data_ranges = {}
    combined_data = pd.DataFrame()
    
    # Add thousands=',' to correctly parse numbers with commas (e.g., '2,085')
    loaded_csvs = {file.name: pd.read_csv(file, thousands=',') for file in selected_files}

    for file in selected_files:
        try:
            df = clean_data(loaded_csvs[file.name])
            if not df.empty:
                timestamp_value = 'YTD' if "YTD" in file.name else f'{re.search(r"\((\d+)\)", file.name).group(1)} Days'
                df['Timestamp'] = timestamp_value
                data_ranges[timestamp_value] = df
                combined_data = pd.concat([combined_data, df], ignore_index=True)
        except Exception as e:
            st.error(f"Failed to process {file.name}: {e}")

    if not combined_data.empty:
        combined_data.set_index('Player', inplace=True)
    
    st.session_state.data_ranges = data_ranges
    st.session_state.combined_data = combined_data
    if 'combined_data' in st.session_state and st.session_state.combined_data is not None:
        st.session_state.trade_analyzer = TradeAnalyzer(st.session_state.combined_data)
        st.session_state.trade_analyzer_data_is_stale = True
    
    st.success(f"Successfully loaded and processed {len(selected_files)} files.")

def display_data_loader_ui(data_dir: Path):
    """Streamlit UI for selecting and loading a league dataset from available files."""
    grouped_files = _group_files_by_league(data_dir)

    if not grouped_files:
        st.info("No Fantrax player CSV files found in the 'data' directory.")
        return

    league_names = sorted(list(grouped_files.keys()))
    
    # Determine the currently loaded league to set the dropdown default
    current_league_index = 0
    if 'loaded_league_name' in st.session_state and st.session_state.loaded_league_name in league_names:
        current_league_index = league_names.index(st.session_state.loaded_league_name)

    selected_league = st.selectbox(
        "Select a league dataset to load",
        options=league_names,
        index=current_league_index,
        key="league_selector"
    )

    if st.button("Load Selected League", type="primary", key="load_league_button"):
        if selected_league:
            files_to_load = grouped_files[selected_league]
            st.info(f"Loading dataset for '{selected_league}'...")
            process_files_into_session_state(files_to_load)
            # Store the name of the loaded league and rerun to refresh the app
            st.session_state.loaded_league_name = selected_league
            st.rerun()
        else:
            st.warning("No league selected.")
