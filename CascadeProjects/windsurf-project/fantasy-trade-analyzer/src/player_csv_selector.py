import streamlit as st
from pathlib import Path
import pandas as pd
import time

def list_player_csv_files(data_dir):
    """Return a list of player CSV files in the data directory."""
    data_path = Path(data_dir)
    # Only include files that look like player CSVs
    csv_files = [f for f in data_path.glob('Fantrax-Players-*.csv') if f.is_file()]
    return sorted(csv_files)


def select_player_csv_files(data_dir):
    """
    Streamlit UI: List available player CSVs and allow user to select one or more. Returns list of selected file paths (not loaded yet).
    """
    csv_files = list_player_csv_files(data_dir)
    if not csv_files:
        st.warning("No Fantrax player CSV files found in the data directory.")
        return []

    file_labels = [f"{f.name} (Last modified: {time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))})" for f in csv_files]
    label_to_file = {label: file for label, file in zip(file_labels, csv_files)}
    selected_labels = st.multiselect("Select Player CSV(s) to Load", file_labels, default=file_labels[:1])
    selected_files = [label_to_file[label] for label in selected_labels]
    return selected_files

def load_selected_csvs(selected_files):
    """
    Given a list of file paths, load them as DataFrames. Returns dict of {filename: DataFrame}.
    """
    loaded = {}
    for file in selected_files:
        try:
            df = pd.read_csv(file)
            loaded[file.name] = df
        except Exception as e:
            st.error(f"Failed to load {file.name}: {e}")
    return loaded
