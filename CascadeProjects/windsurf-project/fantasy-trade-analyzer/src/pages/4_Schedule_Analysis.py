import streamlit as st
import pandas as pd
from pathlib import Path
from data_loader import load_schedule_data, calculate_team_stats
from modules.schedule_analysis.logic import swap_team_schedules, compare_team_stats, calculate_all_schedule_swaps
from modules.schedule_analysis.ui import (
    display_list_view,
    display_table_view,
    display_team_stats,
    display_all_swaps_analysis
)
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Schedule Analysis", layout="wide")

display_global_sidebar()

st.title(":green[Schedule Analysis]")
st.write("View the complete schedule and results for all fantasy matchups.")

# Check if data is loaded
if 'schedule_data' not in st.session_state or st.session_state.schedule_data.empty:
    st.warning("Please load schedule data from the main page to use this feature.")
    st.stop()

schedule_df = st.session_state.schedule_data

# --- Filtering Options ---
st.markdown("---")
st.subheader("Filter Schedule")

# Get all unique teams and scoring periods
all_teams = sorted(list(set(schedule_df["Team 1"]).union(set(schedule_df["Team 2"]))))
all_periods = schedule_df["Scoring Period"].unique()

# Create columns for filters
col1, col2 = st.columns(2)

with col1:
    selected_teams = st.multiselect(
        "Filter by Team(s)",
        options=all_teams,
        default=st.session_state.get('schedule_selected_teams', [])
    )
    st.session_state.schedule_selected_teams = selected_teams

with col2:
    selected_periods = st.multiselect(
        "Filter by Scoring Period(s)",
        options=all_periods,
        default=st.session_state.get('schedule_selected_periods', [])
    )
    st.session_state.schedule_selected_periods = selected_periods

# Apply filters
filtered_df = schedule_df.copy()
if selected_teams:
    filtered_df = filtered_df[
        (filtered_df["Team 1"].isin(selected_teams)) |
        (filtered_df["Team 2"].isin(selected_teams))
    ]
if selected_periods:
    filtered_df = filtered_df[filtered_df["Scoring Period"].isin(selected_periods)]

# --- Display Data ---
if not filtered_df.empty:
    tab1, tab2, tab3 = st.tabs(["List View", "Table View", "Team Stats & Swap Analysis"])

    with tab1:
        display_list_view(filtered_df)

    with tab2:
        display_table_view(filtered_df)

    with tab3:
        st.subheader("Team Performance")
        display_team_stats(schedule_df, calculate_team_stats)
        
        # --- Schedule Swap Analysis ---
        st.markdown("***")
        st.header("Schedule Swap Analysis")

        # Calculate all possible swaps
        with st.spinner("Calculating all possible schedule swaps..."):
            all_swaps_df = calculate_all_schedule_swaps(schedule_df)

        # Get all unique teams for the filter
        all_teams = sorted(list(set(schedule_df["Team 1"]).union(set(schedule_df["Team 2"]))))

        # Display the new analysis UI
        display_all_swaps_analysis(all_swaps_df, all_teams)
else:
    st.info("No matchups found with the selected filters.")