import streamlit as st
import pandas as pd
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, MENUITEMS
from data_loader import load_schedule_data
from logic.schedule_analysis import calculate_team_stats, calculate_all_schedule_swaps
from ui.schedule_analysis_ui import (
    display_list_view,
    display_table_view,
    display_team_stats,
    display_swap_selection,
    display_all_swaps_analysis,
    display_current_period_overview,
)
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title=f"Schedule Analysis - {PAGE_TITLE}", page_icon=PAGE_ICON, layout=LAYOUT, menu_items=MENUITEMS)

# Ensure data is loaded before displaying the page
if not st.session_state.get('data_loaded', False):
    st.warning("Please load a league dataset on the Home page before using the analyzer.")
    st.page_link("Home.py", label="Go to Home", icon="🏠")
    st.stop()

display_global_sidebar()

st.title(":green[Schedule Analysis]")
st.write(
    "View the complete schedule and results for all fantasy matchups. "
    "Analyze your league's schedule to see how luck plays a factor and "
    "swap schedules between teams to see how it impacts standings and win percentages."
)

# Start from whatever schedule is currently in session state (set by Home/Admin)
schedule_df = st.session_state.get("schedule_data")

# Try to refresh from disk each time the page loads so that schedule25_26.csv
# updates are picked up without a full app restart.
try:
    # Clear the cache for the schedule loader so we always see new CSV content
    load_schedule_data.clear()
    fresh_schedule_df = load_schedule_data()
    if fresh_schedule_df is not None and not fresh_schedule_df.empty:
        schedule_df = fresh_schedule_df
        st.session_state.schedule_data = fresh_schedule_df
except Exception:
    # If anything goes wrong, fall back to whatever we already had in session
    pass

# Final guard: if we still don't have schedule data, bail out with guidance
if schedule_df is None or getattr(schedule_df, "empty", True):
    st.warning("Schedule data is still loading or not available. Please load a league on the Home page.")
    st.page_link("Home.py", label="Go to Home", icon="🏠")
    st.stop()

# Derive period metadata for separating completed weeks vs current week
current_period = None
completed_through_period = None
completed_schedule_df = schedule_df

if "Period Number" in schedule_df.columns:
    period_meta = (
        schedule_df
        .groupby("Period Number")[["Scoring Period", "Date Range"]]
        .first()
        .reset_index()
        .sort_values("Period Number")
    )

    today = pd.Timestamp.today().normalize()
    completed_through_period = None
    current_period = None

    for _, row in period_meta.iterrows():
        period_number = int(row["Period Number"])
        date_range = row.get("Date Range")
        start = end = None
        if pd.notna(date_range):
            parts = str(date_range).split(" - ")
            if len(parts) == 2:
                start = pd.to_datetime(parts[0].strip(), errors="coerce")
                end = pd.to_datetime(parts[1].strip(), errors="coerce")

        if start is not None and pd.notna(start) and end is not None and pd.notna(end):
            if start <= today <= end and current_period is None:
                current_period = period_number
            if end < today:
                if completed_through_period is None or period_number > completed_through_period:
                    completed_through_period = period_number

    if completed_through_period is None:
        completed_schedule_df = schedule_df.iloc[0:0].copy()
    else:
        completed_schedule_df = schedule_df[schedule_df["Period Number"] <= completed_through_period].copy()

all_teams = sorted(list(set(schedule_df["Team 1"]).union(set(schedule_df["Team 2"]))))

swap_tab, standings_tab = st.tabs(["Swap / What-Ifs", "Standings & Schedule"])

with swap_tab:
    # Main analysis section
    st.header("Schedule Swap Analysis")
    with st.spinner("Analyzing all schedule swap scenarios..."):
        # Use only completed weeks for the standings baseline in swap analysis,
        # so that the "what-if" results align with the Team Performance Summary.
        all_swaps_df = calculate_all_schedule_swaps(completed_schedule_df)

    display_swap_selection(all_teams, completed_schedule_df)
    display_all_swaps_analysis(all_teams, all_swaps_df, completed_schedule_df)

with standings_tab:
    # Team Performance Section (based only on completed weeks)
    st.subheader("Team Performance Summary")
    if completed_through_period is not None and current_period is not None:
        if completed_through_period != current_period:
            st.caption(
                f"Standings use completed results through Scoring Period {completed_through_period}. "
                f"Current period is {current_period}."
            )
        else:
            st.caption(f"Standings use results through Scoring Period {completed_through_period}.")

    display_team_stats(completed_schedule_df, calculate_team_stats)

    # Show the in-progress/current period separately
    if current_period is not None:
        display_current_period_overview(schedule_df, current_period)

    # Collapsible sections for data filtering and raw schedule views
    with st.expander("Filter Schedule Data"):
        all_periods = schedule_df["Scoring Period"].unique()
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

    # Apply filters for the views that use them (all periods; user controls via filter)
    filtered_df = schedule_df.copy()
    if 'schedule_selected_teams' in st.session_state and st.session_state.schedule_selected_teams:
        filtered_df = filtered_df[
            (filtered_df["Team 1"].isin(st.session_state.schedule_selected_teams)) |
            (filtered_df["Team 2"].isin(st.session_state.schedule_selected_teams))
        ]
    if 'schedule_selected_periods' in st.session_state and st.session_state.schedule_selected_periods:
        filtered_df = filtered_df[filtered_df["Scoring Period"].isin(st.session_state.schedule_selected_periods)]

    with st.expander("Show Filtered Schedule (List View)"):
        if not filtered_df.empty:
            display_list_view(filtered_df)
        else:
            st.info("No matchups found with the selected filters.")

    with st.expander("Show Filtered Schedule (Table View)"):
        if not filtered_df.empty:
            display_table_view(filtered_df)
        else:
            st.info("No matchups found with the selected filters.")