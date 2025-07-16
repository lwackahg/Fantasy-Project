import streamlit as st
from logic.schedule_analysis import calculate_team_stats
from ui.schedule_analysis_ui import (
    display_list_view,
    display_table_view,
    display_team_stats,
    display_swap_selection,
    display_all_swaps_analysis
)
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Schedule Analysis", layout="wide")

display_global_sidebar()

st.title(":green[Schedule Analysis]")
st.write(
    "View the complete schedule and results for all fantasy matchups. "
    "Analyze your league's schedule to see how luck plays a factor and "
    "swap schedules between teams to see how it impacts standings and win percentages."
)

if 'schedule_data' not in st.session_state or st.session_state.schedule_data.empty:
    st.warning("Schedule data not found. Please go to the Home page and upload your schedule file first.")
    st.stop()

schedule_df = st.session_state.schedule_data

# Main analysis section
st.header("Schedule Swap Analysis")
all_teams = sorted(list(set(schedule_df["Team 1"]).union(set(schedule_df["Team 2"]))))
display_swap_selection(all_teams, schedule_df)
display_all_swaps_analysis(all_teams)

st.markdown("***")

# Team Performance Section
st.subheader("Team Performance Summary")
display_team_stats(schedule_df, calculate_team_stats)

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

# Apply filters for the views that use them
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