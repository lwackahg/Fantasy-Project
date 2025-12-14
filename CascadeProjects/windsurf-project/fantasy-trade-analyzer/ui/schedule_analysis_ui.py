"""
UI components for the schedule analysis feature.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_compat import plotly_chart
from logic.schedule_analysis import calculate_team_stats, swap_team_schedules, compare_team_stats

def display_list_view(filtered_df):
    """
    Display schedule data in a list view grouped by scoring period.
    """
    periods = filtered_df["Scoring Period"].unique()
    
    for period in periods:
        period_df = filtered_df[filtered_df["Scoring Period"] == period]
        date_range = period_df["Date Range"].iloc[0]
        
        st.markdown(f"### {period}")
        st.markdown(f"*{date_range}*")
        
        for _, row in period_df.iterrows():
            team1, score1, team2, score2, winner = row["Team 1"], row["Score 1 Display"], row["Team 2"], row["Score 2 Display"], row["Winner"]
            if winner == team1:
                st.markdown(f"**{team1}** - {score1} vs {team2} - {score2}")
            elif winner == team2:
                st.markdown(f"{team1} - {score1} vs **{team2}** - {score2}")
            else:
                st.markdown(f"{team1} - {score1} vs {team2} - {score2}")
        st.markdown("---")

def display_table_view(filtered_df):
    """
    Display schedule data in a table view.
    """
    display_df = filtered_df[["Scoring Period", "Date Range", "Team 1", "Score 1 Display", "Team 2", "Score 2 Display", "Winner"]].copy()
    display_df = display_df.rename(columns={
        "Team 1": "Home Team", "Team 2": "Away Team",
        "Score 1 Display": "Home Score", "Score 2 Display": "Away Score"
    })
    st.dataframe(display_df, width="stretch", hide_index=True)

def display_team_stats(schedule_df, calculate_team_stats):
    """
    Display team performance statistics.
    """
    team_stats = calculate_team_stats(schedule_df)
    _display_team_standings_from_stats(team_stats, heading="Current Standings")


def _display_team_standings_from_stats(team_stats, heading: str = "Current Standings"):
    formatted_stats = team_stats.copy()
    formatted_stats["Record"] = formatted_stats.apply(
        lambda row: f"{int(row['Wins'])}-{int(row['Losses'])}-{int(row['Ties'])}", axis=1
    )
    # Sort by Win % so table order matches the standings positions used in analysis
    if "Win %" in formatted_stats.columns:
        formatted_stats = formatted_stats.sort_values("Win %", ascending=False)
    formatted_stats.reset_index(inplace=True)
    formatted_stats.rename(columns={'index': 'Team'}, inplace=True)
    display_columns = ["Team", "Record", "Win %", "Points For", "Points Against"]

    st.write(f"#### {heading}")
    st.dataframe(
        formatted_stats[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "Team": st.column_config.TextColumn(width="medium"),
            "Win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        }
    )

def create_win_change_barchart(comparison, team1, team2):
    """
    Create a bar chart showing win percentage changes for the top 6 teams.
    """
    top_changes = comparison.sort_values("Win % Change", ascending=False).head(6)
    fig = px.bar(
        top_changes, x=top_changes.index, y="Win % Change", title="Top Win % Changes",
        labels={"index": "Team", "Win % Change": "Win % Change"}, color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        height=300, margin={"t": 40, "b": 30, "l": 40, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#FFFFFF"}
    )
    colors = ["#FFD700" if team in [team1, team2] else "#1E90FF" for team in top_changes.index]
    fig.update_traces(marker_color=colors)
    return fig

def display_team_impact_metrics(comparison, original_stats, new_stats, team_name):
    """
    Display the detailed impact metrics for a single team after a swap.
    """
    team_data = comparison.loc[team_name]
    st.markdown(f"### {team_name}")
    
    win_diff = int(team_data["New Record"].split("-")[0]) - int(team_data["Original Record"].split("-")[0])
    
    original_standings = original_stats.sort_values("Win %", ascending=False).index.tolist()
    new_standings = new_stats.sort_values("Win %", ascending=False).index.tolist()
    
    old_position = original_standings.index(team_name) + 1
    new_position = new_standings.index(team_name) + 1
    position_change = old_position - new_position
    
    st.metric(label="Record", value=team_data["New Record"], delta=f"{win_diff:+d} wins" if win_diff != 0 else "No change")
    win_pct_change = team_data['Win % Change']
    win_pct_delta = f"{win_pct_change:+.1f}%" if win_pct_change != 0 else "No change"
    st.metric(label="Win %", value=f"{team_data['New Win %']:.1f}%", delta=win_pct_delta)
    
    position_delta_str = f"Up {position_change}" if position_change > 0 else f"Down {abs(position_change)}"
    position_delta = f"{position_delta_str} spots" if position_change != 0 else "No change"
    st.metric(label="Standings Position", value=f"#{new_position}", delta=position_delta)


def _get_swap_matchups_view(schedule_df: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
    """Return a compact view of all matchups involving the two swapped teams.

    This is used to show the *actual* schedule before and after a swap, so the
    user can visually confirm that matchups for Team 1 and Team 2 were
    reassigned correctly.
    """
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()

    mask = (schedule_df["Team 1"].isin([team1, team2])) | (schedule_df["Team 2"].isin([team1, team2]))
    subset = schedule_df[mask].copy()
    if subset.empty:
        return subset

    if "Period Number" in subset.columns:
        subset = subset.sort_values(["Period Number", "Scoring Period"])

    cols = [
        "Scoring Period",
        "Date Range",
        "Team 1",
        "Score 1 Display",
        "Team 2",
        "Score 2 Display",
        "Winner",
    ]
    existing_cols = [c for c in cols if c in subset.columns]
    display_df = subset[existing_cols].copy()

    rename_map = {
        "Team 1": "Home Team",
        "Score 1 Display": "Home Score",
        "Team 2": "Away Team",
        "Score 2 Display": "Away Score",
    }
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
    return display_df


def _display_swap_matchups(original_schedule: pd.DataFrame, swapped_schedule: pd.DataFrame, team1: str, team2: str):
    """Show before/after matchups for the two teams involved in a swap."""
    if original_schedule is None or swapped_schedule is None:
        return

    before_df = _get_swap_matchups_view(original_schedule, team1, team2)
    after_df = _get_swap_matchups_view(swapped_schedule, team1, team2)

    if before_df.empty and after_df.empty:
        return

    st.markdown("#### Matchups Before vs After Swap")
    st.caption("Only matchups involving the two swapped teams are shown.")

    col_before, col_after = st.columns(2)
    with col_before:
        st.write("Before Swap")
        st.dataframe(before_df, width="stretch", hide_index=True)
    with col_after:
        st.write("After Swap")
        st.dataframe(after_df, width="stretch", hide_index=True)

def display_swap_selection(all_teams, schedule_df):
    """
    Display the UI for selecting teams to swap and show the results.
    """
    st.subheader("Manual Schedule Swap")
    col1, col2 = st.columns(2)
    team1 = col1.selectbox("Select Team 1", all_teams, index=0, key="swap_team1")
    team2 = col2.selectbox("Select Team 2", all_teams, index=1, key="swap_team2")

    if team1 == team2:
        st.warning("Please select two different teams.")
        return

    if st.button("Simulate Swap"):
        swapped_df, original_stats, new_stats = swap_team_schedules(schedule_df, team1, team2)
        comparison = compare_team_stats(original_stats, new_stats)
        st.session_state['schedule_swap_results'] = {
            "team1": team1, "team2": team2, "comparison": comparison,
            "original_stats": original_stats, "new_stats": new_stats,
            "original_schedule": schedule_df,
            "swapped_schedule": swapped_df,
        }

    if 'schedule_swap_results' in st.session_state:
        results = st.session_state['schedule_swap_results']
        st.markdown("--- ")
        st.subheader(f"Impact of Swapping {results['team1']} and {results['team2']} Schedules")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            display_team_impact_metrics(results['comparison'], results['original_stats'], results['new_stats'], results['team1'])
        with res_col2:
            display_team_impact_metrics(results['comparison'], results['original_stats'], results['new_stats'], results['team2'])
        plotly_chart(create_win_change_barchart(results['comparison'], results['team1'], results['team2']), width="stretch")

        st.markdown("---")
        standings_col1, standings_col2 = st.columns(2)
        with standings_col1:
            _display_team_standings_from_stats(results["original_stats"], heading="Standings Before Swap")
        with standings_col2:
            _display_team_standings_from_stats(results["new_stats"], heading="Standings After Swap")

        _display_swap_matchups(
            results.get("original_schedule"),
            results.get("swapped_schedule"),
            results["team1"],
            results["team2"],
        )

def display_all_swaps_analysis(all_teams, all_swaps_df, schedule_df):
    """
    Displays the analysis of all possible schedule swaps from pre-calculated data.
    """
    st.subheader("All Possible Schedule Swaps")
    st.write(
        "This table shows the effect of every possible one-for-one schedule swap. "
        "When showing all teams, rows are ordered by the total movement in standings "
        "for the two swapped teams."
    )

    if all_swaps_df.empty:
        st.warning("Swap analysis data could not be calculated.")
        return

    selected_team = st.selectbox("Filter by team:", options=["All Teams"] + all_teams, index=0, key="swap_filter_team")

    filtered_df = all_swaps_df if selected_team == "All Teams" else all_swaps_df[(all_swaps_df["Team 1"] == selected_team) | (all_swaps_df["Team 2"] == selected_team)].copy()

    # Sorting logic: for a specific team, sort by that team's position change (best improvements first).
    # For the "All Teams" view, sort by overall Impact (total movement for both teams).
    if selected_team == "All Teams":
        if "Impact" in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by="Impact", ascending=False)
    else:
        def _selected_team_change(row):
            if row["Team 1"] == selected_team:
                return row["Team 1 Position Change"]
            if row["Team 2"] == selected_team:
                return row["Team 2 Position Change"]
            return 0

        filtered_df = filtered_df.copy()
        filtered_df["Selected Team Change"] = filtered_df.apply(_selected_team_change, axis=1)
        sort_keys = ["Selected Team Change"]
        if "Impact" in filtered_df.columns:
            sort_keys.append("Impact")
        sort_keys.extend(["Team 1 Position Change", "Team 2 Position Change"])
        filtered_df = filtered_df.sort_values(
            by=sort_keys,
            ascending=[False] + [False] * (len(sort_keys) - 1),
        )

    st.write(f"Showing {len(filtered_df)} possible swaps.")

    def style_change(val):
        return 'color: #28a745; font-weight: bold;' if val > 0 else ('color: #dc3545; font-weight: bold;' if val < 0 else 'color: #6c757d;')

    # Always show a consistent set of columns; when a team is selected we add
    # a "Selected Team ΔPos" column while keeping the per-team position changes
    # and biggest winner/loser metrics. Impact is still used internally for
    # sorting but is not shown as a separate column.
    base_columns = [
        "Team 1",
        "Team 2",
        "Team 1 Position Change",
        "Team 2 Position Change",
        "Biggest Winner",
        "Winner Change",
        "Biggest Loser",
        "Loser Change",
    ]

    display_columns = base_columns.copy()
    if "Selected Team Change" in filtered_df.columns:
        display_columns.insert(2, "Selected Team Change")  # after Team 2

    display_columns = [c for c in display_columns if c in filtered_df.columns]
    display_df = filtered_df[display_columns].copy()

    style_cols = [
        c
        for c in [
            "Selected Team Change",
            "Team 1 Position Change",
            "Team 2 Position Change",
            "Winner Change",
            "Loser Change",
        ]
        if c in display_df.columns
    ]

    st.dataframe(
        display_df.style.map(style_change, subset=style_cols),
        width="stretch", hide_index=True,
        column_config={
            "Team 1": st.column_config.TextColumn("Team 1", help="The first team in the schedule swap.", width="medium"),
            "Team 2": st.column_config.TextColumn("Team 2", help="The second team in the schedule swap.", width="medium"),
            "Selected Team Change": st.column_config.NumberColumn("Selected Team ΔPos", help="Change in standings position for the team chosen in the filter (positive = moves up)."),
            "Team 1 Position Change": st.column_config.NumberColumn("T1 Change", help="Team 1's change in standings position."),
            "Team 2 Position Change": st.column_config.NumberColumn("T2 Change", help="Team 2's change in standings position."),
            "Biggest Winner": st.column_config.TextColumn("Biggest Winner", help="The team that benefits most from the swap (largest positive position change).", width="medium"),
            "Winner Change": st.column_config.NumberColumn("Winner Change", help="The standings position change for the biggest winner."),
            "Biggest Loser": st.column_config.TextColumn("Biggest Loser", help="The team that is most hurt by the swap (largest negative position change).", width="medium"),
            "Loser Change": st.column_config.NumberColumn("Loser Change", help="The standings position change for the biggest loser."),
        }
    )


def display_current_period_overview(schedule_df, current_period: int | None):
    """Show the live/current scoring period's matchups and scores.

    This is separate from the main standings (which are based on completed
    weeks only) so that you can see how the in-progress period is shaping up
    without affecting the official W/L record.
    """
    if current_period is None or "Period Number" not in schedule_df.columns:
        return

    period_df = schedule_df[schedule_df["Period Number"] == current_period]
    if period_df.empty:
        return

    scoring_label = str(period_df["Scoring Period"].iloc[0]) if hasattr(period_df["Scoring Period"], "iloc") else None
    if not scoring_label:
        scoring_label = f"Scoring Period {current_period}"

    st.write(f"#### This Period: {scoring_label}")
    if "Date Range" in period_df.columns:
        st.caption(str(period_df["Date Range"].iloc[0]))

    display_df = period_df[["Team 1", "Score 1 Display", "Team 2", "Score 2 Display", "Winner"]].copy()
    display_df = display_df.rename(columns={
        "Team 1": "Home Team",
        "Score 1 Display": "Home Score",
        "Team 2": "Away Team",
        "Score 2 Display": "Away Score",
    })

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )
