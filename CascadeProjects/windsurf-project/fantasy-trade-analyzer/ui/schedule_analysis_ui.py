"""
UI components for the schedule analysis feature.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
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
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_team_stats(schedule_df, calculate_team_stats):
    """
    Display team performance statistics.
    """
    team_stats = calculate_team_stats(schedule_df)
    formatted_stats = team_stats.copy()
    formatted_stats["Record"] = formatted_stats.apply(
        lambda row: f"{int(row['Wins'])}-{int(row['Losses'])}-{int(row['Ties'])}", axis=1
    )
    formatted_stats.reset_index(inplace=True)
    formatted_stats.rename(columns={'index': 'Team'}, inplace=True)
    display_columns = ["Team", "Record", "Win %", "Points For", "Points Against"]

    st.write("#### Current Standings")
    st.dataframe(
        formatted_stats[display_columns],
        use_container_width=True,
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
            "original_stats": original_stats, "new_stats": new_stats
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
        st.plotly_chart(create_win_change_barchart(results['comparison'], results['team1'], results['team2']), use_container_width=True)

def display_all_swaps_analysis(all_teams, all_swaps_df):
    """
    Displays the analysis of all possible schedule swaps from pre-calculated data.
    """
    st.subheader("All Possible Schedule Swaps")
    st.write("This table shows the impact of every possible one-for-one schedule swap. Higher 'Impact' scores indicate more significant changes to the league standings.")

    if all_swaps_df.empty:
        st.warning("Swap analysis data could not be calculated.")
        return

    selected_team = st.selectbox("Filter by team:", options=["All Teams"] + all_teams, index=0, key="swap_filter_team")

    filtered_df = all_swaps_df if selected_team == "All Teams" else all_swaps_df[(all_swaps_df["Team 1"] == selected_team) | (all_swaps_df["Team 2"] == selected_team)].copy()

    st.write(f"Showing {len(filtered_df)} possible swaps.")

    def style_change(val):
        return 'color: #28a745; font-weight: bold;' if val > 0 else ('color: #dc3545; font-weight: bold;' if val < 0 else 'color: #6c757d;')

    display_df = filtered_df[["Team 1", "Team 2", "Team 1 Position Change", "Team 2 Position Change", "Biggest Winner", "Winner Change", "Biggest Loser", "Loser Change"]].copy()

    st.dataframe(
        display_df.style.applymap(style_change, subset=['Team 1 Position Change', 'Team 2 Position Change', 'Winner Change', 'Loser Change']),
        use_container_width=True, hide_index=True,
        column_config={
            "Team 1": st.column_config.TextColumn("Team 1", help="The first team in the schedule swap.", width="medium"),
            "Team 2": st.column_config.TextColumn("Team 2", help="The second team in the schedule swap.", width="medium"),
            "Team 1 Position Change": st.column_config.NumberColumn("T1 Change", help="Team 1's change in standings position."),
            "Team 2 Position Change": st.column_config.NumberColumn("T2 Change", help="Team 2's change in standings position."),
            "Biggest Winner": st.column_config.TextColumn("Biggest Winner", help="The team (not T1 or T2) that benefits most from the swap.", width="medium"),
            "Winner Change": st.column_config.NumberColumn("Winner Change", help="The standings position change for the biggest winner."),
            "Biggest Loser": st.column_config.TextColumn("Biggest Loser", help="The team (not T1 or T2) that is most hurt by the swap.", width="medium"),
            "Loser Change": st.column_config.NumberColumn("Loser Change", help="The standings position change for the biggest loser."),
        }
    )

    st.markdown("---")
    st.subheader("Detailed Swap Impact")
    
    if not filtered_df.empty:
        swap_options = [f"'{row['Team 1']}' and '{row['Team 2']}'" for _, row in filtered_df.iterrows()]
        selected_swap_str = st.selectbox("Select a swap for detailed league impact:", options=swap_options, key="swap_detail_select")
        
        if selected_swap_str:
            selected_row = filtered_df.iloc[swap_options.index(selected_swap_str)]
            st.write(f"Showing full impact for swapping **{selected_row['Team 1']}** and **{selected_row['Team 2']}** schedules:")
            
            all_changes = selected_row["All Changes"]
            changes_df = pd.DataFrame(list(all_changes.items()), columns=['Team', 'Position Change']).sort_values(by='Position Change', ascending=False).reset_index(drop=True)
            
            st.dataframe(changes_df.style.applymap(style_change, subset=['Position Change']), use_container_width=True, hide_index=True)
