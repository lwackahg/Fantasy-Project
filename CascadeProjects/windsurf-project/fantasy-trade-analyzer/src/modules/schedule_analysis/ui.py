"""
UI components for the schedule analysis feature.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

def display_list_view(filtered_df):
    """
    Display schedule data in a list view grouped by scoring period.
    
    Args:
        filtered_df (pd.DataFrame): Filtered schedule data
    """
    # Group by scoring period and date range
    periods = filtered_df["Scoring Period"].unique()
    
    for period in periods:
        period_df = filtered_df[filtered_df["Scoring Period"] == period]
        date_range = period_df["Date Range"].iloc[0]
        
        # Display period header
        st.markdown(f"### {period}")
        st.markdown(f"*{date_range}*")
        
        # Display matchups
        for _, row in period_df.iterrows():
            team1 = row["Team 1"]
            score1 = row["Score 1 Display"]
            team2 = row["Team 2"]
            score2 = row["Score 2 Display"]
            winner = row["Winner"]
            
            # Format the matchup with color based on winner
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
    
    Args:
        filtered_df (pd.DataFrame): Filtered schedule data
    """
    # Prepare display columns for table view
    display_df = filtered_df[["Scoring Period", "Date Range", "Team 1", "Score 1 Display", "Team 2", "Score 2 Display", "Winner"]].copy()
    
    # Rename columns for better display
    display_df = display_df.rename(columns={
        "Team 1": "Home Team",
        "Team 2": "Away Team",
        "Score 1 Display": "Home Score",
        "Score 2 Display": "Away Score"
    })
    
    # Show the table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

def display_team_stats(schedule_df, calculate_team_stats):
    """
    Display team performance statistics.
    
    Args:
        schedule_df (pd.DataFrame): The complete schedule data
        calculate_team_stats (function): The function to calculate team stats.
    """
    st.subheader("Team Performance Summary")
    
    # Calculate team stats
    team_stats = calculate_team_stats(schedule_df)
    
    # Format the team stats table
    formatted_stats = team_stats.copy()
    
    # Add win-loss-tie record column
    formatted_stats["Record"] = formatted_stats.apply(
        lambda row: f"{int(row['Wins'])}-{int(row['Losses'])}-{int(row['Ties'])}",
        axis=1
    )
    
    # Reorder columns for better display
    display_columns = [
        "Record", "Win %", "Points For", "Points Against", 
        "Avg Points For", "Avg Points Against", "Total Matchups"
    ]
    
    formatted_stats = formatted_stats[display_columns]
    
    # Display the stats table
    st.dataframe(
        formatted_stats,
        use_container_width=True
    )

def create_win_change_barchart(comparison, team1, team2):
    """
    Create a bar chart showing win percentage changes for the top 6 teams.
    """
    top_changes = comparison.sort_values("Win % Change", ascending=False).head(6)
    fig = px.bar(
        top_changes,
        x=top_changes.index,
        y="Win % Change",
        title="Top Win % Changes",
        labels={"index": "Team", "Win % Change": "Win % Change"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        height=300,
        margin={"t": 40, "b": 30, "l": 40, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FFFFFF"}
    )
    # Highlight swapped teams
    colors = ["#FFD700" if team in [team1, team2] else "#1E90FF" for team in top_changes.index]
    fig.update_traces(marker_color=colors)
    return fig

def display_team_impact_metrics(comparison, original_stats, new_stats, team_name):
    """
    Display the detailed impact metrics for a single team after a swap.
    """
    team_data = comparison.loc[team_name]
    st.markdown(f"### {team_name}")
    
    old_wins = int(team_data["Original Record"].split("-")[0])
    new_wins = int(team_data["New Record"].split("-")[0])
    win_diff = new_wins - old_wins
    
    original_standings = original_stats.sort_values("Win %", ascending=False).index.tolist()
    new_standings = new_stats.sort_values("Win %", ascending=False).index.tolist()
    
    old_position = original_standings.index(team_name) + 1
    new_position = new_standings.index(team_name) + 1
    position_change = old_position - new_position
    
    st.metric(
        label="Record", 
        value=team_data["New Record"], 
        delta=f"{win_diff:+d} wins" if win_diff != 0 else "No change"
    )
    
    st.metric(
        label="Win %", 
        value=f"{team_data['New Win %']:.1f}%", 
        delta=f"{team_data['Win % Change']:+.1f}%"
    )
    
    position_delta = "No change"
    if position_change > 0:
        position_delta = f"Up {position_change} spots"
    elif position_change < 0:
        position_delta = f"Down {abs(position_change)} spots"
        
    st.metric(
        label="Standings Position",
        value=f"#{new_position}",
        delta=position_delta
    )

def display_swap_selection(all_teams, schedule_df, swap_team_schedules, compare_team_stats):
    """
    Display the UI for selecting teams to swap and show the results.
    """
    st.subheader("Manual Schedule Swap")
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", all_teams, index=0, key="swap_team1")
    with col2:
        team2 = st.selectbox("Select Team 2", all_teams, index=1, key="swap_team2")

    if team1 == team2:
        st.warning("Please select two different teams.")
        return

    if st.button("Simulate Swap"):
        swapped_df, original_stats, new_stats = swap_team_schedules(schedule_df, team1, team2)
        comparison = compare_team_stats(original_stats, new_stats)

        st.session_state['schedule_swap_results'] = {
            "team1": team1,
            "team2": team2,
            "comparison": comparison,
            "original_stats": original_stats,
            "new_stats": new_stats
        }

    if 'schedule_swap_results' in st.session_state:
        results = st.session_state['schedule_swap_results']
        team1 = results['team1']
        team2 = results['team2']
        comparison = results['comparison']
        original_stats = results['original_stats']
        new_stats = results['new_stats']
        
        st.markdown("--- ")
        st.subheader(f"Impact of Swapping {team1} and {team2} Schedules")
        
        # Display results in two columns
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            display_team_impact_metrics(comparison, original_stats, new_stats, team1)
        with res_col2:
            display_team_impact_metrics(comparison, original_stats, new_stats, team2)
        
        # Display bar chart
        st.plotly_chart(create_win_change_barchart(comparison, team1, team2), use_container_width=True)


def display_all_swaps_analysis(all_swaps_df, all_teams):
    """
    Displays the analysis of all possible schedule swaps.

    Args:
        all_swaps_df (pd.DataFrame): DataFrame with swap analysis results.
        all_teams (list): List of all team names.
    """
    st.subheader("All Possible Schedule Swaps Analysis")
    st.write("This table shows the impact of every possible one-for-one schedule swap.")

    if all_swaps_df.empty:
        st.warning("Could not calculate swap analysis.")
        return

    # Filter by team
    st.write("") # Add some space
    selected_team = st.selectbox(
        "Filter by team to see all swaps involving them:",
        options=["All Teams"] + all_teams,
        index=0,
        key="swap_filter_team"
    )

    if selected_team == "All Teams":
        filtered_df = all_swaps_df
    else:
        filtered_df = all_swaps_df[
            (all_swaps_df["Team 1"] == selected_team) | (all_swaps_df["Team 2"] == selected_team)
        ].copy()

    st.write(f"Showing {len(filtered_df)} possible swaps.")

    # Define a function to style the position change cells
    def style_change(val):
        if val > 0:
            return f'color: #28a745; font-weight: bold;' # Green for positive
        elif val < 0:
            return f'color: #dc3545; font-weight: bold;' # Red for negative
        else:
            return 'color: #6c757d;' # Gray for no change

    # Prepare display dataframe
    display_df = filtered_df[[
        "Team 1", "Team 2", "Team 1 Position Change", "Team 2 Position Change",
        "Biggest Winner", "Winner Change", "Biggest Loser", "Loser Change"
    ]].copy()

    # Rename for clarity
    display_df.rename(columns={
        "Team 1 Position Change": "Team 1 ∆",
        "Team 2 Position Change": "Team 2 ∆",
        "Winner Change": "Winner ∆",
        "Loser Change": "Loser ∆",
    }, inplace=True)
    
    st.dataframe(
        display_df.style.applymap(style_change, subset=['Team 1 ∆', 'Team 2 ∆', 'Winner ∆', 'Loser ∆']),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Team 1": st.column_config.TextColumn(width="medium"),
            "Team 2": st.column_config.TextColumn(width="medium"),
            "Biggest Winner": st.column_config.TextColumn(width="medium"),
            "Biggest Loser": st.column_config.TextColumn(width="medium"),
        }
    )

    st.markdown("---")
    st.subheader("Detailed Swap Impact")
    
    if not filtered_df.empty:
        # Select a swap to view details
        swap_options = [f"'{row['Team 1']}' and '{row['Team 2']}'" for index, row in filtered_df.iterrows()]
        selected_swap_str = st.selectbox("Select a swap to see the detailed impact on the whole league:", options=swap_options, key="swap_detail_select")
        
        if selected_swap_str:
            # Find the selected row
            selected_row_index = swap_options.index(selected_swap_str)
            selected_row = filtered_df.iloc[selected_row_index]
            
            st.write(f"Showing full impact for swapping **{selected_row['Team 1']}** and **{selected_row['Team 2']}** schedules:")
            
            # Get all changes
            all_changes = selected_row["All Changes"]
            changes_df = pd.DataFrame(list(all_changes.items()), columns=['Team', 'Position Change'])
            changes_df = changes_df.sort_values(by='Position Change', ascending=False).reset_index(drop=True)
            
            st.dataframe(
                changes_df.style.applymap(style_change, subset=['Position Change']),
                use_container_width=True,
                hide_index=True
            )
