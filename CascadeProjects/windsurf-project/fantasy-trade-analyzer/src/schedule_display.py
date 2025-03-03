import streamlit as st
from data_loader import load_schedule_data, calculate_team_stats
from schedule_analysis import swap_team_schedules, compare_team_stats

def display_schedule_page():
    """
    Display the schedule data in a table format with filtering options.
    """
    # Initialize session state variables if they don't exist
    if 'schedule_view_type' not in st.session_state:
        st.session_state.schedule_view_type = "List View"
    if 'schedule_selected_period' not in st.session_state:
        st.session_state.schedule_selected_period = "All Periods"
    if 'schedule_selected_team' not in st.session_state:
        st.session_state.schedule_selected_team = "All Teams"
    if 'schedule_swap_team1' not in st.session_state:
        st.session_state.schedule_swap_team1 = None
    if 'schedule_swap_team2' not in st.session_state:
        st.session_state.schedule_swap_team2 = None
    if 'schedule_swap_performed' not in st.session_state:
        st.session_state.schedule_swap_performed = False
    
    st.title("Fantasy League Schedule")
    st.write("View the complete schedule and results for all fantasy matchups.")
    
    # Load the schedule data
    schedule_df = load_schedule_data()
    
    if schedule_df is None or schedule_df.empty:
        st.error("No schedule data available.")
        return
    
    # Add filters in an expander for cleaner UI
    with st.expander("Filter Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get unique periods for filtering
            periods = schedule_df["Scoring Period"].unique().tolist()
            
            # Safely get index for the selected period
            period_index = 0
            if st.session_state.schedule_selected_period != "All Periods" and st.session_state.schedule_selected_period in periods:
                period_index = periods.index(st.session_state.schedule_selected_period) + 1
            
            selected_period = st.selectbox(
                "Filter by Scoring Period",
                ["All Periods"] + periods,
                index=period_index
            )
            # Update session state
            st.session_state.schedule_selected_period = selected_period
        
        with col2:
            # Get unique teams for filtering
            all_teams = set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())
            all_teams = sorted(list(all_teams))
            
            # Safely get index for the selected team
            team_index = 0
            if st.session_state.schedule_selected_team != "All Teams" and st.session_state.schedule_selected_team in all_teams:
                team_index = all_teams.index(st.session_state.schedule_selected_team) + 1
            
            selected_team = st.selectbox(
                "Filter by Team",
                ["All Teams"] + all_teams,
                index=team_index
            )
            # Update session state
            st.session_state.schedule_selected_team = selected_team
        
        with col3:
            # Add view type option
            view_index = 0 if st.session_state.schedule_view_type == "List View" else 1
            
            view_type = st.radio(
                "View Type",
                ["List View", "Table View"],
                index=view_index
            )
            # Update session state
            st.session_state.schedule_view_type = view_type
    
    # Apply filters
    filtered_df = schedule_df.copy()
    
    if selected_period != "All Periods":
        filtered_df = filtered_df[filtered_df["Scoring Period"] == selected_period]
    
    if selected_team != "All Teams":
        filtered_df = filtered_df[
            (filtered_df["Team 1"] == selected_team) | 
            (filtered_df["Team 2"] == selected_team)
        ]
    
    # Display the filtered schedule
    if not filtered_df.empty:
        # Create tabs for Schedule, Team Stats, and Schedule Swap
        tab1, tab2, tab3 = st.tabs(["Schedule", "Team Performance", "Schedule Swap"])
        
        with tab1:
            if view_type == "List View":
                display_list_view(filtered_df)
            else:  # Table View
                display_table_view(filtered_df)
            
            # Add a download button for the schedule data
            st.download_button(
                label="Download Schedule Data as CSV",
                data=filtered_df.to_csv(index=False),
                file_name="fantasy_schedule.csv",
                mime="text/csv",
            )
        
        with tab2:
            display_team_stats(schedule_df)
            
        with tab3:
            display_schedule_swap(schedule_df)
    else:
        st.info("No matchups found with the selected filters.")

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

def display_team_stats(schedule_df):
    """
    Display team performance statistics.
    
    Args:
        schedule_df (pd.DataFrame): The complete schedule data
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

def display_schedule_swap(schedule_df):
    """
    Display the schedule swap tool UI.
    
    Args:
        schedule_df (pd.DataFrame): The complete schedule data
    """
    st.subheader("Schedule Swap Analysis")
    st.write("See how standings would change if two teams swapped schedules.")
    
    st.markdown("## Schedule Swap Analysis")
    
    st.markdown("""
    This feature allows you to simulate what would happen if two teams swapped schedules.
    
    **How it works:**
    1. Team A takes Team B's schedule (faces Team B's opponents)
    2. Team B takes Team A's schedule (faces Team A's opponents)
    3. Team A uses Team B's original scores, and Team B uses Team A's original scores
    4. Standings are recalculated based on these new matchups
    
    This helps answer the question: "How would Team A have performed with Team B's schedule and scores?"
    """)
    
    # Get all teams for selection
    all_teams = set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())
    all_teams = sorted(list(all_teams))
    
    # Create two columns for team selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Get index for team1 if it exists in session state
        team1_index = 0
        if st.session_state.schedule_swap_team1 in all_teams:
            team1_index = all_teams.index(st.session_state.schedule_swap_team1)
            
        team1 = st.selectbox(
            "Select First Team",
            all_teams,
            index=team1_index,
            key="swap_team1_select"
        )
        st.session_state.schedule_swap_team1 = team1
    
    with col2:
        # Filter out team1 from options for team2
        team2_options = [team for team in all_teams if team != team1]
        
        # Get index for team2 if it exists in session state and is not team1
        team2_index = 0
        if st.session_state.schedule_swap_team2 in team2_options:
            team2_index = team2_options.index(st.session_state.schedule_swap_team2)
            
        team2 = st.selectbox(
            "Select Second Team",
            team2_options,
            index=team2_index,
            key="swap_team2_select"
        )
        st.session_state.schedule_swap_team2 = team2
    
    # Button to perform the swap analysis
    if st.button("Analyze Schedule Swap"):
        st.session_state.schedule_swap_performed = True
        
        with st.spinner("Analyzing schedule swap..."):
            # Perform the schedule swap
            swapped_df, original_stats, new_stats = swap_team_schedules(schedule_df, team1, team2)
            
            if swapped_df is not None:
                # Display the comparison
                st.subheader("Impact on Standings")
                
                # Get the comparison stats
                comparison = compare_team_stats(original_stats, new_stats)
                
                # Highlight the swapped teams
                st.markdown(f"### Teams Swapped: **{team1}** and **{team2}**")
                
                # Display the comparison table with conditional formatting
                st.dataframe(
                    comparison.style.apply(
                        lambda x: ['background-color: #232300' if i in [team1, team2] else '' 
                                 for i in x.index],
                        axis=0
                    ),
                    use_container_width=True
                )
                
                # Show detailed impact for the swapped teams
                st.subheader("Detailed Impact on Swapped Teams")
                
                # Create a filtered dataframe with just the swapped teams
                swapped_teams_comparison = comparison.loc[[team1, team2]]
                
                # Display in a more readable format
                for team in [team1, team2]:
                    team_data = swapped_teams_comparison.loc[team]
                    
                    st.markdown(f"#### {team}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Original Record:** {team_data['Original Record']}")
                        st.markdown(f"**Original Win %:** {team_data['Original Win %']:.3f}")
                    
                    with col2:
                        st.markdown(f"**New Record:** {team_data['New Record']}")
                        st.markdown(f"**New Win %:** {team_data['New Win %']:.3f}")
                    
                    # Show the change with an arrow indicator
                    win_pct_change = team_data["Win % Change"]
                    if win_pct_change > 0:
                        st.markdown(f"**Win % Change:** +{win_pct_change:.3f} ")
                    elif win_pct_change < 0:
                        st.markdown(f"**Win % Change:** {win_pct_change:.3f} ")
                    else:
                        st.markdown(f"**Win % Change:** {win_pct_change:.3f} ")
                    
                    st.markdown("---")
                
                # Option to view the swapped schedule
                with st.expander("View Swapped Schedule"):
                    # Only show matchups involving the swapped teams
                    swapped_team_matchups = swapped_df[
                        (swapped_df["Team 1"] == team1) | 
                        (swapped_df["Team 2"] == team1) |
                        (swapped_df["Team 1"] == team2) | 
                        (swapped_df["Team 2"] == team2)
                    ]
                    
                    # Display the swapped matchups
                    display_table_view(swapped_team_matchups)
            else:
                st.error("Failed to perform schedule swap analysis.")
    
    # Display instructions if no swap has been performed yet
    if not st.session_state.schedule_swap_performed:
        st.info("Select two teams and click 'Analyze Schedule Swap' to see how the standings would change if these teams swapped schedules.")
