import streamlit as st
from data_loader import load_schedule_data, calculate_team_stats

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
        # Create tabs for Schedule and Team Stats
        tab1, tab2 = st.tabs(["Schedule", "Team Performance"])
        
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
    
    # Sort by Win % descending
    formatted_stats = formatted_stats.sort_values("Win %", ascending=False)
    
    # Display team stats
    st.dataframe(
        formatted_stats,
        use_container_width=True,
        hide_index=False
    )
