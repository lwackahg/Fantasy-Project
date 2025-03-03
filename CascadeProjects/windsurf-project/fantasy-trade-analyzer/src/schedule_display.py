import streamlit as st
import pandas as pd
from pathlib import Path
import os
import re
import csv
from data_loader import TEAM_MAPPINGS

def load_schedule_data():
    """
    Load and parse the schedule data from CSV file.
    
    Returns:
        pd.DataFrame: Processed schedule data with proper columns
    """
    try:
        # Get path to schedule.csv
        schedule_path = Path(__file__).parent.parent / "data" / "schedule" / "schedule.csv"
        
        if not os.path.exists(schedule_path):
            st.error("Schedule data file not found.")
            return None
        
        # Process the data using a more robust approach
        data = []
        current_period = None
        current_date_range = None
        
        # Read the file using csv module which handles quotes properly
        with open(schedule_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            
            for row in csv_reader:
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                
                # Check if this is a scoring period line
                if row[0].startswith("Scoring Period"):
                    current_period = row[0].strip()
                    continue
                
                # Check if this is a date range line
                if row[0].startswith("(") and ")" in row[0]:
                    current_date_range = row[0].strip('()"')
                    continue
                
                # Process matchup line - we expect 4 columns: Team1, Score1, Team2, Score2
                if len(row) >= 4:
                    team1 = row[0].strip()
                    score1_str = row[1].strip()
                    team2 = row[2].strip()
                    score2_str = row[3].strip()
                    
                    # Format the matchup string
                    matchup_str = f"{team1} - {score1_str} vs {team2} - {score2_str}"
                    
                    # Convert scores to integers, handling commas in numbers
                    try:
                        score1 = int(score1_str.replace(",", ""))
                    except ValueError:
                        # If we can't convert to int, it might be a date or other text
                        # Just set to 0 and continue without warning
                        score1 = 0
                    
                    try:
                        score2 = int(score2_str.replace(",", ""))
                    except ValueError:
                        score2 = 0
                    
                    # Only add valid matchups (both teams have names)
                    if team1 and team2 and not (team1.startswith("(") or team2.startswith("(")):
                        data.append({
                            "Scoring Period": current_period,
                            "Date Range": current_date_range,
                            "Team 1": team1,
                            "Score 1": score1,
                            "Score 1 Display": score1_str,
                            "Team 2": team2,
                            "Score 2": score2,
                            "Score 2 Display": score2_str,
                            "Matchup": matchup_str
                        })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            st.error("No data was parsed from the schedule file.")
            return None
        
        # Add winner column
        df["Winner"] = df.apply(
            lambda row: row["Team 1"] if row["Score 1"] > row["Score 2"] 
            else (row["Team 2"] if row["Score 2"] > row["Score 1"] else "Tie"),
            axis=1
        )
        
        # Extract period number for sorting
        df["Period Number"] = df["Scoring Period"].apply(
            lambda x: int(re.search(r"Scoring Period (\d+)", x).group(1)) if re.search(r"Scoring Period (\d+)", x) else 0
        )
        
        # Sort by period
        df = df.sort_values("Period Number")
        
        return df
    
    except Exception as e:
        st.error(f"Error loading schedule data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def display_schedule_page():
    """
    Display the schedule data in a table format with filtering options.
    """
    st.title("Fantasy League Schedule")
    st.write("View the complete schedule and results for all fantasy matchups.")
    
    # Load the schedule data
    schedule_df = load_schedule_data()
    
    if schedule_df is None or schedule_df.empty:
        st.error("No schedule data available.")
        return
    
    # Add filters
    st.subheader("Filter Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get unique periods for filtering
        periods = schedule_df["Scoring Period"].unique().tolist()
        selected_period = st.selectbox(
            "Filter by Scoring Period",
            ["All Periods"] + periods
        )
    
    with col2:
        # Get unique teams for filtering
        all_teams = set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())
        all_teams = sorted(list(all_teams))
        
        selected_team = st.selectbox(
            "Filter by Team",
            ["All Teams"] + all_teams
        )
    
    with col3:
        # Add view type option
        view_type = st.radio(
            "View Type",
            ["List View", "Table View"]
        )
    
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
        st.subheader("Schedule Data")
        
        if view_type == "List View":
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
        else:  # Table View
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
        
        # Add team performance summary
        st.subheader("Team Performance Summary")
        
        # Calculate team stats
        team_stats = calculate_team_stats(schedule_df)
        
        # Display team stats
        st.dataframe(
            team_stats,
            use_container_width=True,
            hide_index=False
        )
    else:
        st.info("No matchups found with the selected filters.")

def calculate_team_stats(schedule_df):
    """
    Calculate performance statistics for each team.
    
    Args:
        schedule_df (pd.DataFrame): The schedule data
        
    Returns:
        pd.DataFrame: Team statistics
    """
    # Initialize stats dictionary
    team_stats = {}
    
    # Process each matchup
    for _, row in schedule_df.iterrows():
        team1 = row["Team 1"]
        team2 = row["Team 2"]
        score1 = row["Score 1"]
        score2 = row["Score 2"]
        
        # Initialize team entries if they don't exist
        for team in [team1, team2]:
            if team not in team_stats:
                team_stats[team] = {
                    "Wins": 0,
                    "Losses": 0,
                    "Ties": 0,
                    "Points For": 0,
                    "Points Against": 0,
                    "Total Matchups": 0
                }
        
        # Update team1 stats
        team_stats[team1]["Points For"] += score1
        team_stats[team1]["Points Against"] += score2
        team_stats[team1]["Total Matchups"] += 1
        
        # Update team2 stats
        team_stats[team2]["Points For"] += score2
        team_stats[team2]["Points Against"] += score1
        team_stats[team2]["Total Matchups"] += 1
        
        # Update win/loss/tie records
        if score1 > score2:
            team_stats[team1]["Wins"] += 1
            team_stats[team2]["Losses"] += 1
        elif score2 > score1:
            team_stats[team2]["Wins"] += 1
            team_stats[team1]["Losses"] += 1
        else:
            team_stats[team1]["Ties"] += 1
            team_stats[team2]["Ties"] += 1
    
    # Convert to DataFrame
    stats_df = pd.DataFrame.from_dict(team_stats, orient="index")
    
    # Calculate win percentage
    stats_df["Win %"] = stats_df.apply(
        lambda row: round(row["Wins"] / row["Total Matchups"] * 100, 1) if row["Total Matchups"] > 0 else 0,
        axis=1
    )
    
    # Calculate average points
    stats_df["Avg Points For"] = stats_df["Points For"] / stats_df["Total Matchups"]
    stats_df["Avg Points Against"] = stats_df["Points Against"] / stats_df["Total Matchups"]
    
    # Round numeric columns
    for col in ["Avg Points For", "Avg Points Against"]:
        stats_df[col] = stats_df[col].round(1)
    
    # Sort by wins (descending), then by points (descending)
    stats_df = stats_df.sort_values(["Wins", "Points For"], ascending=[False, False])
    
    return stats_df
