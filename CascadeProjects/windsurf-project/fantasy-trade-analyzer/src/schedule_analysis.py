"""
Schedule analysis module for fantasy trade analyzer.
Contains functions for analyzing and manipulating schedule data.
"""

import pandas as pd
from data_loader import calculate_team_stats

def swap_team_schedules(schedule_df, team1, team2):
    """
    Swap the schedules between two teams and recalculate standings.
    
    This simulates what would happen if:
    - Team A and Team B completely swap places in the schedule
    - Their scores are also swapped
    
    Args:
        schedule_df (pd.DataFrame): The original schedule data
        team1 (str): First team to swap
        team2 (str): Second team to swap
        
    Returns:
        pd.DataFrame: Modified schedule with swapped matchups
        pd.DataFrame: Original team stats
        pd.DataFrame: New team stats after swap
    """
    if schedule_df is None or schedule_df.empty:
        return None, None, None
    
    # Create a copy of the original schedule to modify
    swapped_df = schedule_df.copy()
    
    # Get original team stats for comparison
    original_stats = calculate_team_stats(schedule_df)
    
    # Create dictionaries to store team matchups by period
    # Structure: {period: [(team, opponent, score, is_team1_position)]}
    team1_matchups = {}
    team2_matchups = {}
    
    # First, collect all matchups for both teams by period
    for idx, row in schedule_df.iterrows():
        # Skip header rows (scoring period rows)
        if "Scoring Period" in str(row["Team 1"]):
            continue
            
        period = row["Scoring Period"]
        
        # Initialize period lists if they don't exist
        if period not in team1_matchups:
            team1_matchups[period] = []
        if period not in team2_matchups:
            team2_matchups[period] = []
        
        # Store team1 matchups
        if row["Team 1"] == team1:
            team1_matchups[period].append((team1, row["Team 2"], row["Score 1"], True))
        elif row["Team 2"] == team1:
            team1_matchups[period].append((team1, row["Team 1"], row["Score 2"], False))
            
        # Store team2 matchups
        if row["Team 1"] == team2:
            team2_matchups[period].append((team2, row["Team 2"], row["Score 1"], True))
        elif row["Team 2"] == team2:
            team2_matchups[period].append((team2, row["Team 1"], row["Score 2"], False))
    
    # Create a mapping of original matchups to help with the swap
    # Structure: {(period, team, opponent): (score, is_team1_position)}
    matchup_map = {}
    
    # Populate the matchup map
    for period, matchups in team1_matchups.items():
        for team, opponent, score, is_team1 in matchups:
            matchup_map[(period, team, opponent)] = (score, is_team1)
            
    for period, matchups in team2_matchups.items():
        for team, opponent, score, is_team1 in matchups:
            matchup_map[(period, team, opponent)] = (score, is_team1)
    
    # Now perform the swap
    for idx, row in swapped_df.iterrows():
        # Skip header rows (scoring period rows)
        if "Scoring Period" in str(row["Team 1"]):
            continue
            
        period = row["Scoring Period"]
        
        # Handle direct matchups between team1 and team2 (if any)
        if (row["Team 1"] == team1 and row["Team 2"] == team2) or (row["Team 1"] == team2 and row["Team 2"] == team1):
            # For direct matchups, just swap team positions
            if row["Team 1"] == team1:
                swapped_df.at[idx, "Team 1"] = team2
                swapped_df.at[idx, "Team 2"] = team1
            else:
                swapped_df.at[idx, "Team 1"] = team1
                swapped_df.at[idx, "Team 2"] = team2
                
            # Keep scores the same, just swap them
            score1 = row["Score 1"]
            score2 = row["Score 2"]
            swapped_df.at[idx, "Score 1"] = score2
            swapped_df.at[idx, "Score 2"] = score1
            
            # Update score displays
            swapped_df.at[idx, "Score 1 Display"] = f"{score2:,}"
            swapped_df.at[idx, "Score 2 Display"] = f"{score1:,}"
            
            continue
        
        # Case 1: Matchup involves team1
        if row["Team 1"] == team1 or row["Team 2"] == team1:
            # Get the opponent of team1 in this matchup
            opponent = row["Team 2"] if row["Team 1"] == team1 else row["Team 1"]
            is_team1_position = row["Team 1"] == team1
            
            # Find team2's matchup for this period
            if period in team2_matchups and team2_matchups[period]:
                # Get team2's matchup details
                team2_data = team2_matchups[period][0]
                team2_opponent = team2_data[1]
                team2_score = team2_data[2]
                team2_is_team1_position = team2_data[3]
                
                # Replace team1 with team2 in this matchup
                if is_team1_position:
                    swapped_df.at[idx, "Team 1"] = team2
                    swapped_df.at[idx, "Score 1"] = team2_score
                    swapped_df.at[idx, "Score 1 Display"] = f"{team2_score:,}"
                else:
                    swapped_df.at[idx, "Team 2"] = team2
                    swapped_df.at[idx, "Score 2"] = team2_score
                    swapped_df.at[idx, "Score 2 Display"] = f"{team2_score:,}"
        
        # Case 2: Matchup involves team2
        elif row["Team 1"] == team2 or row["Team 2"] == team2:
            # Get the opponent of team2 in this matchup
            opponent = row["Team 2"] if row["Team 1"] == team2 else row["Team 1"]
            is_team1_position = row["Team 1"] == team2
            
            # Find team1's matchup for this period
            if period in team1_matchups and team1_matchups[period]:
                # Get team1's matchup details
                team1_data = team1_matchups[period][0]
                team1_opponent = team1_data[1]
                team1_score = team1_data[2]
                team1_is_team1_position = team1_data[3]
                
                # Replace team2 with team1 in this matchup
                if is_team1_position:
                    swapped_df.at[idx, "Team 1"] = team1
                    swapped_df.at[idx, "Score 1"] = team1_score
                    swapped_df.at[idx, "Score 1 Display"] = f"{team1_score:,}"
                else:
                    swapped_df.at[idx, "Team 2"] = team1
                    swapped_df.at[idx, "Score 2"] = team1_score
                    swapped_df.at[idx, "Score 2 Display"] = f"{team1_score:,}"
    
    # Recalculate winners
    for idx, row in swapped_df.iterrows():
        # Skip header rows
        if "Scoring Period" in str(row["Team 1"]):
            continue
            
        # Determine winner based on the scores
        if row["Score 1"] > row["Score 2"]:
            swapped_df.at[idx, "Winner"] = row["Team 1"]
        elif row["Score 1"] < row["Score 2"]:
            swapped_df.at[idx, "Winner"] = row["Team 2"]
        else:
            swapped_df.at[idx, "Winner"] = "Tie"
    
    # Calculate new team stats after the swap
    new_stats = calculate_team_stats(swapped_df)
    
    return swapped_df, original_stats, new_stats

def compare_team_stats(original_stats, new_stats):
    """
    Compare original and new team statistics to highlight changes.
    
    Args:
        original_stats (pd.DataFrame): Original team statistics
        new_stats (pd.DataFrame): New team statistics after schedule swap
        
    Returns:
        pd.DataFrame: DataFrame showing the differences in key metrics
    """
    if original_stats is None or new_stats is None:
        return None
    
    # Create a comparison DataFrame
    comparison = pd.DataFrame(index=original_stats.index)
    
    # Add original record
    comparison["Original Record"] = original_stats.apply(
        lambda row: f"{int(row['Wins'])}-{int(row['Losses'])}-{int(row['Ties'])}",
        axis=1
    )
    
    # Add new record
    comparison["New Record"] = new_stats.apply(
        lambda row: f"{int(row['Wins'])}-{int(row['Losses'])}-{int(row['Ties'])}",
        axis=1
    )
    
    # Add win percentage change
    comparison["Original Win %"] = original_stats["Win %"]
    comparison["New Win %"] = new_stats["Win %"]
    comparison["Win % Change"] = round(new_stats["Win %"] - original_stats["Win %"], 3)
    
    # Add points for/against changes
    comparison["Pts For Change"] = round(new_stats["Points For"] - original_stats["Points For"], 1)
    comparison["Pts Against Change"] = round(new_stats["Points Against"] - original_stats["Points Against"], 1)
    
    # Sort by win percentage change (descending)
    comparison = comparison.sort_values("Win % Change", ascending=False)
    
    return comparison

def calculate_all_schedule_swaps(schedule_df):
    """
    Calculate all possible schedule swaps between teams.
    
    Args:
        schedule_df (pd.DataFrame): The original schedule data
        
    Returns:
        dict: Dictionary with team pairs as keys and comparison DataFrames as values
        pd.DataFrame: Original team stats for reference
        pd.DataFrame: Summary DataFrame with the most impactful swaps
    """
    if schedule_df is None or schedule_df.empty:
        return {}, None, None
    
    # Get all unique teams
    teams = list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique()))
    # Remove any non-team entries (like scoring period headers)
    teams = [team for team in teams if not str(team).startswith("Scoring Period")]
    
    # Get original team stats for reference
    original_stats = calculate_team_stats(schedule_df)
    
    # Dictionary to store all swap results
    all_swaps = {}
    
    # List to store summary data for each swap
    summary_data = []
    
    # Calculate all possible swaps
    for i, team1 in enumerate(teams):
        for team2 in teams[i+1:]:
            # Calculate the swap
            _, _, new_stats = swap_team_schedules(schedule_df, team1, team2)
            comparison = compare_team_stats(original_stats, new_stats)
            
            # Store the full comparison result
            all_swaps[(team1, team2)] = comparison
            
            # Calculate standings changes
            original_standings = original_stats.sort_values("Win %", ascending=False).index.tolist()
            new_standings = new_stats.sort_values("Win %", ascending=False).index.tolist()
            
            # Get position changes for both teams
            team1_old_pos = original_standings.index(team1) + 1
            team1_new_pos = new_standings.index(team1) + 1
            team1_pos_change = team1_old_pos - team1_new_pos
            
            team2_old_pos = original_standings.index(team2) + 1
            team2_new_pos = new_standings.index(team2) + 1
            team2_pos_change = team2_old_pos - team2_new_pos
            
            # Find the team with the biggest win % change (absolute value)
            team1_change = abs(comparison.loc[team1, "Win % Change"])
            team2_change = abs(comparison.loc[team2, "Win % Change"])
            max_change_team = team1 if team1_change >= team2_change else team2
            max_change_value = comparison.loc[max_change_team, "Win % Change"]
            
            # Add to summary data
            summary_data.append({
                "Team 1": team1,
                "Team 2": team2,
                "Team 1 Win % Change": comparison.loc[team1, "Win % Change"],
                "Team 2 Win % Change": comparison.loc[team2, "Win % Change"],
                "Team 1 Position Change": team1_pos_change,
                "Team 2 Position Change": team2_pos_change,
                "Max Change Team": max_change_team,
                "Max Change Value": max_change_value,
                "Total Absolute Change": abs(comparison.loc[team1, "Win % Change"]) + abs(comparison.loc[team2, "Win % Change"])
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by total absolute change (most impactful swaps first)
    summary_df = summary_df.sort_values("Total Absolute Change", ascending=False)
    
    return all_swaps, original_stats, summary_df
