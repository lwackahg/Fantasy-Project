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
    
    # First pass: perform the swap
    for idx, row in swapped_df.iterrows():
        # Skip header rows (scoring period rows)
        if "Scoring Period" in str(row["Team 1"]):
            continue
            
        # Handle direct matchups between team1 and team2 (if any)
        if (row["Team 1"] == team1 and row["Team 2"] == team2):
            # Swap team positions and scores
            swapped_df.at[idx, "Team 1"] = team2
            swapped_df.at[idx, "Team 2"] = team1
            
            # Swap scores
            score1 = row["Score 1"]
            score2 = row["Score 2"]
            swapped_df.at[idx, "Score 1"] = score2
            swapped_df.at[idx, "Score 2"] = score1
            
            # Update score displays
            swapped_df.at[idx, "Score 1 Display"] = f"{score2:,}"
            swapped_df.at[idx, "Score 2 Display"] = f"{score1:,}"
            
            # Winner stays the same (just swapped position)
            if row["Winner"] == team1:
                swapped_df.at[idx, "Winner"] = team2
            elif row["Winner"] == team2:
                swapped_df.at[idx, "Winner"] = team1
            
            continue
            
        elif (row["Team 1"] == team2 and row["Team 2"] == team1):
            # Swap team positions and scores
            swapped_df.at[idx, "Team 1"] = team1
            swapped_df.at[idx, "Team 2"] = team2
            
            # Swap scores
            score1 = row["Score 1"]
            score2 = row["Score 2"]
            swapped_df.at[idx, "Score 1"] = score2
            swapped_df.at[idx, "Score 2"] = score1
            
            # Update score displays
            swapped_df.at[idx, "Score 1 Display"] = f"{score2:,}"
            swapped_df.at[idx, "Score 2 Display"] = f"{score1:,}"
            
            # Winner stays the same (just swapped position)
            if row["Winner"] == team1:
                swapped_df.at[idx, "Winner"] = team2
            elif row["Winner"] == team2:
                swapped_df.at[idx, "Winner"] = team1
                
            continue
            
        # Case 1: team1 is Team 1 in this matchup
        if row["Team 1"] == team1:
            # Replace team1 with team2
            swapped_df.at[idx, "Team 1"] = team2
            
            # Swap the score with team2's score from another matchup
            for other_idx, other_row in schedule_df.iterrows():
                if "Scoring Period" in str(other_row["Team 1"]):
                    continue
                    
                # Find a matchup where team2 is in the same position (Team 1)
                if other_row["Team 1"] == team2:
                    swapped_df.at[idx, "Score 1"] = other_row["Score 1"]
                    swapped_df.at[idx, "Score 1 Display"] = other_row["Score 1 Display"]
                    break
            
        # Case 2: team1 is Team 2 in this matchup
        elif row["Team 2"] == team1:
            # Replace team1 with team2
            swapped_df.at[idx, "Team 2"] = team2
            
            # Swap the score with team2's score from another matchup
            for other_idx, other_row in schedule_df.iterrows():
                if "Scoring Period" in str(other_row["Team 1"]):
                    continue
                    
                # Find a matchup where team2 is in the same position (Team 2)
                if other_row["Team 2"] == team2:
                    swapped_df.at[idx, "Score 2"] = other_row["Score 2"]
                    swapped_df.at[idx, "Score 2 Display"] = other_row["Score 2 Display"]
                    break
            
        # Case 3: team2 is Team 1 in this matchup
        elif row["Team 1"] == team2:
            # Replace team2 with team1
            swapped_df.at[idx, "Team 1"] = team1
            
            # Swap the score with team1's score from another matchup
            for other_idx, other_row in schedule_df.iterrows():
                if "Scoring Period" in str(other_row["Team 1"]):
                    continue
                    
                # Find a matchup where team1 is in the same position (Team 1)
                if other_row["Team 1"] == team1:
                    swapped_df.at[idx, "Score 1"] = other_row["Score 1"]
                    swapped_df.at[idx, "Score 1 Display"] = other_row["Score 1 Display"]
                    break
            
        # Case 4: team2 is Team 2 in this matchup
        elif row["Team 2"] == team2:
            # Replace team2 with team1
            swapped_df.at[idx, "Team 2"] = team1
            
            # Swap the score with team1's score from another matchup
            for other_idx, other_row in schedule_df.iterrows():
                if "Scoring Period" in str(other_row["Team 1"]):
                    continue
                    
                # Find a matchup where team1 is in the same position (Team 2)
                if other_row["Team 2"] == team1:
                    swapped_df.at[idx, "Score 2"] = other_row["Score 2"]
                    swapped_df.at[idx, "Score 2 Display"] = other_row["Score 2 Display"]
                    break
    
    # Second pass: recalculate winners
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
