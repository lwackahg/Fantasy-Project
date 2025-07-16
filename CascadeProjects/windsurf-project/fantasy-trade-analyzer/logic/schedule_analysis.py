"""
Schedule analysis module for fantasy trade analyzer.
Contains functions for analyzing and manipulating schedule data.
"""

import pandas as pd
import streamlit as st
from itertools import combinations
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    swapped_df = schedule_df.copy()
    original_stats = calculate_team_stats(schedule_df)
    
    team1_matchups = {}
    team2_matchups = {}
    
    for idx, row in schedule_df.iterrows():
        if "Scoring Period" in str(row["Team 1"]):
            continue
            
        period = row["Scoring Period"]
        
        if period not in team1_matchups:
            team1_matchups[period] = []
        if period not in team2_matchups:
            team2_matchups[period] = []
        
        if row["Team 1"] == team1:
            team1_matchups[period].append((team1, row["Team 2"], row["Score 1"], True))
        elif row["Team 2"] == team1:
            team1_matchups[period].append((team1, row["Team 1"], row["Score 2"], False))
            
        if row["Team 1"] == team2:
            team2_matchups[period].append((team2, row["Team 2"], row["Score 1"], True))
        elif row["Team 2"] == team2:
            team2_matchups[period].append((team2, row["Team 1"], row["Score 2"], False))
    
    for idx, row in swapped_df.iterrows():
        if "Scoring Period" in str(row["Team 1"]):
            continue
            
        period = row["Scoring Period"]
        
        if (row["Team 1"] == team1 and row["Team 2"] == team2) or (row["Team 1"] == team2 and row["Team 2"] == team1):
            if row["Team 1"] == team1:
                swapped_df.at[idx, "Team 1"] = team2
                swapped_df.at[idx, "Team 2"] = team1
            else:
                swapped_df.at[idx, "Team 1"] = team1
                swapped_df.at[idx, "Team 2"] = team2
                
            score1 = row["Score 1"]
            score2 = row["Score 2"]
            swapped_df.at[idx, "Score 1"] = score2
            swapped_df.at[idx, "Score 2"] = score1
            swapped_df.at[idx, "Score 1 Display"] = f"{score2:,}"
            swapped_df.at[idx, "Score 2 Display"] = f"{score1:,}"
            continue
        
        if row["Team 1"] == team1 or row["Team 2"] == team1:
            is_team1_position = row["Team 1"] == team1
            if period in team2_matchups and team2_matchups[period]:
                team2_data = team2_matchups[period][0]
                team2_score = team2_data[2]
                if is_team1_position:
                    swapped_df.at[idx, "Team 1"] = team2
                    swapped_df.at[idx, "Score 1"] = team2_score
                    swapped_df.at[idx, "Score 1 Display"] = f"{team2_score:,}"
                else:
                    swapped_df.at[idx, "Team 2"] = team2
                    swapped_df.at[idx, "Score 2"] = team2_score
                    swapped_df.at[idx, "Score 2 Display"] = f"{team2_score:,}"
        
        elif row["Team 1"] == team2 or row["Team 2"] == team2:
            is_team1_position = row["Team 1"] == team2
            if period in team1_matchups and team1_matchups[period]:
                team1_data = team1_matchups[period][0]
                team1_score = team1_data[2]
                if is_team1_position:
                    swapped_df.at[idx, "Team 1"] = team1
                    swapped_df.at[idx, "Score 1"] = team1_score
                    swapped_df.at[idx, "Score 1 Display"] = f"{team1_score:,}"
                else:
                    swapped_df.at[idx, "Team 2"] = team1
                    swapped_df.at[idx, "Score 2"] = team1_score
                    swapped_df.at[idx, "Score 2 Display"] = f"{team1_score:,}"
    
    for idx, row in swapped_df.iterrows():
        if "Scoring Period" in str(row["Team 1"]):
            continue
        if row["Score 1"] > row["Score 2"]:
            swapped_df.at[idx, "Winner"] = row["Team 1"]
        elif row["Score 1"] < row["Score 2"]:
            swapped_df.at[idx, "Winner"] = row["Team 2"]
        else:
            swapped_df.at[idx, "Winner"] = "Tie"
    
    new_stats = calculate_team_stats(swapped_df)
    return swapped_df, original_stats, new_stats

def compare_team_stats(original_stats, new_stats):
    """
    Compare original and new team statistics to highlight changes.
    """
    if original_stats is None or new_stats is None:
        return None
    
    comparison = pd.DataFrame(index=original_stats.index)
    comparison["Original Record"] = original_stats.apply(
        lambda row: f"{int(row['Wins'])}-{int(row['Losses'])}-{int(row['Ties'])}", axis=1
    )
    comparison["New Record"] = new_stats.apply(
        lambda row: f"{int(row['Wins'])}-{int(row['Losses'])}-{int(row['Ties'])}", axis=1
    )
    comparison["Original Win %"] = original_stats["Win %"]
    comparison["New Win %"] = new_stats["Win %"]
    comparison["Win % Change"] = round(new_stats["Win %"] - original_stats["Win %"], 3)
    comparison["Pts For Change"] = round(new_stats["Points For"] - original_stats["Points For"], 1)
    comparison["Pts Against Change"] = round(new_stats["Points Against"] - original_stats["Points Against"], 1)
    comparison = comparison.sort_values("Win % Change", ascending=False)
    return comparison

def calculate_all_schedule_swaps(schedule_df):
    """
    Calculate all possible schedule swaps and their impact on the entire league.
    """
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()

    teams = sorted(list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())))
    teams = [team for team in teams if not str(team).startswith("Scoring Period")]

    original_stats = calculate_team_stats(schedule_df)
    original_standings = original_stats.sort_values("Win %", ascending=False).index.tolist()

    summary_data = []

    for i, team1 in enumerate(teams):
        for team2 in teams[i+1:]:
            _, _, new_stats = swap_team_schedules(schedule_df, team1, team2)
            new_standings = new_stats.sort_values("Win %", ascending=False).index.tolist()
            
            position_changes = {}
            for team in teams:
                old_pos = original_standings.index(team) + 1
                new_pos = new_standings.index(team) + 1
                position_changes[team] = old_pos - new_pos

            team1_pos_change = position_changes[team1]
            team2_pos_change = position_changes[team2]

            other_teams_changes = {t: p for t, p in position_changes.items() if t not in [team1, team2]}
            
            biggest_winner, biggest_winner_change = (max(other_teams_changes, key=other_teams_changes.get), max(other_teams_changes.values())) if other_teams_changes and max(other_teams_changes.values()) > 0 else (None, 0)
            biggest_loser, biggest_loser_change = (min(other_teams_changes, key=other_teams_changes.get), min(other_teams_changes.values())) if other_teams_changes and min(other_teams_changes.values()) < 0 else (None, 0)

            summary_data.append({
                "Team 1": team1,
                "Team 2": team2,
                "Team 1 Position Change": team1_pos_change,
                "Team 2 Position Change": team2_pos_change,
                "Biggest Winner": biggest_winner,
                "Winner Change": biggest_winner_change,
                "Biggest Loser": biggest_loser,
                "Loser Change": biggest_loser_change,
                "All Changes": position_changes
            })

    if not summary_data:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_data)
    summary_df['Impact'] = summary_df['Team 1 Position Change'].abs() + summary_df['Team 2 Position Change'].abs()
    return summary_df.sort_values(by='Impact', ascending=False).reset_index(drop=True)

@st.cache_data
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
    
    # Get all unique teams first
    all_teams = set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())
    
    # Initialize team entries
    for team in all_teams:
        team_stats[team] = {
            "Wins": 0,
            "Losses": 0,
            "Ties": 0,
            "Points For": 0,
            "Points Against": 0,
            "Total Matchups": 0
        }
    
    # Process each matchup - use vectorized operations where possible
    for _, row in schedule_df.iterrows():
        team1 = row["Team 1"]
        team2 = row["Team 2"]
        score1 = row["Score 1"]
        score2 = row["Score 2"]
        
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
    stats_df["Avg Points For"] = round(stats_df["Points For"] / stats_df["Total Matchups"], 1)
    stats_df["Avg Points Against"] = round(stats_df["Points Against"] / stats_df["Total Matchups"], 1)
    
    return stats_df
