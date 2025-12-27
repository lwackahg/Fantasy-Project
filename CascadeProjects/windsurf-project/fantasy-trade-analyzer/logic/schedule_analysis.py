"""
Schedule analysis module for fantasy trade analyzer.
Contains functions for analyzing and manipulating schedule data.
"""

import pandas as pd
import streamlit as st
from itertools import combinations
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
from scipy import stats

# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

@st.cache_data
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

            # Consider all teams (including Team 1 and Team 2) when looking for
            # the biggest winner/loser. This way, if one of the swapped teams
            # jumps the most spots (e.g. from 3-3 to 6-0), it will correctly
            # appear as the biggest winner rather than showing "None".
            max_change = max(position_changes.values())
            if max_change > 0:
                biggest_winner = next(t for t, c in position_changes.items() if c == max_change)
                biggest_winner_change = max_change
            else:
                biggest_winner, biggest_winner_change = None, 0

            min_change = min(position_changes.values())
            if min_change < 0:
                biggest_loser = next(t for t, c in position_changes.items() if c == min_change)
                biggest_loser_change = min_change
            else:
                biggest_loser, biggest_loser_change = None, 0

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
    all_teams = {team for team in all_teams if not str(team).startswith("Scoring Period")}
    
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
        if str(team1).startswith("Scoring Period") or str(team2).startswith("Scoring Period"):
            continue
        score1 = row["Score 1"]
        score2 = row["Score 2"]

        # Treat 0-0 or missing scores as "not yet played" and skip them.
        # This prevents future weeks with placeholder zeros from being counted
        # as ties and keeps standings aligned with actual completed matchups.
        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            continue

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


@st.cache_data
def get_team_matchup_breakdown(schedule_df, team_name: str):
    """Return a per-matchup PF/PA breakdown for a single team.

    This uses the same rules as calculate_team_stats: future/unplayed 0-0
    matchups (or rows with missing scores) are skipped entirely so that the
    totals line up with the main standings table.
    """
    if schedule_df is None or schedule_df.empty or not team_name:
        return pd.DataFrame()

    rows = []

    for _, row in schedule_df.iterrows():
        team1 = row["Team 1"]
        team2 = row["Team 2"]
        score1 = row["Score 1"]
        score2 = row["Score 2"]

        # Keep in sync with calculate_team_stats: skip unplayed 0-0 or missing matchups
        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            continue

        if team1 != team_name and team2 != team_name:
            continue

        if team1 == team_name:
            pts_for = score1
            pts_against = score2
            opponent = team2
        else:
            pts_for = score2
            pts_against = score1
            opponent = team1

        if pts_for > pts_against:
            outcome = "W"
        elif pts_for < pts_against:
            outcome = "L"
        else:
            outcome = "T"

        rows.append({
            "Scoring Period": row.get("Scoring Period", None),
            "Period Number": row.get("Period Number", None),
            "Date Range": row.get("Date Range", None),
            "Opponent": opponent,
            "Points For": pts_for,
            "Points Against": pts_against,
            "Outcome": outcome,
        })

    breakdown_df = pd.DataFrame(rows)
    if breakdown_df.empty:
        return breakdown_df

    sort_cols = []
    if "Period Number" in breakdown_df.columns:
        sort_cols.append("Period Number")
    if "Scoring Period" in breakdown_df.columns:
        sort_cols.append("Scoring Period")

    if sort_cols:
        breakdown_df = breakdown_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        breakdown_df = breakdown_df.reset_index(drop=True)

    breakdown_df["Cumulative Points For"] = breakdown_df["Points For"].cumsum()
    breakdown_df["Cumulative Points Against"] = breakdown_df["Points Against"].cumsum()
    return breakdown_df


@st.cache_data
def get_team_weekly_points_summary(schedule_df, team_name: str):
    """Return season-to-date weekly scoring summary for a single team.

    Uses the same matchup filtering rules as ``get_team_matchup_breakdown``
    (future 0-0 weeks are ignored) and aggregates the ``Points For`` totals
    for each completed scoring period.

    The result is a lightweight dict with basic distribution stats that can
    be used to compare or calibrate simulated weekly outcome curves.
    """
    if schedule_df is None or getattr(schedule_df, "empty", True) or not team_name:
        return None

    breakdown_df = get_team_matchup_breakdown(schedule_df, team_name)
    if breakdown_df is None or breakdown_df.empty or "Points For" not in breakdown_df.columns:
        return None

    pts = breakdown_df["Points For"].astype(float)
    if pts.empty:
        return None

    values = pts.to_numpy()
    weeks_played = int(len(values))

    mean_fpts = float(values.mean())
    std_fpts = float(values.std(ddof=0))

    p10 = float(np.percentile(values, 10))
    p50 = float(np.percentile(values, 50))
    p90 = float(np.percentile(values, 90))

    thresholds = [1300, 1400, 1500, 1600]
    hits = {f"hit_{t}": int((values >= t).sum()) for t in thresholds}
    hit_rates = {
        f"hit_{t}_rate": float((values >= t).mean() * 100.0) if weeks_played > 0 else 0.0
        for t in thresholds
    }

    summary = {
        "team_name": team_name,
        "weeks_played": weeks_played,
        "mean_fpts": mean_fpts,
        "std_fpts": std_fpts,
        "p10_fpts": p10,
        "p50_fpts": p50,
        "p90_fpts": p90,
    }
    summary.update(hits)
    summary.update(hit_rates)
    return summary


def simulate_remaining_games(
    schedule_df: pd.DataFrame,
    num_simulations: int = 10000,
    playoff_spots: int = 6,
    regression_weight: float = 0.3,
) -> Dict[str, Dict]:
    """
    Monte Carlo simulation of remaining games to calculate playoff probabilities.
    
    Args:
        schedule_df: Schedule data with completed and upcoming games
        num_simulations: Number of Monte Carlo simulations to run
        playoff_spots: Number of playoff teams
        regression_weight: 0-1 weight for shrinking team ratings/scoring toward league average
        
    Returns:
        Dictionary with playoff probabilities for each team
    """
    if schedule_df is None or schedule_df.empty:
        return {}
    
    # Get all unique teams
    all_teams = sorted(list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())))
    all_teams = [team for team in all_teams if not str(team).startswith("Scoring Period")]
    
    # Separate completed and remaining games
    completed_games = []
    remaining_games = []
    
    for _, row in schedule_df.iterrows():
        team1 = row["Team 1"]
        team2 = row["Team 2"]
        if str(team1).startswith("Scoring Period") or str(team2).startswith("Scoring Period"):
            continue
        score1 = row["Score 1"]
        score2 = row["Score 2"]
        
        # Skip unplayed games (0-0 or missing scores)
        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            remaining_games.append((team1, team2))
        else:
            completed_games.append((team1, team2, score1, score2))
    
    if not remaining_games:
        # Season is complete, use actual standings
        final_probs = _calculate_final_playoff_probabilities(schedule_df, all_teams, playoff_spots)
        return final_probs, {}, {}
    
    league_stats = _calculate_league_scoring_stats(completed_games)
    team_ratings = _calculate_team_ratings(
        completed_games,
        all_teams,
        regression_weight=regression_weight,
    )
    scoring_profiles = _calculate_scoring_profiles(
        completed_games,
        all_teams,
        league_mean=float(league_stats["mean_score"]),
        league_std=float(league_stats["std_score"]),
        regression_weight=regression_weight,
    )
    
    # Calculate remaining strength of schedule
    remaining_sos = _calculate_remaining_sos(remaining_games, team_ratings, all_teams)
    
    # Run Monte Carlo simulations
    playoff_results = defaultdict(lambda: defaultdict(int))
    seed_totals = defaultdict(list)  # Track seed distribution across simulations
    
    for sim in range(num_simulations):
        # Simulate remaining games
        sim_results = _simulate_games_once(
            remaining_games,
            team_ratings,
            scoring_profiles,
            completed_games,
            league_stats,
        )
        
        # Calculate final standings
        final_standings = _calculate_simulated_standings(sim_results, all_teams)
        
        # Determine playoff teams (top N make playoffs)
        cutoff = max(2, min(playoff_spots, len(all_teams)))
        playoff_teams = set(final_standings[:cutoff])
        
        # Record results
        for team in all_teams:
            seed = final_standings.index(team) + 1
            seed_totals[team].append(seed)
            
            if team in playoff_teams:
                playoff_results[team]["playoffs"] += 1
                playoff_results[team][f"seed_{seed}"] += 1
            else:
                playoff_results[team]["missed_playoffs"] += 1
    
    playoff_probabilities = {}
    for team in all_teams:
        total_sims = sum(playoff_results[team].values())
        if total_sims > 0:
            probs = {k: (v / num_simulations) * 100 for k, v in playoff_results[team].items()}
            
            # Add seed tracking metrics
            if team in seed_totals and seed_totals[team]:
                probs["expected_seed"] = round(np.mean(seed_totals[team]), 2)
                probs["seed_std"] = round(np.std(seed_totals[team]), 2)
                probs["best_seed"] = min(seed_totals[team])
                probs["worst_seed"] = max(seed_totals[team])
            
            # Add confidence intervals for playoff odds
            if "playoffs" in probs:
                playoff_pct = probs["playoffs"] / 100
                # Handle edge cases (0% or 100%)
                if playoff_pct <= 0:
                    probs["playoff_ci_low"] = 0.0
                    probs["playoff_ci_high"] = 0.0
                elif playoff_pct >= 1:
                    probs["playoff_ci_low"] = 100.0
                    probs["playoff_ci_high"] = 100.0
                else:
                    ci_low = stats.binom.ppf(0.05, num_simulations, playoff_pct) / num_simulations * 100
                    ci_high = stats.binom.ppf(0.95, num_simulations, playoff_pct) / num_simulations * 100
                    probs["playoff_ci_low"] = round(max(0, ci_low), 1)
                    probs["playoff_ci_high"] = round(min(100, ci_high), 1)
            
            playoff_probabilities[team] = probs
    
    return playoff_probabilities, remaining_sos, team_ratings


def _calculate_team_ratings(
    completed_games: List[Tuple],
    all_teams: List[str],
    regression_weight: float = 0.3,
    recency_weight: float = 0.15,
) -> Dict[str, float]:
    ratings = {team: 1500.0 for team in all_teams}
    games_played = {team: 0 for team in all_teams}
    base_k = 40.0
    total_games = len(completed_games)

    for i, (team1, team2, score1, score2) in enumerate(completed_games):
        if team1 not in ratings or team2 not in ratings:
            continue
        games_played[team1] += 1
        games_played[team2] += 1

        rating_diff = ratings[team1] - ratings[team2]
        expected1 = 1 / (1 + 10 ** (-rating_diff / 400))
        expected2 = 1 - expected1

        if score1 > score2:
            actual1, actual2 = 1.0, 0.0
            winner_rating = ratings[team1]
            loser_rating = ratings[team2]
        elif score2 > score1:
            actual1, actual2 = 0.0, 1.0
            winner_rating = ratings[team2]
            loser_rating = ratings[team1]
        else:
            actual1, actual2 = 0.5, 0.5
            winner_rating = None
            loser_rating = None

        k1 = base_k / (1.0 + games_played[team1] * 0.1)
        k2 = base_k / (1.0 + games_played[team2] * 0.1)
        k_factor = (k1 + k2) / 2.0
        
        # Add recency weighting - recent games count more
        if total_games > 0 and recency_weight > 0:
            game_recency = (i + 1) / total_games  # 0→1 as season progresses
            recency_multiplier = 1.0 + recency_weight * (game_recency - 0.5) * 2
            k_factor *= recency_multiplier

        mov_multiplier = 1.0
        if winner_rating is not None and loser_rating is not None:
            margin = abs(float(score1) - float(score2))
            elo_diff = abs(winner_rating - loser_rating)
            mov_multiplier = float(np.log(margin + 1.0) * (2.2 / ((elo_diff * 0.001) + 2.2)))
            mov_multiplier = max(0.5, min(mov_multiplier, 3.0))

        ratings[team1] += k_factor * mov_multiplier * (actual1 - expected1)
        ratings[team2] += k_factor * mov_multiplier * (actual2 - expected2)

    if regression_weight > 0:
        prior_games = 8.0
        for team in all_teams:
            n_games = float(games_played.get(team, 0))
            sample_weight = n_games / (n_games + prior_games) if (n_games + prior_games) > 0 else 0.0
            shrink = float(regression_weight) * (1.0 - sample_weight)
            ratings[team] = ratings[team] * (1.0 - shrink) + 1500.0 * shrink

    return ratings


def _simulate_games_once(
    remaining_games: List[Tuple],
    team_ratings: Dict[str, float],
    scoring_profiles: Dict[str, Dict[str, float]],
    completed_games: List[Tuple],
    league_stats: Dict[str, float],
) -> List[Tuple]:
    """Simulate one complete set of remaining games."""
    sim_results = completed_games.copy()
    
    for team1, team2 in remaining_games:
        # Calculate win probability based on ratings
        rating_diff = team_ratings[team1] - team_ratings[team2]
        win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
        
        # Simulate outcome
        profile1 = scoring_profiles.get(team1, {})
        profile2 = scoring_profiles.get(team2, {})

        if random.random() < win_prob:
            # Team1 wins - simulate realistic scores
            score1, score2 = _simulate_score(
                rating_winner=team_ratings[team1],
                rating_loser=team_ratings[team2],
                winner_profile=profile1,
                loser_profile=profile2,
                league_stats=league_stats,
            )
        else:
            # Team2 wins
            score2, score1 = _simulate_score(
                rating_winner=team_ratings[team2],
                rating_loser=team_ratings[team1],
                winner_profile=profile2,
                loser_profile=profile1,
                league_stats=league_stats,
            )
        
        sim_results.append((team1, team2, score1, score2))
    
    return sim_results


def _simulate_score(
    rating_winner: float,
    rating_loser: float,
    winner_profile: Dict,
    loser_profile: Dict,
    league_stats: Dict[str, float],
) -> Tuple[int, int]:
    """Simulate realistic fantasy scores using Elo gap plus team scoring profiles."""
    # Baseline from historical scoring profiles (offense vs opponent defense)
    w_off_mean = winner_profile.get("off_mean", winner_profile.get("mean", 1450.0))
    w_off_std = winner_profile.get("off_std", winner_profile.get("std", 200.0))
    w_def_mean = winner_profile.get("def_mean", winner_profile.get("mean", 1450.0))
    w_def_std = winner_profile.get("def_std", winner_profile.get("std", 200.0))

    l_off_mean = loser_profile.get("off_mean", loser_profile.get("mean", 1450.0))
    l_off_std = loser_profile.get("off_std", loser_profile.get("std", 200.0))
    l_def_mean = loser_profile.get("def_mean", loser_profile.get("mean", 1450.0))
    l_def_std = loser_profile.get("def_std", loser_profile.get("std", 200.0))

    w_mean = (float(w_off_mean) + float(l_def_mean)) / 2.0
    l_mean = (float(l_off_mean) + float(w_def_mean)) / 2.0
    w_std = float(np.sqrt((float(w_off_std) ** 2 + float(l_def_std) ** 2) / 2.0))
    l_std = float(np.sqrt((float(l_off_std) ** 2 + float(w_def_std) ** 2) / 2.0))

    # Rating-derived bonus
    rating_diff = (rating_winner - rating_loser) / 400
    winner_bonus = rating_diff * 120  # a bit stronger than before
    loser_bonus = -winner_bonus * 0.5

    winner_score = max(
        800,
        int(random.gauss(w_mean + winner_bonus, max(80.0, w_std))),
    )
    loser_score = max(
        800,
        int(random.gauss(l_mean + loser_bonus, max(80.0, l_std))),
    )

    if winner_score <= loser_score:
        # Use league's actual margin distribution for more realistic gaps
        league_margin_mean = league_stats.get("mean_margin", 120.0)
        league_margin_std = league_stats.get("std_margin", 80.0)
        margin = max(1, int(random.gauss(league_margin_mean, league_margin_std)))
        winner_score = int(loser_score + margin)

    return winner_score, loser_score


def _calculate_league_scoring_stats(completed_games: List[Tuple]) -> Dict[str, float]:
    all_scores = []
    all_margins = []
    for _, _, score1, score2 in completed_games:
        try:
            s1, s2 = float(score1), float(score2)
            all_scores.append(s1)
            all_scores.append(s2)
            all_margins.append(abs(s1 - s2))
        except (TypeError, ValueError):
            continue
    
    if not all_scores:
        return {"mean_score": 1450.0, "std_score": 200.0, "mean_margin": 120.0, "std_margin": 80.0}
    
    arr = np.array(all_scores, dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 200.0
    
    margin_stats = {"mean_margin": 120.0, "std_margin": 80.0}
    if all_margins:
        margin_arr = np.array(all_margins, dtype=float)
        margin_stats["mean_margin"] = float(margin_arr.mean())
        margin_stats["std_margin"] = float(margin_arr.std(ddof=1)) if margin_arr.size > 1 else 80.0
    
    return {"mean_score": float(arr.mean()), "std_score": std, **margin_stats}


def _calculate_scoring_profiles(
    completed_games: List[Tuple],
    all_teams: List[str],
    league_mean: float,
    league_std: float,
    regression_weight: float = 0.3,
) -> Dict[str, Dict[str, float]]:
    points_for = {team: [] for team in all_teams}
    points_against = {team: [] for team in all_teams}

    for team1, team2, score1, score2 in completed_games:
        if team1 in points_for:
            points_for[team1].append(float(score1))
            points_against[team1].append(float(score2))
        if team2 in points_for:
            points_for[team2].append(float(score2))
            points_against[team2].append(float(score1))

    profiles = {}
    prior_games = 5.0
    for team, scores in points_for.items():
        pf_scores = scores
        pa_scores = points_against.get(team, [])

        if pf_scores:
            pf_arr = np.array(pf_scores, dtype=float)
            off_mean = float(pf_arr.mean())
            off_std = float(pf_arr.std(ddof=1)) if pf_arr.size > 1 else float(league_std)
            n_games = float(pf_arr.size)
            sample_weight = n_games / (n_games + prior_games)
            shrink = float(regression_weight) * (1.0 - sample_weight)
            off_mean = off_mean * (1.0 - shrink) + float(league_mean) * shrink
            off_std = off_std * sample_weight + float(league_std) * (1.0 - sample_weight)
        else:
            off_mean = float(league_mean)
            off_std = float(league_std)

        if pa_scores:
            pa_arr = np.array(pa_scores, dtype=float)
            def_mean = float(pa_arr.mean())
            def_std = float(pa_arr.std(ddof=1)) if pa_arr.size > 1 else float(league_std)
            n_games = float(pa_arr.size)
            sample_weight = n_games / (n_games + prior_games)
            shrink = float(regression_weight) * (1.0 - sample_weight)
            def_mean = def_mean * (1.0 - shrink) + float(league_mean) * shrink
            def_std = def_std * sample_weight + float(league_std) * (1.0 - sample_weight)
        else:
            def_mean = float(league_mean)
            def_std = float(league_std)

        profiles[team] = {
            "off_mean": float(off_mean),
            "off_std": max(80.0, float(off_std)),
            "def_mean": float(def_mean),
            "def_std": max(80.0, float(def_std)),
            "mean": float(off_mean),
            "std": max(80.0, float(off_std)),
        }

    return profiles


def _calculate_simulated_standings(sim_results: List[Tuple], all_teams: List[str]) -> List[str]:
    team_data = {
        team: {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "pf": 0.0,
            "pa": 0.0,
            "h2h": defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}),
        }
        for team in all_teams
    }

    for team1, team2, score1, score2 in sim_results:
        if team1 not in team_data or team2 not in team_data:
            continue
        s1 = float(score1)
        s2 = float(score2)
        team_data[team1]["pf"] += s1
        team_data[team1]["pa"] += s2
        team_data[team2]["pf"] += s2
        team_data[team2]["pa"] += s1
        if s1 > s2:
            team_data[team1]["wins"] += 1
            team_data[team2]["losses"] += 1
            team_data[team1]["h2h"][team2]["wins"] += 1
            team_data[team2]["h2h"][team1]["losses"] += 1
        elif s2 > s1:
            team_data[team2]["wins"] += 1
            team_data[team1]["losses"] += 1
            team_data[team2]["h2h"][team1]["wins"] += 1
            team_data[team1]["h2h"][team2]["losses"] += 1
        else:
            team_data[team1]["ties"] += 1
            team_data[team2]["ties"] += 1
            team_data[team1]["h2h"][team2]["ties"] += 1
            team_data[team2]["h2h"][team1]["ties"] += 1

    rows = []
    for team in all_teams:
        wins = int(team_data[team]["wins"])
        losses = int(team_data[team]["losses"])
        ties = int(team_data[team]["ties"])
        total = wins + losses + ties
        win_pct = (wins + 0.5 * ties) / total if total > 0 else 0.0
        rows.append(
            {
                "team": team,
                "win_pct": float(win_pct),
                "wins": wins,
                "pf": float(team_data[team]["pf"]),
                "pa": float(team_data[team]["pa"]),
                "h2h": team_data[team]["h2h"],
            }
        )

    rows.sort(key=lambda r: r["win_pct"], reverse=True)
    return [r["team"] for r in _resolve_ties_with_league_rules(rows)]


def _resolve_ties_with_league_rules(rows: List[Dict]) -> List[Dict]:
    result = []
    i = 0
    while i < len(rows):
        group = [rows[i]]
        j = i + 1
        while j < len(rows) and abs(rows[j]["win_pct"] - rows[i]["win_pct"]) < 1e-9:
            group.append(rows[j])
            j += 1
        if len(group) == 1:
            result.append(group[0])
        else:
            result.extend(_apply_tiebreakers_to_group(group))
        i = j
    return result


def _apply_tiebreakers_to_group(group: List[Dict]) -> List[Dict]:
    team_names = [g["team"] for g in group]
    for g in group:
        h2h_wins = 0
        h2h_losses = 0
        h2h_ties = 0
        for opp in team_names:
            if opp == g["team"]:
                continue
            cell = g["h2h"].get(opp)
            if not cell:
                continue
            h2h_wins += int(cell.get("wins", 0))
            h2h_losses += int(cell.get("losses", 0))
            h2h_ties += int(cell.get("ties", 0))
        total = h2h_wins + h2h_losses + h2h_ties
        g["h2h_pct"] = (h2h_wins + 0.5 * h2h_ties) / total if total > 0 else 0.5
    group.sort(
        key=lambda r: (
            r.get("h2h_pct", 0.5),
            r.get("pf", 0.0),
            -r.get("pa", 0.0),
            r.get("wins", 0),
        ),
        reverse=True,
    )
    return group


def _calculate_final_playoff_probabilities(
    schedule_df: pd.DataFrame,
    all_teams: List[str],
    playoff_spots: int = 6,
) -> Dict[str, Dict]:
    completed_games = []
    for _, row in schedule_df.iterrows():
        team1 = row.get("Team 1")
        team2 = row.get("Team 2")
        if str(team1).startswith("Scoring Period") or str(team2).startswith("Scoring Period"):
            continue
        score1 = row.get("Score 1")
        score2 = row.get("Score 2")
        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            continue
        completed_games.append((team1, team2, float(score1), float(score2)))

    final_order = _calculate_simulated_standings(completed_games, all_teams)
    cutoff = max(2, min(int(playoff_spots), len(all_teams)))
    playoff_teams = set(final_order[:cutoff])

    # Calculate ratings even for completed season (useful for display)
    league_stats = _calculate_league_scoring_stats(completed_games)
    team_ratings = _calculate_team_ratings(completed_games, all_teams, regression_weight=0.3)

    playoff_probabilities = {}
    for team in all_teams:
        seed = final_order.index(team) + 1
        if team in playoff_teams:
            playoff_probabilities[team] = {
                "playoffs": 100.0,
                f"seed_{seed}": 100.0,
                "missed_playoffs": 0.0,
                "expected_seed": float(seed),
                "seed_std": 0.0,
                "best_seed": seed,
                "worst_seed": seed,
            }
        else:
            playoff_probabilities[team] = {
                "playoffs": 0.0,
                "missed_playoffs": 100.0,
                "expected_seed": float(seed),
                "seed_std": 0.0,
                "best_seed": seed,
                "worst_seed": seed,
            }

    return playoff_probabilities, {}, team_ratings  # Empty SOS since no remaining games


def _calculate_remaining_sos(remaining_games: List[Tuple], team_ratings: Dict[str, float], all_teams: List[str]) -> Dict[str, float]:
    """Calculate strength of remaining schedule for each team."""
    remaining_opps = {team: [] for team in all_teams}
    for t1, t2 in remaining_games:
        remaining_opps[t1].append(t2)
        remaining_opps[t2].append(t1)
    
    league_avg = np.mean(list(team_ratings.values()))
    return {
        team: np.mean([team_ratings[o] for o in opps]) - league_avg if opps else 0
        for team, opps in remaining_opps.items()
    }


def calculate_magic_numbers(
    schedule_df: pd.DataFrame,
    playoff_spots: int = 6,
) -> pd.DataFrame:
    """
    Calculate clinch and elimination numbers for each team.
    
    Magic number = Games remaining + 1 + (Team's losses - Competitor's wins)
    When magic number hits 0, team clinches/is eliminated.
    """
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()
    
    all_teams = sorted(list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())))
    all_teams = [team for team in all_teams if not str(team).startswith("Scoring Period")]
    
    # Count completed and remaining games per team
    completed = {team: {"wins": 0, "losses": 0, "ties": 0} for team in all_teams}
    remaining = {team: 0 for team in all_teams}
    
    for _, row in schedule_df.iterrows():
        team1, team2 = row["Team 1"], row["Team 2"]
        if str(team1).startswith("Scoring Period") or str(team2).startswith("Scoring Period"):
            continue
        
        score1, score2 = row["Score 1"], row["Score 2"]
        
        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            remaining[team1] += 1
            remaining[team2] += 1
        else:
            if score1 > score2:
                completed[team1]["wins"] += 1
                completed[team2]["losses"] += 1
            elif score2 > score1:
                completed[team2]["wins"] += 1
                completed[team1]["losses"] += 1
            else:
                completed[team1]["ties"] += 1
                completed[team2]["ties"] += 1
    
    # Sort teams by current standing
    standings = sorted(
        all_teams,
        key=lambda t: (completed[t]["wins"] + 0.5 * completed[t]["ties"], completed[t]["wins"]),
        reverse=True
    )
    
    results = []
    for team in all_teams:
        team_wins = completed[team]["wins"]
        team_remaining = remaining[team]
        team_max_wins = team_wins + team_remaining
        
        current_rank = standings.index(team) + 1
        
        # Clinch number: vs team currently at playoff_spots+1 position
        if current_rank <= playoff_spots:
            # Compare against first team out
            if playoff_spots < len(standings):
                competitor = standings[playoff_spots]  # 0-indexed, so this is team at spot playoff_spots+1
                comp_max_wins = completed[competitor]["wins"] + remaining[competitor]
                # Magic number to clinch: need enough wins that competitor can't catch you
                clinch_num = max(0, comp_max_wins - team_wins + 1)
            else:
                clinch_num = 0  # Already clinched if fewer teams than spots
        else:
            clinch_num = None  # Not in playoff position
        
        # Elimination number: vs team currently at playoff_spots position
        if current_rank > playoff_spots:
            # Compare against last team in
            competitor = standings[playoff_spots - 1]  # Team at last playoff spot
            comp_wins = completed[competitor]["wins"]
            # Elimination: when your max wins can't reach their current wins
            elim_num = max(0, team_max_wins - comp_wins + 1)
        else:
            elim_num = None  # Already in playoff position
        
        results.append({
            "Team": team,
            "Current Rank": current_rank,
            "Wins": team_wins,
            "Losses": completed[team]["losses"],
            "Remaining": team_remaining,
            "Max Wins": team_max_wins,
            "Clinch #": clinch_num,
            "Elimination #": elim_num,
            "Status": "CLINCHED" if clinch_num == 0 else ("ELIMINATED" if elim_num == 0 else "Active"),
        })
    
    return pd.DataFrame(results).sort_values("Current Rank")


def estimate_playoff_threshold(
    schedule_df: pd.DataFrame,
    playoff_spots: int = 6,
    num_simulations: int = 2000,
    team_ratings: Dict[str, float] = None,
    scoring_profiles: Dict[str, Dict] = None,
    league_stats: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Estimate the win total needed to make playoffs by simulating seasons
    and tracking what the "last team in" finishes with.
    
    Returns:
        Dictionary with threshold statistics:
        - mean: Average wins needed
        - median: Median wins needed  
        - p25: 25th percentile (optimistic - easier path)
        - p75: 75th percentile (pessimistic - harder path)
        - p90: 90th percentile (very safe target)
    """
    if schedule_df is None or schedule_df.empty:
        return {"mean": 0, "median": 0, "p25": 0, "p75": 0, "p90": 0}
    
    all_teams = sorted(list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())))
    all_teams = [team for team in all_teams if not str(team).startswith("Scoring Period")]
    
    # Separate completed and remaining games
    completed_games = []
    remaining_games = []
    
    for _, row in schedule_df.iterrows():
        team1, team2 = row["Team 1"], row["Team 2"]
        if str(team1).startswith("Scoring Period") or str(team2).startswith("Scoring Period"):
            continue
        score1, score2 = row["Score 1"], row["Score 2"]
        
        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            remaining_games.append((team1, team2))
        else:
            completed_games.append((team1, team2, float(score1), float(score2)))
    
    if not remaining_games:
        # Season complete - threshold is what the last playoff team actually has
        final_standings = _calculate_simulated_standings(completed_games, all_teams)
        if playoff_spots <= len(final_standings):
            last_team_in = final_standings[playoff_spots - 1]
            # Count their wins
            wins = sum(1 for t1, t2, s1, s2 in completed_games 
                      if (t1 == last_team_in and s1 > s2) or (t2 == last_team_in and s2 > s1))
            return {"mean": wins, "median": wins, "p25": wins, "p75": wins, "p90": wins, "threshold_wins": wins}
        return {"mean": 0, "median": 0, "p25": 0, "p75": 0, "p90": 0}
    
    # Calculate ratings/profiles if not provided
    if league_stats is None:
        league_stats = _calculate_league_scoring_stats(completed_games)
    if team_ratings is None:
        team_ratings = _calculate_team_ratings(completed_games, all_teams, regression_weight=0.3)
    if scoring_profiles is None:
        scoring_profiles = _calculate_scoring_profiles(
            completed_games, all_teams,
            league_mean=league_stats["mean_score"],
            league_std=league_stats["std_score"],
            regression_weight=0.3
        )
    
    # Run simulations and track "last team in" win totals
    last_team_in_wins = []
    
    for _ in range(num_simulations):
        sim_results = _simulate_games_once(
            remaining_games, team_ratings, scoring_profiles, completed_games, league_stats
        )
        
        # Calculate wins for each team
        team_wins = {team: 0 for team in all_teams}
        for t1, t2, s1, s2 in sim_results:
            if t1 in team_wins and t2 in team_wins:
                if s1 > s2:
                    team_wins[t1] += 1
                elif s2 > s1:
                    team_wins[t2] += 1
        
        # Sort by wins to get standings
        sorted_teams = sorted(all_teams, key=lambda t: team_wins[t], reverse=True)
        
        if playoff_spots <= len(sorted_teams):
            last_playoff_team = sorted_teams[playoff_spots - 1]
            last_team_in_wins.append(team_wins[last_playoff_team])
    
    if not last_team_in_wins:
        return {"mean": 0, "median": 0, "p25": 0, "p75": 0, "p90": 0}
    
    arr = np.array(last_team_in_wins)
    return {
        "mean": round(float(arr.mean()), 1),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min_observed": int(arr.min()),
        "max_observed": int(arr.max()),
        "threshold_wins": round(float(np.percentile(arr, 50)), 0),  # Use median as "target"
    }


def calculate_wins_probability(
    current_wins: int,
    remaining_games: int,
    expected_win_rate: float,
) -> Dict[int, float]:
    """
    Calculate probability of reaching each possible final win total.
    
    Uses binomial distribution for analytical calculation.
    
    Args:
        current_wins: Wins so far
        remaining_games: Games left to play
        expected_win_rate: Team's expected win rate (0-1) based on Elo/SOS
        
    Returns:
        Dict mapping final_wins -> probability of reaching that exact total
    """
    from scipy import stats
    
    probs = {}
    for additional_wins in range(remaining_games + 1):
        final_wins = current_wins + additional_wins
        # P(exactly additional_wins more wins)
        prob = stats.binom.pmf(additional_wins, remaining_games, expected_win_rate)
        probs[final_wins] = prob * 100  # Convert to percentage
    
    return probs


def calculate_probability_of_reaching_threshold(
    current_wins: int,
    remaining_games: int,
    expected_win_rate: float,
    threshold_wins: int,
) -> float:
    """
    Calculate probability of reaching or exceeding a win threshold.
    
    P(final_wins >= threshold) = P(additional_wins >= threshold - current_wins)
    """
    from scipy import stats
    
    wins_needed = max(0, threshold_wins - current_wins)
    
    if wins_needed > remaining_games:
        return 0.0  # Mathematically eliminated
    
    if wins_needed <= 0:
        return 100.0  # Already there or past it
    
    # P(X >= wins_needed) = 1 - P(X < wins_needed) = 1 - CDF(wins_needed - 1)
    prob = 1 - stats.binom.cdf(wins_needed - 1, remaining_games, expected_win_rate)
    return prob * 100


def calculate_expected_win_rate(
    team: str,
    remaining_games: List[Tuple],
    team_ratings: Dict[str, float],
) -> float:
    """
    Calculate a team's expected win rate for their remaining games
    based on Elo ratings of opponents.
    """
    opponents = []
    for t1, t2 in remaining_games:
        if t1 == team:
            opponents.append(t2)
        elif t2 == team:
            opponents.append(t1)
    
    if not opponents:
        return 0.5  # No remaining games
    
    team_rating = team_ratings.get(team, 1500.0)
    
    total_win_prob = 0.0
    for opp in opponents:
        opp_rating = team_ratings.get(opp, 1500.0)
        rating_diff = team_rating - opp_rating
        win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
        total_win_prob += win_prob
    
    return total_win_prob / len(opponents)


def calculate_playoff_odds_with_threshold(
    schedule_df: pd.DataFrame,
    playoff_spots: int = 6,
    num_threshold_sims: int = 2000,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculate playoff odds using threshold-based analysis.
    
    This approach:
    1. Estimates the playoff threshold (wins needed to make it)
    2. Calculates each team's current wins and remaining games
    3. Computes expected win rate based on remaining schedule strength
    4. Uses binomial probability to calculate P(reaching threshold)
    
    Returns:
        Tuple of (DataFrame with team analysis, threshold stats dict)
    """
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(), {}
    
    all_teams = sorted(list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())))
    all_teams = [team for team in all_teams if not str(team).startswith("Scoring Period")]
    
    # Gather completed and remaining games
    completed_games = []
    remaining_games = []
    
    for _, row in schedule_df.iterrows():
        team1, team2 = row["Team 1"], row["Team 2"]
        if str(team1).startswith("Scoring Period") or str(team2).startswith("Scoring Period"):
            continue
        score1, score2 = row["Score 1"], row["Score 2"]
        
        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            remaining_games.append((team1, team2))
        else:
            completed_games.append((team1, team2, float(score1), float(score2)))
    
    # Calculate current standings
    current_wins = {team: 0 for team in all_teams}
    current_losses = {team: 0 for team in all_teams}
    games_remaining = {team: 0 for team in all_teams}
    
    for t1, t2, s1, s2 in completed_games:
        if s1 > s2:
            current_wins[t1] += 1
            current_losses[t2] += 1
        elif s2 > s1:
            current_wins[t2] += 1
            current_losses[t1] += 1
    
    for t1, t2 in remaining_games:
        games_remaining[t1] += 1
        games_remaining[t2] += 1
    
    # Calculate team ratings
    league_stats = _calculate_league_scoring_stats(completed_games)
    team_ratings = _calculate_team_ratings(completed_games, all_teams, regression_weight=0.3)
    scoring_profiles = _calculate_scoring_profiles(
        completed_games, all_teams,
        league_mean=league_stats["mean_score"],
        league_std=league_stats["std_score"],
        regression_weight=0.3
    )
    
    # Estimate playoff threshold
    threshold_stats = estimate_playoff_threshold(
        schedule_df, playoff_spots, num_threshold_sims,
        team_ratings, scoring_profiles, league_stats
    )
    threshold_wins = int(threshold_stats.get("threshold_wins", threshold_stats.get("median", 0)))
    
    # Calculate each team's odds
    results = []
    for team in all_teams:
        wins = current_wins[team]
        losses = current_losses[team]
        remaining = games_remaining[team]
        max_wins = wins + remaining
        
        # Expected win rate for team (use pairs only)
        remaining_games_tuples = [(t1, t2) for t1, t2 in remaining_games]
        exp_win_rate = calculate_expected_win_rate(team, remaining_games_tuples, team_ratings)
        expected_additional_wins = remaining * exp_win_rate
        projected_wins = wins + expected_additional_wins
        
        # Wins needed to hit threshold
        wins_needed = max(0, threshold_wins - wins)
        
        # Probability of reaching threshold
        if remaining > 0:
            prob_make_playoffs = calculate_probability_of_reaching_threshold(
                wins, remaining, exp_win_rate, threshold_wins
            )
        else:
            prob_make_playoffs = 100.0 if wins >= threshold_wins else 0.0
        
        # Probability of reaching "safe" threshold (75th percentile)
        safe_threshold = int(threshold_stats.get("p75", threshold_wins + 1))
        if remaining > 0:
            prob_safe = calculate_probability_of_reaching_threshold(
                wins, remaining, exp_win_rate, safe_threshold
            )
        else:
            prob_safe = 100.0 if wins >= safe_threshold else 0.0
        
        # Status via playoff probability and feasibility
        status = _determine_status(prob_make_playoffs, wins_needed, remaining)
        
        results.append({
            "Team": team,
            "Current Wins": wins,
            "Current Losses": losses,
            "Remaining Games": remaining,
            "Max Possible Wins": max_wins,
            "Projected Wins": round(projected_wins, 1),
            "Wins Needed": wins_needed,
            "Exp Win Rate": round(exp_win_rate * 100, 1),
            "Playoff %": round(prob_make_playoffs, 1),
            "Safe %": round(prob_safe, 1),  # Prob of reaching conservative threshold
            "Power Rating": round(team_ratings.get(team, 1500), 0),
            "Status": status,
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("Playoff %", ascending=False).reset_index(drop=True)
    
    # Add current rank
    df["Current Rank"] = df["Current Wins"].rank(ascending=False, method="min").astype(int)
    
    return df, threshold_stats


def get_team_playoff_scenarios(
    schedule_df: pd.DataFrame,
    team: str,
    playoff_spots: int = 6,
    num_threshold_sims: int = 1000,
    num_scenario_sims: int = 300,
) -> Dict[str, Any]:
    """
    Complete path-to-playoffs analysis for a single team including:
    - Remaining schedule with win probabilities
    - What-if scenarios with actual playoff probabilities
    - Teams to root for/against
    - Key matchups to watch
    """
    if schedule_df is None or schedule_df.empty:
        return {"error": "No schedule data"}
    
    all_teams = sorted(list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())))
    all_teams = [t for t in all_teams if not str(t).startswith("Scoring Period")]
    if team not in all_teams:
        return {"error": f"Team '{team}' not found"}
    
    # Split completed/remaining
    completed_games = []
    remaining_games = []  # (t1, t2, period)
    team_remaining_games = []
    for _, row in schedule_df.iterrows():
        t1, t2 = row["Team 1"], row["Team 2"]
        if str(t1).startswith("Scoring Period") or str(t2).startswith("Scoring Period"):
            continue
        s1, s2 = row["Score 1"], row["Score 2"]
        period = str(row.get("Scoring Period", "")).strip()
        
        if pd.isna(s1) or pd.isna(s2) or (s1 == 0 and s2 == 0):
            remaining_games.append((t1, t2, period))
            if t1 == team or t2 == team:
                opp = t2 if t1 == team else t1
                team_remaining_games.append({"opponent": opp, "period": period})
        else:
            completed_games.append((t1, t2, float(s1), float(s2)))
    
    # Current wins/losses for all teams
    current_wins = {t: 0 for t in all_teams}
    current_losses = {t: 0 for t in all_teams}
    for t1, t2, s1, s2 in completed_games:
        if s1 > s2:
            current_wins[t1] += 1
            current_losses[t2] += 1
        elif s2 > s1:
            current_wins[t2] += 1
            current_losses[t1] += 1
    
    team_wins = current_wins[team]
    team_losses = current_losses[team]
    team_remaining = len(team_remaining_games)
    
    # Ratings and profiles
    league_stats = _calculate_league_scoring_stats(completed_games)
    team_ratings = _calculate_team_ratings(completed_games, all_teams, regression_weight=0.3)
    scoring_profiles = _calculate_scoring_profiles(
        completed_games,
        all_teams,
        league_mean=league_stats["mean_score"],
        league_std=league_stats["std_score"],
        regression_weight=0.3,
    )
    
    # Threshold calc
    threshold_stats = estimate_playoff_threshold(
        schedule_df, playoff_spots, num_threshold_sims,
        team_ratings, scoring_profiles, league_stats
    )
    threshold_wins = int(threshold_stats.get("threshold_wins", threshold_stats.get("median", 0)))
    safe_target = int(threshold_stats.get("p75", threshold_wins + 1))
    lock_target = int(threshold_stats.get("p90", safe_target + 1))
    
    # Expected win rate for team
    remaining_pairs = [(t1, t2) for t1, t2, _ in remaining_games]
    exp_win_rate = calculate_expected_win_rate(team, remaining_pairs, team_ratings)
    expected_additional_wins = round(team_remaining * exp_win_rate, 1)
    projected_wins = round(team_wins + expected_additional_wins, 1)
    wins_needed = max(0, threshold_wins - team_wins)
    
    # Standings order
    sorted_by_wins = sorted(all_teams, key=lambda t: (current_wins[t], -current_losses[t]), reverse=True)
    team_rank = sorted_by_wins.index(team) + 1
    
    # 1) Remaining schedule with win probs
    schedule_rows = []
    team_rating = team_ratings.get(team, 1500.0)
    for game in team_remaining_games:
        opp = game["opponent"]
        opp_rating = team_ratings.get(opp, 1500.0)
        rating_diff = team_rating - opp_rating
        win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
        
        opp_wins = current_wins.get(opp, 0)
        opp_losses = current_losses.get(opp, 0)
        opp_rank = sorted_by_wins.index(opp) + 1 if opp in sorted_by_wins else len(all_teams)
        
        if opp_rank <= playoff_spots + 2 and opp_rank >= playoff_spots - 2:
            importance = "🔥 Critical"
        elif win_prob < 0.35:
            importance = "⚠️ Tough"
        elif win_prob > 0.65:
            importance = "✅ Favorable"
        else:
            importance = "📊 Toss-up"
        
        schedule_rows.append({
            "Period": game["period"],
            "Opponent": opp,
            "Opp Record": f"{opp_wins}-{opp_losses}",
            "Opp Rank": opp_rank,
            "Opp Rating": int(round(opp_rating, 0)),
            "Win Prob %": round(win_prob * 100, 1),
            "Importance": importance,
        })
    remaining_schedule_df = pd.DataFrame(schedule_rows)
    if not remaining_schedule_df.empty and "Period" in remaining_schedule_df.columns:
        remaining_schedule_df = remaining_schedule_df.sort_values("Period").reset_index(drop=True)

    def _period_sort_key(p: Any) -> int:
        s = str(p)
        digits = "".join([c for c in s if c.isdigit()])
        return int(digits) if digits else 10**9
    
    # 2) Scenarios with actual playoff probability
    win_probs = calculate_wins_probability(team_wins, team_remaining, exp_win_rate)
    other_remaining = [(t1, t2) for t1, t2, _ in remaining_games if t1 != team and t2 != team]
    scenario_rows = []
    for additional_wins in range(team_remaining + 1):
        final_wins = team_wins + additional_wins
        record_losses = team_remaining - additional_wins
        exact_prob = win_probs.get(final_wins, 0)
        cumulative_prob = sum(p for w, p in win_probs.items() if w >= final_wins)
        
        playoff_prob = _estimate_playoff_prob_at_wins_fast(
            team=team,
            target_wins=final_wins,
            all_teams=all_teams,
            playoff_spots=playoff_spots,
            team_ratings=team_ratings,
            current_wins=current_wins,
            other_remaining=other_remaining,
            num_sims=num_scenario_sims,
        )
        
        if playoff_prob >= 95:
            outlook = "🟢 Locked In"
        elif playoff_prob >= 80:
            outlook = "🟢 Very Likely"
        elif playoff_prob >= 60:
            outlook = "🟡 Good Chance"
        elif playoff_prob >= 40:
            outlook = "🟡 Coin Flip"
        elif playoff_prob >= 20:
            outlook = "🟠 Uphill Battle"
        elif playoff_prob > 0:
            outlook = "🔴 Long Shot"
        else:
            outlook = "❌ Eliminated"
        
        scenario_rows.append({
            "Finish": f"{additional_wins}-{record_losses}",
            "Final Wins": final_wins,
            "Chance of This": f"{round(exact_prob, 1)}%",
            "Playoff %": round(playoff_prob, 1),
            "Outlook": outlook,
        })
    scenarios_df = pd.DataFrame(scenario_rows)
    
    # 3) Teams to root for/against
    teams_impact = []
    for other_team in all_teams:
        if other_team == team:
            continue
        other_wins = current_wins[other_team]
        other_losses = current_losses[other_team]
        other_remaining_count = sum(1 for t1, t2, _ in remaining_games if t1 == other_team or t2 == other_team)
        other_max_wins = other_wins + other_remaining_count
        other_rank = sorted_by_wins.index(other_team) + 1
        
        if other_max_wins < threshold_wins - 4 and other_wins < team_wins - 4:
            continue
        
        if team_rank > playoff_spots:
            if other_rank <= playoff_spots:
                relationship = "🔴 Root Against"
                reason = f"#{other_rank} blocking your path"
            elif other_rank > playoff_spots and other_wins > team_wins:
                relationship = "🔴 Root Against"
                reason = f"Ahead of you ({other_wins}W vs {team_wins}W)"
            elif other_rank > playoff_spots and other_wins < team_wins:
                relationship = "🟢 Root For (vs others)"
                reason = "Behind you - keeps buffer"
            elif other_wins == team_wins:
                relationship = "🔴 Root Against"
                reason = "Tied with you - need separation"
            else:
                relationship = "⚪ Neutral"
                reason = "Minimal impact"
        else:
            if other_rank > playoff_spots:
                if other_max_wins >= team_wins:
                    relationship = "🔴 Root Against"
                    reason = f"#{other_rank} chasing (max {other_max_wins}W)"
                else:
                    relationship = "⚪ Neutral"
                    reason = "Can't catch you"
            else:
                relationship = "⚪ Neutral"
                reason = "Both in playoffs"
        
        other_opponents = [
            t2 if t1 == other_team else t1
            for t1, t2, _ in remaining_games
            if t1 == other_team or t2 == other_team
        ][:3]
        
        teams_impact.append({
            "Team": other_team,
            "Record": f"{other_wins}-{other_losses}",
            "Rank": other_rank,
            "Games Left": other_remaining_count,
            "Max Wins": other_max_wins,
            "Relationship": relationship,
            "Why": reason,
            "Next Opponents": ", ".join(other_opponents) if other_opponents else "—",
        })
    
    def sort_key(x):
        if "Root Against" in x["Relationship"]:
            return (0, -x["Max Wins"])
        if "Root For" in x["Relationship"]:
            return (1, -x["Max Wins"])
        return (2, -x["Max Wins"])
    
    teams_impact.sort(key=sort_key)
    teams_impact_df = pd.DataFrame(teams_impact[:12])
    
    # 4) Key matchups elsewhere
    key_matchups = []
    for t1, t2, period in remaining_games:
        if t1 == team or t2 == team:
            continue
        t1_wins, t2_wins = current_wins[t1], current_wins[t2]
        t1_losses, t2_losses = current_losses[t1], current_losses[t2]
        t1_rank = sorted_by_wins.index(t1) + 1
        t2_rank = sorted_by_wins.index(t2) + 1
        
        t1_relevant = abs(t1_wins - team_wins) <= 3 or abs(t1_rank - playoff_spots) <= 2
        t2_relevant = abs(t2_wins - team_wins) <= 3 or abs(t2_rank - playoff_spots) <= 2
        if not (t1_relevant or t2_relevant):
            continue
        
        r1 = team_ratings.get(t1, 1500)
        r2 = team_ratings.get(t2, 1500)
        t1_prob = 1 / (1 + 10 ** (-(r1 - r2) / 400))
        favorite = t1 if t1_prob > 0.5 else t2
        fav_prob = max(t1_prob, 1 - t1_prob) * 100
        
        if team_rank > playoff_spots:
            if t1_rank <= playoff_spots and t2_rank > playoff_spots:
                preferred, why = t2, f"{t1} is in playoffs (#{t1_rank})"
            elif t2_rank <= playoff_spots and t1_rank > playoff_spots:
                preferred, why = t1, f"{t2} is in playoffs (#{t2_rank})"
            elif t1_wins > team_wins and t2_wins <= team_wins:
                preferred, why = t2, f"{t1} ahead ({t1_wins}W)"
            elif t2_wins > team_wins and t1_wins <= team_wins:
                preferred, why = t1, f"{t2} ahead ({t2_wins}W)"
            elif t1_wins > t2_wins:
                preferred, why = t2, f"{t1} has more wins"
            elif t2_wins > t1_wins:
                preferred, why = t1, f"{t2} has more wins"
            else:
                preferred, why = "Either", "Both similar to you"
        else:
            if t1_rank > playoff_spots and t2_rank > playoff_spots:
                preferred, why = "Either loses", "Both chasing playoff spots"
            elif t1_rank > playoff_spots:
                preferred, why = t2, f"{t1} (#{t1_rank}) chasing"
            elif t2_rank > playoff_spots:
                preferred, why = t1, f"{t2} (#{t2_rank}) chasing"
            else:
                preferred, why = "Either", "Both in playoffs"
        
        if (t1_rank in {playoff_spots, playoff_spots + 1}) or (t2_rank in {playoff_spots, playoff_spots + 1}):
            impact = "🔥 High"
        elif t1_relevant and t2_relevant:
            impact = "⚡ Medium"
        else:
            impact = "📊 Low"
        
        key_matchups.append({
            "Period": period,
            "Matchup": f"{t1} vs {t2}",
            "Records": f"({t1_wins}-{t1_losses}) vs ({t2_wins}-{t2_losses})",
            "Favorite": f"{favorite} ({round(fav_prob)}%)",
            "Root For": preferred,
            "Why": why,
            "Impact": impact,
        })
    
    impact_order = {"🔥 High": 0, "⚡ Medium": 1, "📊 Low": 2}
    key_matchups.sort(key=lambda x: (impact_order.get(x["Impact"], 3), str(x["Period"])))
    key_matchups_df = pd.DataFrame(key_matchups[:10])

    # 4b) This week + week-by-week conditional odds
    # Determine next scoring period for this team
    team_periods = sorted({g.get("period", "") for g in team_remaining_games}, key=_period_sort_key)
    next_period = team_periods[0] if team_periods else None

    this_week_summary = {}
    this_week_watch_df = pd.DataFrame()
    week_by_week_df = pd.DataFrame()

    if next_period:
        # Find this team's game in the next period
        team_game = None
        for t1, t2, period in remaining_games:
            if str(period).strip() != str(next_period).strip():
                continue
            if t1 == team or t2 == team:
                team_game = (t1, t2, period)
                break

        # Build a forced-winner key that matches the exact ordering we stored
        def _force_key(game: Tuple[str, str, str]) -> Tuple[str, str, str]:
            return (game[0], game[1], game[2])

        if team_game:
            t1, t2, period = team_game
            opp = t2 if t1 == team else t1

            # Baseline assumes no forced outcomes
            baseline_odds = _estimate_playoff_odds_with_forced_results(
                team=team,
                all_teams=all_teams,
                playoff_spots=playoff_spots,
                team_ratings=team_ratings,
                current_wins=current_wins,
                remaining_games=remaining_games,
                forced_winners={},
                num_sims=min(500, num_scenario_sims),
            )

            odds_if_win = _estimate_playoff_odds_with_forced_results(
                team=team,
                all_teams=all_teams,
                playoff_spots=playoff_spots,
                team_ratings=team_ratings,
                current_wins=current_wins,
                remaining_games=remaining_games,
                forced_winners={_force_key(team_game): team},
                num_sims=min(500, num_scenario_sims),
            )
            odds_if_loss = _estimate_playoff_odds_with_forced_results(
                team=team,
                all_teams=all_teams,
                playoff_spots=playoff_spots,
                team_ratings=team_ratings,
                current_wins=current_wins,
                remaining_games=remaining_games,
                forced_winners={_force_key(team_game): opp},
                num_sims=min(500, num_scenario_sims),
            )

            this_week_summary = {
                "Period": next_period,
                "Opponent": opp,
                "Baseline %": round(baseline_odds, 1),
                "If Win %": round(odds_if_win, 1),
                "If Loss %": round(odds_if_loss, 1),
                "Swing (W-L)": round(odds_if_win - odds_if_loss, 1),
            }

            # Root-for/against: evaluate other games in this week assuming we WIN our game
            watch_rows = []
            for g1, g2, g_period in remaining_games:
                if str(g_period).strip() != str(next_period).strip():
                    continue
                if g1 == team or g2 == team:
                    continue

                odds_g1 = _estimate_playoff_odds_with_forced_results(
                    team=team,
                    all_teams=all_teams,
                    playoff_spots=playoff_spots,
                    team_ratings=team_ratings,
                    current_wins=current_wins,
                    remaining_games=remaining_games,
                    forced_winners={_force_key(team_game): team, (g1, g2, g_period): g1},
                    num_sims=min(400, num_scenario_sims),
                )
                odds_g2 = _estimate_playoff_odds_with_forced_results(
                    team=team,
                    all_teams=all_teams,
                    playoff_spots=playoff_spots,
                    team_ratings=team_ratings,
                    current_wins=current_wins,
                    remaining_games=remaining_games,
                    forced_winners={_force_key(team_game): team, (g1, g2, g_period): g2},
                    num_sims=min(400, num_scenario_sims),
                )
                preferred = g1 if odds_g1 >= odds_g2 else g2
                delta = (odds_g1 - odds_g2) if preferred == g1 else (odds_g2 - odds_g1)
                watch_rows.append(
                    {
                        "Matchup": f"{g1} vs {g2}",
                        "Root For": preferred,
                        "Odds If Root %": round(max(odds_g1, odds_g2), 1),
                        "Alt Odds %": round(min(odds_g1, odds_g2), 1),
                        "Impact (pp)": round(delta, 1),
                    }
                )

            if watch_rows:
                this_week_watch_df = pd.DataFrame(watch_rows).sort_values(
                    "Impact (pp)", ascending=False
                ).reset_index(drop=True)

            # Week-by-week: odds if we win/lose each upcoming week (forcing ONLY our game that week)
            week_rows = []
            for p in team_periods[:6]:
                tg = None
                for a, b, per in remaining_games:
                    if str(per).strip() != str(p).strip():
                        continue
                    if a == team or b == team:
                        tg = (a, b, per)
                        break
                if not tg:
                    continue
                a, b, per = tg
                o = b if a == team else a
                p_win = _estimate_playoff_odds_with_forced_results(
                    team=team,
                    all_teams=all_teams,
                    playoff_spots=playoff_spots,
                    team_ratings=team_ratings,
                    current_wins=current_wins,
                    remaining_games=remaining_games,
                    forced_winners={_force_key(tg): team},
                    num_sims=min(350, num_scenario_sims),
                )
                p_loss = _estimate_playoff_odds_with_forced_results(
                    team=team,
                    all_teams=all_teams,
                    playoff_spots=playoff_spots,
                    team_ratings=team_ratings,
                    current_wins=current_wins,
                    remaining_games=remaining_games,
                    forced_winners={_force_key(tg): o},
                    num_sims=min(350, num_scenario_sims),
                )
                week_rows.append(
                    {
                        "Period": per,
                        "Opponent": o,
                        "If Win %": round(p_win, 1),
                        "If Loss %": round(p_loss, 1),
                        "Swing (pp)": round(p_win - p_loss, 1),
                    }
                )
            if week_rows:
                week_by_week_df = pd.DataFrame(week_rows)
    
    # 5) Path summary
    if wins_needed > team_remaining:
        path_summary = f"❌ **Mathematically eliminated** — Need {wins_needed} wins but only {team_remaining} games left"
        status = "ELIMINATED"
    elif wins_needed == 0 and team_wins >= lock_target:
        path_summary = f"✅ **Clinched** — Already at {team_wins} wins (lock target: {lock_target})"
        status = "CLINCHED"
    elif wins_needed == 0:
        path_summary = f"🟢 **In position** — {team_wins} wins, threshold ~{threshold_wins}"
        status = "IN_POSITION"
    elif wins_needed <= team_remaining * 0.3:
        path_summary = f"🟢 **Favorable** — Need {wins_needed} of {team_remaining} ({round(wins_needed/team_remaining*100)}%)"
        status = "FAVORABLE"
    elif wins_needed <= team_remaining * 0.5:
        path_summary = f"🟡 **Contending** — Need {wins_needed} of {team_remaining}"
        status = "CONTENDING"
    elif wins_needed <= team_remaining * 0.7:
        path_summary = f"🟠 **Uphill battle** — Must go {wins_needed}-{team_remaining - wins_needed} or better"
        status = "UPHILL"
    else:
        path_summary = f"🔴 **Long shot** — Need {wins_needed} of {team_remaining} ({round(wins_needed/team_remaining*100)}%)"
        status = "LONG_SHOT"
    
    if not scenarios_df.empty:
        most_likely_idx = scenarios_df["Chance of This"].str.replace("%", "").astype(float).idxmax()
        most_likely = scenarios_df.iloc[most_likely_idx]
        most_likely_finish = most_likely["Finish"]
        most_likely_playoff_pct = most_likely["Playoff %"]
    else:
        most_likely_finish = "N/A"
        most_likely_playoff_pct = 0
    
    return {
        "team": team,
        "current_record": f"{team_wins}-{team_losses}",
        "current_wins": team_wins,
        "current_losses": team_losses,
        "remaining_games": team_remaining,
        "team_rank": team_rank,
        "total_teams": len(all_teams),
        "in_playoff_position": team_rank <= playoff_spots,
        "team_rating": int(round(team_rating, 0)),
        "expected_win_rate_pct": round(exp_win_rate * 100, 1),
        "threshold_wins": threshold_wins,
        "safe_target": safe_target,
        "lock_target": lock_target,
        "wins_needed": wins_needed,
        "projected_wins": projected_wins,
        "best_case_wins": team_wins + team_remaining,
        "worst_case_wins": team_wins,
        "most_likely_finish": most_likely_finish,
        "most_likely_playoff_pct": most_likely_playoff_pct,
        "status": status,
        "path_summary": path_summary,
        "remaining_schedule": remaining_schedule_df,
        "scenarios": scenarios_df,
        "teams_to_watch": teams_impact_df,
        "key_matchups": key_matchups_df,
        "this_week": this_week_summary,
        "this_week_watch": this_week_watch_df,
        "week_by_week": week_by_week_df,
        "threshold_stats": threshold_stats,
    }


def _estimate_playoff_prob_at_wins_fast(
    team: str,
    target_wins: int,
    all_teams: List[str],
    playoff_spots: int,
    team_ratings: Dict[str, float],
    current_wins: Dict[str, int],
    other_remaining: List[Tuple[str, str]],
    num_sims: int = 300,
) -> float:
    """
    Fast estimation of playoff probability if team finishes with exactly target_wins.
    Simulates other teams' remaining games to see how often target_wins is enough.
    """
    if num_sims <= 0:
        return 0.0
    
    made_playoffs = 0
    
    for _ in range(num_sims):
        sim_wins = current_wins.copy()
        sim_wins[team] = target_wins
        
        for t1, t2 in other_remaining:
            r1 = team_ratings.get(t1, 1500)
            r2 = team_ratings.get(t2, 1500)
            win_prob = 1 / (1 + 10 ** (-(r1 - r2) / 400))
            if random.random() < win_prob:
                sim_wins[t1] = sim_wins.get(t1, 0) + 1
            else:
                sim_wins[t2] = sim_wins.get(t2, 0) + 1
        
        sorted_teams = sorted(all_teams, key=lambda t: sim_wins.get(t, 0), reverse=True)
        if team in sorted_teams[:playoff_spots]:
            made_playoffs += 1
    
    return (made_playoffs / num_sims) * 100


def _estimate_playoff_odds_with_forced_results(
    *,
    team: str,
    all_teams: List[str],
    playoff_spots: int,
    team_ratings: Dict[str, float],
    current_wins: Dict[str, int],
    remaining_games: List[Tuple[str, str, str]],
    forced_winners: Dict[Tuple[str, str, str], str],
    num_sims: int = 300,
) -> float:
    """Estimate playoff odds while forcing outcomes of some remaining games.

    `forced_winners` maps a game key (team1, team2, period) to the forced winner.
    """
    if num_sims <= 0:
        return 0.0

    made_playoffs = 0
    for _ in range(num_sims):
        sim_wins = current_wins.copy()

        # Apply forced results first
        for (t1, t2, period), winner in forced_winners.items():
            if winner == t1:
                sim_wins[t1] = sim_wins.get(t1, 0) + 1
            elif winner == t2:
                sim_wins[t2] = sim_wins.get(t2, 0) + 1

        # Simulate remaining (non-forced) games
        for t1, t2, period in remaining_games:
            game_key = (t1, t2, period)
            if game_key in forced_winners:
                continue
            # Also consider reversed ordering as same matchup
            rev_key = (t2, t1, period)
            if rev_key in forced_winners:
                continue

            r1 = team_ratings.get(t1, 1500)
            r2 = team_ratings.get(t2, 1500)
            win_prob = 1 / (1 + 10 ** (-(r1 - r2) / 400))
            if random.random() < win_prob:
                sim_wins[t1] = sim_wins.get(t1, 0) + 1
            else:
                sim_wins[t2] = sim_wins.get(t2, 0) + 1

        sorted_teams = sorted(all_teams, key=lambda t: sim_wins.get(t, 0), reverse=True)
        if team in sorted_teams[:playoff_spots]:
            made_playoffs += 1

    return (made_playoffs / num_sims) * 100


def display_team_playoff_path(team: str, schedule_df: pd.DataFrame, playoff_spots: int = 6):
    """
    Streamlit helper to show a team's path to playoffs using get_team_playoff_scenarios.
    """
    with st.spinner(f"Analyzing {team}'s playoff scenarios..."):
        scenarios = get_team_playoff_scenarios(
            schedule_df,
            team,
            playoff_spots=playoff_spots,
            num_threshold_sims=1000,
            num_scenario_sims=300,
        )
    if "error" in scenarios:
        st.error(scenarios["error"])
        return

    st.markdown(f"## 🏀 {team} — Path to Playoffs")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Record", scenarios["current_record"])
    col2.metric("Games Left", scenarios["remaining_games"])
    col3.metric("Proj Wins", scenarios["projected_wins"])
    col4.metric("Wins Needed", scenarios["wins_needed"])

    st.markdown(
        f"**Target:** ~{scenarios['threshold_wins']} wins · "
        f"Safe: {scenarios.get('safe_target', scenarios['threshold_wins'])} · "
        f"Lock: {scenarios.get('lock_target', scenarios['threshold_wins'])}"
    )

    # This week / week-by-week needs
    if scenarios.get("this_week"):
        tw = scenarios["this_week"]
        st.markdown("#### This Week: What needs to happen")
        st.markdown(
            f"**{tw.get('Period')}** vs **{tw.get('Opponent')}** — "
            f"If you **win**: **{tw.get('If Win %', 0)}%** · "
            f"If you **lose**: **{tw.get('If Loss %', 0)}%** · "
            f"Swing: **{tw.get('Swing (W-L)', 0)} pp**"
        )

        if scenarios.get("this_week_watch") is not None and not scenarios["this_week_watch"].empty:
            st.markdown("##### Rooting Guide (assuming you win)")
            st.dataframe(
                scenarios["this_week_watch"].head(8),
                use_container_width=True,
                hide_index=True,
            )

    if scenarios.get("week_by_week") is not None and not scenarios["week_by_week"].empty:
        st.markdown("#### Week-by-Week: win/loss impact")
        st.dataframe(
            scenarios["week_by_week"],
            use_container_width=True,
            hide_index=True,
        )

    # Remaining schedule
    st.markdown("#### Remaining Schedule (win probabilities)")
    if not scenarios["remaining_schedule"].empty:
        st.dataframe(scenarios["remaining_schedule"], use_container_width=True, hide_index=True)
    else:
        st.info("No remaining games.")

    # What-if scenarios
    st.markdown("#### What-If Scenarios (finish record → playoff probability)")
    if not scenarios["scenarios"].empty:
        st.dataframe(scenarios["scenarios"], use_container_width=True, hide_index=True)
    else:
        st.info("Season complete.")


def _determine_status(playoff_pct: float, wins_needed: int, remaining: int) -> str:
    """Determine realistic status based on playoff probability and feasibility."""
    if wins_needed > remaining:
        return "❌ ELIMINATED"
    if playoff_pct >= 99:
        return "✅ CLINCHED"
    if playoff_pct >= 90:
        return "🟢 SAFE"
    if playoff_pct >= 70:
        return "🟢 LIKELY"
    if playoff_pct >= 50:
        return "🟡 CONTENDING"
    if playoff_pct >= 25:
        return "🟠 BUBBLE"
    if playoff_pct >= 5:
        return "🔴 LONG SHOT"
    return "💀 MIRACLE NEEDED"


def get_remaining_games_count(schedule_df: pd.DataFrame) -> Dict[str, int]:
    """Count remaining games per team."""
    remaining = defaultdict(int)
    for _, row in schedule_df.iterrows():
        team1, team2 = row["Team 1"], row["Team 2"]
        if str(team1).startswith("Scoring Period"):
            continue
        score1, score2 = row["Score 1"], row["Score 2"]
        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            remaining[team1] += 1
            remaining[team2] += 1
    return dict(remaining)


def get_playoff_summary(playoff_probabilities: Dict[str, Dict], remaining_sos: Dict[str, float] = None, team_ratings: Dict[str, float] = None) -> pd.DataFrame:
    """Create a summary DataFrame of playoff probabilities."""
    if not playoff_probabilities:
        return pd.DataFrame()
    
    summary_data = []
    for team, probs in playoff_probabilities.items():
        row = {"Team": team}
        row.update(probs)
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Add columns for common playoff metrics
    if "playoffs" in df.columns:
        df["Playoff %"] = df["playoffs"]
        df["Miss %"] = df.get("missed_playoffs", 0)
    
    # Find most likely seed
    seed_cols = [col for col in df.columns if col.startswith("seed_") and col[5:].isdigit()]
    if seed_cols:
        seed_df = df[seed_cols]
        most_likely = seed_df.idxmax(axis=1).str.replace("seed_", "")
        # convert to numeric, keeping NaN where no seeds are present
        df["Most Likely Seed"] = pd.to_numeric(most_likely, errors="coerce")
        df["Seed Confidence"] = seed_df.max(axis=1)
    
    # Add seed range if available
    if "best_seed" in df.columns and "worst_seed" in df.columns:
        df["Seed Range"] = df.apply(lambda r: f"{int(r['best_seed'])}-{int(r['worst_seed'])}", axis=1)
    
    # Add remaining SOS if available
    if remaining_sos is not None:
        df["Remaining SOS"] = df["Team"].map(remaining_sos).fillna(0).round(1)
    
    # Add power rating if available
    if team_ratings is not None:
        df["Power Rating"] = df["Team"].map(team_ratings).fillna(1500).round(0)
    
    # Add games behind leader
    if "Playoff %" in df.columns:
        df = df.sort_values("Playoff %", ascending=False).reset_index(drop=True)
        if len(df) > 0:
            leader_pct = df.iloc[0]["Playoff %"]
            df["GB"] = ((leader_pct - df["Playoff %"]) / 100 * 10).round(1)  # Rough proxy
    
    return df.sort_values("Playoff %", ascending=False)
