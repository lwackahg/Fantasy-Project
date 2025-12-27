"""Playoff odds + path-to-playoffs logic.

This module contains the threshold-based playoff odds and team path analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import random

import numpy as np
import pandas as pd

from logic.schedule_analysis import (
    _calculate_league_scoring_stats,
    _calculate_scoring_profiles,
    _calculate_simulated_standings,
    _calculate_team_ratings,
    _determine_status,
    _simulate_games_once,
)


def estimate_playoff_threshold(
    schedule_df: pd.DataFrame,
    playoff_spots: int = 6,
    num_simulations: int = 2000,
    team_ratings: Dict[str, float] | None = None,
    scoring_profiles: Dict[str, Dict] | None = None,
    league_stats: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Estimate the win total needed to make playoffs.

    Runs simulations and tracks what the "last team in" finishes with.
    """
    if schedule_df is None or schedule_df.empty:
        return {"mean": 0, "median": 0, "p25": 0, "p75": 0, "p90": 0}

    all_teams = sorted(
        list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique()))
    )
    all_teams = [team for team in all_teams if not str(team).startswith("Scoring Period")]

    completed_games: List[Tuple[str, str, float, float]] = []
    remaining_games: List[Tuple[str, str]] = []

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
        final_standings = _calculate_simulated_standings(completed_games, all_teams)
        if playoff_spots <= len(final_standings):
            last_team_in = final_standings[playoff_spots - 1]
            wins = sum(
                1
                for t1, t2, s1, s2 in completed_games
                if (t1 == last_team_in and s1 > s2) or (t2 == last_team_in and s2 > s1)
            )
            return {
                "mean": wins,
                "median": wins,
                "p25": wins,
                "p75": wins,
                "p90": wins,
                "threshold_wins": wins,
            }
        return {"mean": 0, "median": 0, "p25": 0, "p75": 0, "p90": 0}

    if league_stats is None:
        league_stats = _calculate_league_scoring_stats(completed_games)
    if team_ratings is None:
        team_ratings = _calculate_team_ratings(completed_games, all_teams, regression_weight=0.3)
    if scoring_profiles is None:
        scoring_profiles = _calculate_scoring_profiles(
            completed_games,
            all_teams,
            league_mean=league_stats["mean_score"],
            league_std=league_stats["std_score"],
            regression_weight=0.3,
        )

    last_team_in_wins: List[int] = []
    for _ in range(num_simulations):
        sim_results = _simulate_games_once(
            remaining_games,
            team_ratings,
            scoring_profiles,
            completed_games,
            league_stats,
        )

        team_wins = {team: 0 for team in all_teams}
        for t1, t2, s1, s2 in sim_results:
            if t1 in team_wins and t2 in team_wins:
                if s1 > s2:
                    team_wins[t1] += 1
                elif s2 > s1:
                    team_wins[t2] += 1

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
        "threshold_wins": round(float(np.percentile(arr, 50)), 0),
    }


def calculate_wins_probability(
    current_wins: int,
    remaining_games: int,
    expected_win_rate: float,
) -> Dict[int, float]:
    """Probability of reaching each possible final win total (binomial)."""
    from scipy import stats

    probs: Dict[int, float] = {}
    for additional_wins in range(remaining_games + 1):
        final_wins = current_wins + additional_wins
        prob = stats.binom.pmf(additional_wins, remaining_games, expected_win_rate)
        probs[final_wins] = prob * 100

    return probs


def calculate_probability_of_reaching_threshold(
    current_wins: int,
    remaining_games: int,
    expected_win_rate: float,
    threshold_wins: int,
) -> float:
    """P(final_wins >= threshold_wins) via binomial CDF."""
    from scipy import stats

    wins_needed = max(0, threshold_wins - current_wins)

    if wins_needed > remaining_games:
        return 0.0
    if wins_needed <= 0:
        return 100.0

    prob = 1 - stats.binom.cdf(wins_needed - 1, remaining_games, expected_win_rate)
    return prob * 100


def calculate_expected_win_rate(
    team: str,
    remaining_games: List[Tuple],
    team_ratings: Dict[str, float],
) -> float:
    """Expected win rate from Elo for the remaining schedule."""
    opponents: List[str] = []
    for t1, t2 in remaining_games:
        if t1 == team:
            opponents.append(t2)
        elif t2 == team:
            opponents.append(t1)

    if not opponents:
        return 0.5

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
    """Threshold-based playoff odds (binomial + simulated threshold)."""
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(), {}

    all_teams = sorted(
        list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique()))
    )
    all_teams = [team for team in all_teams if not str(team).startswith("Scoring Period")]

    completed_games: List[Tuple[str, str, float, float]] = []
    remaining_games: List[Tuple[str, str]] = []

    for _, row in schedule_df.iterrows():
        team1, team2 = row["Team 1"], row["Team 2"]
        if str(team1).startswith("Scoring Period") or str(team2).startswith("Scoring Period"):
            continue
        score1, score2 = row["Score 1"], row["Score 2"]

        if pd.isna(score1) or pd.isna(score2) or (score1 == 0 and score2 == 0):
            remaining_games.append((team1, team2))
        else:
            completed_games.append((team1, team2, float(score1), float(score2)))

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

    league_stats = _calculate_league_scoring_stats(completed_games)
    team_ratings = _calculate_team_ratings(completed_games, all_teams, regression_weight=0.3)
    scoring_profiles = _calculate_scoring_profiles(
        completed_games,
        all_teams,
        league_mean=league_stats["mean_score"],
        league_std=league_stats["std_score"],
        regression_weight=0.3,
    )

    threshold_stats = estimate_playoff_threshold(
        schedule_df,
        playoff_spots,
        num_threshold_sims,
        team_ratings,
        scoring_profiles,
        league_stats,
    )
    threshold_wins = int(threshold_stats.get("threshold_wins", threshold_stats.get("median", 0)))

    results: List[Dict[str, Any]] = []
    for team in all_teams:
        wins = current_wins[team]
        losses = current_losses[team]
        remaining = games_remaining[team]
        max_wins = wins + remaining

        remaining_games_tuples = [(t1, t2) for t1, t2 in remaining_games]
        exp_win_rate = calculate_expected_win_rate(team, remaining_games_tuples, team_ratings)
        expected_additional_wins = remaining * exp_win_rate
        projected_wins = wins + expected_additional_wins

        wins_needed = max(0, threshold_wins - wins)

        if remaining > 0:
            prob_make_playoffs = calculate_probability_of_reaching_threshold(
                wins, remaining, exp_win_rate, threshold_wins
            )
        else:
            prob_make_playoffs = 100.0 if wins >= threshold_wins else 0.0

        safe_threshold = int(threshold_stats.get("p75", threshold_wins + 1))
        if remaining > 0:
            prob_safe = calculate_probability_of_reaching_threshold(
                wins, remaining, exp_win_rate, safe_threshold
            )
        else:
            prob_safe = 100.0 if wins >= safe_threshold else 0.0

        status = _determine_status(prob_make_playoffs, wins_needed, remaining)

        results.append(
            {
                "Team": team,
                "Current Wins": wins,
                "Current Losses": losses,
                "Remaining Games": remaining,
                "Max Possible Wins": max_wins,
                "Projected Wins": round(projected_wins, 1),
                "Wins Needed": wins_needed,
                "Exp Win Rate": round(exp_win_rate * 100, 1),
                "Playoff %": round(prob_make_playoffs, 1),
                "Safe %": round(prob_safe, 1),
                "Power Rating": round(team_ratings.get(team, 1500), 0),
                "Status": status,
            }
        )

    df = pd.DataFrame(results).sort_values("Playoff %", ascending=False).reset_index(drop=True)
    df["Current Rank"] = df["Current Wins"].rank(ascending=False, method="min").astype(int)

    return df, threshold_stats


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
    """Fast estimation of playoff probability if team finishes with exactly target_wins."""
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
    """Estimate playoff odds while forcing outcomes of some remaining games."""
    if num_sims <= 0:
        return 0.0

    made_playoffs = 0
    for _ in range(num_sims):
        sim_wins = current_wins.copy()

        for (t1, t2, period), winner in forced_winners.items():
            if winner == t1:
                sim_wins[t1] = sim_wins.get(t1, 0) + 1
            elif winner == t2:
                sim_wins[t2] = sim_wins.get(t2, 0) + 1

        for t1, t2, period in remaining_games:
            game_key = (t1, t2, period)
            if game_key in forced_winners:
                continue
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


def get_team_playoff_scenarios(
    schedule_df: pd.DataFrame,
    team: str,
    playoff_spots: int = 6,
    num_threshold_sims: int = 1000,
    num_scenario_sims: int = 300,
) -> Dict[str, Any]:
    """Complete path-to-playoffs analysis for a single team."""
    if schedule_df is None or schedule_df.empty:
        return {"error": "No schedule data"}

    all_teams = sorted(
        list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique()))
    )
    all_teams = [t for t in all_teams if not str(t).startswith("Scoring Period")]
    if team not in all_teams:
        return {"error": f"Team '{team}' not found"}

    completed_games: List[Tuple[str, str, float, float]] = []
    remaining_games: List[Tuple[str, str, str]] = []
    team_remaining_games: List[Dict[str, Any]] = []

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

    league_stats = _calculate_league_scoring_stats(completed_games)
    team_ratings = _calculate_team_ratings(completed_games, all_teams, regression_weight=0.3)
    scoring_profiles = _calculate_scoring_profiles(
        completed_games,
        all_teams,
        league_mean=league_stats["mean_score"],
        league_std=league_stats["std_score"],
        regression_weight=0.3,
    )

    threshold_stats = estimate_playoff_threshold(
        schedule_df,
        playoff_spots,
        num_threshold_sims,
        team_ratings,
        scoring_profiles,
        league_stats,
    )
    threshold_wins = int(threshold_stats.get("threshold_wins", threshold_stats.get("median", 0)))
    safe_target = int(threshold_stats.get("p75", threshold_wins + 1))
    lock_target = int(threshold_stats.get("p90", safe_target + 1))

    remaining_pairs = [(t1, t2) for t1, t2, _ in remaining_games]
    exp_win_rate = calculate_expected_win_rate(team, remaining_pairs, team_ratings)
    expected_additional_wins = round(team_remaining * exp_win_rate, 1)
    projected_wins = round(team_wins + expected_additional_wins, 1)
    wins_needed = max(0, threshold_wins - team_wins)

    sorted_by_wins = sorted(
        all_teams,
        key=lambda t: (current_wins[t], -current_losses[t]),
        reverse=True,
    )
    team_rank = sorted_by_wins.index(team) + 1

    schedule_rows: List[Dict[str, Any]] = []
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

        schedule_rows.append(
            {
                "Period": game["period"],
                "Opponent": opp,
                "Opp Record": f"{opp_wins}-{opp_losses}",
                "Opp Rank": opp_rank,
                "Opp Rating": int(round(opp_rating, 0)),
                "Win Prob %": round(win_prob * 100, 1),
                "Importance": importance,
            }
        )

    remaining_schedule_df = pd.DataFrame(schedule_rows)
    if not remaining_schedule_df.empty and "Period" in remaining_schedule_df.columns:
        remaining_schedule_df = remaining_schedule_df.sort_values("Period").reset_index(drop=True)

    def _period_sort_key(p: Any) -> int:
        s = str(p)
        digits = "".join([c for c in s if c.isdigit()])
        return int(digits) if digits else 10**9

    win_probs = calculate_wins_probability(team_wins, team_remaining, exp_win_rate)
    other_remaining = [(t1, t2) for t1, t2, _ in remaining_games if t1 != team and t2 != team]

    scenario_rows: List[Dict[str, Any]] = []
    for additional_wins in range(team_remaining + 1):
        final_wins = team_wins + additional_wins
        record_losses = team_remaining - additional_wins
        exact_prob = win_probs.get(final_wins, 0)

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

        scenario_rows.append(
            {
                "Finish": f"{additional_wins}-{record_losses}",
                "Final Wins": final_wins,
                "Chance of This": f"{round(exact_prob, 1)}%",
                "Playoff %": round(playoff_prob, 1),
                "Outlook": outlook,
            }
        )

    scenarios_df = pd.DataFrame(scenario_rows)

    teams_impact: List[Dict[str, Any]] = []
    for other_team in all_teams:
        if other_team == team:
            continue
        other_wins = current_wins[other_team]
        other_losses = current_losses[other_team]
        other_remaining_count = sum(
            1 for t1, t2, _ in remaining_games if t1 == other_team or t2 == other_team
        )
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

        teams_impact.append(
            {
                "Team": other_team,
                "Record": f"{other_wins}-{other_losses}",
                "Rank": other_rank,
                "Games Left": other_remaining_count,
                "Max Wins": other_max_wins,
                "Relationship": relationship,
                "Why": reason,
                "Next Opponents": ", ".join(other_opponents) if other_opponents else "—",
            }
        )

    def sort_key(x: Dict[str, Any]):
        if "Root Against" in x["Relationship"]:
            return (0, -x["Max Wins"])
        if "Root For" in x["Relationship"]:
            return (1, -x["Max Wins"])
        return (2, -x["Max Wins"])

    teams_impact.sort(key=sort_key)
    teams_impact_df = pd.DataFrame(teams_impact[:12])

    key_matchups: List[Dict[str, Any]] = []
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

        if (t1_rank in {playoff_spots, playoff_spots + 1}) or (
            t2_rank in {playoff_spots, playoff_spots + 1}
        ):
            impact = "🔥 High"
        elif t1_relevant and t2_relevant:
            impact = "⚡ Medium"
        else:
            impact = "📊 Low"

        key_matchups.append(
            {
                "Period": period,
                "Matchup": f"{t1} vs {t2}",
                "Records": f"({t1_wins}-{t1_losses}) vs ({t2_wins}-{t2_losses})",
                "Favorite": f"{favorite} ({round(fav_prob)}%)",
                "Root For": preferred,
                "Why": why,
                "Impact": impact,
            }
        )

    impact_order = {"🔥 High": 0, "⚡ Medium": 1, "📊 Low": 2}
    key_matchups.sort(key=lambda x: (impact_order.get(x["Impact"], 3), str(x["Period"])))
    key_matchups_df = pd.DataFrame(key_matchups[:10])

    team_periods = sorted({g.get("period", "") for g in team_remaining_games}, key=_period_sort_key)
    next_period = team_periods[0] if team_periods else None

    this_week_summary: Dict[str, Any] = {}
    this_week_watch_df = pd.DataFrame()
    week_by_week_df = pd.DataFrame()

    if next_period:
        team_game = None
        for t1, t2, period in remaining_games:
            if str(period).strip() != str(next_period).strip():
                continue
            if t1 == team or t2 == team:
                team_game = (t1, t2, period)
                break

        def _force_key(game: Tuple[str, str, str]) -> Tuple[str, str, str]:
            return (game[0], game[1], game[2])

        if team_game:
            t1, t2, period = team_game
            opp = t2 if t1 == team else t1

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

            watch_rows: List[Dict[str, Any]] = []
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
                this_week_watch_df = (
                    pd.DataFrame(watch_rows)
                    .sort_values("Impact (pp)", ascending=False)
                    .reset_index(drop=True)
                )

            week_rows: List[Dict[str, Any]] = []
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

    if wins_needed > team_remaining:
        path_summary = (
            f"❌ **Mathematically eliminated** — Need {wins_needed} wins "
            f"but only {team_remaining} games left"
        )
        status = "ELIMINATED"
    elif wins_needed == 0 and team_wins >= lock_target:
        path_summary = (
            f"✅ **Clinched** — Already at {team_wins} wins (lock target: {lock_target})"
        )
        status = "CLINCHED"
    elif wins_needed == 0:
        path_summary = f"🟢 **In position** — {team_wins} wins, threshold ~{threshold_wins}"
        status = "IN_POSITION"
    elif wins_needed <= team_remaining * 0.3:
        path_summary = (
            f"🟢 **Favorable** — Need {wins_needed} of {team_remaining} "
            f"({round(wins_needed / team_remaining * 100)}%)"
        )
        status = "FAVORABLE"
    elif wins_needed <= team_remaining * 0.5:
        path_summary = f"🟡 **Contending** — Need {wins_needed} of {team_remaining}"
        status = "CONTENDING"
    elif wins_needed <= team_remaining * 0.7:
        path_summary = (
            f"🟠 **Uphill battle** — Must go {wins_needed}-{team_remaining - wins_needed} "
            f"or better"
        )
        status = "UPHILL"
    else:
        path_summary = (
            f"🔴 **Long shot** — Need {wins_needed} of {team_remaining} "
            f"({round(wins_needed / team_remaining * 100)}%)"
        )
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
