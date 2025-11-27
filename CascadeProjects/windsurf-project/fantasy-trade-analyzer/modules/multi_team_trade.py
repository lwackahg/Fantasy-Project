"""
Multi-team trade suggestion engine.
Handles circular trades (e.g. A -> B -> C -> A) for N teams.
"""
import pandas as pd
from typing import Dict, List, Optional
from itertools import permutations, product

from modules.trade_suggestions_core import (
    _calculate_core_value,
    _get_core_size,
    _simulate_core_value_gain,
)
from modules.trade_suggestions import (
    calculate_league_scarcity_context,
    calculate_player_value,
)

MAX_MULTI_TEAM_COMBINATIONS = 200000

def find_circular_trade_suggestions(
    teams_map: Dict[str, pd.DataFrame],
    target_teams: List[str],
    min_value_gain: float = 0.0,
    max_suggestions: int = 20,
    player_fpts_overrides: Dict[str, float] = None,
) -> List[Dict]:
    """
    Find circular trade suggestions involving all specified teams.
    Example (3 teams): Team A -> Team B -> Team C -> Team A
    Each team gives 1 player to the next team in the cycle.
    
    Args:
        teams_map: Dict of team_name -> DataFrame (roster)
        target_teams: List of team names to include in the cycle
        min_value_gain: Minimum core value gain required for EACH team involved
        max_suggestions: Max suggestions to return
        player_fpts_overrides: Optional FP/G overrides
        
    Returns:
        List of trade suggestion dictionaries
    """
    if not target_teams or len(target_teams) < 3:
        return []

    # Filter and prepare team dataframes
    active_teams = {}
    for name in target_teams:
        if name not in teams_map:
            return []
        df = teams_map[name].copy()
        
        # Apply overrides
        if player_fpts_overrides:
            for player, fpts in player_fpts_overrides.items():
                mask = df['Player'] == player
                if mask.any():
                    df.loc[mask, 'Mean FPts'] = fpts
                    
        active_teams[name] = df

    # Calculate context
    scarcity_context = calculate_league_scarcity_context(teams_map)
    
    # Precompute core values and player values for all active teams
    team_meta = {}
    core_size = _get_core_size()
    
    for name, df in active_teams.items():
        # Calculate values
        df['Value'] = df.apply(
            lambda row: calculate_player_value(row, scarcity_context=scarcity_context),
            axis=1
        )
        baseline_core = _calculate_core_value(df, core_size)
        
        # Filter tradable players (simplify to top 30 by value to reduce complexity)
        tradable = df.nlargest(30, 'Value')
        
        team_meta[name] = {
            'df': df,
            'baseline_core': baseline_core,
            'tradable_players': tradable.to_dict('records')
        }

    suggestions = []
    
    # Generate all cycles (permutations of teams)
    # Fix first team to avoid duplicates of same cycle rotated
    # e.g. A-B-C is same cycle as B-C-A and C-A-B
    first_team = target_teams[0]
    other_teams = target_teams[1:]
    
    cycle_count = 0
    
    for p in permutations(other_teams):
        cycle_order = [first_team] + list(p)
        # Cycle is: cycle_order[0] gives to cycle_order[1], [1] to [2], ..., [N] to [0]
        
        # Create list of player lists for product()
        # index i corresponds to team cycle_order[i]
        player_lists = [team_meta[t]['tradable_players'] for t in cycle_order]
        
        combo_count = 0
        for players in product(*player_lists):
            combo_count += 1
            if combo_count > MAX_MULTI_TEAM_COMBINATIONS:
                break
            
            # 'players' is a tuple of player dicts, one from each team in cycle_order
            # Player at i is given BY team i TO team (i+1)%N
            
            # Validation: Calculate gain for EACH team
            valid_trade = True
            team_gains = {}
            
            num_teams = len(cycle_order)
            for i in range(num_teams):
                team_name = cycle_order[i]
                player_given = players[i]
                
                # Team i receives player from team (i-1)%N
                # The player received is located at index (i-1)%N in the 'players' tuple
                prev_idx = (i - 1) % num_teams
                player_received = players[prev_idx]
                
                meta = team_meta[team_name]
                
                # Check 1-for-1 gain
                # Note: simulating gain using FULL roster (meta['df'])
                core_gain = _simulate_core_value_gain(
                    meta['df'],
                    [player_given],  # Give
                    [player_received], # Get
                    core_size,
                    meta['baseline_core']
                )
                
                if core_gain < min_value_gain:
                    valid_trade = False
                    break
                    
                team_gains[team_name] = core_gain
            
            if valid_trade:
                # Construct result object
                transfers = []
                for i in range(num_teams):
                    giver = cycle_order[i]
                    receiver = cycle_order[(i + 1) % num_teams]
                    p_obj = players[i]
                    transfers.append({
                        "from": giver,
                        "to": receiver,
                        "player": p_obj["Player"],
                        "value": p_obj["Value"],
                        "fpts": p_obj["Mean FPts"]
                    })
                
                # Total value created (sum of all teams' gains)
                total_gain = sum(team_gains.values())
                
                suggestions.append({
                    "type": "circular_multi_team",
                    "teams": cycle_order,
                    "transfers": transfers,
                    "gains": team_gains,
                    "total_value_gain": total_gain
                })
                
        cycle_count += 1

    # Sort by total value created across all teams
    suggestions.sort(key=lambda x: x['total_value_gain'], reverse=True)
    
    return suggestions[:max_suggestions]
