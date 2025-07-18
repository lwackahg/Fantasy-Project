import pandas as pd

# Tier-based adjustments
TIER_WEIGHTS = {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.8}

def calculate_initial_values(pps_df, num_teams, roster_spots_per_team, budget_per_team, vorp_budget_pct=0.97):
    """
    Calculates initial auction values based on the PPS, Tiers, and VORP model.
    """
    """
    Calculates initial auction values based on the PPS and VORP model.
    """
    # 1. Define Replacement Player
    total_rostered_players = num_teams * roster_spots_per_team
    replacement_player_index = total_rostered_players 

    if len(pps_df) <= replacement_player_index:
        # Handle case where player pool is smaller than total roster spots
        replacement_level_pps = pps_df['PPS'].min() 
    else:
        replacement_level_pps = pps_df.iloc[replacement_player_index]['PPS']

    # 2. Assign Tiers
    pps_df['Tier'] = pd.qcut(pps_df['PPS'], 5, labels=False, duplicates='drop') + 1

    # 3. Calculate VORP Score
    pps_df['VORP'] = pps_df['PPS'] - replacement_level_pps
    pps_df['VORP'] = pps_df['VORP'].clip(lower=0)

    # 4. Calculate Tier-Adjusted VORP
    pps_df['TierWeight'] = pps_df['Tier'].map(TIER_WEIGHTS)
    pps_df['TierAdjustedVORP'] = pps_df['VORP'] * pps_df['TierWeight']

    # 3. Allocate Budget
    total_league_budget = num_teams * budget_per_team
    total_vorp_budget = total_league_budget * vorp_budget_pct

    # 5. Calculate Dollar-per-Tier-Adjusted-VORP
    total_tier_adjusted_vorp_pool = pps_df['TierAdjustedVORP'].sum()
    dollar_per_vorp = total_vorp_budget / total_tier_adjusted_vorp_pool if total_tier_adjusted_vorp_pool > 0 else 0

    # 6. Final Calculated Auction Value
    pps_df['BaseValue'] = (pps_df['TierAdjustedVORP'] * dollar_per_vorp) + 1
    pps_df['BaseValue'] = pps_df['BaseValue'].round(2)

    return pps_df

def recalculate_dynamic_values(available_players_df, remaining_money_pool, total_league_budget):
    """
    Recalculates player values dynamically based on remaining money, VORP pool, and re-tiering.
    """
    """
    Recalculates player values dynamically based on remaining money and VORP pool.
    """
    # Determine the portion of the remaining budget that should be allocated to VORP
    vorp_budget_pct_remaining = remaining_money_pool / total_league_budget
    remaining_vorp_budget = remaining_money_pool * vorp_budget_pct_remaining

    # 1. Re-tier the remaining players
    if len(available_players_df) >= 5:
        available_players_df['Tier'] = pd.qcut(available_players_df['PPS'], 5, labels=False, duplicates='drop') + 1
    else: # Not enough players to form 5 tiers
        available_players_df['Tier'] = 5

    # 2. Recalculate Tier-Adjusted VORP
    available_players_df['TierWeight'] = available_players_df['Tier'].map(TIER_WEIGHTS)
    available_players_df['TierAdjustedVORP'] = available_players_df['VORP'] * available_players_df['TierWeight']

    # 3. Calculate new dollar per point
    total_remaining_vorp = available_players_df['TierAdjustedVORP'].sum()

    if total_remaining_vorp > 0:
        dollar_per_vorp = remaining_vorp_budget / total_remaining_vorp
    else:
        dollar_per_vorp = 0

    # 4. Calculate the new adjusted value
    available_players_df['AdjValue'] = (available_players_df['TierAdjustedVORP'] * dollar_per_vorp) + 1
    available_players_df['AdjValue'] = available_players_df['AdjValue'].round(2)
    
    return available_players_df
