import pandas as pd
import numpy as np

# Tier weights for adjusting VORP. Tier 1 is the best.
TIER_WEIGHTS = {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.8}

def _assign_percentile_tiers(df):
    """Assigns tiers based on PPS percentiles."""
    if df.empty or 'PPS' not in df.columns:
        return df

    pps_series = df['PPS']
    # Define percentile thresholds for tiers
    p98 = pps_series.quantile(0.98)
    p90 = pps_series.quantile(0.90)
    p75 = pps_series.quantile(0.75)
    p50 = pps_series.quantile(0.50)
    
    conditions = [
        (df['PPS'] >= p98),
        (df['PPS'] >= p90),
        (df['PPS'] >= p75),
        (df['PPS'] >= p50)
    ]
    choices = [1, 2, 3, 4]
    df['Tier'] = np.select(conditions, choices, default=5)
    return df

def calculate_initial_values(pps_df, num_teams, roster_spots_per_team, budget_per_team, base_value_model, vorp_budget_pct=0.97, market_value_weight=0.5):
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

    # Assign Tiers based on percentiles
    pps_df = _assign_percentile_tiers(pps_df)

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
    # 4. Calculate VORP-based Value Component
    total_vorp = pps_df['TierAdjustedVORP'].sum()
    total_vorp_budget = (num_teams * budget_per_team) * vorp_budget_pct
    
    if total_vorp > 0:
        pps_df['VORPValue'] = (pps_df['TierAdjustedVORP'] / total_vorp) * total_vorp_budget
    else:
        pps_df['VORPValue'] = 0

    # 5. Calculate Base Value based on the selected model
    if base_value_model == "Pure VORP":
        pps_df['BaseValue'] = pps_df['VORPValue']
    elif base_value_model == "Pure Market Value":
        # For pure market value, we use the historical data, falling back to a minimum of $1 for players with VORP.
        pps_df['BaseValue'] = pps_df['MarketValue']
        pps_df.loc[pps_df['VORP'] > 0, 'BaseValue'] = pps_df.loc[pps_df['VORP'] > 0, 'BaseValue'].fillna(1)
        pps_df['BaseValue'].fillna(0, inplace=True) # Players with no VORP and no market value are $0
    else: # Blended (Default)
        pps_df['MarketValue'].fillna(pps_df['VORPValue'], inplace=True)
        vorp_weight = 1 - market_value_weight
        pps_df['BaseValue'] = (pps_df['VORPValue'] * vorp_weight) + (pps_df['MarketValue'] * market_value_weight)

    # Set a floor value of $1 for any player with positive VORP
    pps_df.loc[pps_df['VORP'] > 0, 'BaseValue'] = pps_df.loc[pps_df['VORP'] > 0, 'BaseValue'].clip(lower=1)
    pps_df['BaseValue'] = pps_df['BaseValue'].round()

    # Store initial counts for scarcity calculations
    initial_tier_counts = pps_df['Tier'].value_counts().to_dict()
    initial_pos_counts = pps_df['Position'].value_counts().to_dict()

    return pps_df, initial_tier_counts, initial_pos_counts

def recalculate_dynamic_values(available_players_df, remaining_money_pool, total_league_money, scarcity_model, initial_tier_counts, initial_pos_counts):
    """
    Recalculates player values dynamically based on remaining money, VORP pool, and re-tiering.
    """
    """
    Recalculates player values dynamically based on remaining money and VORP pool.
    """
    # Determine the portion of the remaining budget that should be allocated to VORP
    vorp_budget_pct_remaining = remaining_money_pool / total_league_money
    remaining_vorp_budget = remaining_money_pool * vorp_budget_pct_remaining

    # 1. Re-tier the remaining players (Tier 1 is best)
    available_players_df = _assign_percentile_tiers(available_players_df)

    # 2. Recalculate Tier-Adjusted VORP
    available_players_df['TierWeight'] = available_players_df['Tier'].map(TIER_WEIGHTS)
    available_players_df['TierAdjustedVORP'] = available_players_df['VORP'] * available_players_df['TierWeight']

    # 3. Recalculate Adjusted Value
    total_remaining_vorp = available_players_df['TierAdjustedVORP'].sum()
    if total_remaining_vorp > 0:
        available_players_df['AdjValue'] = (available_players_df['TierAdjustedVORP'] / total_remaining_vorp) * remaining_vorp_budget
    else:
        available_players_df['AdjValue'] = 0

    # 4. Apply Scarcity Model based on user selection
    max_premium = 1.25 # Cap premium at 25%

    if scarcity_model == "Tier Scarcity":
        current_tier_counts = available_players_df['Tier'].value_counts().to_dict()
        scarcity_factors = {}
        for tier, initial_count in initial_tier_counts.items():
            current_count = current_tier_counts.get(tier, 0)
            if current_count > 0 and initial_count > 0:
                scarcity_factors[tier] = np.sqrt(initial_count / current_count)
            else:
                scarcity_factors[tier] = 1
        
        available_players_df['ScarcityPremium'] = available_players_df['Tier'].map(scarcity_factors).fillna(1)

    elif scarcity_model == "Position Scarcity":
        current_pos_counts = available_players_df['Position'].value_counts().to_dict()
        scarcity_factors = {}
        for pos, initial_count in initial_pos_counts.items():
            current_count = current_pos_counts.get(pos, 0)
            if current_count > 0 and initial_count > 0:
                scarcity_factors[pos] = np.sqrt(initial_count / current_count)
            else:
                scarcity_factors[pos] = 1

        available_players_df['ScarcityPremium'] = available_players_df['Position'].map(scarcity_factors).fillna(1)
    
    else: # "None"
        available_players_df['ScarcityPremium'] = 1

    available_players_df['ScarcityPremium'] = available_players_df['ScarcityPremium'].clip(upper=max_premium)
    available_players_df['AdjValue'] *= available_players_df['ScarcityPremium']

    available_players_df.loc[available_players_df['VORP'] > 0, 'AdjValue'] = available_players_df.loc[available_players_df['VORP'] > 0, 'AdjValue'].clip(lower=1)
    available_players_df['AdjValue'] = available_players_df['AdjValue'].round()

    return available_players_df
