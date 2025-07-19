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

def _get_model_col_name(model_name, prefix):
    """Creates a clean column name from a model name."""
    # Remove special characters and spaces, then convert to CamelCase
    clean_name = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
    return f"{prefix}{clean_name}"

def calculate_initial_values(pps_df, num_teams, roster_spots_per_team, budget_per_team, base_value_models, vorp_budget_pct=0.97, market_value_weight=0.5):
    """
    Calculates initial auction values for one or more Base Value models.
    """
    # Add this line at the very beginning to ensure correctness
    pps_df = pps_df.sort_values(by='PPS', ascending=False).reset_index(drop=True)

    # 1. Define Replacement Player
    total_rostered_players = num_teams * roster_spots_per_team
    replacement_player_index = total_rostered_players

    if len(pps_df) <= replacement_player_index:
        replacement_level_pps = pps_df['PPS'].min() if not pps_df.empty else 0
    else:
        replacement_level_pps = pps_df.iloc[replacement_player_index]['PPS']

    # Assign Tiers and calculate VORP
    pps_df = _assign_percentile_tiers(pps_df)
    pps_df['VORP'] = pps_df['PPS'] - replacement_level_pps
    pps_df['VORP'] = pps_df['VORP'].clip(lower=0)

    # 2. Calculate core VORP-based value component
    pps_df['TierWeight'] = pps_df['Tier'].map(TIER_WEIGHTS).fillna(1.0)
    pps_df['TierAdjustedVORP'] = pps_df['VORP'] * pps_df['TierWeight']
    total_vorp = pps_df['TierAdjustedVORP'].sum()
    total_vorp_budget = (num_teams * budget_per_team) * vorp_budget_pct
    pps_df['VORPValue'] = (pps_df['TierAdjustedVORP'] / total_vorp) * total_vorp_budget if total_vorp > 0 else 0

    # 3. Calculate Base Value for each selected model
    for i, model in enumerate(base_value_models):
        col_name = _get_model_col_name(model, 'BV_')
        temp_value = pd.Series(index=pps_df.index, dtype=float)

        if model == "Risk-Adjusted VORP":
            if 'AvgGP_Percentage' in pps_df.columns:
                risk_factor = pps_df['AvgGP_Percentage'].fillna(0.75)
                temp_value = pps_df['VORPValue'] * risk_factor
            else:
                temp_value = pps_df['VORPValue'] # Fallback

        elif model == "Expert Consensus Value (ECV)":
            if 'ECV' in pps_df.columns:
                ecv = pps_df['ECV'].fillna(pps_df['VORPValue'])
                temp_value = (pps_df['VORPValue'] * 0.5) + (ecv * 0.5)
            else:
                market_val = pps_df['MarketValue'].fillna(pps_df['VORPValue'])
                temp_value = (pps_df['VORPValue'] * 0.5) + (market_val * 0.5) # Fallback

        elif model == "Pure VORP":
            temp_value = pps_df['VORPValue']

        elif model == "Pure Market Value":
            temp_value = pps_df['MarketValue']
            temp_value.loc[pps_df['VORP'] > 0] = temp_value.loc[pps_df['VORP'] > 0].fillna(1)
            temp_value.fillna(0, inplace=True)

        else:  # Blended (VORP + Market) - Default
            market_val = pps_df['MarketValue'].fillna(pps_df['VORPValue'])
            vorp_weight = 1 - market_value_weight
            temp_value = (pps_df['VORPValue'] * vorp_weight) + (market_val * market_value_weight)

        # Post-processing for each calculated value
        temp_value.loc[pps_df['VORP'] > 0] = temp_value.loc[pps_df['VORP'] > 0].clip(lower=1)
        pps_df[col_name] = temp_value.round()

        # The first model in the list sets the primary 'BaseValue' for sorting and default display
        if i == 0:
            pps_df['BaseValue'] = pps_df[col_name]

    # Store initial counts for scarcity calculations
    initial_tier_counts = pps_df['Tier'].value_counts().to_dict()
    initial_pos_counts = pps_df['Position'].value_counts().to_dict()

    return pps_df, initial_tier_counts, initial_pos_counts

def recalculate_dynamic_values(available_players_df, remaining_money_pool, total_league_money, scarcity_models, initial_tier_counts, initial_pos_counts, roster_composition=None, num_teams=None):
    """
    Recalculates player values dynamically for one or more scarcity models.
    """
    if available_players_df.empty:
        return available_players_df

    # 1. Calculate Preliminary Adjusted Value (shared by all models)
    available_players_df = _assign_percentile_tiers(available_players_df)
    available_players_df['TierWeight'] = available_players_df['Tier'].map(TIER_WEIGHTS).fillna(1.0)
    available_players_df['TierAdjustedVORP'] = available_players_df['VORP'] * available_players_df['TierWeight']
    total_remaining_vorp = available_players_df['TierAdjustedVORP'].sum()
    prelim_adj_value = (available_players_df['TierAdjustedVORP'] / total_remaining_vorp) * remaining_money_pool if total_remaining_vorp > 0 else 0

    # 2. Calculate Adjusted Value for each selected scarcity model
    for i, model in enumerate(scarcity_models):
        col_name = _get_model_col_name(model, 'AV_')
        scarcity_premium = pd.Series(1.0, index=available_players_df.index) # Default to 1.0 (no premium)

        # Determine which model logic to apply
        effective_model = model
        if model == "Roster Slot Demand":
            effective_model = "Position Scarcity" if roster_composition and num_teams else "Tier Scarcity"
        elif model == "Opponent Budget Targeting":
            effective_model = "Tier Scarcity" # Placeholder

        # Calculate scarcity premium based on the effective model
        if effective_model == "Contrarian Fade":
            current_tier_counts = available_players_df['Tier'].value_counts()
            initial_tier_counts_s = pd.Series(initial_tier_counts)
            ratio = (initial_tier_counts_s / current_tier_counts).replace([np.inf, -np.inf], 1).fillna(1)
            scarcity_factors = (1 / np.sqrt(ratio)).clip(lower=0.75) # Cap discount
            scarcity_premium = available_players_df['Tier'].map(scarcity_factors).fillna(1)

        elif effective_model == "Tier Scarcity":
            current_tier_counts = available_players_df['Tier'].value_counts()
            initial_tier_counts_s = pd.Series(initial_tier_counts)
            ratio = (initial_tier_counts_s / current_tier_counts).replace([np.inf, -np.inf], 1).fillna(1)
            scarcity_factors = np.sqrt(ratio).clip(upper=1.25) # Cap premium
            scarcity_premium = available_players_df['Tier'].map(scarcity_factors).fillna(1)

        elif effective_model == "Position Scarcity":
            current_pos_counts = available_players_df['Position'].value_counts()
            initial_pos_counts_s = pd.Series(initial_pos_counts)
            ratio = (initial_pos_counts_s / current_pos_counts).replace([np.inf, -np.inf], 1).fillna(1)
            scarcity_factors = np.sqrt(ratio).clip(upper=1.25) # Cap premium
            scarcity_premium = available_players_df['Position'].map(scarcity_factors).fillna(1)

        # Apply premium and post-process
        final_adj_value = prelim_adj_value * scarcity_premium
        final_adj_value.loc[available_players_df['VORP'] > 0] = final_adj_value.loc[available_players_df['VORP'] > 0].clip(lower=1)
        available_players_df[col_name] = final_adj_value.round()

        # The first model sets the primary 'AdjValue' for core logic
        if i == 0:
            available_players_df['AdjValue'] = available_players_df[col_name]

    return available_players_df
