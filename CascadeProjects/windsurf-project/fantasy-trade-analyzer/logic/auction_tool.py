import pandas as pd
import numpy as np

# --- Constants for Value Models ---
BASE_VALUE_MODELS = [
    "No Adjustment",
    "Blended (VORP + Market)",
    "Pure VORP",
    "Pure Market Value",
    "Risk-Adjusted VORP",
    "Expert Consensus Value (ECV)"
]

SCARCITY_MODELS = [
    "No Scarcity Adjustment",
    "Positional Scarcity",
    "Tier-Based Scarcity",
    "Combined Scarcity (Positional + Tier)",
    "Contrarian Fade",
    "Roster Slot Demand",
    "Opponent Budget Targeting"
]


# Tier weights for adjusting VORP. Tier 1 is the best.
TIER_WEIGHTS = {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.8}

def _assign_percentile_tiers(df, tier_cutoffs=None):
    """Assigns tiers based on PPS percentiles, with customizable cutoffs."""
    if df.empty or 'PPS' not in df.columns:
        return df

    if tier_cutoffs is None:
        # Default cutoffs if none are provided
        tier_cutoffs = {
            'Tier 1': 0.98, # Top 2%
            'Tier 2': 0.90, # Next 8%
            'Tier 3': 0.75, # Next 15%
            'Tier 4': 0.50  # Next 25%
        }

    pps_series = df['PPS']
    # Define percentile thresholds for tiers from the provided dictionary
    p_t1 = pps_series.quantile(tier_cutoffs.get('Tier 1', 0.98))
    p_t2 = pps_series.quantile(tier_cutoffs.get('Tier 2', 0.90))
    p_t3 = pps_series.quantile(tier_cutoffs.get('Tier 3', 0.75))
    p_t4 = pps_series.quantile(tier_cutoffs.get('Tier 4', 0.50))
    
    conditions = [
        (df['PPS'] >= p_t1),
        (df['PPS'] >= p_t2),
        (df['PPS'] >= p_t3),
        (df['PPS'] >= p_t4)
    ]
    choices = [1, 2, 3, 4]
    df['Tier'] = np.select(conditions, choices, default=5)
    return df

def _get_model_col_name(model_name, prefix):
    """Creates a clean column name from a model name."""
    # Remove special characters and spaces, then convert to CamelCase
    clean_name = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
    return f"{prefix}{clean_name}"

def calculate_initial_values(pps_df, num_teams, budget_per_team, roster_composition, base_value_models, tier_cutoffs=None, injured_players=None, vorp_budget_pct=0.97, market_value_weight=0.5):
    """
    Calculates the initial Base Value for all players based on the selected valuation models.
    """
    # Load and merge historical data
    try:
        from pathlib import Path
        data_path = Path(__file__).resolve().parent.parent / "data"
        gp_hist_df = pd.read_csv(data_path / "PlayerGPOverYears.csv")
        fpg_hist_df = pd.read_csv(data_path / "PlayerFPperGameOverYears.csv")

        # Align column names for merging
        gp_hist_df.rename(columns={'Player': 'PlayerName'}, inplace=True)
        fpg_hist_df.rename(columns={'Player': 'PlayerName'}, inplace=True)

        # Rename stat columns for clarity, e.g., 'S1 GP' -> 'S1_GP'
        gp_hist_df.rename(columns={f'S{i} GP': f'S{i}_GP' for i in range(1, 13)}, inplace=True)
        fpg_hist_df.rename(columns={f'S{i} FP/G': f'S{i}_FP/G' for i in range(1, 13)}, inplace=True)

        # Select only the necessary historical columns to merge
        gp_cols_to_merge = ['PlayerName'] + [f'S{i}_GP' for i in range(1, 5)]
        fpg_cols_to_merge = ['PlayerName'] + [f'S{i}_FP/G' for i in range(1, 5)]

        # Merge historical data into the main pps_df
        pps_df = pps_df.merge(gp_hist_df[gp_cols_to_merge], on='PlayerName', how='left')
        pps_df = pps_df.merge(fpg_hist_df[fpg_cols_to_merge], on='PlayerName', how='left')

    except FileNotFoundError:
        # If files are not found, historical data will be missing, but the app won't crash
        pass

    # Apply injury adjustments before any calculations
    if injured_players:
        for player, status in injured_players.items():
            if player in pps_df['PlayerName'].values:
                if status == "Full Season":
                    pps_df.loc[pps_df['PlayerName'] == player, 'PPS'] = 0
                elif status == "3/4 Season":
                    pps_df.loc[pps_df['PlayerName'] == player, 'PPS'] *= 0.25
                elif status == "Half Season":
                    pps_df.loc[pps_df['PlayerName'] == player, 'PPS'] *= 0.5
                elif status == "1/4 Season":
                    pps_df.loc[pps_df['PlayerName'] == player, 'PPS'] *= 0.75

    vorp_budget_pct = 0.8  # 80% of the budget is allocated based on VORP
    roster_spots_per_team = sum(roster_composition.values())
    replacement_level_players = int(num_teams * roster_spots_per_team * 0.85) # Top 85% of drafted players

    # 1. Calculate VORP and Tiers
    if replacement_level_players >= len(pps_df):
        replacement_level_pps = pps_df['PPS'].min()
    else:
        replacement_level_pps = pps_df['PPS'].nlargest(replacement_level_players).iloc[-1]

    # Assign Tiers and calculate VORP
    pps_df = _assign_percentile_tiers(pps_df, tier_cutoffs=tier_cutoffs)
    vorp_series = pps_df['PPS'] - replacement_level_pps
    vorp_series = vorp_series.clip(lower=0)

    # Create the final DataFrame, preserving original columns
    final_df = pps_df.copy() # Start with all original columns (including historical stats)
    final_df['VORP'] = vorp_series
    final_df['Tier'] = pps_df['Tier'] # Ensure Tier is carried over

    # 2. Calculate core VORP-based value component
    final_df['TierWeight'] = final_df['Tier'].map(TIER_WEIGHTS).fillna(1.0)
    final_df['TierAdjustedVORP'] = final_df['VORP'] * final_df['TierWeight']
    total_vorp = final_df['TierAdjustedVORP'].sum()
    total_vorp_budget = (num_teams * budget_per_team) * vorp_budget_pct
    final_df['VORPValue'] = (final_df['TierAdjustedVORP'] / total_vorp) * total_vorp_budget if total_vorp > 0 else 0

    # 3. Calculate Base Value for each selected model
    for model in base_value_models:
        col_name = _get_model_col_name(model, 'BV_')
        if model == "Pure VORP":
            final_df[col_name] = final_df['VORPValue']
        elif model == "Pure Market Value":
            final_df[col_name] = final_df['MarketValue']
        elif model == "Blended (VORP + Market)":
            final_df[col_name] = (final_df['VORPValue'] * 0.5) + (final_df['MarketValue'] * 0.5)
        elif model == "Risk-Adjusted VORP":
            # Simple risk model: reduce value by a percentage of games missed
            s4_gp_series = pd.to_numeric(final_df['S4_GP'], errors='coerce').fillna(82) if 'S4_GP' in final_df else pd.Series(82, index=final_df.index)
            games_played_factor = (s4_gp_series / 82) ** 2
            final_df[col_name] = final_df['VORPValue'] * games_played_factor
        elif model == "Expert Consensus Value (ECV)":
            # Placeholder: Blends VORP with a simulated expert rank/value
            final_df[col_name] = (final_df['VORPValue'] * 0.7) + (final_df['MarketValue'] * 0.3)
        elif model == "No Adjustment":
            # This model simply uses the calculated VORP value without any other adjustments.
            final_df[col_name] = final_df['VORPValue']
        else:
            final_df[col_name] = final_df['VORPValue'] # Default to VORP

        # Ensure a minimum value of $1 for any player with positive VORP
        final_df[col_name] = final_df[col_name].round()
        final_df.loc[final_df['VORP'] > 0, col_name] = final_df.loc[final_df['VORP'] > 0, col_name].clip(lower=1)

    # Set 'BaseValue' to the average of the selected models' values
    if base_value_models:
        bv_cols = [_get_model_col_name(model, 'BV_') for model in base_value_models]
        final_df['BaseValue'] = final_df[bv_cols].mean(axis=1).round()
    else:
        final_df['BaseValue'] = 0

    # 4. Calculate initial tier and position counts for scarcity models
    initial_tier_counts = final_df['Tier'].value_counts().to_dict()
    initial_pos_counts = final_df['Position'].value_counts().to_dict()

    return final_df, initial_tier_counts, initial_pos_counts

    # Store initial counts for scarcity calculations
    initial_tier_counts = pps_df['Tier'].value_counts().to_dict()
    initial_pos_counts = pps_df['Position'].value_counts().to_dict()

    return pps_df, initial_tier_counts, initial_pos_counts

def recalculate_dynamic_values(available_players_df, remaining_money_pool, total_league_money, base_value_models, scarcity_models, initial_tier_counts, initial_pos_counts, tier_cutoffs=None, roster_composition=None, num_teams=None):
    """
    Recalculates player values dynamically for one or more scarcity models.
    """
    if available_players_df.empty:
        return available_players_df

    # 1. Calculate Preliminary Adjusted Value (shared by all models)
    available_players_df = _assign_percentile_tiers(available_players_df, tier_cutoffs=tier_cutoffs)
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
            initial_tier_counts_s = pd.Series(initial_tier_counts)
            current_tier_counts = available_players_df['Tier'].value_counts().reindex(initial_tier_counts_s.index, fill_value=0)
            tier_scarcity_factor = (1 - (current_tier_counts / initial_tier_counts_s)).fillna(0)
            available_players_df['TierScarcity'] = available_players_df['Tier'].map(tier_scarcity_factor) * (remaining_money_pool / total_league_money) * available_players_df['BaseValue']

        elif effective_model == "Position Scarcity":
            initial_pos_counts_s = pd.Series(initial_pos_counts)
            current_pos_counts = available_players_df['Position'].value_counts().reindex(initial_pos_counts_s.index, fill_value=0)
            pos_scarcity_factor = (1 - (current_pos_counts / initial_pos_counts_s)).fillna(0)
            available_players_df['PosScarcity'] = available_players_df['Position'].map(pos_scarcity_factor) * (remaining_money_pool / total_league_money) * available_players_df['BaseValue'].clip(upper=1.25) # Cap premium

        # Apply premium and post-process
        final_adj_value = prelim_adj_value * scarcity_premium
        final_adj_value = final_adj_value.round()
        final_adj_value.loc[available_players_df['VORP'] > 0] = final_adj_value.loc[available_players_df['VORP'] > 0].clip(lower=1)
        available_players_df[col_name] = final_adj_value

    # Set 'AdjValue' to the average of the selected models' values
    if scarcity_models:
        av_cols = [_get_model_col_name(model, 'AV_') for model in scarcity_models]
        available_players_df['AdjValue'] = available_players_df[av_cols].mean(axis=1).round()
    else:
        available_players_df['AdjValue'] = available_players_df['BaseValue'] # Default to BaseValue if no scarcity model

    # --- Confidence Score Calculation ---
    base_model_cols = [_get_model_col_name(m, 'BV_') for m in base_value_models]
    scarcity_model_cols = [_get_model_col_name(m, 'AV_') for m in scarcity_models]
    all_model_cols = [col for col in base_model_cols + scarcity_model_cols if col in available_players_df.columns]

    if len(all_model_cols) > 1:
        values_df = available_players_df[all_model_cols]
        available_players_df['ValueMean'] = values_df.mean(axis=1)
        available_players_df['ValueStd'] = values_df.std(axis=1)
        
        available_players_df['Confidence'] = np.where(available_players_df['ValueMean'] > 0, (1 - (available_players_df['ValueStd'] / available_players_df['ValueMean'])) * 100, 100)
        available_players_df['Confidence'] = available_players_df['Confidence'].clip(0, 100) # Ensure confidence is between 0 and 100
    else:
        available_players_df['Confidence'] = 100.0
        if len(all_model_cols) == 1:
            available_players_df['ValueMean'] = available_players_df[all_model_cols[0]]
        else:
            available_players_df['ValueMean'] = available_players_df['BaseValue']

    return available_players_df

