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
    "Replacement-Aware Positional Scarcity",
    "Tier-Based Scarcity",
    "Combined Scarcity (Positional + Tier)",
    "Contrarian Fade",
    "Roster Slot Demand",
    "Opponent Budget Targeting",
    # New optional multipliers (selectable)
    "Phase-Aware Multiplier",
    "Flexibility Bonus",
    "Opportunity Cost Discount"
]


# Tier weights for adjusting VORP. Tier 1 is the best.
TIER_WEIGHTS = {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.8}

# --- Early-Career Model V2 constants ---
EC_ELITE_FPG_THRESHOLD = 85.0  # Elite early performance threshold for Tier 1 Upside
EC_TREND_DELTA_MIN = 5.0       # Minimum FP/G increase to count as a clear positive trend

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

def _parse_positions(pos_str):
    """Return a list of positions from a string like 'G/F' or 'C,Flx'."""
    try:
        return [p.strip() for p in str(pos_str).replace(',', '/').split('/') if p and p.strip()]
    except Exception:
        return [str(pos_str)] if pos_str else []

def calculate_initial_values(pps_df, num_teams, budget_per_team, roster_composition, base_value_models, tier_cutoffs=None, injured_players=None, vorp_budget_pct=0.97, market_value_weight=0.5, ec_settings=None):
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

    # Defer injury adjustments until after Early-Career projection to avoid over-penalizing young players
    deferred_injuries = injured_players or {}

    # --- Early-Career Model V2: adjust PPS for young players (SeasonsPlayed <= 3) ---
    try:
        # Resolve EC thresholds/multipliers (caller can override)
        ec_elite_thr = None
        ec_trend_delta = None
        ec_mults = None
        if isinstance(ec_settings, dict):
            ec_elite_thr = ec_settings.get('elite_fpg_threshold', None)
            ec_trend_delta = ec_settings.get('trend_delta_min', None)
            # Expect percentages as 15,10,5 or multipliers like 1.15; normalize
            m1 = ec_settings.get('tier1_pct', None)
            m2 = ec_settings.get('tier2_pct', None)
            m3 = ec_settings.get('tier3_pct', None)
            if m1 is not None and m2 is not None and m3 is not None:
                # If values > 1.5, assume percents
                if max(m1, m2, m3) > 1.5:
                    ec_mults = {1: 1.0 + (float(m1) / 100.0), 2: 1.0 + (float(m2) / 100.0), 3: 1.0 + (float(m3) / 100.0)}
                else:
                    ec_mults = {1: float(m1), 2: float(m2), 3: float(m3)}

        ec_elite_thr = float(ec_elite_thr) if ec_elite_thr is not None else EC_ELITE_FPG_THRESHOLD
        ec_trend_delta = float(ec_trend_delta) if ec_trend_delta is not None else EC_TREND_DELTA_MIN
        if ec_mults is None:
            ec_mults = {1: 1.15, 2: 1.10, 3: 1.05}
        # Determine seasons played using available FP/G history (S1..S4 with S4 most recent)
        fp_cols = [c for c in [f'S{i}_FP/G' for i in range(1, 5)] if c in pps_df.columns]
        if fp_cols:
            seasons_played = pps_df[fp_cols].apply(lambda r: int(np.sum(pd.to_numeric(r, errors='coerce').fillna(0) > 0)), axis=1)
        else:
            seasons_played = pd.Series(0, index=pps_df.index)
        pps_df['SeasonsPlayed'] = seasons_played

        # Optional: load draft pedigree if present
        draft_info = None
        try:
            from pathlib import Path as _Path
            _data_path = _Path(__file__).resolve().parent.parent / "data"
            draft_path = _data_path / "DraftPedigree.csv"
            if draft_path.exists():
                ddf = pd.read_csv(draft_path)
                # Expect columns: PlayerName, DraftPickOverall, DraftRound
                if 'Player' in ddf.columns and 'PlayerName' not in ddf.columns:
                    ddf.rename(columns={'Player': 'PlayerName'}, inplace=True)
                draft_info = ddf[['PlayerName', 'DraftPickOverall', 'DraftRound']].copy()
        except Exception:
            draft_info = None

        # Helper to compute upside tier (1=15%, 2=10%, 3=5%)
        def _upside_tier(row):
            s4 = pd.to_numeric(row.get('S4_FP/G', np.nan), errors='coerce')
            s3 = pd.to_numeric(row.get('S3_FP/G', np.nan), errors='coerce')
            pick_overall = None
            draft_round = None
            if draft_info is not None:
                try:
                    rec = draft_info.loc[draft_info['PlayerName'] == row['PlayerName']]
                    if not rec.empty:
                        pick_overall = pd.to_numeric(rec.iloc[0].get('DraftPickOverall', np.nan), errors='coerce')
                        draft_round = pd.to_numeric(rec.iloc[0].get('DraftRound', np.nan), errors='coerce')
                except Exception:
                    pass
            # Tier 1: Top 10 pick and elite early performance
            if (pick_overall is not None and not np.isnan(pick_overall) and pick_overall <= 10) and (pd.notna(s4) and s4 >= ec_elite_thr):
                return 1
            # Tier 2: First-round pick and positive FP/G trend
            trend_ok = (pd.notna(s4) and pd.notna(s3) and (s4 - s3) >= ec_trend_delta)
            first_round = (draft_round is not None and not np.isnan(draft_round) and int(draft_round) == 1)
            if first_round and trend_ok:
                return 2
            # Otherwise Tier 3
            return 3

        # Compute projected FP/G and apply upside for SeasonsPlayed <= 3
        is_young = pps_df['SeasonsPlayed'] <= 3
        if is_young.any():
            # Recent-weighted projection
            s4 = pd.to_numeric(pps_df.get('S4_FP/G', np.nan), errors='coerce') if 'S4_FP/G' in pps_df.columns else pd.Series(np.nan, index=pps_df.index)
            s3 = pd.to_numeric(pps_df.get('S3_FP/G', np.nan), errors='coerce') if 'S3_FP/G' in pps_df.columns else pd.Series(np.nan, index=pps_df.index)
            proj = pd.Series(np.nan, index=pps_df.index)
            # Entering Year 2: use S4
            proj.loc[pps_df['SeasonsPlayed'] == 1] = s4.loc[pps_df['SeasonsPlayed'] == 1]
            # Entering Year 3: 0.8*S4 + 0.2*S3 (fallbacks to S4 if S3 missing)
            idx_y3 = pps_df['SeasonsPlayed'] == 2
            proj.loc[idx_y3] = (s4.loc[idx_y3].fillna(0) * 0.8) + (s3.loc[idx_y3].fillna(0) * 0.2)
            # SeasonsPlayed == 3: still treat as young with same weighting
            idx_y4 = pps_df['SeasonsPlayed'] == 3
            proj.loc[idx_y4] = (s4.loc[idx_y4].fillna(0) * 0.8) + (s3.loc[idx_y4].fillna(0) * 0.2)

            # Apply upside modifier
            upside_tiers = pps_df.apply(_upside_tier, axis=1)
            mult = upside_tiers.map(ec_mults).fillna(ec_mults.get(3, 1.05))
            boosted = (proj * mult).fillna(proj)

            # Replace PPS for young players where projection is available
            use_idx = is_young & proj.notna()
            pps_df.loc[use_idx, 'PPS'] = boosted.loc[use_idx]
    except Exception:
        # Fail-safe: if anything goes wrong, do not block execution
        pass

    # Apply injury adjustments now (post EC), using gentler scaling for young players
    if deferred_injuries:
        try:
            # Ensure SeasonsPlayed exists (compute if missing)
            if 'SeasonsPlayed' not in pps_df.columns:
                fp_cols2 = [c for c in [f'S{i}_FP/G' for i in range(1, 5)] if c in pps_df.columns]
                if fp_cols2:
                    pps_df['SeasonsPlayed'] = pps_df[fp_cols2].apply(lambda r: int(np.sum(pd.to_numeric(r, errors='coerce').fillna(0) > 0)), axis=1)
                else:
                    pps_df['SeasonsPlayed'] = 0

            for player, status in deferred_injuries.items():
                if player not in pps_df['PlayerName'].values:
                    continue
                idx = pps_df['PlayerName'] == player
                is_young = (pps_df.loc[idx, 'SeasonsPlayed'] <= 3).bool()
                if is_young:
                    # Gentler for young players
                    if status == "Full Season":
                        pps_df.loc[idx, 'PPS'] = 0
                    elif status == "3/4 Season":
                        pps_df.loc[idx, 'PPS'] *= 0.40
                    elif status == "Half Season":
                        pps_df.loc[idx, 'PPS'] *= 0.70
                    elif status == "1/4 Season":
                        pps_df.loc[idx, 'PPS'] *= 0.85
                else:
                    # Veteran (original scaling)
                    if status == "Full Season":
                        pps_df.loc[idx, 'PPS'] = 0
                    elif status == "3/4 Season":
                        pps_df.loc[idx, 'PPS'] *= 0.25
                    elif status == "Half Season":
                        pps_df.loc[idx, 'PPS'] *= 0.50
                    elif status == "1/4 Season":
                        pps_df.loc[idx, 'PPS'] *= 0.75
        except Exception:
            pass

    # Use the provided parameter; do not override it with a constant
    # vorp_budget_pct represents fraction of total budget allocated to VORP-based values
    vorp_budget_pct = vorp_budget_pct
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

    # 2. Calculate Base Value for each selected model
    for model in base_value_models:
        col_name = _get_model_col_name(model, 'BV_')
        if model == "Pure VORP":
            final_df[col_name] = final_df['VORPValue']
        elif model == "Pure Market Value":
            # Ensure MarketValue exists; default to 0 if missing
            if 'MarketValue' not in final_df.columns:
                final_df['MarketValue'] = 0
            final_df[col_name] = final_df['MarketValue']
        elif model == "Blended (VORP + Market)":
            if 'MarketValue' not in final_df.columns:
                final_df['MarketValue'] = 0
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

def recalculate_dynamic_values(available_players_df, remaining_money_pool, total_league_money, base_value_models, scarcity_models, initial_tier_counts, initial_pos_counts, teams_data, tier_cutoffs=None, roster_composition=None, num_teams=None):
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

    # --- Global dynamic multipliers (independent of selected scarcity model) ---
    # Phase estimation from budgets (0 early -> 1 late)
    try:
        total_initial_budget = sum(d['budget'] + sum(p.get('Price', 0) for p in d.get('players', [])) for d in teams_data.values())
        budget_depletion_ratio = 1 - (sum(d['budget'] for d in teams_data.values()) / total_initial_budget) if total_initial_budget > 0 else 0.0
        budget_depletion_ratio = float(np.clip(budget_depletion_ratio, 0.0, 1.0))
    except Exception:
        budget_depletion_ratio = 0.0

    # Phase multiplier favors T1-2 early, T3-4 late (gentle adjustments)
    phase_mult = pd.Series(1.0, index=available_players_df.index)
    try:
        is_t12 = available_players_df['Tier'].isin([1, 2])
        is_t34 = available_players_df['Tier'].isin([3, 4])
        phase_mult[is_t12] = 1.0 + (0.10 * (1.0 - budget_depletion_ratio))  # up to +10% very early
        phase_mult[is_t34] = 1.0 + (0.08 * budget_depletion_ratio)          # up to +8% very late
    except Exception:
        pass

    # Flexibility bonus: more eligible positions worth slightly more, stronger late
    try:
        elig_counts = available_players_df['Position'].apply(lambda s: max(1, len(_parse_positions(s))))
        # base bonus per extra eligibility grows from 4% -> 10% as draft progresses
        base_bonus = 0.04 + 0.06 * budget_depletion_ratio
        flex_mult = 1.0 + ((elig_counts - 1).clip(lower=0) * base_bonus).clip(upper=0.10)
        flex_mult = pd.Series(flex_mult.values, index=available_players_df.index)
    except Exception:
        flex_mult = pd.Series(1.0, index=available_players_df.index)

    # Replacement-aware demand: compute open slots across teams, including Flex distribution
    pos_demand_weight = {}
    try:
        req = {k: int(v) for k, v in (roster_composition or {}).items()}
        flex_slots = req.get('Flx', 0)
        # Count filled per team
        demand_counts = {}
        for pos in req.keys():
            if pos == 'Flx':
                continue
            demand_counts[pos] = 0
        for team_name, data in teams_data.items():
            filled = pd.Series([p.get('Position', '') for p in data.get('players', [])]).value_counts().to_dict()
            for pos in demand_counts.keys():
                demand_counts[pos] += max(0, req.get(pos, 0) - int(filled.get(pos, 0)))
        # Distribute flex demand proportionally across core positions present in available pool
        base_total = sum(demand_counts.values())
        if flex_slots > 0:
            # Detect which positions exist in the pool
            pool_pos = set()
            for pos_str in available_players_df['Position'].unique():
                for p in _parse_positions(pos_str):
                    if p != 'Flx':
                        pool_pos.add(p)
            spread_keys = [p for p in demand_counts.keys() if p in pool_pos]
            if spread_keys:
                add_each = float(flex_slots) / float(len(spread_keys))
                for p in spread_keys:
                    demand_counts[p] += add_each
        total_demand = float(sum(demand_counts.values()))
        if total_demand > 0:
            pos_demand_weight = {pos: demand_counts[pos] / total_demand for pos in demand_counts}
    except Exception:
        pos_demand_weight = {}

    # Opportunity cost discount: lots of substitutes -> small discount (up to 10%)
    try:
        # Precompute counts by (pos, tier)
        counts_by = {}
        for _, row in available_players_df.iterrows():
            tiers = int(row.get('Tier', 5))
            for p in _parse_positions(row.get('Position', '')):
                key = (p, tiers)
                counts_by[key] = counts_by.get(key, 0) + 1
        # Build per-player discount
        oc_mult = []
        for _, row in available_players_df.iterrows():
            pos_list = _parse_positions(row.get('Position', ''))
            t = int(row.get('Tier', 5))
            # Count same tier and neighbors across eligible positions
            subs = 0
            for p in pos_list:
                subs += counts_by.get((p, t), 0)
                subs += counts_by.get((p, max(1, t-1)), 0)
                subs += counts_by.get((p, min(5, t+1)), 0)
            # Higher substitutes -> larger discount up to 10%
            disc = min(0.10, 0.01 * subs)
            oc_mult.append(1.0 - disc)
        oc_mult = pd.Series(oc_mult, index=available_players_df.index)
    except Exception:
        oc_mult = pd.Series(1.0, index=available_players_df.index)

    # Note: We do NOT apply phase/flex/OC globally. They are selectable models below.

    # 2. Calculate Adjusted Value for each selected scarcity model
    for i, model in enumerate(scarcity_models):
        col_name = _get_model_col_name(model, 'AV_')
        scarcity_premium = pd.Series(1.0, index=available_players_df.index) # Default to 1.0 (no premium)

        # --- Scarcity Premium Calculation --- 
        # Base premium is 1.0 (no change). Models adjust this multiplier.
        scarcity_premium = pd.Series(1.0, index=available_players_df.index)

        # --- Tier-Based Scarcity --- 
        if model == "Tier-Based Scarcity":
            initial_tier_counts_s = pd.Series(initial_tier_counts)
            current_tier_counts = available_players_df['Tier'].value_counts().reindex(initial_tier_counts_s.index, fill_value=0)
            # How much of the tier has been drafted (0 to 1)
            tier_drafted_ratio = 1 - (current_tier_counts / initial_tier_counts_s)
            # Premium increases as more of a tier is drafted. Max premium of 1.25
            tier_premium = 1 + (tier_drafted_ratio * 0.25)
            scarcity_premium = available_players_df['Tier'].map(tier_premium).fillna(1.0)

        # --- Positional Scarcity ---
        elif model == "Positional Scarcity":
            # Replacement-aware: use demand weights and multi-eligibility averaging
            try:
                # Classic positional scarcity (fallback to counts if weights missing)
                initial_pos_counts_s = pd.Series(initial_pos_counts)
                current_pos_counts = available_players_df['Position'].value_counts().reindex(initial_pos_counts_s.index, fill_value=0)
                pos_drafted_ratio = 1 - (current_pos_counts / initial_pos_counts_s)
                pos_premium = 1 + (pos_drafted_ratio * 0.30)
                scarcity_premium = available_players_df['Position'].map(pos_premium).fillna(1.0)
            except Exception:
                scarcity_premium = pd.Series(1.0, index=available_players_df.index)

        # --- Replacement-Aware Positional Scarcity ---
        elif model == "Replacement-Aware Positional Scarcity":
            try:
                if pos_demand_weight:
                    # Map each row to the average of its eligible positions' weights
                    def _avg_weight(pos_str):
                        ps = _parse_positions(pos_str)
                        if not ps:
                            return 1.0
                        vals = [pos_demand_weight.get(p, 0) for p in ps if p != 'Flx']
                        base = np.mean(vals) if vals else 0
                        # Convert 0..1 weight to 1..1.35 premium
                        return 1.0 + (base * 0.35)
                    scarcity_premium = available_players_df['Position'].apply(_avg_weight)
                else:
                    # Fallback to initial vs current counts if weights unavailable
                    initial_pos_counts_s = pd.Series(initial_pos_counts)
                    current_pos_counts = available_players_df['Position'].value_counts().reindex(initial_pos_counts_s.index, fill_value=0)
                    pos_drafted_ratio = 1 - (current_pos_counts / initial_pos_counts_s)
                    pos_premium = 1 + (pos_drafted_ratio * 0.30)
                    scarcity_premium = available_players_df['Position'].map(pos_premium).fillna(1.0)
            except Exception:
                scarcity_premium = pd.Series(1.0, index=available_players_df.index)

        # --- Combined Scarcity (Positional + Tier) ---
        elif model == "Combined Scarcity (Positional + Tier)":
            initial_tier_counts_s = pd.Series(initial_tier_counts)
            current_tier_counts = available_players_df['Tier'].value_counts().reindex(initial_tier_counts_s.index, fill_value=0)
            tier_drafted_ratio = 1 - (current_tier_counts / initial_tier_counts_s)
            tier_premium = 1 + (tier_drafted_ratio * 0.25)

            initial_pos_counts_s = pd.Series(initial_pos_counts)
            current_pos_counts = available_players_df['Position'].value_counts().reindex(initial_pos_counts_s.index, fill_value=0)
            pos_drafted_ratio = 1 - (current_pos_counts / initial_pos_counts_s)
            pos_premium = 1 + (pos_drafted_ratio * 0.30)

            # Weighted average of the two premiums
            scarcity_premium = (available_players_df['Tier'].map(tier_premium).fillna(1.0) * 0.5) + (available_players_df['Position'].map(pos_premium).fillna(1.0) * 0.5)

        # --- Contrarian Fade ---
        elif model == "Contrarian Fade":
            # This model slightly devalues players from tiers that are still plentiful.
            initial_tier_counts_s = pd.Series(initial_tier_counts)
            current_tier_counts = available_players_df['Tier'].value_counts().reindex(initial_tier_counts_s.index, fill_value=0)
            # Ratio of players remaining in a tier. Closer to 1 means more players left.
            tier_remaining_ratio = (current_tier_counts / initial_tier_counts_s).fillna(1.0)
            # Value is discounted if many players from that tier are left. Max discount of 15%.
            contrarian_premium = 1 - ((tier_remaining_ratio - tier_remaining_ratio.min()) / (tier_remaining_ratio.max() - tier_remaining_ratio.min()) * 0.15)
            scarcity_premium = available_players_df['Tier'].map(contrarian_premium).fillna(1.0)

        # --- Roster Slot Demand ---
        elif model == "Roster Slot Demand":
            # Calculates demand based on total remaining unfilled roster spots for each position.
            demand = {}
            for pos in roster_composition:
                demand[pos] = 0
            for team_name, data in teams_data.items():
                team_roster_comp = pd.Series([p['Position'] for p in data['players']]).value_counts().to_dict()
                for pos, required in roster_composition.items():
                    current_filled = team_roster_comp.get(pos, 0)
                    demand[pos] += max(0, required - current_filled)
            
            total_demand = sum(demand.values())
            if total_demand > 0:
                demand_factor = {pos: (count / total_demand) for pos, count in demand.items()}
                # Premium is higher for positions with higher relative demand. Max premium 1.35
                demand_premium = {pos: 1 + (factor * 0.35) for pos, factor in demand_factor.items()}
                scarcity_premium = available_players_df['Position'].map(demand_premium).fillna(1.0)

        # --- Opponent Budget Targeting ---
        elif model == "Opponent Budget Targeting":
            # Target high-tier players if opponents have money, target mid-tier if they don't.
            avg_opponent_budget = np.mean([d['budget'] for d in teams_data.values()])

            # Define luxury players (Tiers 1-2) and value players (Tiers 3-4)
            is_luxury = available_players_df['Tier'].isin([1, 2])
            is_value = available_players_df['Tier'].isin([3, 4])

            # Boost luxury players early, boost value players late
            luxury_premium = 1 + (0.20 * (1 - budget_depletion_ratio)) # Higher when budgets are full
            value_premium = 1 + (0.15 * budget_depletion_ratio) # Higher when budgets are low

            scarcity_premium = pd.Series(1.0, index=available_players_df.index)
            scarcity_premium[is_luxury] = luxury_premium
            scarcity_premium[is_value] = value_premium

        # --- Phase-Aware Multiplier ---
        elif model == "Phase-Aware Multiplier":
            scarcity_premium = phase_mult

        # --- Flexibility Bonus ---
        elif model == "Flexibility Bonus":
            scarcity_premium = flex_mult

        # --- Opportunity Cost Discount ---
        elif model == "Opportunity Cost Discount":
            scarcity_premium = oc_mult

        # Save scarcity premium fraction (multiplier - 1.0) for UI tooltips
        sp_col = _get_model_col_name(model, 'SP_')
        try:
            available_players_df[sp_col] = (scarcity_premium - 1.0).fillna(0.0)
        except Exception:
            available_players_df[sp_col] = 0.0

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
            # If no models, or only one, there's no variance, so mean is just the base value or 0
            available_players_df['ValueMean'] = available_players_df.get('BaseValue', 0)

    # Final type check to prevent UI errors
    if 'PlayerName' in available_players_df.columns:
        available_players_df['PlayerName'] = available_players_df['PlayerName'].astype(str)

    return available_players_df
