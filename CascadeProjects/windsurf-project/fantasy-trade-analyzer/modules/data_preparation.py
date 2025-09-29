import pandas as pd
import numpy as np
import streamlit as st

# --- Constants and Configuration ---
# --- Player Power Score (PPS) Configuration ---


# File paths
FP_PER_GAME_FILE = 'data/PlayerFPperGameOverYears.csv'
GP_FILE = 'data/PlayerGPOverYears.csv'
HISTORICAL_BIDS_FILE = 'data/PlayerBidPricesOverYears.csv'
# Position data will be sourced from seasonal stats files S1-S4
OUTPUT_FILE = 'data/player_projections.csv'

def generate_pps_projections(games_in_season=82, trend_weights=None, injured_players=None):
    """
    Loads historical player data to calculate the Player Power Score (PPS) and 
    saves the result to a CSV file for the auction tool.

    Args:
        games_in_season (int): The number of games in a standard season.
        trend_weights (dict): A dictionary with season keys ('S1'-'S4') and their float weights.

    Returns:
        bool: True if successful, False otherwise.
    """
    if trend_weights is None:
        # Provide default weights if none are passed (e.g., for standalone execution)
        trend_weights = {'S1': 0.15, 'S2': 0.20, 'S3': 0.30, 'S4': 0.35}
    try:
        # Load the datasets
        fp_df = pd.read_csv(FP_PER_GAME_FILE)
        gp_df = pd.read_csv(GP_FILE)
        bids_df = pd.read_csv(HISTORICAL_BIDS_FILE)

        # --- Calculate Market Value from Historical Bids (Trend-Weighted) ---
        # Map trend_weights (S1..S4) onto bids columns (S1..S12). We ignore 'S1 (Picked Spot)'.
        bid_cols = [col for col in bids_df.columns if col.startswith('S') and 'Picked Spot' not in col]
        bids_df[bid_cols] = bids_df[bid_cols].apply(pd.to_numeric, errors='coerce')

        # Build a weight lookup for S1..S12 using provided trend_weights where available
        weight_lookup = {f'S{i}': float(trend_weights.get(f'S{i}', 0.0)) for i in range(1, 13)}

        def _weighted_market_value(row):
            wsum = 0.0
            vsum = 0.0
            for col in bid_cols:
                w = weight_lookup.get(col, 0.0)
                v = row.get(col, np.nan)
                if pd.notna(v) and w > 0:
                    vsum += v * w
                    wsum += w
            if wsum > 0:
                return vsum / wsum
            # Fallback: simple mean of non-null bids if all weights are zero/missing
            return pd.to_numeric(row[bid_cols], errors='coerce').dropna().mean()

        bids_df['MarketValue'] = bids_df.apply(_weighted_market_value, axis=1).round()
        market_value_df = bids_df[['Player', 'MarketValue']].copy().dropna()
        market_value_df['Player'] = market_value_df['Player'].str.strip()

        # Load and consolidate position data from seasonal stats files
        all_pos_df = pd.DataFrame()
        for i in range(1, 5):
            try:
                season_stats_file = f'data/S{i}Stats.csv'
                season_df = pd.read_csv(season_stats_file)
                season_df.rename(columns={'Player': 'PlayerName'}, inplace=True, errors='ignore')
                season_df['PlayerName'] = season_df['PlayerName'].str.strip()
                pos_data = season_df[['PlayerName', 'Position']].copy()
                all_pos_df = pd.concat([all_pos_df, pos_data])
            except (FileNotFoundError, KeyError):
                # Handle cases where file or 'Position' column might be missing
                continue
        
        # Drop duplicates, keeping the last (most recent season's) entry for each player
        if not all_pos_df.empty:
            all_pos_df.drop_duplicates(subset='PlayerName', keep='last', inplace=True)

        # --- Data Merging and Cleaning ---
        fp_df.rename(columns={'player': 'Player'}, inplace=True)
        gp_df.rename(columns={'player': 'Player'}, inplace=True)
        fp_df['Player'] = fp_df['Player'].str.strip()
        gp_df['Player'] = gp_df['Player'].str.strip()

        # Merge the primary datasets
        merged_df = pd.merge(fp_df, gp_df, on='Player', how='left')
        merged_df = pd.merge(merged_df, market_value_df, on='Player', how='left')
        
        # Merge the consolidated position data
        if not all_pos_df.empty:
            all_pos_df.rename(columns={'PlayerName': 'Player'}, inplace=True)
            merged_df = pd.merge(merged_df, all_pos_df, on='Player', how='left')
        
        # Ensure a 'Position' column exists and fill any missing values
        if 'Position' not in merged_df.columns:
            merged_df['Position'] = 'Flx'
        merged_df['Position'].fillna('Flx', inplace=True)

        # --- Data Type Conversion ---
        for season in trend_weights.keys():
            fp_col = f'{season} FP/G'
            gp_col = f'{season} GP'
            if fp_col in merged_df.columns:
                merged_df[fp_col] = pd.to_numeric(merged_df[fp_col], errors='coerce')
            if gp_col in merged_df.columns:
                merged_df[gp_col] = pd.to_numeric(merged_df[gp_col], errors='coerce')

        # --- Calculate Player Power Score (PPS) ---

        # 1. Calculate Trend-Weighted FP/G
        merged_df['TrendWeightedFPG'] = 0.0
        for index, row in merged_df.iterrows():
            weighted_fp_sum = 0
            total_weight = 0
            for season, weight in trend_weights.items():
                fp_col = f'{season} FP/G'
                if fp_col in merged_df.columns and pd.notna(row[fp_col]):
                    weighted_fp_sum += row[fp_col] * weight
                    total_weight += weight
            if total_weight > 0:
                merged_df.at[index, 'TrendWeightedFPG'] = weighted_fp_sum / total_weight

        # 2. Calculate Risk-Adjusted Availability (Avg GP %)
        gp_cols = [f'{s} GP' for s in trend_weights.keys() if f'{s} GP' in merged_df.columns]
        merged_df['TotalGamesPlayed'] = merged_df[gp_cols].sum(axis=1)
        merged_df['NumSeasonsPlayed'] = merged_df[gp_cols].notna().sum(axis=1)
        # Avoid division by zero for players with no games played data
        merged_df['AvgGP_Percentage'] = np.where(merged_df['NumSeasonsPlayed'] > 0, (merged_df['TotalGamesPlayed'] / (merged_df['NumSeasonsPlayed'] * games_in_season)), 0)
        merged_df['AvgGP_Percentage'] = merged_df['AvgGP_Percentage'].clip(0, 1) # Cap at 100%

        # 3. Calculate Final Player Power Score (PPS)
        merged_df['PPS'] = merged_df['TrendWeightedFPG'] * merged_df['AvgGP_Percentage']

        # 4. Apply Injury Adjustments
        if injured_players:
            for player, status in injured_players.items():
                if status == "Out for Season":
                    merged_df.loc[merged_df['Player'] == player, 'PPS'] = 0
                elif status == "Half Season":
                    merged_df.loc[merged_df['Player'] == player, 'PPS'] *= 0.5

        # --- Final Data Selection and Save ---
        # Select and rename columns for the final output
        final_df = merged_df[['Player', 'Position', 'MarketValue', 'PPS']].copy()
        final_df.rename(columns={'Player': 'PlayerName'}, inplace=True)

        # Sort by PPS descending and save
        final_df.sort_values(by='PPS', ascending=False, inplace=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"Successfully created projection file at: {OUTPUT_FILE}")
        print(f"Total players with projections: {len(final_df)}")
        print("\nTop 5 players by PPS:")
        print(final_df.head())
        return True

    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure all required CSV files are in the 'data' directory.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return False

if __name__ == '__main__':
    # Allows the script to be run standalone for testing or manual generation.
    generate_pps_projections()
