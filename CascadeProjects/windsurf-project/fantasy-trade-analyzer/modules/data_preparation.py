import pandas as pd
import numpy as np

# --- Constants and Configuration ---
# --- Player Power Score (PPS) Configuration ---
# Trend-Weighted FP/G weights. S1 is the oldest season, S4 is the most recent.
TREND_WEIGHTS = {'S1': 0.15, 'S2': 0.20, 'S3': 0.30, 'S4': 0.35}

# File paths
FP_PER_GAME_FILE = 'data/PlayerFPperGameOverYears.csv'
GP_FILE = 'data/PlayerGPOverYears.csv'
PLAYER_IDS_FILE = 'data/AllPlayerIds.csv'
OUTPUT_FILE = 'data/player_projections.csv'

def get_player_position(player_name, ids_df):
    """Fetches the position for a given player from the IDs dataframe."""
    # In a real scenario, we would merge with a file that has player positions.
    return 'Flx' # Defaulting to Flex position

def generate_pps_projections(games_in_season=82):
    """
    Loads historical player data to calculate the Player Power Score (PPS) and 
    saves the result to a CSV file for the auction tool.

    Args:
        games_in_season (int): The number of games in a standard season.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Load the datasets
        fp_df = pd.read_csv(FP_PER_GAME_FILE)
        gp_df = pd.read_csv(GP_FILE)
        player_ids_df = pd.read_csv(PLAYER_IDS_FILE)

        # --- Data Merging and Cleaning ---
        fp_df.rename(columns={'player': 'Player'}, inplace=True)
        gp_df.rename(columns={'player': 'Player'}, inplace=True)
        fp_df['Player'] = fp_df['Player'].str.strip()
        gp_df['Player'] = gp_df['Player'].str.strip()

        merged_df = pd.merge(fp_df, gp_df, on='Player', how='left')

        # --- Data Type Conversion ---
        for season in TREND_WEIGHTS.keys():
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
            for season, weight in TREND_WEIGHTS.items():
                fp_col = f'{season} FP/G'
                if fp_col in merged_df.columns and pd.notna(row[fp_col]):
                    weighted_fp_sum += row[fp_col] * weight
                    total_weight += weight
            if total_weight > 0:
                merged_df.at[index, 'TrendWeightedFPG'] = weighted_fp_sum / total_weight

        # 2. Calculate Risk-Adjusted Availability (Avg GP %)
        gp_cols = [f'{s} GP' for s in TREND_WEIGHTS.keys() if f'{s} GP' in merged_df.columns]
        merged_df['TotalGamesPlayed'] = merged_df[gp_cols].sum(axis=1)
        merged_df['NumSeasonsPlayed'] = merged_df[gp_cols].notna().sum(axis=1)
        # Avoid division by zero for players with no games played data
        merged_df['AvgGP_Percentage'] = np.where(merged_df['NumSeasonsPlayed'] > 0, (merged_df['TotalGamesPlayed'] / (merged_df['NumSeasonsPlayed'] * games_in_season)), 0)
        merged_df['AvgGP_Percentage'] = merged_df['AvgGP_Percentage'].clip(0, 1) # Cap at 100%

        # 3. Calculate Final Player Power Score (PPS)
        merged_df['PPS'] = merged_df['TrendWeightedFPG'] * merged_df['AvgGP_Percentage']

        # --- Data Cleaning and Output ---
        merged_df['Position'] = merged_df['Player'].apply(lambda x: get_player_position(x, player_ids_df))
        merged_df.rename(columns={'Player': 'PlayerName'}, inplace=True)

        output_cols = ['PlayerName', 'Position', 'PPS', 'TrendWeightedFPG', 'AvgGP_Percentage']
        output_df = merged_df[output_cols].copy()
        output_df = output_df[output_df['PPS'] > 0].sort_values(by='PPS', ascending=False).reset_index(drop=True)
        
        output_df.to_csv(OUTPUT_FILE, index=False)

        print(f"Successfully created projection file at: {OUTPUT_FILE}")
        print(f"Total players with projections: {len(output_df)}")
        print("\nTop 5 players by PPS:")
        print(output_df.head())
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
