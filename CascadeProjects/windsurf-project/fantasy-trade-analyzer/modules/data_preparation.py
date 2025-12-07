import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from modules.historical_ytd_downloader.logic import DOWNLOAD_DIR as HISTORICAL_YTD_DIR, load_and_compare_seasons
from modules.draft_history import load_draft_history

# --- Constants and Configuration ---
# --- Player Power Score (PPS) Configuration ---

# Get the absolute path to the data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUCTION_DATA_DIR = DATA_DIR / "auction"

# File paths (auction-related CSVs live under data/auction, with legacy fallbacks where needed)
FP_PER_GAME_FILE = AUCTION_DATA_DIR / "PlayerFPperGameOverYears.csv"
GP_FILE = AUCTION_DATA_DIR / "PlayerGPOverYears.csv"
HISTORICAL_BIDS_FILE = AUCTION_DATA_DIR / "PlayerBidPricesOverYears.csv"
# Position data will be sourced from seasonal stats files S1-S4
OUTPUT_FILE = AUCTION_DATA_DIR / "player_projections.csv"

HISTORICAL_SEASON_MAP = {
	"S1": "2021-22",
	"S2": "2022-23",
	"S3": "2023-24",
	"S4": "2024-25",
	"S5": "2025-26",
}


def _infer_historical_league_name():
	"""Infer which league's historical YTD files to use.

	Prefers the league currently loaded into session state, and falls back to
	inspecting files on disk under HISTORICAL_YTD_DIR.
	"""
	try:
		name = st.session_state.get("loaded_league_name")
		if name:
			return str(name)
	except Exception:
		pass

	try:
		if HISTORICAL_YTD_DIR.exists():
			for path in HISTORICAL_YTD_DIR.glob("Fantrax-Players-*-YTD-*.csv"):
				stem_parts = path.stem.split("-")
				if len(stem_parts) >= 4:
					# Fantrax-Players-{league}-YTD-{season}
					return "-".join(stem_parts[2:-2]) if len(stem_parts) > 4 else stem_parts[2]
	except Exception:
		pass

	return None


def _build_fp_gp_from_historical_ytd(season_keys):
	league_name = _infer_historical_league_name()
	if not league_name:
		return None, None

	if not season_keys:
		return None, None

	seasons_to_compare = []
	for key in season_keys:
		label = HISTORICAL_SEASON_MAP.get(key)
		if label and label not in seasons_to_compare:
			seasons_to_compare.append(label)

	if not seasons_to_compare:
		return None, None

	df = load_and_compare_seasons(league_name, seasons_to_compare=seasons_to_compare)
	if df is None or df.empty:
		return None, None

	df = df.copy()
	if "Player" not in df.columns:
		return None, None

	df["Player"] = df["Player"].astype(str).str.strip()

	fp_cols = {"Player": df["Player"]}
	gp_cols = {"Player": df["Player"]}

	for key in season_keys:
		label = HISTORICAL_SEASON_MAP.get(key)
		if not label:
			continue
		fp_src = f"FP/G_{label}"
		gp_src = f"GP_{label}"
		if fp_src in df.columns:
			fp_cols[f"{key} FP/G"] = df[fp_src]
		if gp_src in df.columns:
			gp_cols[f"{key} GP"] = df[gp_src]

	if len(fp_cols) <= 1 or len(gp_cols) <= 1:
		return None, None

	fp_df = pd.DataFrame(fp_cols)
	gp_df = pd.DataFrame(gp_cols)
	return fp_df, gp_df


def _build_market_value_from_drafts(trend_weights):
	try:
		drafts = load_draft_history()
	except Exception:
		return None

	if drafts is None or drafts.empty or "Bid" not in drafts.columns:
		return None

	df = drafts.copy()
	df["Bid"] = pd.to_numeric(df["Bid"], errors="coerce")
	df = df[df["Bid"].notna() & (df["Bid"] > 0)]
	if df.empty:
		return None

	pivot = df.pivot_table(index="Player", columns="SeasonKey", values="Bid", aggfunc="mean")
	if pivot.empty:
		return None

	pivot.columns = [str(c) for c in pivot.columns]

	weight_lookup = {}
	if trend_weights:
		for col in pivot.columns:
			if col.startswith("S"):
				weight_lookup[col] = float(trend_weights.get(col, 0.0))

	def _weighted(row):
		vsum = 0.0
		wsum = 0.0
		for col, w in weight_lookup.items():
			if w <= 0.0:
				continue
			v = row.get(col)
			if pd.notna(v):
				vsum += float(v) * w
				wsum += w
		if wsum > 0.0:
			return vsum / wsum

		vals = []
		for col in weight_lookup.keys():
			v = row.get(col)
			if pd.notna(v):
				vals.append(float(v))
		if not vals:
			return np.nan
		return float(np.mean(vals))

	pivot["MarketValue"] = pivot.apply(_weighted, axis=1).round()
	pivot.reset_index(inplace=True)

	market_value_df = pivot[["Player", "MarketValue"]].copy()
	market_value_df["Player"] = market_value_df["Player"].astype(str).str.strip()
	market_value_df = market_value_df.dropna(subset=["MarketValue"])
	if market_value_df.empty:
		return None

	return market_value_df


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
        # Ensure auction data directory exists for generated outputs
        AUCTION_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Resolve input CSVs with legacy fallbacks (root data/)
        fp_path = FP_PER_GAME_FILE if FP_PER_GAME_FILE.exists() else DATA_DIR / 'PlayerFPperGameOverYears.csv'
        gp_path = GP_FILE if GP_FILE.exists() else DATA_DIR / 'PlayerGPOverYears.csv'
        bids_path = HISTORICAL_BIDS_FILE if HISTORICAL_BIDS_FILE.exists() else DATA_DIR / 'PlayerBidPricesOverYears.csv'

        # Track where inputs actually came from so the UI can surface fallbacks
        source_info = {
            "pps_fp_gp_source": None,
            "pps_market_value_source": None,
        }

        # --- Load FP/G & GP (DB/historical-YTD first, then legacy CSVs) ---
        fp_df = None
        gp_df = None

        try:
            season_keys = list(trend_weights.keys())
        except Exception:
            season_keys = ['S1', 'S2', 'S3', 'S4']

        used_historical = False
        try:
            fp_df, gp_df = _build_fp_gp_from_historical_ytd(season_keys)
            if fp_df is not None and gp_df is not None:
                used_historical = True
        except Exception:
            fp_df, gp_df = None, None

        if not used_historical:
            fp_df = pd.read_csv(fp_path)
            gp_df = pd.read_csv(gp_path)
            source_info["pps_fp_gp_source"] = "csv"
        else:
            source_info["pps_fp_gp_source"] = "historical_ytd"

        # --- Calculate Market Value (draft history first, then legacy bids CSV) ---
        market_value_df = None
        used_draft_history = False
        try:
            market_value_df = _build_market_value_from_drafts(trend_weights)
            if market_value_df is not None:
                used_draft_history = True
        except Exception:
            market_value_df = None

        if not used_draft_history:
            bids_df = pd.read_csv(bids_path)
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
            market_value_df['Player'] = market_value_df['Player'].astype(str).str.strip()
            source_info["pps_market_value_source"] = "bids_csv"
        else:
            source_info["pps_market_value_source"] = "draft_history"

        # Load and consolidate position data from seasonal stats files
        all_pos_df = pd.DataFrame()
        for i in range(1, 5):
            try:
                season_stats_file = DATA_DIR / f'S{i}Stats.csv'
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
        # Keep season-level FP/G & GP columns so the auction tool can reuse them.
        base_cols = ['Player', 'Position', 'MarketValue', 'PPS']
        hist_cols = [
            c for c in merged_df.columns
            if c.startswith('S') and ('FP/G' in c or c.endswith(' GP'))
        ]
        keep_cols = base_cols + [c for c in hist_cols if c not in base_cols]
        final_df = merged_df[keep_cols].copy()
        final_df.rename(columns={'Player': 'PlayerName'}, inplace=True)

        # Sort by PPS descending and save
        final_df.sort_values(by='PPS', ascending=False, inplace=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        
        # Expose where PPS inputs came from so the UI can surface fallbacks
        try:
            st.session_state.auction_data_sources = source_info
        except Exception:
            pass
        
        print(f"Successfully created projection file at: {OUTPUT_FILE}")
        print(f"Total players with projections: {len(final_df)}")
        print("\nTop 5 players by PPS:")
        print(final_df.head())
        return True

    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure all required CSV files are in the 'data' directory.")
        try:
            st.session_state.auction_data_sources = {"error": str(e)}
        except Exception:
            pass
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        try:
            st.session_state.auction_data_sources = {"error": str(e)}
        except Exception:
            pass
        return False

if __name__ == '__main__':
    # Allows the script to be run standalone for testing or manual generation.
    generate_pps_projections()
