"""Cross-player value and reliability analyzer using cached game logs."""

import streamlit as st
import os
import re
from typing import List
import plotly.express as px

from modules.player_value.logic import build_player_value_profiles
from modules.player_game_log_scraper.logic import load_league_cache_index
from modules.player_game_log_scraper.ui_viewer import show_player_consistency_viewer
from modules.historical_ytd_downloader.logic import load_and_compare_seasons, DOWNLOAD_DIR
from modules.trade_suggestions import calculate_player_value, calculate_league_scarcity_context


def _get_all_seasons_for_league(league_id: str) -> List[str]:
	"""Return all seasons present in the league cache index, sorted most recent first."""
	index = load_league_cache_index(league_id, rebuild_if_missing=True)
	if not index or not index.get("players"):
		return []

	seasons = set()
	for pdata in index["players"].values():
		seasons.update(pdata.get("seasons", {}).keys())

	return sorted(list(seasons), reverse=True)


@st.cache_data(show_spinner=False)
def _get_player_value_profiles_cached(league_id: str, seasons: list[str], min_games_per_season: int, min_seasons: int):
	"""Cached wrapper around build_player_value_profiles to avoid recomputing on every rerun."""
	seasons_arg = seasons if seasons else None
	return build_player_value_profiles(
		league_id=league_id,
		seasons=seasons_arg,
		min_games_per_season=min_games_per_season,
		min_seasons=min_seasons,
	)


@st.cache_data(show_spinner=False)
def _get_yoy_context_for_league(league_id: str):
	"""Return (league_name_sanitized, seasons) for YoY comparison, or None if unavailable."""
	env_ids = os.getenv("FANTRAX_LEAGUE_IDS", "")
	env_names = os.getenv("FANTRAX_LEAGUE_NAMES", "")
	id_list = [s.strip() for s in env_ids.split(",") if s.strip()]
	name_list = [s.strip() for s in env_names.split(",") if s.strip()]
	league_name_map = {
		lid: (name_list[idx] if idx < len(name_list) else lid)
		for idx, lid in enumerate(id_list)
	}
	display_name = league_name_map.get(league_id, league_id)
	league_name_sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", str(display_name).strip().replace(" ", "_"))
	available_files = list(DOWNLOAD_DIR.glob(f"Fantrax-Players-{league_name_sanitized}-YTD-*.csv"))
	if not available_files:
		return None
	seasons: list[str] = []
	for fpath in available_files:
		parts = fpath.stem.split('-YTD-')
		if len(parts) == 2:
			seasons.append(parts[1])
	seasons = sorted(set(seasons), reverse=True)
	if len(seasons) < 2:
		return None
	return league_name_sanitized, seasons


@st.cache_data(show_spinner=False)
def _get_yoy_comparison_cached(league_name_sanitized: str, seasons: list[str]):
	"""Cached wrapper around load_and_compare_seasons for YoY trends."""
	return load_and_compare_seasons(league_name_sanitized, seasons)


def main():
	st.set_page_config(page_title="Player Value & Consistency", page_icon="🏆", layout="wide")
	st.title("🏆 Player Value & Consistency Hub")
	st.write("Analyze multi-season player value profiles and detailed game-by-game consistency from one place.")

	default_league_id = os.getenv("FANTRAX_DEFAULT_LEAGUE_ID", "")
	league_id = st.text_input("Fantrax League ID", value=default_league_id, help="League to analyze based on cached game logs.")

	if not league_id:
		st.warning("Enter a league ID to load player value profiles and consistency data.")
		return

	st.markdown("---")
	value_tab, consistency_tab, yoy_tab = st.tabs([
		"📈 Player Value Rankings",
		"📊 Player Consistency Browser",
		"📊 YoY Trends",
	])

	with value_tab:
		seasons = _get_all_seasons_for_league(league_id)
		if not seasons:
			st.info("No cached seasons found for this league. Run the game log scraper first.")
		else:
			st.subheader("Filters")
			col1, col2, col3 = st.columns([2, 1, 1])
			with col1:
				selected_seasons = st.multiselect(
					"Seasons to include",
					options=seasons,
					default=seasons,
					help="Choose which seasons to include in the value calculation."
				)
			with col2:
				min_games = st.number_input(
					"Min games/season",
					min_value=1,
					max_value=82,
					value=20,
					step=1,
					help="Players must meet this games played threshold in a season to be included."
				)
			with col3:
				min_seasons = st.number_input(
					"Min seasons",
					min_value=1,
					max_value=10,
					value=1,
					step=1,
					help="Players must have at least this many qualifying seasons."
				)
			
			with st.spinner("Building player value profiles from cache..."):
				df = _get_player_value_profiles_cached(
					league_id=league_id,
					seasons=selected_seasons,
					min_games_per_season=min_games,
					min_seasons=min_seasons,
				)

			if df is None or df.empty:
				st.warning("No players met the criteria for the selected filters.")
			else:
				# Compute FP/G-centric trade model value using the same logic as the trade engine.
				# We treat the multi-season averages as an approximate league snapshot.
				try:
					league_df = df[[
						"Player",
						"AvgMeanFPts",
						"AvgCV%",
						"AvgGamesPerSeason",
					]].rename(columns={
						"AvgMeanFPts": "Mean FPts",
						"AvgCV%": "CV %",
						"AvgGamesPerSeason": "GP",
					})
					# Build a synthetic league context from all players in this view
					scarcity_context = calculate_league_scarcity_context({"League": league_df})
					trade_values = league_df.apply(
						lambda row: calculate_player_value(row, scarcity_context=scarcity_context),
						axis=1,
					)
					# Align index just in case
					df["TradeModelValue"] = trade_values.values
				except Exception:
					# Fallback: if anything goes wrong, at least keep the column present
					df["TradeModelValue"] = float("nan")
				
				st.success(f"Computed value profiles for {len(df)} players.")
				# Rankings table
				st.subheader("Player Rankings")
				display_cols = [
					"Player",
					"TradeModelValue",
					"ValueScore",
					"ProductionScore",
					"ConsistencyScore",
					"AvailabilityScore",
					"PlayoffReliabilityScore",
					"DurabilityTier",
					"SeasonsIncluded",
					"AvgMeanFPts",
					"AvgCV%",
					"AvgGamesPerSeason",
				]
				df_display = df[display_cols].copy()
				for col in [
					"TradeModelValue",
					"ValueScore",
					"ProductionScore",
					"ConsistencyScore",
					"AvailabilityScore",
					"PlayoffReliabilityScore",
					"AvgMeanFPts",
					"AvgCV%",
					"AvgGamesPerSeason",
				]:
					if col in df_display.columns:
						df_display[col] = df_display[col].round(2)
				st.dataframe(
					df_display.sort_values("TradeModelValue", ascending=False),
					use_container_width=True,
					height=600,
				)
				st.markdown("---")
				st.subheader("Value vs Production (Trade Target Map)")
				# Scatter plot to visualize players by production, consistency, and durability
				fig = px.scatter(
					df,
					x="AvgMeanFPts",
					y="ValueScore",
					color="DurabilityTier",
					hover_name="Player",
					size="AvgGamesPerSeason",
					size_max=20,
					labels={
						"AvgMeanFPts": "Avg FPts/G",
						"ValueScore": "Value Score",
					},
				)
				fig.update_layout(
					height=500,
					xaxis_title="Average Fantasy Points per Game",
					yaxis_title="Composite Value Score",
					hovermode="closest",
				)
				st.plotly_chart(fig, use_container_width=True)
				# Download button
				csv = df.to_csv(index=False)
				st.download_button(
					label="📥 Download Player Value Rankings as CSV",
					data=csv,
					file_name=f"player_value_rankings_{league_id}.csv",
					mime="text/csv",
				)

				# Warm YoY comparison cache in the background so the YoY tab feels instant
				ctx = _get_yoy_context_for_league(league_id)
				if ctx is not None:
					league_name_sanitized, yoy_seasons = ctx
					_ = _get_yoy_comparison_cached(league_name_sanitized, yoy_seasons)

	with consistency_tab:
		show_player_consistency_viewer(initial_league_id=league_id)

	with yoy_tab:
		st.subheader("📊 Year-over-Year FP/G Trends")
		st.caption("Compare player FP/G across historical YTD CSVs to spot breakouts, regressions, and stable producers.")
		# Use cached context + comparison to keep this tab responsive
		ctx = _get_yoy_context_for_league(league_id)
		if ctx is None:
			st.warning("No historical YTD CSVs found for this league, or fewer than two seasons are available.")
			return
		league_name_sanitized, seasons = ctx
		st.info(f"Comparing all available seasons: {', '.join(seasons)}")
		with st.spinner("Loading YoY comparison data..."):
			comparison_df = _get_yoy_comparison_cached(league_name_sanitized, seasons)
		if comparison_df is None or comparison_df.empty:
			st.warning("No valid YoY comparison data available.")
			return
		
		# Basic filters similar to the standalone YoY page
		st.markdown("---")
		col1, col2, col3 = st.columns(3)
		with col1:
			min_fpg = st.number_input(
				"Min FP/G (most recent season)",
				min_value=0.0,
				value=30.0,
				step=5.0,
			)
		with col2:
			min_gp = st.number_input(
				"Min GP (most recent season)",
				min_value=0,
				value=1,
				step=1,
			)
		with col3:
			show_only_improvers = st.checkbox("Show only improvers", value=False)
		
		# Apply filters on a copy
		filtered_df = comparison_df.copy()
		most_recent_fp_col = f"FP/G_{seasons[0]}"
		most_recent_gp_col = f"GP_{seasons[0]}"
		if most_recent_fp_col in filtered_df.columns:
			filtered_df = filtered_df[filtered_df[most_recent_fp_col] >= min_fpg]
		if most_recent_gp_col in filtered_df.columns:
			filtered_df = filtered_df[filtered_df[most_recent_gp_col] >= min_gp]
		
		if len(seasons) >= 2:
			pct_col = f"YoY_Pct_{seasons[0]}_vs_{seasons[1]}"
			if pct_col in filtered_df.columns and show_only_improvers:
				filtered_df = filtered_df[filtered_df[pct_col] > 0]
		
		# Round numeric columns and display
		if filtered_df.empty:
			st.info("No players match the current YoY filters.")
		else:
			for col in filtered_df.columns:
				if col != "Player" and str(filtered_df[col].dtype).startswith("float"):
					filtered_df[col] = filtered_df[col].round(2)
			# Summary metrics (compact version of standalone YoY page)
			st.markdown("#### Summary")
			col1, col2, col3, col4 = st.columns(4)
			with col1:
				st.metric("Total Players", len(filtered_df))
			with col2:
				if len(seasons) >= 2:
					pct_col = f"YoY_Pct_{seasons[0]}_vs_{seasons[1]}"
					if pct_col in filtered_df.columns:
						improvers = len(filtered_df[filtered_df[pct_col] > 0])
						st.metric("Improvers", improvers)
			with col3:
				if len(seasons) >= 2:
					pct_col = f"YoY_Pct_{seasons[0]}_vs_{seasons[1]}"
					if pct_col in filtered_df.columns:
						decliners = len(filtered_df[filtered_df[pct_col] < 0])
						st.metric("Decliners", decliners)
			with col4:
				if len(seasons) >= 2:
					pct_col = f"YoY_Pct_{seasons[0]}_vs_{seasons[1]}"
					if pct_col in filtered_df.columns:
						breakouts = len(filtered_df[filtered_df[pct_col] > 20])
						st.metric("Breakouts (>20%)", breakouts)
			st.markdown("#### Player Comparison")
			st.dataframe(filtered_df, use_container_width=True, height=550)


if __name__ == "__main__":
	main()
