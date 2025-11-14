"""Cross-player value and reliability analyzer using cached game logs."""

import streamlit as st
import os
from typing import List
import plotly.express as px

from modules.player_value.logic import build_player_value_profiles
from modules.player_game_log_scraper.logic import load_league_cache_index
from modules.player_game_log_scraper.ui_viewer import show_player_consistency_viewer


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
	value_tab, consistency_tab = st.tabs([
		"📈 Player Value Rankings",
		"📊 Player Consistency Browser",
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
				st.success(f"Computed value profiles for {len(df)} players.")
				# Rankings table
				st.subheader("Player Rankings")
				display_cols = [
					"Player",
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
					df_display.sort_values("ValueScore", ascending=False),
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

	with consistency_tab:
		show_player_consistency_viewer(initial_league_id=league_id)


if __name__ == "__main__":
	main()
