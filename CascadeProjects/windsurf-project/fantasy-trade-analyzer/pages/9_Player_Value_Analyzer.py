"""Cross-player value and reliability analyzer using cached game logs."""

import streamlit as st
import os
from typing import List

from modules.player_value.logic import build_player_value_profiles
from modules.player_game_log_scraper.logic import load_league_cache_index


def _get_all_seasons_for_league(league_id: str) -> List[str]:
	"""Return all seasons present in the league cache index, sorted most recent first."""
	index = load_league_cache_index(league_id, rebuild_if_missing=True)
	if not index or not index.get("players"):
		return []

	seasons = set()
	for pdata in index["players"].values():
		seasons.update(pdata.get("seasons", {}).keys())

	return sorted(list(seasons), reverse=True)


def main():
	st.set_page_config(page_title="Player Value Analyzer", page_icon="🏆", layout="wide")
	st.title("🏆 Player Value & Reliability Analyzer")
	st.write("Rank players across seasons by production, consistency, and availability using cached game logs.")

	default_league_id = os.getenv("FANTRAX_DEFAULT_LEAGUE_ID", "")
	league_id = st.text_input("Fantrax League ID", value=default_league_id, help="League to analyze based on cached game logs.")

	if not league_id:
		st.warning("Enter a league ID to load player value profiles.")
		return

	seasons = _get_all_seasons_for_league(league_id)
	if not seasons:
		st.info("No cached seasons found for this league. Run the game log scraper first.")
		return

	st.sidebar.header("Filters")
	selected_seasons = st.sidebar.multiselect("Seasons to include", options=seasons, default=seasons)
	min_games = st.sidebar.number_input("Min games per season", min_value=1, max_value=82, value=20, step=1)
	min_seasons = st.sidebar.number_input("Min seasons", min_value=1, max_value=10, value=1, step=1)

	with st.spinner("Building player value profiles from cache..."):
		df = build_player_value_profiles(
			league_id=league_id,
			seasons=selected_seasons,
			min_games_per_season=min_games,
			min_seasons=min_seasons,
		)

	if df is None or df.empty:
		st.warning("No players met the criteria for the selected filters.")
		return

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
	st.dataframe(
		df[display_cols].sort_values("ValueScore", ascending=False),
		use_container_width=True,
		height=600,
	)

	# Download button
	csv = df.to_csv(index=False)
	st.download_button(
		label="📥 Download Player Value Rankings as CSV",
		data=csv,
		file_name=f"player_value_rankings_{league_id}.csv",
		mime="text/csv",
	)


if __name__ == "__main__":
	main()
