import datetime
from typing import Dict, List

import streamlit as st

from data_loader import TEAM_MAPPINGS
from modules.historical_trade_analyzer.logic import build_historical_combined_data
from modules.player_game_log_scraper.logic import load_league_cache_index
from modules.trade_analysis.logic import (
    TradeAnalyzer,
    _load_trade_history,
    _save_trade_history,
    FANTRAX_DEFAULT_LEAGUE_ID,
)
from modules.trade_analysis.ui import display_trade_results


def _historical_player_selection_interface(selected_teams: List[str], rosters_by_team: Dict[str, List[str]]):
    """UI for selecting traded players and their destinations based on provided rosters."""
    st.write("### Select Traded Players")
    trade_teams: Dict[str, Dict[str, str]] = {}

    if not selected_teams:
        return trade_teams

    num_cols = min(len(selected_teams), 3)
    cols = st.columns(num_cols)

    for i, team in enumerate(selected_teams):
        roster = rosters_by_team.get(team, [])
        with cols[i % num_cols]:
            st.write(f"#### {TEAM_MAPPINGS.get(team, team)}")
            if not roster:
                st.info("No roster entered for this team.")
                continue

            selected_players = st.multiselect(
                f"Players from {TEAM_MAPPINGS.get(team, team)} to trade",
                options=roster,
                key=f"hist_players_{team}",
            )

            if selected_players:
                trade_teams[team] = {}
                for player in selected_players:
                    other_teams = [t for t in selected_teams if t != team]
                    if not other_teams:
                        continue
                    dest = st.selectbox(
                        f"Destination for {player}",
                        options=other_teams,
                        format_func=lambda t: TEAM_MAPPINGS.get(t, t),
                        key=f"hist_dest_{team}_{player}",
                    )
                    trade_teams[team][player] = dest

    return trade_teams


def show_historical_trade_analyzer():
    """Display the Historical Trade Analyzer UI in Admin Tools."""
    st.subheader("📜 Historical Trade Analyzer")
    st.markdown("Use cached game logs to analyze past trades as of a specific date.")

    league_id = st.session_state.get("league_id") or FANTRAX_DEFAULT_LEAGUE_ID
    if not league_id:
        st.info("Set `league_id` in session (e.g., via other tools or league_config) to use this analyzer.")
        return

    index = load_league_cache_index(league_id, rebuild_if_missing=True)
    if not index or not index.get("players"):
        st.info("No league cache index found. Run the Player Game Logs scraper first.")
        return

    # Collect all seasons from the index
    seasons = set()
    for pdata in index.get("players", {}).values():
        for season_str in pdata.get("seasons", {}).keys():
            seasons.add(season_str)

    if not seasons:
        st.info("No seasons found in game log cache.")
        return

    seasons_sorted = sorted(seasons, reverse=True)

    with st.expander("Trade Context", expanded=True):
        season = st.selectbox("Season", options=seasons_sorted, index=0)
        trade_date = st.date_input(
            "Trade date",
            value=datetime.date.today(),
            help="The date on which the trade occurred.",
        )
        trade_label = st.text_input(
            "Trade label / description",
            value="",
            help="Give this historical trade a recognizable name.",
        )
        num_players = st.number_input(
            "Number of Top Players to Analyze",
            min_value=1,
            max_value=12,
            value=10,
            help="How many top players per team to include in the analysis.",
        )

    st.markdown("---")
    st.markdown("### Teams and Rosters at Trade Time")
    st.caption(
        "Rosters and results are shown as they were on the trade date. "
        "If players you remember are missing here, they were likely added or dropped at a different time and are no longer on the team today."
    )

    team_ids = sorted(TEAM_MAPPINGS.keys())
    selected_teams = st.multiselect(
        "Teams involved in the trade",
        options=team_ids,
        format_func=lambda t: TEAM_MAPPINGS.get(t, t),
    )

    if not selected_teams:
        st.info("Select at least two teams to analyze a trade.")
        return

    rosters_by_team: Dict[str, List[str]] = {}

    for team_id in selected_teams:
        display_name = TEAM_MAPPINGS.get(team_id, team_id)
        text = st.text_area(
            f"Roster for {display_name} at trade time (one player per line)",
            key=f"hist_roster_{team_id}",
            height=120,
        )
        players = [line.strip() for line in text.splitlines() if line.strip()]
        rosters_by_team[team_id] = players

    if any(not rosters_by_team.get(t) for t in selected_teams):
        st.info("Enter at least one player for each selected team.")

    st.markdown("---")
    trade_teams = _historical_player_selection_interface(selected_teams, rosters_by_team)

    if not trade_teams:
        st.info("Select traded players and their destinations above.")
        return

    if st.button("Analyze Historical Trade", type="primary"):
        with st.spinner("Building historical snapshot from game logs and running analysis..."):
            snapshot_df = build_historical_combined_data(trade_date, league_id, season, rosters_by_team)

            if snapshot_df is None or snapshot_df.empty:
                st.error("Could not build historical snapshot. Check that game logs exist for the selected season and players.")
                return

            analyzer = TradeAnalyzer(snapshot_df)
            results = analyzer.evaluate_trade_fairness(trade_teams, int(num_players))
            # Annotate results with season so downstream components can select matching game logs
            for team_key in results.keys():
                if isinstance(results.get(team_key), dict):
                    results[team_key]["season"] = season
                    results[team_key]["trade_date"] = trade_date.isoformat()

            if not results:
                st.error("No results returned from trade analysis.")
                return

            # Display using existing Trade Analysis UI components
            st.markdown("---")
            st.subheader("Historical Trade Analysis Result")
            display_trade_results(results)

            # Persist to shared trade history cache as a historical entry
            history = _load_trade_history()
            try:
                summary = analyzer._generate_trade_summary(results)
            except Exception:
                summary = ""

            entry = {
                "trade_teams": trade_teams,
                "summary": summary,
                "label": trade_label or "",
                "date": trade_date.isoformat(),
                "num_players": int(num_players),
                "source": "historical",
                "season": season,
                "rosters_by_team": rosters_by_team,
                "league_id": league_id,
            }
            history.append(entry)
            _save_trade_history(history)

            # If a main TradeAnalyzer exists in session, keep its history in sync for this run
            if "trade_analyzer" in st.session_state and st.session_state.trade_analyzer:
                st.session_state.trade_analyzer.trade_history = history

            st.success("Historical trade analysis completed and recorded.")
