import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

from modules.manager_ids import load_manager_ids, get_manager_list
from modules.sidebar.ui import display_global_sidebar
from streamlit_compat import dataframe
from ui.history_hub import render_best_team_optimizer, render_draft_history


st.set_page_config(page_title="Manager History", page_icon="👤", layout="wide")

load_dotenv(find_dotenv('fantrax.env'))

display_global_sidebar()

st.title("👤 Manager History")
st.caption("Manager timeline + Draft history in one place.")

tab_mgr, tab_history, tab_draft, tab_best = st.tabs(["👤 Manager History", "📊 League History", "📜 Draft History", "🧠 Best $200 Team"])

with tab_mgr:
    # Load manager identity data
    mid_df = load_manager_ids()
    if mid_df.empty:
        st.warning("ManagerIDs.csv not found or contains no rows.")
        st.stop()

    mgr_list = get_manager_list(mid_df)
    if mgr_list.empty:
        st.warning("No manager records available.")
        st.stop()

    # Sidebar-like selector in the main body
    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.subheader("Select Manager")
        options = mgr_list["label"].tolist()
        default_index = 0
        selected_label = st.selectbox("Manager", options=options, index=default_index)
        # Map back to ManagerID
        selected_row = mgr_list[mgr_list["label"] == selected_label].iloc[0]
        selected_id = selected_row["managerid"]

    with right_col:
        st.subheader("Manager Overview")
        st.markdown(f"**Manager ID:** `{selected_id}`")

        st.markdown("### League Standings")
        username = os.environ.get("FANTRAX_USERNAME", "")
        password = os.environ.get("FANTRAX_PASSWORD", "")
        if not username or not password:
            st.info("Fantrax credentials not found. Set FANTRAX_USERNAME and FANTRAX_PASSWORD in fantrax.env.")
            standings_df = pd.DataFrame(columns=["league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA"])
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                refresh = st.button("Refresh League Standings", key="refresh_league_standings")
            with col2:
                stop = st.button("Stop/Kill Chrome", key="stop_chrome")
                if stop:
                    try:
                        from modules.league_standings_scraper.logic import request_stop, _kill_chromedriver_processes
                        request_stop()
                        _kill_chromedriver_processes()
                        st.success("Killed Chrome/chromedriver processes.")
                    except Exception as e:
                        st.error(f"Failed to kill processes: {e}")

            standings_df = pd.DataFrame(columns=["league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA"])
            if refresh:
                with st.spinner("Scraping Fantrax standings across leagues…"):
                    try:
                        from modules.league_standings_scraper.logic import clear_stop, scrape_all_league_standings
                        clear_stop()
                        standings_df = scrape_all_league_standings(
                            username=username,
                            password=password,
                        )
                    except Exception as e:
                        st.error(f"Failed to load league standings: {e}")
                        standings_df = pd.DataFrame(columns=["league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA"])
            else:
                # Try to load from cache without any network calls
                try:
                    from modules.league_standings_scraper.logic import load_cached_all_league_standings
                    standings_df = load_cached_all_league_standings()
                    if standings_df.empty:
                        standings_df = pd.DataFrame(columns=["league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA"])
                except Exception:
                    standings_df = pd.DataFrame(columns=["league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA"])

        seasons_df = mid_df[mid_df["managerid"] == selected_id].copy()

        def _norm_team(x: str) -> str:
            return "".join(ch.lower() for ch in str(x) if ch.isalnum())

        league_meta = {
            "Mister Squidward's Step Back 3": {
                "season": "2021-22",
                "champion": "Gianni’s Secret Move",
            },
            "Mister Squidward Travels": {
                "season": "2022-23",
                "champion": "2024 MVP Ding Yanyuhang",
            },
            "Mister Squidward's N Word Pass": {
                "season": "2023-24",
                "champion": "Adam Silver's Sovereign Wealth Fund",
            },
            "Mr Squidwards 69": {
                "season": "2024-25",
                "champion": "15 Dream Team",
            },
            "Mr Squidward's Gay Layup Line": {
                "season": "2025-26",
                "champion": "(in progress)",
            },
        }

        # Build season -> manager team name mapping
        mgr_team_by_season = {}
        if not seasons_df.empty:
            for _, r in seasons_df.iterrows():
                s = str(r.get("season", "")).strip()
                t = str(r.get("team_name", "")).strip()
                if s and t:
                    mgr_team_by_season[s] = t

        # Season filter checkboxes
        available_seasons = sorted({meta["season"] for meta in league_meta.values()})
        exclude_seasons = []
        if available_seasons:
            st.markdown("**Select seasons to exclude:**")
            cols = st.columns(len(available_seasons))
            for i, season in enumerate(available_seasons):
                with cols[i]:
                    if st.checkbox(season, key=f"exclude_{season}"):
                        exclude_seasons.append(season)

        matched_rows = []
        if (not standings_df.empty) and ("team_name" in standings_df.columns):
            for league_name, meta in league_meta.items():
                season = meta.get("season", "")
                if season in exclude_seasons:
                    continue
                mgr_team = mgr_team_by_season.get(season, "")
                if not mgr_team:
                    continue
                want = _norm_team(mgr_team)
                df_league = standings_df[standings_df.get("league_name", "") == league_name].copy()
                if df_league.empty:
                    continue
                df_league["_norm"] = df_league["team_name"].astype(str).map(_norm_team)
                df_mgr = df_league[df_league["_norm"] == want].copy()
                if df_mgr.empty:
                    continue
                df_mgr["season"] = season
                df_mgr["manager_team_name"] = mgr_team
                df_mgr["champion"] = meta.get("champion", "")
                matched_rows.append(df_mgr)

        st.markdown("#### Selected Manager Standings by Season")
        if matched_rows:
            matched = pd.concat(matched_rows, ignore_index=True)
            # Add per-season win rate (whole number %)
            def win_rate_row(row):
                try:
                    w = int(row["W"])
                    l = int(row["L"])
                    t = int(row["T"])
                    total = w + l + t
                    return f"{round(w / total * 100)}%" if total > 0 else "N/A"
                except Exception:
                    return "N/A"
            matched["Win Rate"] = matched.apply(win_rate_row, axis=1)
            # Add per-season W/L ratio
            def wl_ratio_row(row):
                try:
                    w = int(row["W"])
                    l = int(row["L"])
                    if l == 0:
                        return f"{w}/{l} = —"
                    ratio = w / l
                    return f"{w}/{l} = {ratio:.2f}"
                except Exception:
                    return "N/A"
            matched["W/L Ratio"] = matched.apply(wl_ratio_row, axis=1)
            show = [
                "season",
                "league_name",
                "manager_team_name",
                "W",
                "L",
                "T",
                "FPtsF",
                "FPtsA",
                "Win Rate",
                "W/L Ratio",
                "champion",
            ]
            show = [c for c in show if c in matched.columns]
            dataframe(matched[show], hide_index=True, width="stretch")

            # Summary row at the bottom
            numeric_cols = ["W", "L", "T", "FPtsF", "FPtsA"]
            summary = {}
            for col in numeric_cols:
                if col in matched.columns:
                    try:
                        summary[col] = int(matched[col].sum())
                    except Exception:
                        summary[col] = ""
                else:
                    summary[col] = ""
            # Calculate win rate
            w = summary.get("W")
            l = summary.get("L")
            t = summary.get("T")
            if isinstance(w, int) and isinstance(l, int) and isinstance(t, int):
                total_games = w + l + t
                if total_games > 0:
                    win_rate = w / total_games
                    summary["Win Rate"] = f"{round(win_rate * 100)}%"
                else:
                    summary["Win Rate"] = "N/A"
            else:
                summary["Win Rate"] = "N/A"
            # Calculate W/L ratio
            if isinstance(w, int) and isinstance(l, int):
                if l == 0:
                    summary["W/L Ratio"] = f"{w}/{l} = —"
                else:
                    ratio = w / l
                    summary["W/L Ratio"] = f"{w}/{l} = {ratio:.2f}"
            else:
                summary["W/L Ratio"] = "N/A"
            summary_df = pd.DataFrame([summary])
            st.markdown("**Totals across matched seasons:**")
            dataframe(summary_df, hide_index=True, width="stretch")
        else:
            st.info("No matched standings found for this manager yet (based on season-to-league mapping).")

        with st.expander("Show full league standings (all teams)", expanded=False):
            show_cols = [
                "league_name",
                "team_name",
                "W",
                "L",
                "T",
                "FPtsF",
                "FPtsA",
            ]
            show_cols = [c for c in show_cols if c in standings_df.columns]
            dataframe(standings_df[show_cols], hide_index=True, width="stretch")

        # Filter seasons for this manager
        seasons_df = mid_df[mid_df["managerid"] == selected_id].copy()
        if seasons_df.empty:
            st.info("No season records found for this manager.")
        else:
            # Sort by season string; relies on consistent formatting like '2021-22'
            seasons_df = seasons_df.sort_values("season")

            st.markdown("### Seasons & Team Identities")
            display_cols = [
                "season",
                "team_name",
                "team_abbreviation",
            ]
            pretty = seasons_df[display_cols].rename(
                columns={
                    "season": "Season",
                    "team_name": "Team Name",
                    "team_abbreviation": "Team Abbrev",
                }
            )
            dataframe(pretty, hide_index=True, width="stretch")

            # Simple timeline-style text summary
            st.markdown("### Timeline")
            timeline_lines = []
            for _, row in seasons_df.iterrows():
                season = row.get("season", "?")
                tname = row.get("team_name", "?")
                tabbr = row.get("team_abbreviation", "")
                if tabbr:
                    timeline_lines.append(f"- **{season}** – {tname} (`{tabbr}`)")
                else:
                    timeline_lines.append(f"- **{season}** – {tname}")
            if timeline_lines:
                st.markdown("\n".join(timeline_lines))
            else:
                st.caption("No timeline entries available.")

    st.markdown("---")
    st.caption(
        "Rosters and advanced performance history by manager can be layered onto this view "
        "once historical rosters are available from the DB/log scrapers."
    )

with tab_history:
    st.markdown("### 📊 League History Overview")
    st.caption("All seasons, standings, and top managers by W/L ratio across time.")

    # Load standings cache
    username = os.environ.get("FANTRAX_USERNAME", "")
    password = os.environ.get("FANTRAX_PASSWORD", "")
    if not username or not password:
        st.info("Fantrax credentials not found. Set FANTRAX_USERNAME and FANTRAX_PASSWORD in fantrax.env to view league history.")
        st.stop()

    standings_df = pd.DataFrame(columns=["league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA"])
    try:
        from modules.league_standings_scraper.logic import load_cached_all_league_standings
        standings_df = load_cached_all_league_standings()
        if standings_df.empty:
            standings_df = pd.DataFrame(columns=["league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA"])
    except Exception:
        standings_df = pd.DataFrame(columns=["league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA"])

    # Load manager data for name mapping
    mid_df = load_manager_ids()
    mgr_map = {}
    if not mid_df.empty:
        for _, r in mid_df.iterrows():
            mgr_map.setdefault(r["managerid"], set()).add(str(r["team_name"]).strip())

    league_meta = {
        "Mister Squidward's Step Back 3": {"season": "2021-22", "champion": "Gianni’s Secret Move"},
        "Mister Squidward Travels": {"season": "2022-23", "champion": "2024 MVP Ding Yanyuhang"},
        "Mister Squidward's N Word Pass": {"season": "2023-24", "champion": "Adam Silver's Sovereign Wealth Fund"},
        "Mr Squidwards 69": {"season": "2024-25", "champion": "15 Dream Team"},
        "Mr Squidward's Gay Layup Line": {"season": "2025-26", "champion": "(in progress)"},
    }

    def _norm_team(x: str) -> str:
        return "".join(ch.lower() for ch in str(x) if ch.isalnum())

    # Enrich standings with season, champion, and managerid
    enriched = []
    if not standings_df.empty and "team_name" in standings_df.columns:
        for _, row in standings_df.iterrows():
            league = row.get("league_name", "")
            meta = league_meta.get(league, {})
            season = meta.get("season", "")
            champion = meta.get("champion", "")
            team_name = str(row.get("team_name", "")).strip()
            norm = _norm_team(team_name)
            # Try to map team name to a manager id
            manager_ids = []
            for mgr_id, team_set in mgr_map.items():
                if any(_norm_team(t) == norm for t in team_set):
                    manager_ids.append(mgr_id)
            enriched.append({
                "season": season,
                "league_name": league,
                "team_name": team_name,
                "W": row.get("W", ""),
                "L": row.get("L", ""),
                "T": row.get("T", ""),
                "FPtsF": row.get("FPtsF", ""),
                "FPtsA": row.get("FPtsA", ""),
                "champion": champion,
                "manager_ids": ",".join(manager_ids),
            })

    if not enriched:
        st.info("No cached standings data available. Refresh from the Manager History tab first.")
    else:
        df = pd.DataFrame(enriched)
        # Add win rate and w/l ratio
        def win_rate_row(row):
            try:
                w = int(row["W"])
                l = int(row["L"])
                t = int(row["T"])
                total = w + l + t
                return f"{round(w / total * 100)}%" if total > 0 else "N/A"
            except Exception:
                return "N/A"
        def wl_ratio_row(row):
            try:
                w = int(row["W"])
                l = int(row["L"])
                if l == 0:
                    return f"{w}/{l} = —"
                ratio = w / l
                return f"{w}/{l} = {ratio:.2f}"
            except Exception:
                return "N/A"
        df["Win Rate"] = df.apply(win_rate_row, axis=1)
        df["W/L Ratio"] = df.apply(wl_ratio_row, axis=1)

        st.markdown("#### All Teams by Season")
        # Exclude current season (2025-26)
        df_hist = df[df["season"] != "2025-26"].copy()
        all_cols = ["season", "league_name", "team_name", "W", "L", "T", "FPtsF", "FPtsA", "Win Rate", "W/L Ratio", "champion"]
        dataframe(df_hist[all_cols], hide_index=True, width="stretch")

        # Best managers by W/L ratio across seasons
        st.markdown("#### 🏆 Best Managers by W/L Ratio (All Seasons)")
        # Exclude current season (2025-26)
        df_hist = df[df["season"] != "2025-26"].copy()
        # Expand manager_ids to rows
        mgr_rows = []
        for _, row in df_hist.iterrows():
            ids = str(row["manager_ids"]).split(",") if row["manager_ids"] else []
            for mgr_id in ids:
                if not mgr_id.strip():
                    continue
                mgr_rows.append({
                    "manager_id": mgr_id.strip(),
                    "season": row["season"],
                    "team_name": row["team_name"],
                    "W": row["W"],
                    "L": row["L"],
                    "T": row["T"],
                    "Win Rate": row["Win Rate"],
                    "W/L Ratio": row["W/L Ratio"],
                })
        if mgr_rows:
            mgr_df = pd.DataFrame(mgr_rows)
            # Parse numeric ratio for sorting
            def parse_ratio(val):
                try:
                    if "=" in str(val):
                        return float(val.split("=")[1].strip())
                except Exception:
                    pass
                return -1
            mgr_df["_ratio_val"] = mgr_df["W/L Ratio"].apply(parse_ratio)
            best = mgr_df.sort_values("_ratio_val", ascending=False).drop_duplicates(subset="manager_id", keep="first")
            best_display = best[["manager_id", "season", "team_name", "W", "L", "T", "Win Rate", "W/L Ratio"]]
            dataframe(best_display, hide_index=True, width="stretch")
        else:
            st.info("No manager mappings found in ManagerIDs.csv.")

    st.markdown("---")
    st.markdown("### 🏆 Playoffs")
    st.caption("Playoffs rounds and matchups per league (cached until you click Refresh).")
    col1, col2 = st.columns([1, 1])
    with col1:
        refresh_playoffs = st.button("Refresh Playoffs", key="refresh_playoffs")
    with col2:
        stop_playoffs = st.button("Stop/Kill Chrome (Playoffs)", key="stop_playoffs")
        if stop_playoffs:
            try:
                from modules.league_standings_scraper.logic import request_stop, _kill_chromedriver_processes
                request_stop()
                _kill_chromedriver_processes()
                st.success("Killed Chrome/chromedriver processes.")
            except Exception as e:
                st.error(f"Failed to kill processes: {e}")

    playoffs_df = pd.DataFrame(columns=["league_name", "round", "date_range", "away_team", "away_total", "home_team", "home_total"])
    if refresh_playoffs:
        with st.spinner("Scraping Fantrax playoffs across leagues…"):
            try:
                from modules.league_standings_scraper.logic import clear_stop, scrape_all_league_playoffs
                clear_stop()
                playoffs_df = scrape_all_league_playoffs(username=username, password=password)
            except Exception as e:
                st.error(f"Failed to load playoffs: {e}")
                playoffs_df = pd.DataFrame(columns=["league_name", "round", "date_range", "away_team", "away_total", "home_team", "home_total"])
    else:
        # Load from cache only
        try:
            from modules.league_standings_scraper.logic import load_cached_all_league_playoffs
            playoffs_df = load_cached_all_league_playoffs()
            if playoffs_df.empty:
                playoffs_df = pd.DataFrame(columns=["league_name", "round", "date_range", "away_team", "away_total", "home_team", "home_total"])
        except Exception:
            playoffs_df = pd.DataFrame(columns=["league_name", "round", "date_range", "away_team", "away_total", "home_team", "home_total"])

    if playoffs_df.empty:
        st.info("No cached playoffs data available. Click Refresh Playoffs to fetch.")
    else:
        show_cols = ["league_name", "round", "date_range", "away_team", "away_total", "home_team", "home_total"]
        show_cols = [c for c in show_cols if c in playoffs_df.columns]
        dataframe(playoffs_df[show_cols], hide_index=True, width="stretch")

with tab_draft:
    render_draft_history()

with tab_best:
    render_best_team_optimizer()
