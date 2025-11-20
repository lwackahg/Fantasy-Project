import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.trade_suggestions import find_trade_suggestions, calculate_exponential_value, set_trade_balance_preset
from modules.player_game_log_scraper.logic import get_cache_directory
from modules.player_game_log_scraper.ui_fantasy_teams import _build_fantasy_team_view
from modules.trade_analysis.consistency_integration import (
    CONSISTENCY_VERY_MAX_CV,
    CONSISTENCY_MODERATE_MAX_CV,
)
from modules.player_game_log_scraper import db_store
from modules.trade_analysis.logic import run_trade_analysis
from modules.trade_analysis.ui import display_trade_results
from modules.team_mappings import TEAM_MAPPINGS

CODE_BY_MANAGER = {v: k for k, v in TEAM_MAPPINGS.items()}

def _display_trade_suggestion(suggestion, rank, rosters_by_team, your_team_name):
    """Display a single trade suggestion with detailed impact metrics."""
    core_size_approx = 7.14
    min_games = 25
    weekly_core_fp_change = suggestion["value_gain"]
    core_ppg_change = weekly_core_fp_change / min_games
    opp_weekly_core_fp_change = suggestion.get("opp_core_gain", 0)

    st.markdown(f"### 🔄 Trade Impact: **+{weekly_core_fp_change:.1f} weekly core FP** for you")
    st.caption(
        f"Your core FP/G improves by ~{core_ppg_change:.2f} across your top {int(core_size_approx)} players"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📤 You Give")
        give_df = pd.DataFrame(
            {
                "Player": suggestion["you_give"],
                "FP/G": suggestion["your_fpts"],
                "CV%": suggestion["your_cv"],
            }
        )
        st.dataframe(give_df, hide_index=True, use_container_width=True)
        your_avg_fpts = sum(suggestion["your_fpts"]) / max(len(suggestion["your_fpts"]), 1)
        st.caption(f"Package avg: {your_avg_fpts:.1f} FP/G")
    with col2:
        st.markdown("### 📥 You Get")
        get_df = pd.DataFrame(
            {
                "Player": suggestion["you_get"],
                "FP/G": suggestion["their_fpts"],
                "CV%": suggestion["their_cv"],
            }
        )
        st.dataframe(get_df, hide_index=True, use_container_width=True)
        their_avg_fpts = sum(suggestion["their_fpts"]) / max(len(suggestion["their_fpts"]), 1)
        st.caption(f"Package avg: {their_avg_fpts:.1f} FP/G")

    st.markdown("---")
    st.markdown("#### 📊 Weekly Core FP Impact")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Your Gain",
            x=["Weekly Core FP"],
            y=[weekly_core_fp_change],
            marker_color="#4CAF50",
            text=[f"+{weekly_core_fp_change:.1f}"],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Opponent Change",
            x=["Weekly Core FP"],
            y=[opp_weekly_core_fp_change],
            marker_color="#FF9800" if opp_weekly_core_fp_change < 0 else "#2196F3",
            text=[f"{opp_weekly_core_fp_change:+.1f}"],
            textposition="outside",
        )
    )
    fig.update_layout(
        barmode="group",
        height=300,
        showlegend=True,
        yaxis_title="Weekly Core FP Change",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"trade_value_chart_{rank}")

    if weekly_core_fp_change > 30:
        st.success("🟢 **Excellent Trade** - Major weekly core FP upgrade!")
    elif weekly_core_fp_change > 15:
        st.success("🟢 **Strong Trade** - Solid weekly core FP gain")
    elif weekly_core_fp_change > 5:
        st.info("🟡 **Decent Trade** - Modest weekly core FP improvement")
    else:
        st.info("🟡 **Marginal Trade** - Small weekly core FP gain")

    if opp_weekly_core_fp_change < -15:
        st.warning("⚠️ **Opponent loses significant core FP** - they may not accept")
    elif opp_weekly_core_fp_change < -5:
        st.info("ℹ️ **Opponent loses some core FP** - negotiate carefully")
    elif opp_weekly_core_fp_change > 0:
        st.success("✅ **Win-win trade** - opponent also gains core FP")

    # Aggregate averages for reuse across sections
    your_avg_cv = sum(suggestion["your_cv"]) / max(len(suggestion["your_cv"]), 1)
    their_avg_cv = sum(suggestion["their_cv"]) / max(len(suggestion["their_cv"]), 1)
    fpts_diff = their_avg_fpts - your_avg_fpts
    cv_change = their_avg_cv - your_avg_cv

    # Short narrative summary
    st.markdown("#### 💡 Why this trade works")
    reasons = []
    reasons.append(
        f"📈 Core upgrade: your top ~{int(core_size_approx)} players gain ~{core_ppg_change:.2f} FP/G "
        f"(+{weekly_core_fp_change:.1f} weekly core FP)."
    )

    pattern = suggestion.get("pattern", "")
    if pattern in ("2-for-1", "3-for-1"):
        reasons.append("📦 Consolidation: trading depth for a stronger core piece.")
    elif pattern in ("1-for-2", "1-for-3"):
        reasons.append("📊 Depth play: turning one stud into multiple reliable starters.")

    if fpts_diff > 3:
        reasons.append(f"💰 Package FP/G tilts toward what you receive (+{fpts_diff:.1f} FP/G).")
    elif fpts_diff < -3:
        reasons.append(
            f"⚖️ You give up some package FP/G ({fpts_diff:.1f}), but your core still strengthens."
        )

    if cv_change < -5:
        reasons.append("🛡️ Risk reduction: your roster becomes noticeably more consistent.")
    elif cv_change > 5:
        reasons.append("🎲 Higher variance: more upside but a shakier floor.")

    for line in reasons:
        st.markdown(line)
    if not reasons:
        st.caption("Balanced value-based tweak to your core roster.")
    with st.expander("🔬 Deep Dive (compact)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Avg FP/G", f"{your_avg_fpts:.1f}")
            st.metric("Your Avg CV%", f"{your_avg_cv:.1f}%")
        with col2:
            st.metric("Their Avg FP/G", f"{their_avg_fpts:.1f}")
            st.metric("Their Avg CV%", f"{their_avg_cv:.1f}%")
        with col3:
            st.metric("FP/G Change", f"{fpts_diff:+.1f}")
            st.metric("CV% Change", f"{cv_change:+.1f}%", help="Negative = more consistent")

        if your_avg_cv < CONSISTENCY_VERY_MAX_CV:
            your_risk = "Low"
        elif your_avg_cv <= CONSISTENCY_MODERATE_MAX_CV:
            your_risk = "Moderate"
        else:
            your_risk = "High"
        if their_avg_cv < CONSISTENCY_VERY_MAX_CV:
            their_risk = "Low"
        elif their_avg_cv <= CONSISTENCY_MODERATE_MAX_CV:
            their_risk = "Moderate"
        else:
            their_risk = "High"
        st.caption(f"Your risk level: {your_risk} • Their risk level: {their_risk}")

    with st.expander("📆 Recent Form (YTD vs Last 7/15/30)", expanded=False):
        player_index = {}
        for team_df in rosters_by_team.values():
            if team_df is None or team_df.empty or "Player" not in team_df.columns:
                continue
            for _, row in team_df.iterrows():
                name = row.get("Player")
                if name and name not in player_index:
                    player_index[name] = row

        def _build_trend_line(player_name, is_outgoing):
            row = player_index.get(player_name)
            if row is None:
                return None
            base = row.get("Mean FPts")
            if base is None or pd.isna(base):
                return None
            spans = [
                ("Last 7", "L7 FPts"),
                ("Last 15", "L15 FPts"),
                ("Last 30", "L30 FPts"),
            ]
            for label, col in spans:
                if col in row.index:
                    val = row.get(col)
                    if val is None or pd.isna(val):
                        continue
                    delta = float(val) - float(base)
                    if abs(delta) >= 3.0:
                        direction = "📈" if delta > 0 else "📉"
                        side = "you give" if is_outgoing else "you get"
                        return f"{direction} {player_name} ({side}) — {label}: {val:.1f} vs YTD {base:.1f} ({delta:+.1f} FP/G)"
            return None

        lines = []
        for name in suggestion["you_give"]:
            text = _build_trend_line(name, is_outgoing=True)
            if text:
                lines.append(text)
        for name in suggestion["you_get"]:
            text = _build_trend_line(name, is_outgoing=False)
            if text:
                lines.append(text)
        if lines:
            for line in lines:
                st.markdown(line)
        else:
            st.caption("No significant recent FP/G swings (±3 FP/G) vs YTD for players in this trade.")

    with st.expander("📋 Roster Snapshot (top 10)", expanded=False):
        your_roster = rosters_by_team.get(your_team_name)
        opp_roster = rosters_by_team.get(suggestion["team"])
        if your_roster is None or your_roster.empty or opp_roster is None or opp_roster.empty:
            st.caption("Roster data not available for before/after view.")
            return

        your_before = your_roster.copy()
        opp_before = opp_roster.copy()

        if "Player" in your_before.columns:
            your_after = your_before[~your_before["Player"].isin(suggestion["you_give"])].copy()
        else:
            your_after = your_before.copy()
        incoming_dfs = []
        if "Player" in opp_before.columns:
            for player in suggestion["you_get"]:
                src = opp_before[opp_before["Player"] == player]
                if src.empty and "Player" in your_before.columns:
                    src = your_before[your_before["Player"] == player]
                if not src.empty:
                    incoming_dfs.append(src)
        if incoming_dfs:
            your_after = pd.concat([your_after] + incoming_dfs, ignore_index=True)

        if "Player" in opp_before.columns:
            opp_after = opp_before[~opp_before["Player"].isin(suggestion["you_get"])].copy()
        else:
            opp_after = opp_before.copy()
        opp_incoming_dfs = []
        if "Player" in your_before.columns:
            for player in suggestion["you_give"]:
                src = your_before[your_before["Player"] == player]
                if src.empty and "Player" in opp_before.columns:
                    src = opp_before[opp_before["Player"] == player]
                if not src.empty:
                    opp_incoming_dfs.append(src)
        if opp_incoming_dfs:
            opp_after = pd.concat([opp_after] + opp_incoming_dfs, ignore_index=True)

        def _prepare_roster(df, highlight_out, highlight_in):
            if df is None or df.empty:
                return pd.DataFrame()
            view = df.copy()
            if "Player" in view.columns:
                def _status(p):
                    if p in highlight_out:
                        return "OUT"
                    if p in highlight_in:
                        return "IN"
                    return ""
                view["Trade Status"] = view["Player"].apply(_status)
            for sort_col in ["Mean FPts", "FP/G"]:
                if sort_col in view.columns:
                    view = view.sort_values(sort_col, ascending=False)
                    break
            if len(view) > 10:
                view = view.head(10)
            cols = [c for c in ["Player", "Team", "Mean FPts", "FP/G", "GP", "Trade Status"] if c in view.columns]
            return view[cols] if cols else view

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Your Team**")
            st.caption("Top roster slots before and after this trade.")
            st.write("Before")
            st.dataframe(_prepare_roster(your_before, suggestion["you_give"], []), hide_index=True, use_container_width=True)
            st.write("After")
            st.dataframe(_prepare_roster(your_after, suggestion["you_give"], suggestion["you_get"]), hide_index=True, use_container_width=True)
        with col2:
            st.markdown(f"**{suggestion['team']}**")
            st.caption("Top roster slots before and after this trade.")
            st.write("Before")
            st.dataframe(_prepare_roster(opp_before, suggestion["you_get"], []), hide_index=True, use_container_width=True)
            st.write("After")
            st.dataframe(_prepare_roster(opp_after, suggestion["you_get"], suggestion["you_give"]), hide_index=True, use_container_width=True)

    with st.expander("📉 Full Trend & Risk Analysis (Trade Analysis tool)", expanded=False):
        has_analyzer = bool(st.session_state.get("trade_analyzer")) and st.session_state.get("combined_data") is not None
        if not has_analyzer:
            st.info(
                "To run full 7/14/30/60d analysis, first load a league dataset in the Trade Analysis tool "
                "from the main menu. Then reopen this tab."
            )
        else:
            your_code = CODE_BY_MANAGER.get(your_team_name, your_team_name)
            opp_name = suggestion["team"]
            opp_code = CODE_BY_MANAGER.get(opp_name, opp_name)
            trade_teams = {
                your_code: {p: opp_code for p in suggestion["you_give"]},
                opp_code: {p: your_code for p in suggestion["you_get"]},
            }
            label = f"{your_team_name} vs {opp_name} - {suggestion.get('pattern', '')} (Suggestion #{rank})"
            if st.button(
                "Run Full Trade Analysis for this suggestion",
                key=f"run_full_trade_analysis_{rank}_{opp_name}",
            ):
                results = run_trade_analysis(
                    trade_teams=trade_teams,
                    num_players=10,
                    trade_label=label,
                )
                if not results:
                    st.warning("No analysis results were produced. Check that trade data is loaded for this league.")
                else:
                    display_trade_results(results)


def _score_trade_fp_vs_cv(suggestion, risk_preference: int) -> float:
    """Compute a composite score for sorting trades by FP/G vs consistency preference.

    The slider provides a 0-100 weighting between weekly core FP gain and
    package-level CV% improvement (lower CV = more consistent). This helper
    remains UI-only and does not affect realism guards inside the engine.
    """
    try:
        core_gain = float(suggestion.get("value_gain", 0.0))
    except Exception:
        core_gain = 0.0

    your_cv_list = suggestion.get("your_cv") or []
    their_cv_list = suggestion.get("their_cv") or []

    def _avg(seq):
        return (sum(seq) / max(len(seq), 1)) if seq else 0.0

    your_avg_cv = _avg(your_cv_list)
    their_avg_cv = _avg(their_cv_list)
    # Positive consistency_gain means your side becomes more consistent
    consistency_gain = -(their_avg_cv - your_avg_cv)

    try:
        pref = int(risk_preference)
    except Exception:
        pref = 0
    pref = max(0, min(100, pref))
    weight_consistency = pref / 100.0
    weight_fp = 1.0 - weight_consistency

    return weight_fp * core_gain + weight_consistency * consistency_gain


def display_trade_suggestions_tab():
    """Embedded Trade Suggestions UI for use inside other pages (tabs)."""
    st.subheader("🤝 AI-Powered Trade Suggestions")
    st.markdown("Get intelligent trade recommendations based on exponential value calculations and consistency analysis.")

    try:
        from league_config import FANTRAX_DEFAULT_LEAGUE_ID
    except ImportError:
        FANTRAX_DEFAULT_LEAGUE_ID = ""

    league_id = st.text_input("League ID", value=FANTRAX_DEFAULT_LEAGUE_ID, key="tab_trade_suggest_league_id")
    if not league_id:
        st.warning("Please enter a league ID to get trade suggestions.")
        return

    cache_dir = get_cache_directory()
    cache_files = list(cache_dir.glob(f"player_game_log_full_*_{league_id}_*.json"))
    if not cache_files:
        st.error(f"No cached player data found for league {league_id}. Please run Bulk Scrape in Admin Tools first.")
        return

    selected_season = None
    try:
        seasons = db_store.get_league_available_seasons(league_id)
        if seasons:
            selected_season = seasons[0]
    except Exception:
        selected_season = None

    if not selected_season:
        season_set = set()
        for cf in cache_files:
            parts = cf.stem.split("_")
            if len(parts) >= 2:
                season_part = "_".join(parts[-2:])
                season = season_part.replace("_", "-")
                season_set.add(season)
        if season_set:
            selected_season = sorted(list(season_set), reverse=True)[0]

    if not selected_season:
        st.error("Could not determine a season to analyze for trade suggestions.")
        return

    rosters_by_team = _build_fantasy_team_view(league_id, cache_files, selected_season)
    if not rosters_by_team:
        st.error("Could not load team rosters. Make sure player data is properly formatted.")
        return

    st.success(f"✅ Loaded {len(rosters_by_team)} teams with player data for {selected_season}")

    st.markdown("---")
    st.markdown("## ⚙️ Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        your_team_name = st.selectbox(
            "Select Your Team",
            options=sorted(rosters_by_team.keys()),
            key="tab_your_team_select",
        )

    with col2:
        trade_patterns = st.multiselect(
            "Trade Patterns",
            options=[
                "1-for-1",
                "2-for-1",
                "1-for-2",
                "2-for-2",
                "3-for-1",
                "1-for-3",
                "3-for-2",
                "2-for-3",
                "3-for-3",
            ],
            default=["1-for-1", "2-for-1", "1-for-2", "2-for-2"],
            help="Select which trade patterns to consider",
            key="tab_trade_patterns",
        )

    with col3:
        min_value_gain = st.slider(
            "Min Value Gain",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=5.0,
            help=(
                "Minimum weekly core FP gain the engine will accept on your side. "
                "As a rough guide, 25 ≈ +1 FP/G across your core; 50 ≈ +2 FP/G."
            ),
            key="tab_min_value_gain",
        )

    with col4:
        trade_balance_level = st.slider(
            "Trade Balance (1=super strict, 50=super loose)",
            min_value=1,
            max_value=50,
            value=50,
            help=(
                "Controls how forgiving the engine is to the opponent. Lower = they must clearly benefit "
                "in FP/G+CV and package optics (fewer but more realistic trades). Higher = allows more "
                "lopsided/speculative ideas and relaxes some FP/G ratio guards."
            ),
            key="tab_trade_balance_level",
        )

    with st.expander("🔧 Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            available_teams = [t for t in sorted(rosters_by_team.keys()) if t != your_team_name]
            target_teams = st.multiselect(
                "Target Specific Teams (optional)",
                options=available_teams,
                help="Leave empty to consider all teams",
                key="tab_target_teams",
            )

            exclude_teams = st.multiselect(
                "Exclude Teams (optional)",
                options=available_teams,
                help="Teams you do NOT want to trade with",
                key="tab_exclude_teams",
            )

            max_suggestions = st.number_input(
                "Max Suggestions (engine)",
                min_value=5,
                max_value=25,
                value=15,
                step=5,
                key="tab_max_suggestions",
            )

            display_count = st.number_input(
                "Suggestions to display",
                min_value=5,
                max_value=25,
                value=15,
                step=5,
                key="tab_display_count",
            )

            max_per_opponent_display = st.number_input(
                "Max suggestions per opponent (display-only)",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help=(
                    "0 = no limit. >0 caps how many suggestions a single opponent can occupy "
                    "in the displayed list so one manager cannot dominate the top N slots."
                ),
                key="tab_max_per_opponent_display",
            )

        with col2:
            your_team_df = rosters_by_team[your_team_name]
            exclude_players = st.multiselect(
                "Exclude Your Players (untouchables)",
                options=sorted(your_team_df["Player"].tolist()),
                help="Players you don't want to trade away",
                key="tab_exclude_players",
            )

            include_players = st.multiselect(
                "Must-Include From Your Team (trade bait)",
                options=sorted(your_team_df["Player"].tolist()),
                help="Only show trades where at least one of these players is included on your side",
                key="tab_include_players",
            )

            other_players = sorted(
                p
                for team, df in rosters_by_team.items()
                if team != your_team_name
                for p in df["Player"].tolist()
            )
            target_opposing_players = st.multiselect(
                "Target Opposing Players",
                options=other_players,
                help="Only show trades where at least one of these players is included on the other side",
                key="tab_target_opposing_players",
            )
            exclude_opposing_players = st.multiselect(
                "Exclude Opposing Players",
                options=other_players,
                help="Opposing players you do NOT want to receive in trades",
                key="tab_exclude_opposing_players",
            )

            realism_min_opp_core = st.number_input(
                "Opponent min package FP/G advantage",
                min_value=-150.0,
                max_value=150.0,
                value=-150.0,
                step=1.0,
                help=(
                    "UI-level optics filter applied after the engine. For each trade we compare the average FP/G of "
                    "the players you give vs. the players you get. Positive values mean the opponent's package looks "
                    "better in raw FP/G. 0 means they must at least break even; 3 means you give them ~3 FP/G more "
                    "than you receive in the package."
                ),
                key="tab_min_opp_core_change",
            )

            risk_preference = st.slider(
                "Consistency vs FP/G emphasis",
                min_value=0,
                max_value=100,
                value=40,
                step=5,
                help=(
                    "Controls how suggestions are sorted: 0 = prioritize weekly core FP gain only; "
                    "100 = heavily favor trades that improve your consistency (lower CV%) even if FP gain is smaller."
                ),
                key="tab_fp_cv_preference",
            )

    suggestions_session_key = "tab_trade_suggestions_results"
    if st.button("🔍 Find Trade Suggestions (Tab)", type="primary", key="tab_find_trade_suggestions"):
        with st.spinner("Analyzing trade opportunities..."):
            set_trade_balance_preset(trade_balance_level)

            your_team_df = rosters_by_team[your_team_name]
            other_teams = {k: v for k, v in rosters_by_team.items() if k != your_team_name}

            suggestions = find_trade_suggestions(
                your_team=your_team_df,
                other_teams=other_teams,
                trade_patterns=trade_patterns,
                min_value_gain=min_value_gain,
                max_suggestions=max_suggestions,
                target_teams=target_teams if target_teams else None,
                exclude_players=exclude_players if exclude_players else None,
                include_players=include_players if include_players else None,
                exclude_teams=exclude_teams if exclude_teams else None,
                target_opposing_players=target_opposing_players if target_opposing_players else None,
                exclude_opposing_players=exclude_opposing_players if exclude_opposing_players else None,
            )

            if not suggestions:
                st.session_state[suggestions_session_key] = []
            else:
                filtered_suggestions = []
                for s in suggestions:
                    your_avg_fpts = sum(s["your_fpts"]) / max(len(s["your_fpts"]), 1)
                    their_avg_fpts = sum(s["their_fpts"]) / max(len(s["their_fpts"]), 1)
                    # From opponent optics: positive = you give them more FP/G than you get
                    opp_pkg_fp_advantage = your_avg_fpts - their_avg_fpts
                    if opp_pkg_fp_advantage >= realism_min_opp_core:
                        filtered_suggestions.append(s)
                st.session_state[suggestions_session_key] = filtered_suggestions

    filtered_suggestions = st.session_state.get(suggestions_session_key)
    if filtered_suggestions is None:
        # No search has been run yet
        pass
    elif not filtered_suggestions:
        st.warning(
            "No beneficial trades found with current filters, or all trades were filtered out by the "
            "opponent FP change threshold. Try loosening your criteria."
        )
    else:
        # Sort suggestions according to the current FP/G vs CV preference slider
        try:
            sorted_suggestions = sorted(
                filtered_suggestions,
                key=lambda s: _score_trade_fp_vs_cv(s, risk_preference),
                reverse=True,
            )
        except Exception:
            sorted_suggestions = filtered_suggestions

        st.success(f"✅ Found {len(sorted_suggestions)} trade suggestions after realism filter.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Suggestions", len(sorted_suggestions))
        with col2:
            best_gain = sorted_suggestions[0]["value_gain"]
            st.metric("Best Value Gain", f"{best_gain:.1f}")
        with col3:
            avg_gain = sum(s["value_gain"] for s in sorted_suggestions) / len(sorted_suggestions)
            st.metric("Avg Value Gain", f"{avg_gain:.1f}")
        with col4:
            pattern_counts = {}
            for s in sorted_suggestions:
                pattern_counts[s["pattern"]] = pattern_counts.get(s["pattern"], 0) + 1
            most_common = max(pattern_counts, key=pattern_counts.get)
            st.metric("Most Common Pattern", most_common)

        st.markdown("---")
        st.markdown("### 📊 Trade Suggestions")

        # Clamp display count to the number of available suggestions
        display_n = max(1, min(int(display_count), len(sorted_suggestions)))

        # Build an indexed list so we can preserve global rank while grouping by opponent
        indexed_suggestions = list(enumerate(sorted_suggestions, 1))

        # Optionally cap how many suggestions any single opponent can occupy
        max_per_opp = int(max_per_opponent_display or 0)
        if max_per_opp > 0:
            team_counts = {}
            pruned = []
            for rank, s in indexed_suggestions:
                team_name = s.get("team", "?")
                current = team_counts.get(team_name, 0)
                if current >= max_per_opp:
                    continue
                team_counts[team_name] = current + 1
                pruned.append((rank, s))
            indexed_suggestions = pruned

        # Trim to the overall display limit
        indexed_suggestions = indexed_suggestions[:display_n]

        # Group by opponent team while preserving the global ranking order
        suggestions_by_team = {}
        for rank, s in indexed_suggestions:
            team_name = s.get("team", "?")
            if team_name not in suggestions_by_team:
                suggestions_by_team[team_name] = []
            suggestions_by_team[team_name].append((rank, s))

        for team_name, items in suggestions_by_team.items():
            st.markdown(f"#### Trades with {team_name}")
            for rank, suggestion in items:
                your_avg_fpts = sum(suggestion["your_fpts"]) / max(len(suggestion["your_fpts"]), 1)
                their_avg_fpts = sum(suggestion["their_fpts"]) / max(len(suggestion["their_fpts"]), 1)
                opp_pkg_fp_advantage = your_avg_fpts - their_avg_fpts
                if opp_pkg_fp_advantage < 0:
                    fairness_tag = "You Favored"
                elif opp_pkg_fp_advantage >= realism_min_opp_core:
                    fairness_tag = "Opponent Favored"
                else:
                    fairness_tag = "Balanced"

                label = (
                    f"#{rank} - {suggestion['pattern']} with {suggestion['team']} "
                    f"(Value Gain: +{suggestion['value_gain']:.1f}) "
                    f"[{fairness_tag}]"
                )
                with st.expander(label, expanded=(rank <= 3)):
                    _display_trade_suggestion(suggestion, rank, rosters_by_team, your_team_name)

    with st.expander("📈 Exponential Value Curve", expanded=False):
        fpts_range = list(range(20, 101, 5))
        values = [calculate_exponential_value(f) for f in fpts_range]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpts_range,
                y=values,
                mode="lines+markers",
                name="Exponential Value",
                line=dict(color="#4a90e2", width=3),
                marker=dict(size=8),
            )
        )

        linear_values = [f * 5 for f in fpts_range]
        fig.add_trace(
            go.Scatter(
                x=fpts_range,
                y=linear_values,
                mode="lines",
                name="Linear Value (for comparison)",
                line=dict(color="gray", width=2, dash="dash"),
                opacity=0.5,
            )
        )

        fig.update_layout(
            title="Exponential vs Linear Value Scaling",
            xaxis_title="Fantasy Points per Game",
            yaxis_title="Player Value",
            height=400,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True, key="tab_exponential_curve_chart")
