import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
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
from modules.historical_ytd_downloader.logic import load_and_compare_seasons, get_available_seasons

# Player similarity imports
from modules.trade_suggestions.player_similarity import (
    calculate_league_stats,
    trade_package_similarity,
    find_similar_players,
    create_player_vector,
    player_similarity_score,
    weighted_euclidean_distance,
)
from modules.trade_suggestions.similarity_viz import (
    render_player_comparison_radar,
    render_multi_player_radar,
)

CODE_BY_MANAGER = {v: k for k, v in TEAM_MAPPINGS.items()}

def _display_trade_suggestion(suggestion, rank, rosters_by_team, your_team_name, yoy_index=None):
    """Display a single trade suggestion with detailed impact metrics."""
    core_size_approx = 8.0
    min_games = 25
    weekly_core_fp_change = suggestion["value_gain"]
    core_ppg_change = weekly_core_fp_change / min_games
    opp_weekly_core_fp_change = suggestion.get("opp_core_gain", 0)

    st.markdown(f"### 🔄 Trade Impact: **{weekly_core_fp_change:+.1f} weekly core FP** for you")
    st.caption(
        f"Your core FP/G changes by ~{core_ppg_change:+.2f} across your top {int(core_size_approx)} players"
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
        st.dataframe(give_df, hide_index=True, width="stretch")
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
        st.dataframe(get_df, hide_index=True, width="stretch")
        their_avg_fpts = sum(suggestion["their_fpts"]) / max(len(suggestion["their_fpts"]), 1)
        st.caption(f"Package avg: {their_avg_fpts:.1f} FP/G")

    st.markdown("---")
    st.markdown("#### 📊 Weekly Core FP Impact")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Your Change",
            x=["Weekly Core FP"],
            y=[weekly_core_fp_change],
            marker_color="#4CAF50",
            text=[f"{weekly_core_fp_change:+.1f}"],
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
    st.plotly_chart(fig, width="stretch", key=f"trade_value_chart_{rank}")

    if weekly_core_fp_change > 30:
        st.success("🟢 **Excellent Trade** - Major weekly core FP upgrade!")
    elif weekly_core_fp_change > 15:
        st.success("🟢 **Strong Trade** - Solid weekly core FP gain")
    elif weekly_core_fp_change > 5:
        st.info("🟡 **Decent Trade** - Modest weekly core FP improvement")
    elif weekly_core_fp_change >= 0:
        st.info("🟡 **Marginal Trade** - Small weekly core FP gain")
    elif weekly_core_fp_change > -5:
        st.info("🟠 **Trade-Off:** Small weekly core FP loss – may be worth it if you gain consistency or positional fit")
    else:
        st.warning("🔴 **Core FP Loss:** You give up weekly core FP – only pursue if other factors clearly outweigh the downgrade")

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
    if weekly_core_fp_change > 0:
        reasons.append(
            f"📈 Core upgrade: your top ~{int(core_size_approx)} players gain ~{core_ppg_change:.2f} FP/G "
            f"({weekly_core_fp_change:+.1f} weekly core FP)."
        )
    elif weekly_core_fp_change < 0:
        if weekly_core_fp_change > -5 and cv_change < -5:
            reasons.append(
                f"🧩 Consistency trade-off: your core gives up ~{abs(core_ppg_change):.2f} FP/G "
                f"({weekly_core_fp_change:.1f} weekly core FP) in exchange for a more stable, less volatile roster."
            )
        else:
            reasons.append(
                f"📉 Core downgrade: your top ~{int(core_size_approx)} players lose ~{abs(core_ppg_change):.2f} FP/G "
                f"({weekly_core_fp_change:.1f} weekly core FP)."
            )
    else:
        reasons.append(
            f"⚖️ Core neutral: your top ~{int(core_size_approx)} players stay roughly flat (~{core_ppg_change:+.2f} FP/G, "
            f"{weekly_core_fp_change:+.1f} weekly core FP)."
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

    with st.expander("📆 Recent Form (YTD vs Last 7/14/30)", expanded=False):
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
                ("Last 14", "L14 FPts"),
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

    st.markdown("#### 🗣️ Talking points for your opponent")
    pitch_points = []
    if opp_weekly_core_fp_change > 0:
        pitch_points.append(
            f"Emphasize that this actually **increases their weekly core FP** by ~{opp_weekly_core_fp_change:.1f}."
        )
    elif opp_weekly_core_fp_change > -10:
        pitch_points.append(
            f"Frame it as mostly a **fit / consolidation move** – their weekly core FP only changes by ~{opp_weekly_core_fp_change:.1f}."
        )
    opp_pkg_fp_advantage = your_avg_fpts - their_avg_fpts
    if opp_pkg_fp_advantage > 2:
        pitch_points.append(
            f"From their side the package looks good: they gain about {opp_pkg_fp_advantage:.1f} FP/G in what they receive."
        )
    opp_cv_change = your_avg_cv - their_avg_cv
    if opp_cv_change < -3:
        pitch_points.append("Highlight that their roster becomes **more consistent overall** (lower Avg Player CV%).")
    elif opp_cv_change > 3:
        pitch_points.append("Pitch it as them taking on **more upside / boom-bust profiles** if they like variance.")
    if pattern in ("1-for-2", "1-for-3"):
        pitch_points.append("Sell it as a **depth play** for them: they turn one slot into multiple playable pieces.")
    elif pattern in ("2-for-1", "3-for-1"):
        pitch_points.append("Sell it as **consolidation**: they turn several pieces into one clear core starter and a free roster spot.")
    if any("📈" in line and "(you give)" in line for line in lines):
        pitch_points.append("Point out any 📈 lines under **You Give** – those show players you're sending who are on recent heaters vs YTD.")
    if any("📉" in line and "(you get)" in line for line in lines):
        pitch_points.append("Mention 📉 lines under **You Get** – those are their players who have been underperforming YTD that you're willing to buy low on.")
    if yoy_index:
        # YoY sell-high candidates you are sending (they receive)
        yoy_sell_high = []
        for name in suggestion.get("you_give", []):
            info = yoy_index.get(name)
            if not info:
                continue
            delta = info.get("delta")
            if delta is None or pd.isna(delta) or delta < 5.0:
                continue
            prev_fp = info.get("prev")
            curr_fp = info.get("curr")
            if prev_fp is None or curr_fp is None or pd.isna(prev_fp) or pd.isna(curr_fp):
                continue
            yoy_sell_high.append((name, prev_fp, curr_fp, delta))
        if yoy_sell_high:
            label_parts = [
                f"{n} ({prev:.1f}→{curr:.1f} FP/G)"
                for n, prev, curr, d in yoy_sell_high
            ]
            pitch_points.append(
                "You're offering them players in clear YoY upswing: "
                + ", ".join(label_parts)
                + ". Frame it as them adding established improvers while you reset around different pieces."
            )
        # YoY buy-low candidates you are receiving (their players)
        yoy_buy_low = []
        for name in suggestion.get("you_get", []):
            info = yoy_index.get(name)
            if not info:
                continue
            delta = info.get("delta")
            if delta is None or pd.isna(delta) or delta > -5.0:
                continue
            prev_fp = info.get("prev")
            curr_fp = info.get("curr")
            if prev_fp is None or curr_fp is None or pd.isna(prev_fp) or pd.isna(curr_fp):
                continue
            yoy_buy_low.append((name, prev_fp, curr_fp, delta))
        if yoy_buy_low:
            label_parts = [
                f"{n} ({prev:.1f}→{curr:.1f} FP/G)"
                for n, prev, curr, d in yoy_buy_low
            ]
            pitch_points.append(
                "You can present this as moving off potential regression risk: "
                + ", ".join(label_parts)
                + ". You're willing to take on the bounce-back gamble so they don't have to."
            )
    if pitch_points:
        for text in pitch_points:
            st.markdown(f"- {text}")
    else:
        st.caption("No obvious selling angles detected beyond core value; lean on team fit and positional needs.")

    with st.expander("🎯 Trade Framework Analysis", expanded=False):
        st.markdown("### Understanding this trade from multiple angles")
        
        st.markdown("#### 📦 Package-level comparison")
        pkg_bullets = []
        if fpts_diff > 5:
            pkg_bullets.append(f"**Package FP/G advantage**: You're receiving players averaging **{fpts_diff:.1f} FP/G more** than what you're sending. On paper, the incoming package looks clearly stronger per roster slot.")
        elif fpts_diff < -5:
            pkg_bullets.append(f"**Package FP/G disadvantage**: You're giving up players averaging **{abs(fpts_diff):.1f} FP/G more** than what you're receiving. This is a consolidation/fit play where you're paying a package tax.")
        else:
            pkg_bullets.append(f"**Package FP/G roughly balanced**: The players you're sending and receiving are similar in per-slot production (~{abs(fpts_diff):.1f} FP/G difference).")
        
        if cv_change < -5:
            pkg_bullets.append(f"**Consistency upgrade**: The incoming package is noticeably **more consistent** (CV% drops by {abs(cv_change):.1f}%). You're trading volatility for reliability.")
        elif cv_change > 5:
            pkg_bullets.append(f"**Consistency downgrade**: The incoming package is **more volatile** (CV% rises by {cv_change:.1f}%). You're adding boom-bust upside but a shakier floor.")
        
        for bullet in pkg_bullets:
            st.markdown(f"- {bullet}")
        
        st.markdown("#### 🏗️ Roster construction impact")
        roster_bullets = []
        if weekly_core_fp_change > 20:
            roster_bullets.append(f"**Major core upgrade**: Your top ~{int(core_size_approx)} players gain **+{weekly_core_fp_change:.1f} weekly core FP** (~+{core_ppg_change:.2f} FP/G per core slot). This significantly raises your team's ceiling.")
        elif weekly_core_fp_change > 10:
            roster_bullets.append(f"**Solid core upgrade**: Your core gains **+{weekly_core_fp_change:.1f} weekly FP** (~+{core_ppg_change:.2f} FP/G). Meaningful improvement to your starting lineup strength.")
        elif weekly_core_fp_change > 0:
            roster_bullets.append(f"**Modest core upgrade**: Your core gains **+{weekly_core_fp_change:.1f} weekly FP** (~+{core_ppg_change:.2f} FP/G). Small but positive improvement to your rotation.")
        
        if pattern in ("2-for-1", "3-for-1"):
            roster_bullets.append(f"**Consolidation trade**: You're turning {pattern.split('-')[0]} rotation pieces into 1 stronger player. This raises your per-game ceiling but reduces depth. You'll need to fill the open roster spot(s) via waivers or trades.")
        elif pattern in ("1-for-2", "1-for-3"):
            roster_bullets.append(f"**Depth play**: You're turning 1 player into {pattern.split('-')[2]} pieces. This lowers your per-slot ceiling but gives you more bodies for streaming, injury insurance, and total season games.")
        
        for bullet in roster_bullets:
            st.markdown(f"- {bullet}")
        
        st.markdown("#### ⚖️ Strategic trade-offs")
        strategy_bullets = []
        
        if your_avg_fpts > 70 and their_avg_fpts < 65:
            strategy_bullets.append("**Star-for-depth bet**: You're acquiring high-end talent at the cost of depth. This can win playoff weeks if your star performs, but leaves you vulnerable if they slump or get injured.")
        elif your_avg_fpts < 65 and their_avg_fpts > 70:
            strategy_bullets.append("**Depth-for-star bet**: You're giving up a high-end piece for multiple rotation players. This can stabilize your floor and total season production, but you lose the ceiling of a true difference-maker.")
        
        if abs(cv_change) > 5:
            if cv_change < 0:
                strategy_bullets.append(f"**Risk profile shift**: You're becoming **more consistent** (CV% drops {abs(cv_change):.1f}%). Better for risk-averse managers who want a stable floor, but you may sacrifice some explosive upside weeks.")
            else:
                strategy_bullets.append(f"**Risk profile shift**: You're becoming **more volatile** (CV% rises {cv_change:.1f}%). Better for managers chasing upside and willing to absorb boom-bust swings.")
        
        if opp_weekly_core_fp_change < -10:
            strategy_bullets.append(f"**Opponent sacrifice**: Your opponent loses **{abs(opp_weekly_core_fp_change):.1f} weekly core FP**. They're making a strategic gamble (maybe betting on fit, schedule, or different roster construction philosophy). This makes the trade harder to sell but potentially more valuable if they accept.")
        elif opp_weekly_core_fp_change > 5:
            strategy_bullets.append(f"**Win-win structure**: Your opponent also gains **+{opp_weekly_core_fp_change:.1f} weekly core FP**. Both sides improve in different ways, making this easier to negotiate.")
        
        if strategy_bullets:
            for bullet in strategy_bullets:
                st.markdown(f"- {bullet}")
        else:
            st.caption("This trade is primarily about fit and positional needs rather than major strategic shifts.")
        
        st.markdown("---")
        st.markdown("#### 💡 Bottom line")
        if weekly_core_fp_change > 15 and fpts_diff > 3:
            st.success("**Strong trade for you**: You gain both core value and package FP/G. The math clearly favors your side.")
        elif weekly_core_fp_change > 10 and fpts_diff < -3:
            st.info("**Strategic overpay**: You're paying a package FP/G tax, but your core still strengthens. This is a bet on fit and roster construction over raw per-slot value.")
        elif weekly_core_fp_change > 5:
            st.info("**Modest upgrade**: You gain some core value. Whether it's worth it depends on your team needs, schedule, and risk tolerance.")
        else:
            st.info("**Marginal trade**: Small core gain. This is more about fit, positional needs, or strategic roster reshaping than a clear value win.")

    # Player Similarity Analysis Section
    with st.expander("🔍 Player Similarity Analysis", expanded=False):
        st.markdown("### Profile-based trade analysis")
        
        # Get league stats for normalization
        try:
            # Build combined player data from rosters
            all_roster_dfs = [df for df in rosters_by_team.values() if df is not None and not df.empty]
            if all_roster_dfs:
                combined_df = pd.concat(all_roster_dfs, ignore_index=True)
                if 'Player' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['Player'])
                
                league_means, league_stds = calculate_league_stats(combined_df)
                
                # Get player rows for the trade
                give_players = []
                get_players = []
                
                for player_name in suggestion["you_give"]:
                    player_rows = combined_df[combined_df['Player'] == player_name]
                    if not player_rows.empty:
                        give_players.append(player_rows.iloc[0])
                
                for player_name in suggestion["you_get"]:
                    player_rows = combined_df[combined_df['Player'] == player_name]
                    if not player_rows.empty:
                        get_players.append(player_rows.iloc[0])
                
                if give_players and get_players:
                    # Calculate package similarity
                    pkg_analysis = trade_package_similarity(
                        give_players, get_players, league_means, league_stds
                    )
                    
                    st.markdown("#### 📊 Package Profile Comparison")
                    
                    sim_cols = st.columns(4)
                    with sim_cols[0]:
                        st.metric("Package Similarity", f"{pkg_analysis['similarity']:.1f}%",
                                  help="How similar the trade packages are in overall profile")
                    with sim_cols[1]:
                        fpg_change = pkg_analysis.get('fpg_change', 0)
                        st.metric("FP/G Profile", f"{fpg_change:+.2f}σ",
                                  help="Z-score change in FP/G dimension")
                    with sim_cols[2]:
                        cons_change = pkg_analysis.get('consistency_change', 0)
                        st.metric("Consistency", f"{cons_change:+.2f}σ",
                                  help="Z-score change in consistency (positive = more consistent)")
                    with sim_cols[3]:
                        value_change = pkg_analysis.get('value_change', 0)
                        st.metric("Value Profile", f"{value_change:+.2f}σ",
                                  help="Z-score change in composite value")
                    
                    # Interpretation
                    if pkg_analysis['similarity'] > 80:
                        st.success("📊 **Very similar packages** - This is a balanced swap of comparable players")
                    elif pkg_analysis['similarity'] > 60:
                        st.info("📊 **Moderately similar packages** - Some profile differences but reasonable swap")
                    elif pkg_analysis['similarity'] > 40:
                        st.warning("📊 **Different profiles** - You're trading for a different type of player")
                    else:
                        st.warning("📊 **Very different profiles** - Major shift in player archetype")
                    
                    # Show radar comparison if 1-for-1 or small trade
                    if len(give_players) <= 2 and len(get_players) <= 2:
                        st.markdown("#### 🎯 Player Profile Radar")
                        all_trade_players = give_players + get_players
                        if len(all_trade_players) >= 2:
                            fig = render_multi_player_radar(all_trade_players, league_means, league_stds)
                            st.plotly_chart(fig, use_container_width=True, key=f"radar_{rank}")
                    
                    # Find similar players to what you're giving up
                    st.markdown("#### 🔄 Similar Players to What You're Giving")
                    for give_player in give_players[:2]:  # Limit to first 2
                        player_name = give_player.get('Player', 'Unknown')
                        similar = find_similar_players(
                            player_name, combined_df, league_means, league_stds,
                            n=3, exclude_same_team=True
                        )
                        if similar:
                            similar_names = [f"{name} ({score:.0f}%)" for name, score, _ in similar]
                            st.caption(f"**{player_name}** → Similar: {', '.join(similar_names)}")
                else:
                    st.caption("Unable to find player data for similarity analysis.")
            else:
                st.caption("No roster data available for similarity analysis.")
        except Exception as e:
            st.caption(f"Similarity analysis unavailable: {str(e)}")

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
            st.dataframe(_prepare_roster(your_before, suggestion["you_give"], []), hide_index=True, width="stretch")
            st.write("After")
            st.dataframe(_prepare_roster(your_after, suggestion["you_give"], suggestion["you_get"]), hide_index=True, width="stretch")
        with col2:
            st.markdown(f"**{suggestion['team']}**")
            st.caption("Top roster slots before and after this trade.")
            st.write("Before")
            st.dataframe(_prepare_roster(opp_before, suggestion["you_get"], []), hide_index=True, width="stretch")
            st.write("After")
            st.dataframe(_prepare_roster(opp_after, suggestion["you_get"], suggestion["you_give"]), hide_index=True, width="stretch")

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
                    num_players=8,
                    trade_label=label,
                )
                if not results:
                    st.warning("No analysis results were produced. Check that trade data is loaded for this league.")
                else:
                    display_trade_results(results)


def _estimate_ytd_fp_change_for_suggestion(
    suggestion,
    rosters_by_team,
    your_team_name: str,
    top_n: int = 8,
) -> float:
    your_roster = rosters_by_team.get(your_team_name)
    opp_roster = rosters_by_team.get(suggestion.get("team"))
    if your_roster is None or your_roster.empty or opp_roster is None or opp_roster.empty:
        return 0.0
    if "Player" not in your_roster.columns:
        return 0.0

    your_before = your_roster.copy()
    opp_before = opp_roster.copy()

    you_give = suggestion.get("you_give") or []
    you_get = suggestion.get("you_get") or []

    your_after = your_before
    if you_give:
        your_after = your_after[~your_after["Player"].isin(you_give)].copy()

    incoming_dfs = []
    for player in you_get:
        src = opp_before[opp_before["Player"] == player]
        if src.empty:
            for df in rosters_by_team.values():
                if df is None or df.empty or "Player" not in df.columns:
                    continue
                match = df[df["Player"] == player]
                if not match.empty:
                    src = match
                    break
        if not src.empty:
            incoming_dfs.append(src)
    if incoming_dfs:
        your_after = pd.concat([your_after] + incoming_dfs, ignore_index=True)

    def _avg_top(df):
        if df is None or df.empty:
            return 0.0
        sort_col = None
        if "Mean FPts" in df.columns:
            sort_col = "Mean FPts"
        elif "FP/G" in df.columns:
            sort_col = "FP/G"
        if sort_col is None:
            return 0.0
        top = df.sort_values(sort_col, ascending=False).head(top_n)
        val_col = "FP/G" if "FP/G" in top.columns else sort_col
        vals = top[val_col].dropna()
        if vals.empty:
            return 0.0
        return float(vals.mean())

    before = _avg_top(your_before)
    after = _avg_top(your_after)
    return after - before


def _score_trade_fp_vs_cv(
    suggestion,
    risk_preference: int,
    rosters_by_team=None,
    your_team_name: str = "",
    min_ytd_fp_change: float = -2.0,
) -> float:
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

    base_score = weight_fp * core_gain + weight_consistency * consistency_gain

    if rosters_by_team is not None and your_team_name:
        try:
            ytd_fp_change = _estimate_ytd_fp_change_for_suggestion(
                suggestion,
                rosters_by_team,
                your_team_name,
                top_n=8,
            )
        except Exception:
            ytd_fp_change = 0.0

        try:
            threshold = float(min_ytd_fp_change)
        except Exception:
            threshold = -2.0

        if ytd_fp_change < threshold:
            base_score -= 1000.0
        else:
            base_score += ytd_fp_change * 5.0

    return base_score


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

    yoy_index = None
    try:
        league_ids_env = os.environ.get("FANTRAX_LEAGUE_IDS", "")
        league_names_env = os.environ.get("FANTRAX_LEAGUE_NAMES", "")
        ids = [i.strip() for i in league_ids_env.split(",") if i.strip()]
        names = [n.strip() for n in league_names_env.split(",") if n.strip()]
        mapping = dict(zip(ids, names))
        raw_name = mapping.get(league_id, "")
        if raw_name:
            league_name = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name.strip().replace(" ", "_"))
            seasons_all = get_available_seasons()
            previous_seasons = [s for s in seasons_all if s != selected_season]
            if previous_seasons:
                recent_season = selected_season
                prev_season = previous_seasons[0]
                comparison_df = load_and_compare_seasons(league_name, [recent_season, prev_season])
                if comparison_df is not None:
                    curr_col = f"FP/G_{recent_season}"
                    prev_col = f"FP/G_{prev_season}"
                    change_col = f"YoY_Change_{recent_season}_vs_{prev_season}"
                    if curr_col in comparison_df.columns and prev_col in comparison_df.columns:
                        yoy_index = {}
                        for _, row in comparison_df.iterrows():
                            name = row.get("Player")
                            if not name:
                                continue
                            curr = row.get(curr_col)
                            prev_fp = row.get(prev_col)
                            if pd.isna(curr) or pd.isna(prev_fp):
                                continue
                            delta = row.get(change_col) if change_col in comparison_df.columns else None
                            if delta is None or pd.isna(delta):
                                delta = curr - prev_fp
                            yoy_index[str(name)] = {
                                "curr": float(curr),
                                "prev": float(prev_fp),
                                "delta": float(delta),
                            }
    except Exception:
        yoy_index = None

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

    min_ytd_fp_change = -2.0

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

            min_ytd_fp_change = st.number_input(
                "Min YTD FP/G change (approx top 8)",
                min_value=-15.0,
                max_value=5.0,
                value=-2.0,
                step=0.5,
                help=(
                    "Approximate FP/G tax across your top ~8 you are willing to pay. "
                    "Negative values allow some FP/G loss for consolidation; 0 shows only FP/G-neutral/positive trades."
                ),
                key="tab_min_ytd_fp_change",
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

    with st.expander("⚙️ Player Performance Assumptions (Override FP/G)", expanded=False):
        st.caption("Select players to manually set their assumed FP/G for trade calculations.")
        
        all_players_set = set()
        if rosters_by_team:
            for df in rosters_by_team.values():
                if df is not None and "Player" in df.columns:
                    all_players_set.update(df["Player"].dropna().tolist())
        all_players_sorted = sorted(list(all_players_set))
        
        selected_override_players = st.multiselect(
            "Select Players to Override",
            options=all_players_sorted,
            key="tab_override_players_select"
        )
        
        player_fpts_overrides = {}
        if selected_override_players:
            st.markdown("##### Set Assumed FP/G")
            cols = st.columns(3)
            for i, player in enumerate(selected_override_players):
                current_fpts = 0.0
                for df in rosters_by_team.values():
                    if df is not None and "Player" in df.columns:
                        match = df[df["Player"] == player]
                        if not match.empty:
                            val = match.iloc[0].get("Mean FPts")
                            if val is not None and not pd.isna(val):
                                current_fpts = float(val)
                            break
                
                with cols[i % 3]:
                    new_fpts = st.number_input(
                        f"{player}",
                        value=current_fpts,
                        step=0.1,
                        format="%.1f",
                        key=f"override_fpts_{player}"
                    )
                    player_fpts_overrides[player] = new_fpts

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
                player_fpts_overrides=player_fpts_overrides if "player_fpts_overrides" in locals() and player_fpts_overrides else None,
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
            "No beneficial trades found with current filters."
        )
    else:
        # Sort suggestions according to the current FP/G vs CV preference slider
        # and an approximate YTD top-10 FP/G change derived from current rosters.
        try:
            sorted_suggestions = sorted(
                filtered_suggestions,
                key=lambda s: _score_trade_fp_vs_cv(
                    s,
                    risk_preference,
                    rosters_by_team=rosters_by_team,
                    your_team_name=your_team_name,
                    min_ytd_fp_change=min_ytd_fp_change,
                ),
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
                    _display_trade_suggestion(suggestion, rank, rosters_by_team, your_team_name, yoy_index=yoy_index)

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

        st.plotly_chart(fig, width="stretch", key="tab_exponential_curve_chart")
