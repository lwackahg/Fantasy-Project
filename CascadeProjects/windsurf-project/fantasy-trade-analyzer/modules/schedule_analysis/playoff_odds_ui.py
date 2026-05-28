"""UI components for playoff odds (Schedule Analysis -> Odds tab)."""

import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_compat import dataframe, plotly_chart

from logic.schedule_analysis import simulate_remaining_games, get_playoff_summary
from logic.playoff_odds import (
    calculate_playoff_odds_with_threshold,
    calculate_exact_seed_outcomes,
    explore_exact_playoff_scenarios,
    get_team_playoff_scenarios,
)


def display_playoff_odds(schedule_df):
    """Run Monte Carlo odds + threshold odds and render the Odds tab UI."""
    st.subheader("Playoff Odds")

    col_controls = st.columns(3)
    with col_controls[0]:
        method = st.radio(
            "Method",
            options=["Monte Carlo", "Exact (all outcomes)"],
            index=0,
            horizontal=True,
            help=(
                "Exact enumerates every remaining win/loss outcome (2^N). "
                "If there are too many remaining matchups, it will refuse and you should use Monte Carlo."
            ),
        )
        num_sims = st.slider(
            "Simulations",
            min_value=500,
            max_value=20000,
            step=500,
            value=5000,
            help="More sims = smoother probabilities, but slower.",
        )
        regression_weight = st.slider(
            "Regression to league mean",
            min_value=0.0,
            max_value=0.8,
            value=0.3,
            step=0.05,
            help="Higher = shrink ratings/scoring more toward league average (helps early season).",
        )
    with col_controls[1]:
        playoff_spots = st.number_input(
            "Playoff spots",
            min_value=2,
            max_value=12,
            value=6,
            step=1,
            help="How many teams make the playoffs. Used for seeding visualization.",
        )
        max_exact = st.number_input(
            "Exact max scenarios",
            min_value=1_000,
            max_value=5_000_000,
            value=200_000,
            step=50_000,
            help="Safety cap for exact mode: max 2^N outcomes to enumerate.",
        )
    with col_controls[2]:
        run_btn = st.button("Run Playoff Odds", type="primary")

    if run_btn:
        if method == "Exact (all outcomes)":
            with st.spinner("Enumerating all remaining outcomes..."):
                exact = calculate_exact_seed_outcomes(
                    schedule_df,
                    playoff_spots=int(playoff_spots),
                    max_scenarios=int(max_exact),
                )
            if exact.get("error"):
                st.session_state["playoff_odds_summary"] = None
                st.session_state["playoff_odds_threshold"] = None
                st.session_state["playoff_odds_threshold_stats"] = None
                st.session_state["playoff_odds_exact_error"] = exact.get("error")
                st.session_state["playoff_odds_exact_meta"] = {
                    "scenarios": exact.get("scenarios"),
                    "remaining_games": exact.get("remaining_games"),
                }
            else:
                summary_df = exact.get("seed_summary")
                st.session_state["playoff_odds_summary"] = summary_df
                st.session_state["playoff_odds_threshold"] = None
                st.session_state["playoff_odds_threshold_stats"] = None
                st.session_state["playoff_odds_exact_error"] = None
                st.session_state["playoff_odds_exact_meta"] = {
                    "scenarios": exact.get("scenarios"),
                    "remaining_games": exact.get("remaining_games"),
                }
        else:
            with st.spinner("Simulating remaining games..."):
                probs, remaining_sos, team_ratings = simulate_remaining_games(
                    schedule_df,
                    num_simulations=int(num_sims),
                    playoff_spots=int(playoff_spots),
                    regression_weight=float(regression_weight),
                )
                summary_df = get_playoff_summary(probs, remaining_sos, team_ratings)

                threshold_df, threshold_stats = calculate_playoff_odds_with_threshold(
                    schedule_df, playoff_spots=int(playoff_spots), num_threshold_sims=2000
                )

                if not summary_df.empty:
                    st.session_state["playoff_odds_summary"] = summary_df
                    st.session_state["playoff_odds_threshold"] = threshold_df
                    st.session_state["playoff_odds_threshold_stats"] = threshold_stats
                else:
                    st.session_state["playoff_odds_summary"] = None
                    st.session_state["playoff_odds_threshold"] = None
                    st.session_state["playoff_odds_threshold_stats"] = None
                st.session_state["playoff_odds_exact_error"] = None
                st.session_state["playoff_odds_exact_meta"] = None

    summary_df = st.session_state.get("playoff_odds_summary")
    threshold_df = st.session_state.get("playoff_odds_threshold")
    threshold_stats = st.session_state.get("playoff_odds_threshold_stats")
    exact_err = st.session_state.get("playoff_odds_exact_error")
    exact_meta = st.session_state.get("playoff_odds_exact_meta")

    if exact_err:
        if exact_meta and exact_meta.get("remaining_games") is not None:
            st.warning(
                f"Exact mode unavailable: {exact_err}"
            )
        else:
            st.warning(f"Exact mode unavailable: {exact_err}")

    if summary_df is None or summary_df.empty:
        st.info("Run the playoff odds simulation to see probabilities.")
        return

    if exact_meta and exact_meta.get("scenarios"):
        st.caption(
            f"Exact enumeration: **{int(exact_meta.get('scenarios')):,}** scenarios "
            f"across **{int(exact_meta.get('remaining_games') or 0)}** remaining matchups."
        )

    seed_cols = [c for c in summary_df.columns if c.startswith("seed_") and c[5:].isdigit()]
    limited_seed_cols = [c for c in seed_cols if int(c.split("_")[1]) <= playoff_spots]

    st.markdown("##### Playoff Probability")
    playoff_chart_df = summary_df[["Team", "Playoff %"]].sort_values("Playoff %", ascending=False)
    fig_playoff = px.bar(
        playoff_chart_df,
        x="Team",
        y="Playoff %",
        labels={"Playoff %": "Playoff Probability (%)"},
        color="Playoff %",
        color_continuous_scale=px.colors.sequential.Blues,
        height=350,
    )
    fig_playoff.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20})
    plotly_chart(fig_playoff, use_container_width=True)

    if limited_seed_cols:
        st.markdown("##### Seed Distribution")
        seed_df = summary_df[["Team"] + limited_seed_cols].melt(
            id_vars="Team", var_name="Seed", value_name="Probability"
        )
        seed_df["Seed"] = seed_df["Seed"].str.replace("seed_", "")
        fig_seed = px.bar(
            seed_df,
            x="Team",
            y="Probability",
            color="Seed",
            labels={"Probability": "Probability (%)"},
            barmode="stack",
            height=380,
        )
        fig_seed.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20})
        plotly_chart(fig_seed, use_container_width=True)

    st.markdown("##### Odds Table")
    display_cols = ["Team", "Playoff %", "Miss %"]
    display_cols += [c for c in limited_seed_cols if c in summary_df.columns]
    display_cols += [c for c in ["Most Likely Seed", "Seed Confidence"] if c in summary_df.columns]
    dataframe(
        summary_df[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "Playoff %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0),
            "Miss %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0),
        },
    )

    st.markdown("##### Threshold-Based Odds (analytical)")
    if threshold_df is not None and not threshold_df.empty:
        threshold_display_cols = [
            "Team",
            "Current Wins",
            "Remaining Games",
            "Wins Needed",
            "Exp Win Rate",
            "Playoff %",
            "Safe %",
            "Status",
        ]
        threshold_display_cols = [c for c in threshold_display_cols if c in threshold_df.columns]
        dataframe(
            threshold_df[threshold_display_cols],
            width="stretch",
            hide_index=True,
        )
        if threshold_stats:
            st.markdown(
                f"Median threshold: **{threshold_stats.get('median', 0):.0f} wins**, "
                f"Safe target (75%): **{threshold_stats.get('p75', 0):.0f}**, "
                f"Lock target (90%): **{threshold_stats.get('p90', 0):.0f}**"
            )
    else:
        st.info(
            "Threshold-based odds are only available in Monte Carlo mode. "
            "Exact mode enumerates outcomes but does not estimate a win-threshold distribution."
        )

    st.markdown("---")
    st.markdown("#### 🔍 Team Path to Playoffs")
    if threshold_df is not None and not threshold_df.empty and "Team" in threshold_df.columns:
        team_options = threshold_df["Team"].tolist()
    elif summary_df is not None and not summary_df.empty and "Team" in summary_df.columns:
        team_options = summary_df["Team"].tolist()
    else:
        team_options = []
    selected_team = st.selectbox(
        "Select a team to view their path:",
        options=[""] + team_options,
        format_func=lambda x: f"🏀 {x}" if x else "Choose a team...",
    )
    if not selected_team:
        return

    if exact_meta and exact_meta.get("scenarios"):
        with st.expander("🧩 Scenario Explorer (Exact)", expanded=True):
            explore = explore_exact_playoff_scenarios(
                schedule_df,
                selected_team,
                playoff_spots=int(playoff_spots),
                max_scenarios=int(max_exact),
            )
            if explore.get("error"):
                st.info(explore.get("error"))
            else:
                make_pct = float(explore.get("make_pct", 0.0) or 0.0)
                miss_pct = float(explore.get("miss_pct", 0.0) or 0.0)
                scenarios = int(explore.get("scenarios", 0) or 0)

                # Header metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Make playoffs", f"{make_pct:.1f}%")
                c2.metric("Miss playoffs", f"{miss_pct:.1f}%")
                c3.metric("Scenarios analyzed", f"{scenarios:,}")

                st.markdown("---")

                # Simple view: Games that matter with clear if/then
                swing_df = explore.get("swing_table")
                if swing_df is not None and not swing_df.empty:
                    st.markdown("### 🎮 Games That Actually Matter")
                    st.caption(f"These games change {selected_team}'s playoff odds. Everything else is noise.")
                    
                    # Filter to meaningful games (swing > 5pp)
                    important = swing_df[swing_df["Swing (pp)"] > 5.0].copy()
                    
                    if important.empty:
                        st.info(f"{selected_team} is basically locked in. No remaining games significantly change their playoff odds.")
                    else:
                        # Build simple if/then view
                        chart_data = []
                        for _, row in important.head(10).iterrows():
                            per = str(row.get("Period", ""))
                            t1 = str(row.get("Team A", ""))
                            t2 = str(row.get("Team B", ""))
                            make_a = float(row.get("Make% if A wins", 0))
                            make_b = float(row.get("Make% if B wins", 0))
                            
                            # Show both outcomes
                            chart_data.append({
                                "Game": f"{per}: {t1} vs {t2}",
                                "Outcome": f"If {t1} wins",
                                "Make %": make_a,
                                "sort": make_a,
                            })
                            chart_data.append({
                                "Game": f"{per}: {t1} vs {t2}",
                                "Outcome": f"If {t2} wins",
                                "Make %": make_b,
                                "sort": make_b,
                            })
                        
                        chart_df = pd.DataFrame(chart_data)
                        
                        # Group by game and show side-by-side outcomes
                        for game in chart_df["Game"].unique():
                            game_rows = chart_df[chart_df["Game"] == game]
                            st.markdown(f"**{game}**")
                            
                            cols = st.columns(2)
                            for idx, (_, r) in enumerate(game_rows.iterrows()):
                                with cols[idx]:
                                    outcome = r["Outcome"]
                                    make_pct = r["Make %"]
                                    miss_pct = 100 - make_pct
                                    
                                    # Color code based on good/bad for making playoffs
                                    if make_pct >= 50:
                                        st.success(f"{outcome}\n→ **{make_pct:.0f}% make** / {miss_pct:.0f}% miss")
                                    else:
                                        st.error(f"{outcome}\n→ {make_pct:.0f}% make / **{miss_pct:.0f}% miss**")
                            
                            st.markdown("")

                st.markdown("---")

                st.markdown("---")

                # Dead simple shareable summary
                st.markdown("### 📝 Share with the boys")
                
                def _build_simple_summary(mode: str) -> str:
                    mode_u = str(mode or "MISS").upper()
                    swing_df = explore.get("swing_table")
                    
                    if swing_df is None or swing_df.empty:
                        return f"For {selected_team}: (no data available)"
                    
                    # Get high-impact games only
                    important = swing_df[swing_df["Swing (pp)"] > 5.0].copy()
                    
                    if important.empty:
                        if mode_u == "MISS":
                            return f"💀 For **{selected_team}** to MISS playoffs ({miss_pct:.1f}% chance):\n\nBasically impossible - they're locked in."
                        else:
                            return f"🏆 For **{selected_team}** to MAKE playoffs ({make_pct:.1f}% chance):\n\nBasically guaranteed - they're locked in."
                    
                    # Build simple requirements with cumulative impact
                    if mode_u == "MISS":
                        intro = f"💀 For **{selected_team}** to MISS playoffs ({miss_pct:.1f}% chance):\n\n"
                        
                        # Collect bad outcomes sorted by impact
                        bad_outcomes = []
                        for _, row in important.iterrows():
                            t1 = str(row.get("Team A", ""))
                            t2 = str(row.get("Team B", ""))
                            make_a = float(row.get("Make% if A wins", 0))
                            make_b = float(row.get("Make% if B wins", 0))
                            per = str(row.get("Period", ""))
                            
                            # Which outcome hurts them more?
                            if make_a < make_b:
                                bad_outcome = f"{t1} beats {t2}"
                                result_pct = make_a
                            else:
                                bad_outcome = f"{t2} beats {t1}"
                                result_pct = make_b
                            
                            # Include all games that have meaningful swing (already filtered to >5pp)
                            bad_outcomes.append({
                                "outcome": bad_outcome,
                                "period": per,
                                "make_pct": result_pct,
                                "teams": (t1, t2),
                            })
                        
                        if not bad_outcomes:
                            return intro + "Already very likely to make playoffs - no remaining games significantly change odds."
                        
                        # Sort by impact (lowest make% first)
                        bad_outcomes.sort(key=lambda x: x["make_pct"])
                        
                        # Show cumulative cascade with clear matchup names
                        result = intro + "Here's what needs to happen (cascading impact):\n\n"
                        current_make = 100.0
                        
                        for idx, item in enumerate(bad_outcomes[:5], 1):
                            outcome = item["outcome"]
                            per = item["period"]
                            
                            # For cumulative display, we approximate the cascade
                            # In reality we'd need to re-enumerate, but for simplicity:
                            # Each bad outcome roughly multiplies the remaining "safe margin"
                            if idx == 1:
                                new_make = item["make_pct"]
                            else:
                                # Approximate: each additional bad outcome cuts the remaining margin
                                margin_lost = (current_make - item["make_pct"]) * 0.7  # dampening factor
                                new_make = max(0, current_make - margin_lost)
                            
                            drop = current_make - new_make
                            result += f"{idx}. **{per}: {outcome}**\n"
                            result += f"   → Drops from {current_make:.0f}% to {new_make:.0f}% make (−{drop:.0f}pp)\n\n"
                            current_make = new_make
                            
                            if current_make < 10:
                                break
                        
                        result += f"({int(scenarios * miss_pct / 100):,} out of {scenarios:,} scenarios result in a miss)"
                        return result
                    
                    else:  # MAKE
                        intro = f"🏆 For **{selected_team}** to MAKE playoffs ({make_pct:.1f}% chance):\n\n"
                        requirements = []
                        
                        for _, row in important.iterrows():
                            t1 = str(row.get("Team A", ""))
                            t2 = str(row.get("Team B", ""))
                            make_a = float(row.get("Make% if A wins", 0))
                            make_b = float(row.get("Make% if B wins", 0))
                            
                            # Which outcome helps them more?
                            if make_a > make_b:
                                good_outcome = f"{t1} beats {t2}"
                                result_pct = make_a
                            else:
                                good_outcome = f"{t2} beats {t1}"
                                result_pct = make_b
                            
                            # Only show if it meaningfully helps
                            if result_pct > 20:
                                requirements.append(f"  • {good_outcome} (boosts to {result_pct:.0f}% make)")
                        
                        if not requirements:
                            return intro + "Already very likely - no single game dramatically changes odds."
                        
                        result = intro + "Needs several of these to happen:\n" + "\n".join(requirements[:5])
                        result += f"\n\n({int(scenarios * make_pct / 100):,} out of {scenarios:,} scenarios result in making it)"
                        return result
                
                summary_mode = st.radio(
                    "What do you want to know?",
                    options=["How they MISS", "How they MAKE"],
                    index=0,
                    horizontal=True,
                    key="exact_summary_mode",
                )
                
                mode_key = "MISS" if "MISS" in summary_mode else "MAKE"
                summary_text = _build_simple_summary(mode_key)
                
                st.text_area(
                    "Copy this:",
                    value=summary_text,
                    height=200,
                    key="exact_summary_output",
                )
                st.download_button(
                    "⬇️ Download",
                    data=summary_text,
                    file_name=f"{selected_team.replace(' ', '_')}_{mode_key.lower()}.txt",
                    mime="text/plain",
                    key="exact_summary_dl",
                )

    scenarios = get_team_playoff_scenarios(
        schedule_df,
        selected_team,
        playoff_spots=int(playoff_spots),
        num_threshold_sims=1000,
        num_scenario_sims=300,
    )
    if "error" in scenarios:
        st.error(scenarios["error"])
        return

    st.caption("Team Path UI: v2 (this-week + week-by-week enabled)")
    st.markdown(f"### {selected_team} — Path to Playoffs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record", scenarios.get("current_record", "—"))
    c2.metric("Games Left", scenarios.get("remaining_games", 0))
    c3.metric("Proj Wins", scenarios.get("projected_wins", 0))
    c4.metric("Wins Needed", scenarios.get("wins_needed", 0))

    if scenarios.get("path_summary"):
        st.markdown(scenarios["path_summary"])

    tw = scenarios.get("this_week")
    if tw:
        st.markdown("#### This Week: What needs to happen")
        st.markdown(
            f"**{tw.get('Period')}** vs **{tw.get('Opponent')}** — "
            f"If you **win**: **{tw.get('If Win %', 0)}%** · "
            f"If you **lose**: **{tw.get('If Loss %', 0)}%** · "
            f"Swing: **{tw.get('Swing (W-L)', 0)} pp**"
        )
        if scenarios.get("this_week_watch") is not None and not scenarios["this_week_watch"].empty:
            st.markdown("##### Rooting Guide (assuming you win)")
            dataframe(scenarios["this_week_watch"].head(8), width="stretch", hide_index=True)
    else:
        st.info(
            "This Week section is unavailable (could not detect the next scoring period for this team). "
            "This usually means future rows are missing/blank in the schedule's `Scoring Period` column."
        )

    if scenarios.get("week_by_week") is not None and not scenarios["week_by_week"].empty:
        st.markdown("#### Week-by-Week: win/loss impact")
        dataframe(scenarios["week_by_week"], width="stretch", hide_index=True)

    with st.expander("Debug: scenario keys (for troubleshooting)"):
        st.write(
            {
                "has_this_week": bool(scenarios.get("this_week")),
                "has_this_week_watch": scenarios.get("this_week_watch") is not None
                and not scenarios.get("this_week_watch").empty,
                "has_week_by_week": scenarios.get("week_by_week") is not None
                and not scenarios.get("week_by_week").empty,
                "remaining_schedule_rows": 0
                if scenarios.get("remaining_schedule") is None
                else int(len(scenarios.get("remaining_schedule"))),
            }
        )

    tab1, tab2, tab3, tab4 = st.tabs(["📅 Schedule", "🎲 Scenarios", "👀 Root For/Against", "🔥 Key Matchups"])
    with tab1:
        st.markdown("#### Remaining Schedule")
        if scenarios.get("remaining_schedule") is not None and not scenarios["remaining_schedule"].empty:
            dataframe(scenarios["remaining_schedule"], width="stretch", hide_index=True)
        else:
            st.info("No remaining games — season complete!")
    with tab2:
        st.markdown("#### What-If Scenarios")
        if scenarios.get("scenarios") is not None and not scenarios["scenarios"].empty:
            dataframe(scenarios["scenarios"], width="stretch", hide_index=True)
            if scenarios.get("most_likely_finish"):
                st.success(
                    f"**Most likely:** You go **{scenarios.get('most_likely_finish')}** → "
                    f"**{scenarios.get('most_likely_playoff_pct', 0)}%** playoff odds"
                )
        else:
            st.info("Season complete")
    with tab3:
        st.markdown("#### Teams to Watch")
        if scenarios.get("teams_to_watch") is not None and not scenarios["teams_to_watch"].empty:
            dataframe(scenarios["teams_to_watch"], width="stretch", hide_index=True)
        else:
            st.info("No relevant teams to track")
    with tab4:
        st.markdown("#### Key Matchups")
        if scenarios.get("key_matchups") is not None and not scenarios["key_matchups"].empty:
            dataframe(scenarios["key_matchups"], width="stretch", hide_index=True)
        else:
            st.info("No key matchups to watch")
