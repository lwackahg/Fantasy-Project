"""UI components for playoff odds (Schedule Analysis -> Odds tab)."""

import streamlit as st
import plotly.express as px
from streamlit_compat import dataframe, plotly_chart

from logic.schedule_analysis import simulate_remaining_games, get_playoff_summary
from logic.playoff_odds import calculate_playoff_odds_with_threshold, get_team_playoff_scenarios


def display_playoff_odds(schedule_df):
    """Run Monte Carlo odds + threshold odds and render the Odds tab UI."""
    st.subheader("Playoff Odds")

    col_controls = st.columns(3)
    with col_controls[0]:
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
    with col_controls[2]:
        run_btn = st.button("Run Playoff Odds", type="primary")

    if run_btn:
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

    summary_df = st.session_state.get("playoff_odds_summary")
    threshold_df = st.session_state.get("playoff_odds_threshold")
    threshold_stats = st.session_state.get("playoff_odds_threshold_stats")
    if summary_df is None or summary_df.empty:
        st.info("Run the playoff odds simulation to see probabilities.")
        return

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
        st.info("Threshold-based odds not available yet. Run the simulation.")

    st.markdown("---")
    st.markdown("#### 🔍 Team Path to Playoffs")
    team_options = threshold_df["Team"].tolist() if threshold_df is not None and not threshold_df.empty else []
    selected_team = st.selectbox(
        "Select a team to view their path:",
        options=[""] + team_options,
        format_func=lambda x: f"🏀 {x}" if x else "Choose a team...",
    )
    if not selected_team:
        return

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
