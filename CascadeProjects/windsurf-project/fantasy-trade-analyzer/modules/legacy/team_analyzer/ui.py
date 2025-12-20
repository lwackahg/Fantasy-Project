"""UI components for the Team Analyzer feature."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_compat import plotly_chart
from streamlit_compat import dataframe
from .logic import calculate_team_stats, calculate_league_ranks, STAT_CATEGORIES
from modules.trade_analysis.consistency_integration import (
	enrich_roster_with_consistency,
	build_league_consistency_index,
)
from modules.player_value.logic import build_player_value_profiles

try:
	from league_config import FANTRAX_DEFAULT_LEAGUE_ID
except ImportError:
	FANTRAX_DEFAULT_LEAGUE_ID = ""

def display_team_analyzer():
    """Main function to display the Team Analyzer page components."""
    st.markdown("Analyze any team's categorical strengths and weaknesses to identify trade opportunities.")

    # Get necessary data from session state
    combined_data = st.session_state.get('combined_data')
    data_ranges = st.session_state.get('data_ranges', {})
    teams = sorted(combined_data['Fantasy_Manager'].unique())
    time_ranges = list(data_ranges.keys())

    # --- User Selections ---
    col1, col2 = st.columns(2)
    with col1:
        selected_team = st.selectbox("Select a Team to Analyze", options=teams)
    with col2:
        # Ensure 'YTD' is the default if available
        default_time_range_index = time_ranges.index('YTD') if 'YTD' in time_ranges else 0
        selected_range = st.selectbox("Select Time Range", options=time_ranges, index=default_time_range_index)

    if not selected_team or not selected_range:
        st.info("Please select a team and time range to begin analysis.")
        return

    # --- Analysis and Visualization ---
    with st.spinner(f"Analyzing {selected_team} for the {selected_range} range..."):
        # Perform calculations
        team_stats = calculate_team_stats(combined_data, selected_range)
        league_ranks = calculate_league_ranks(team_stats)

        if league_ranks.empty:
            st.warning(f"No data available for the selected time range: {selected_range}")
            return

        team_ranks = league_ranks.loc[selected_team]

        # --- Radar Chart ---
        st.subheader(f"Categorical Ranks for {selected_team}")

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=team_ranks.values,
            theta=team_ranks.index,
            fill='toself',
            name=selected_team,
            hovertemplate='<b>%{theta}</b><br>Rank: %{r}<extra></extra>'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[len(teams), 1]  # Invert axis so 1 is on the outside
                )
            ),
            showlegend=False,
            height=500
        )

        plotly_chart(fig, width="stretch")
        st.caption("The outer edge represents a rank of 1 (best in the league), while the center is the worst.")

        # --- Ranks Table ---
        with st.expander("View Full League Rankings"):
            dataframe(league_ranks.style.format("{:}"), width="stretch")

        # --- Player Consistency & Value ---
        with st.expander(f"Player Consistency & Value for {selected_team}"):
            league_id = st.session_state.get("league_id", FANTRAX_DEFAULT_LEAGUE_ID)
            if not league_id:
                st.info("League ID not available; consistency and value metrics require a league context.")
                return

            team_range_df = combined_data[
                (combined_data["Fantasy_Manager"] == selected_team)
                & (combined_data["Timestamp"] == selected_range)
            ].reset_index()

            if team_range_df.empty:
                st.info("No player data found for this team and time range.")
                return

            base_cols = [col for col in ["Player", "Team", "FP/G", "FPts", "GP"] if col in team_range_df.columns]
            roster_df = team_range_df[base_cols].copy()

            with st.spinner("Loading consistency and value metrics for this team..."):
                # Consistency index cache
                ci_cache_key = "consistency_index_cache"
                ci_cache = st.session_state.get(ci_cache_key) or {}
                consistency_index = None
                if isinstance(ci_cache, dict) and league_id in ci_cache:
                    consistency_index = ci_cache.get(league_id)
                else:
                    try:
                        consistency_index = build_league_consistency_index(league_id)
                    except Exception:
                        consistency_index = None
                    if consistency_index is not None:
                        ci_cache[league_id] = consistency_index
                        st.session_state[ci_cache_key] = ci_cache

                roster_with_consistency = enrich_roster_with_consistency(
                    roster_df.copy(), league_id, consistency_index=consistency_index
                )

                # Value profiles cache
                vp_cache_key = "player_value_profiles_cache"
                vp_cache = st.session_state.get(vp_cache_key) or {}
                value_profiles_df = None
                if isinstance(vp_cache, dict) and league_id in vp_cache:
                    value_profiles_df = vp_cache.get(league_id)
                else:
                    try:
                        value_profiles_df = build_player_value_profiles(league_id)
                    except Exception:
                        value_profiles_df = None
                    if value_profiles_df is not None:
                        vp_cache[league_id] = value_profiles_df
                        st.session_state[vp_cache_key] = vp_cache

                if value_profiles_df is not None and not value_profiles_df.empty:
                    merged = roster_with_consistency.merge(
                        value_profiles_df[["Player", "ValueScore", "ProductionScore", "ConsistencyScore"]],
                        on="Player",
                        how="left",
                    )
                else:
                    merged = roster_with_consistency

                dataframe(merged.set_index("Player"), width="stretch")
