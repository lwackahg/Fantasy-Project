"""
UI components for the Team Analyzer feature.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .logic import calculate_team_stats, calculate_league_ranks, STAT_CATEGORIES

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

        st.plotly_chart(fig, use_container_width=True)
        st.caption("The outer edge represents a rank of 1 (best in the league), while the center is the worst.")

        # --- Ranks Table ---
        with st.expander("View Full League Rankings"):
            st.dataframe(league_ranks.style.format("{:}"))
