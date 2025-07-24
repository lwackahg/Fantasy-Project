import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from .logic import (
    get_cached_standings_and_min_games,
    calculate_adjusted_scores,
    submit_adjustments_to_fantrax,
    get_cached_periods,
    FANTRAX_USERNAME,
    FANTRAX_PASSWORD
)

from ..weekly_standings_analyzer.logic import get_league_name_map, FANTRAX_DEFAULT_LEAGUE_ID

def show_standings_adjuster():
    st.title("Standings Adjuster")
    st.write("This tool submits the calculated adjustments from the Weekly Standings Analyzer to Fantrax.")

    league_name_map = get_league_name_map()
    league_names = list(league_name_map.keys())
    default_league_name = next((name for name, id in league_name_map.items() if id == FANTRAX_DEFAULT_LEAGUE_ID), None)

    selected_league_name = st.selectbox("Select League", options=league_names, index=league_names.index(default_league_name) if default_league_name in league_names else 0)
    league_id = league_name_map[selected_league_name]

    cached_periods_data = get_cached_periods(league_id)
    cached_periods_options = sorted(cached_periods_data.keys())

    if not cached_periods_options:
        st.warning("No cached standings found for this league ID. Please run the Weekly Standings Analyzer first.")
        return

    selected_period = st.selectbox("Select Period to Adjust", options=cached_periods_options)

    if selected_period:
        raw_df, min_games = get_cached_standings_and_min_games(league_id, selected_period)
        
        if raw_df is not None:
            st.info(f"Using minimum games value of **{min_games}** from cache for Period {selected_period}.")
            adjustments_df = calculate_adjusted_scores(raw_df.copy(), min_games)

            st.subheader(f"Review Adjustments for Period {selected_period}")
            
            # Calculate the final adjustment value for submission (negative, rounded whole number)
            adjustments_df['Final Adjustment'] = -(adjustments_df['Adjustment'].round(0).astype(int))
            
            st.dataframe(adjustments_df[['team_name', 'Adjustment', 'Final Adjustment']])

            if st.button("Submit Adjustments to Fantrax"):
                with st.spinner("Submitting adjustments..."):
                    success, message = submit_adjustments_to_fantrax(
                        league_id,
                        selected_period,
                        FANTRAX_USERNAME,
                        FANTRAX_PASSWORD,
                        adjustments_df
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        else:
            st.error(f"Could not load cached data for period {selected_period}.")
