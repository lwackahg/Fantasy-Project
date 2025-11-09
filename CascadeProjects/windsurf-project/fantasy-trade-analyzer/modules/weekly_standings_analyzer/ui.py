import streamlit as st
import pandas as pd
from .logic import (
    get_weekly_standings, 
    calculate_adjusted_scores,
    clear_weekly_standings_cache,
    get_league_name_map,
    FANTRAX_USERNAME,
    FANTRAX_PASSWORD,
    FANTRAX_DEFAULT_LEAGUE_ID
)
from ..standings_adjuster.logic import get_cached_periods, get_cached_periods_with_min_games
from ..standings_adjuster.excel_export import generate_comprehensive_excel, generate_period_excel

def parse_periods(periods_str):
    """Parses a string of periods (e.g., '1-3,5') into a list of integers."""
    periods = set()
    if not periods_str.strip():
        return []
    parts = periods_str.replace(" ", "").split(',')
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end:
                    start, end = end, start
                periods.update(range(start, end + 1))
            except ValueError:
                continue # Ignore invalid ranges
        else:
            try:
                periods.add(int(part))
            except ValueError:
                continue # Ignore invalid numbers
    return sorted(list(periods))

def show_weekly_standings_analyzer():
    """Displays the UI for the Weekly Standings Analyzer tool."""
    st.title("Weekly Standings Analyzer")
    st.write("This tool scrapes weekly standings and calculates adjusted scores based on a games limit.")

    league_map = get_league_name_map()
    if not league_map:
        st.warning("No league IDs found in `fantrax.env`. Please add `FANTRAX_LEAGUE_IDS`.")
        league_id = st.text_input("Fantrax League ID", value=FANTRAX_DEFAULT_LEAGUE_ID or "")
    else:
        # Attempt to find the index of the default league ID
        default_name = next((name for name, id in league_map.items() if id == FANTRAX_DEFAULT_LEAGUE_ID), list(league_map.keys())[0])
        selected_league_name = st.selectbox("Select Fantrax League", options=list(league_map.keys()), index=list(league_map.keys()).index(default_name))
        league_id = league_map[selected_league_name]

    # Display cached periods for the selected league
    if league_id:
        cached_periods_with_min = get_cached_periods_with_min_games(league_id)
        if cached_periods_with_min:
            st.markdown("---")
            st.subheader("📦 Cached Periods")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create a nice table showing cached periods and their min games
                cache_df = pd.DataFrame([
                    {'Period': period, 'Min Games': min_games}
                    for period, min_games in sorted(cached_periods_with_min.items())
                ])
                st.dataframe(
                    cache_df,
                    hide_index=True,
                    use_container_width=True,
                    height=min(len(cache_df) * 35 + 38, 300)  # Dynamic height, max 300px
                )
                st.caption(f"Total: {len(cached_periods_with_min)} period(s) cached")
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("📥 Export All to Excel", use_container_width=True):
                    try:
                        with st.spinner("Generating Excel..."):
                            filepath = generate_comprehensive_excel(league_id, selected_league_name)
                            with open(filepath, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download",
                                    data=f,
                                    file_name=filepath.name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,
                                    key="analyzer_export_all"
                                )
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.markdown("---")

    periods_input = st.text_input("Scoring Period(s)", value="1", help="Enter a single period, a range (e.g., 1-4), or a comma-separated list (e.g., 1,3,5).")

    periods_to_run = parse_periods(periods_input)
    min_games_map = {}

    if periods_to_run:
        st.markdown("--- ")
        st.subheader("Set Minimum Games per Period")
        cols = st.columns(min(len(periods_to_run), 4))
        for i, period in enumerate(periods_to_run):
            with cols[i % 4]:
                min_games_map[period] = st.number_input(
                    f"Period {period}",
                    min_value=0,
                    value=30,
                    key=f"min_games_{period}"
                )
        st.markdown("--- ")

    force_refresh = st.checkbox("Force Refresh (ignore cache)", key="weekly_standings_force_refresh", help="Check this box to bypass the local cache and download the latest data from Fantrax.")

    if st.button("Get Weekly Standings"):
        if not league_id:
            st.error("Please enter a Fantrax League ID.")
            return
        if not FANTRAX_USERNAME or not FANTRAX_PASSWORD:
            st.error("Fantrax username or password not found. Please set them in your `fantrax.env` file.")
            return

        if not periods_to_run:
            st.error("Invalid period format. Please enter periods like '1', '1-4', or '1,3,5'.")
            return

        for period in periods_to_run:
            st.subheader(f"Results for Period {period}")
            try:
                min_games_for_period = min_games_map.get(period, 25)
                with st.spinner(f"Fetching and calculating standings for period {period}..."):
                    df, from_cache = get_weekly_standings(
                        league_id, period, FANTRAX_USERNAME, FANTRAX_PASSWORD, min_games_for_period, force_refresh
                    )
                    st.success(f"Got standings for period {period}. From cache: {from_cache}")

                    if 'Calc FP/G' not in df.columns:
                        df['Calc FP/G'] = df.apply(lambda row: row['FPts'] / row['GP'] if row['GP'] > 0 else 0, axis=1)

                    adjusted_df = calculate_adjusted_scores(df.copy(), min_games_for_period)

                    st.header("Adjusted Standings")
                    display_cols = [
                        'team_name', 'FPts', 'GP', 'Calc FP/G',
                        'Games Over', 'Adjustment', 'Adjusted FPts'
                    ]
                    display_cols = [col for col in display_cols if col in adjusted_df.columns]
                    st.dataframe(adjusted_df[display_cols])

            except Exception as e:
                st.error(f"An error occurred for period {period}: {e}")
