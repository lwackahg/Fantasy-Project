"""
UI components for the team scouting feature.
"""

import streamlit as st
import pandas as pd
from .logic import calculate_team_metrics, collect_player_data, TIME_RANGE_ORDER

def display_comparison_table(comparison_data, metric):
    """Display styled comparison table for the given metric."""
    st.subheader(f"{metric} Comparison")
    styled_table = comparison_data.style.highlight_max(axis=1, color='green')\
        .highlight_min(axis=1, color='red')\
        .format("{:.2f}")\
        .set_properties(border='1px solid #ddd')
    st.dataframe(styled_table)

def display_player_comparison(data_ranges, selected_players, metrics):
    """Display comprehensive comparison table for selected players."""
    if not selected_players or not metrics:
        return

    for metric in metrics:
        all_stats, all_std_devs = [], []

        for player in selected_players:
            stats_df, std_dev_df = collect_player_data(data_ranges, player, [metric])
            all_stats.append(stats_df)
            all_std_devs.append(std_dev_df)

        if all_stats:
            combined_stats = pd.concat(all_stats)
            combined_std_devs = pd.concat(all_std_devs)

            pivot_table = combined_stats.pivot(index='Player', columns='Time Range', values=metric)
            pivot_table['STD_DEV'] = combined_std_devs[f'{metric}_STD'].values

            ordered_cols = [col for col in TIME_RANGE_ORDER if col in pivot_table.columns] + ['STD_DEV']
            display_comparison_table(pivot_table[ordered_cols], metric)

def display_team_metric_tables(metric_tables, comparison_metrics):
    """Displays the individual and team performance tables."""
    for metric in comparison_metrics:
        st.write(f"### {metric} Analysis")
        data = metric_tables[metric]
        available_ranges = data['available_ranges']
        col1, col2 = st.columns(2)

        with col1:
            st.write("Individual Performance")
            display_df = data['individual'][available_ranges]
            styled_table = display_df.style\
                .highlight_max(axis=0, color='green')\
                .highlight_min(axis=0, color='red')\
                .format("{:.2f}")\
                .set_properties(border='1px solid #ddd')
            st.dataframe(styled_table)

        with col2:
            st.write("Team Statistics")
            team_stats = pd.DataFrame({
                'Total': data['totals'],
                'Average': data['averages'],
                'Std Dev': data['std_devs']
            })
            st.dataframe(team_stats.style.format("{:.2f}"))

def display_team_scouting_page(current_data, data_ranges):
    """Display team scouting interface for comparing players across teams."""
    st.title("Team Scouting")

    df = current_data.reset_index()
    fantasy_managers = df['Fantasy_Manager'].unique()
    fantasy_manager = st.selectbox("Select Fantasy Manager's Team:", fantasy_managers)

    if fantasy_manager:
        team_players = df[df['Fantasy_Manager'] == fantasy_manager]['Player'].unique().tolist()
        available_metrics = ['FPts', 'FP/G', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP']
        comparison_metrics = st.multiselect("Select Stats to Compare:", available_metrics, default=['FP/G'])

        n_best = st.slider("Show Top N Players:", min_value=1, max_value=len(team_players), value=min(9, len(team_players)))

        if comparison_metrics:
            st.subheader(f"Top {n_best} Players Analysis")
            metric_tables = calculate_team_metrics(data_ranges, team_players, comparison_metrics, n_best)
            if metric_tables:
                display_team_metric_tables(metric_tables, comparison_metrics)

        st.subheader("Individual Player Comparison")
        comparison_players = st.multiselect("Select Players to Compare:", team_players)

        if comparison_players and comparison_metrics:
            display_player_comparison(data_ranges, comparison_players, comparison_metrics)
