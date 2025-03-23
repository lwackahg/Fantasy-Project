"""
Player Data Display Module
Handles all player data visualization and comparison functionality.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Define constants for time range ordering
TIME_RANGE_ORDER = ['7 Days', '14 Days', '30 Days', '60 Days', 'YTD']


def get_ordered_time_ranges(ascending=True):
    """Return time ranges in the correct order."""
    return TIME_RANGE_ORDER if ascending else TIME_RANGE_ORDER[::-1]


def get_available_time_ranges(data_ranges):
    """Return sorted time ranges present in the data."""
    return sorted(data_ranges.keys(),
                  key=lambda x: TIME_RANGE_ORDER.index(x) if x in TIME_RANGE_ORDER else len(TIME_RANGE_ORDER))


def calculate_standard_deviation(metric_values):
    """Calculate standard deviations for given metric values."""
    return {metric: np.std(values) if values else 0 for metric, values in metric_values.items()}


def collect_player_data(data_ranges, player, metrics):
    """Collect player data for given metrics across all time ranges."""
    metric_values = {metric: [] for metric in metrics}
    stats = []

    for range_name, df in data_ranges.items():
        player_data = df[df['Player'] == player]

        if not player_data.empty:
            stats_row = {'Player': player, 'Time Range': range_name}
            for metric in metrics:
                value = player_data[metric].iloc[0]
                stats_row[metric] = value
                metric_values[metric].append(value)
            stats.append(stats_row)

    stats_df = pd.DataFrame(stats)
    std_devs = calculate_standard_deviation(metric_values)
    
    # Converting std_devs to a DataFrame
    std_dev_df = pd.DataFrame([{ 
        'Player': player, 
        **{f'{metric}_STD': std_devs[metric] for metric in metrics}
    }])
    
    return stats_df, std_dev_df


def display_comparison_table(comparison_data, metric):
    """Display styled comparison table for the given metric."""
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
        st.subheader(f"{metric} Comparison")
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
            st.session_state.comparison_data[metric] = pivot_table[ordered_cols]
            display_comparison_table(pivot_table, metric)


def calculate_team_metrics(data_ranges, players, metrics, n_best=None):
    """Calculate combined metrics for a team's top N players."""
    all_stats, all_std_devs = [], []

    for player in players:
        stats_df, std_dev_df = collect_player_data(data_ranges, player, metrics)
        all_stats.append(stats_df)
        all_std_devs.append(std_dev_df)

    if not all_stats:
        return None

    combined_stats = pd.concat(all_stats)
    combined_std_devs = pd.concat(all_std_devs)

    metric_tables = {}
    for metric in metrics:
        pivot = combined_stats.pivot(index='Player', columns='Time Range', values=metric)
        pivot['Avg'] = pivot.mean(axis=1).sort_values(ascending=False).head(n_best) if n_best else pivot.mean(axis=1)
        totals = pivot.sum()
        averages = pivot.mean()
        std_devs = pivot.std()

        metric_tables[metric] = {
            'individual': pivot,
            'totals': totals,
            'averages': averages,
            'std_devs': std_devs,
            'available_ranges': get_available_time_ranges(data_ranges)
        }

    return metric_tables


def display_team_scouting(current_data, data_ranges):
    """Display team scouting interface for comparing players across teams."""
    st.title("Team Scouting")

    df = current_data.reset_index()
    fantasy_managers = df['Fantasy_Manager'].unique()
    fantasy_manager = st.selectbox("Select Fantasy Manager's Team:", fantasy_managers)

    if fantasy_manager:
        team_players = df[df['Fantasy_Manager'] == fantasy_manager]['Player'].unique().tolist()
        available_metrics = ['FPts', 'FP/G', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP']
        comparison_metrics = st.multiselect("Select Stats to Compare:", available_metrics, default=['FP/G'])

        n_best = st.slider("Show Top N Players:", min_value=1, max_value=len(team_players), value=9)

        if comparison_metrics:
            st.subheader(f"Top {n_best} Players Analysis")
            metric_tables = calculate_team_metrics(data_ranges, team_players, comparison_metrics, n_best)

            if metric_tables:
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

        st.subheader("Individual Player Comparison")
        comparison_players = st.multiselect("Select Players to Compare:", team_players)

        if comparison_players and comparison_metrics:
            display_player_comparison(data_ranges, comparison_players, comparison_metrics)


def display_player_data(data_ranges, combined_data):
    """Display the player data in a clean and searchable format."""
    st.subheader("Player Data Overview")

    if not combined_data.empty:
        search_query = st.text_input("Search Players", "").strip()

        if search_query:
            filtered_data = combined_data[combined_data.index.str.contains(search_query, case=False, na=False)]
            if not filtered_data.empty:
                st.write(f"Data for **{search_query}**:")
                st.dataframe(filtered_data)
            else:
                st.write("Player not found.")
        else:
            st.dataframe(combined_data)
    else:
        st.write("No data available to display.")


def display_metrics(data):
    """Display basic statistics as metrics in the Streamlit app."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Players", len(data))
    with col2:
        st.metric("Teams", data['Team'].nunique())
    with col3:
        st.metric("Avg FP/G", f"{data['FP/G'].mean():.1f}")


def display_team_rankings(all_stats):
    """Display rankings and trade analysis for fantasy managers' teams."""
    st.title("Team Rankings and Trade Opportunities")

    if not all_stats:
        st.write("No team data available for analysis.")
        return

    selected_managers = st.multiselect("Select your team:", options=list(all_stats.keys()))

    if not selected_managers:
        st.write("Please select at least one team.")
        return

    def calculate_team_metrics(stats_df):
        avg_fpg = stats_df['FP/G'].mean()
        consistency = (stats_df['FP/G'].std() / avg_fpg) if avg_fpg > 0 else float('inf')
        return {'Average FP/G': round(avg_fpg, 2), 
                'Std Dev': round(stats_df['FP/G'].std(), 2),
                'Consistency Score': round(consistency, 3), 
                'Team Size': stats_df['Player'].nunique()}

    team_rankings = pd.DataFrame(
        [{**{'Manager': manager}, **calculate_team_metrics(stats_df)} 
         for manager, stats_df in all_stats.items() if not stats_df.empty]
    )

    st.subheader("Team Rankings. Average across all time ranges")
    styled_rankings = team_rankings.style.format({'Average FP/G': '{:.2f}', 
                                                  'Std Dev': '{:.2f}', 
                                                  'Consistency Score': '{:.3f}', 
                                                  'Team Size': '{:d}'})\
        .highlight_max(subset=['Average FP/G'], color='green')\
        .highlight_min(subset=['Consistency Score'], color='green')

    st.dataframe(styled_rankings)

    # Detailed metrics for selected teams
    st.title("Team Performance Metrics")
    teams_per_row = 3

    for i in range(0, len(selected_managers), teams_per_row):
        cols = st.columns(teams_per_row)
        for col_idx, manager in enumerate(selected_managers[i:i + teams_per_row]):
            if col_idx < len(cols):  # Check to prevent out-of-bounds
                stats_df = all_stats.get(manager, pd.DataFrame())
                with cols[col_idx]:
                    st.write(f"**{manager}'s Team**")
                    if not stats_df.empty:
                        team_metrics = stats_df.groupby('Time Range')['FP/G'].agg(['mean', 'median', 'std']).round(2)
                        team_metrics.columns = ['Mean FP/G', 'Median FP/G', 'Std Dev']
                        st.table(team_metrics.style.highlight_max(axis=0, color='green')\
                            .highlight_min(axis=0, color='red').format('{:.2f}'))
                    else:
                        st.write("No data available.")


def display_trade_comparison(selected_player, player_stats, trade_targets):
    """
    Display enhanced trade comparison visualization with detailed metrics and charts.
    """
    if not trade_targets:
        st.warning("No comparable trade targets found within the specified criteria.")
        return

    st.subheader(f"Trade Comparison for {selected_player}")

    # Create a dataframe for comparison
    comparison_df = pd.DataFrame([player_stats] + trade_targets)
    comparison_df.set_index('Player', inplace=True)

    # Display the comparison table
    st.write("#### Player Comparison")
    styled_comparison = comparison_df.style.highlight_max(axis=0, color='green')\
        .highlight_min(axis=0, color='red')\
        .format({"FP/G": "{:.2f}", "PTS": "{:.1f}", "REB": "{:.1f}", 
                 "AST": "{:.1f}", "STL": "{:.1f}", "BLK": "{:.1f}", 
                 "MIN": "{:.1f}", "GP": "{:.0f}"})
    st.dataframe(styled_comparison)

    # Radar Chart for Visual Comparison
    st.write("#### Radar Chart Comparison")
    metrics_to_plot = ['FP/G', 'PTS', 'REB', 'AST', 'STL', 'BLK']
    
    # Normalize the data for radar chart
    radar_df = comparison_df[metrics_to_plot].copy()
    for col in radar_df.columns:
        max_val = radar_df[col].max()
        if max_val > 0:  # Avoid division by zero
            radar_df[col] = radar_df[col] / max_val

    # Prepare data for radar chart
    radar_df = radar_df.reset_index()
    radar_df_melted = pd.melt(radar_df, id_vars=['Player'], value_vars=metrics_to_plot,
                              var_name='Metric', value_name='Normalized Value')

    # Create radar chart
    fig = px.line_polar(radar_df_melted, r='Normalized Value', theta='Metric', 
                         color='Player', line_close=True, 
                         range_r=[0, 1], template="plotly_dark")
    fig.update_traces(fill='toself')
    st.plotly_chart(fig)

    # Bar charts for key metrics
    st.write("#### Key Metrics Comparison")
    for metric in metrics_to_plot:
        fig = px.bar(comparison_df.reset_index(), x='Player', y=metric, 
                     title=f"{metric} Comparison", color='Player')
        st.plotly_chart(fig)

    # Recommendation
    st.subheader("Trade Recommendation")
    best_alternative = comparison_df['FP/G'].idxmax()
    if best_alternative == selected_player:
        st.success(f"Keep {selected_player}. They are performing better than the alternatives.")
    else:
        st.success(f"Consider trading for {best_alternative}. They are currently performing better in FP/G.")