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
    """Return sorted time ranges actually present in the data."""
    return sorted(data_ranges.keys(),
                  key=lambda x: TIME_RANGE_ORDER.index(x) if x in TIME_RANGE_ORDER else len(TIME_RANGE_ORDER))


def calculate_player_stats(data_ranges, player, metrics):
    """
    Calculate player statistics across different time ranges.
    Returns both the detailed stats DataFrame and a DataFrame with standard deviations.
    """
    stats, std_devs = [], {metric: [] for metric in metrics}
    metric_values = {metric: [] for metric in metrics}

    # Collect values for metrics to calculate std dev
    for range_name, df in data_ranges.items():
        player_data = df[df['Player'] == player]
        if not player_data.empty:
            for metric in metrics:
                value = player_data[metric].iloc[0]
                metric_values[metric].append(value)

    # Calculate standard deviations
    std_devs = {metric: np.std(metric_values[metric]) if metric_values[metric] else 0 for metric in metrics}

    # Collect stats
    for range_name, df in data_ranges.items():
        player_data = df[df['Player'] == player]
        if not player_data.empty:
            row_data = {'Player': player, 'Time Range': range_name}
            row_data.update({metric: player_data[metric].iloc[0] for metric in metrics})
            stats.append(row_data)

    stats_df = pd.DataFrame(stats)
    stats_df['Time Range'] = pd.Categorical(stats_df['Time Range'], TIME_RANGE_ORDER)
    stats_df.sort_values('Time Range', inplace=True)

    std_dev_df = pd.DataFrame([{ 
        'Player': player, 
        **{f'{metric}_STD': std_devs[metric] for metric in metrics}
    }])

    return stats_df, std_dev_df


def display_player_comparison(data_ranges, selected_players, metrics):
    """Display comprehensive comparison table for selected players."""
    if not selected_players or not metrics:
        return

    # Store DataFrames for later use
    if 'comparison_data' not in st.session_state:
        st.session_state.comparison_data = {}

    for metric in metrics:
        st.subheader(f"{metric} Comparison")

        all_stats, all_std_devs = [], []
        for player in selected_players:
            stats_df, std_dev_df = calculate_player_stats(data_ranges, player, [metric])
            all_stats.append(stats_df)
            all_std_devs.append(std_dev_df)

        if all_stats:
            combined_stats = pd.concat(all_stats)
            combined_std_devs = pd.concat(all_std_devs)

            pivot_table = combined_stats.pivot(index='Player', columns='Time Range', values=metric)
            pivot_table['STD_DEV'] = combined_std_devs[f'{metric}_STD'].values

            # Ensure columns are in the correct order
            ordered_cols = [col for col in TIME_RANGE_ORDER if col in pivot_table.columns] + ['STD_DEV']
            pivot_table = pivot_table[ordered_cols]

            st.session_state.comparison_data[metric] = pivot_table
            styled_table = pivot_table.style.highlight_max(axis=1, color='green')\
                .highlight_min(axis=1, color='red')\
                .format("{:.2f}")\
                .set_properties(border='1px solid #ddd')

            st.dataframe(styled_table)


def calculate_team_metrics(data_ranges, players, metrics, n_best=None):
    """Calculate combined metrics for a team's top N players."""
    all_stats, all_std_devs = [], []

    for player in players:
        stats_df, std_dev_df = calculate_player_stats(data_ranges, player, metrics)
        all_stats.append(stats_df)
        all_std_devs.append(std_dev_df)

    if not all_stats:
        return None

    combined_stats = pd.concat(all_stats)
    combined_std_devs = pd.concat(all_std_devs)
    available_ranges = get_available_time_ranges(data_ranges)

    metric_tables = {}
    for metric in metrics:
        pivot = combined_stats.pivot(index='Player', columns='Time Range', values=metric)
        pivot['Avg'] = pivot.mean(axis=1)
        pivot.sort_values('Avg', ascending=False, inplace=True)
        if n_best:
            pivot = pivot.head(n_best)

        totals = pivot[available_ranges].sum()
        averages = pivot[available_ranges].mean()
        std_devs = pivot[available_ranges].std()

        metric_tables[metric] = {
            'individual': pivot,
            'totals': totals,
            'averages': averages,
            'std_devs': std_devs,
            'available_ranges': available_ranges
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


def display_player_trends(current_data):
    """Display comprehensive trend analysis for players, identifying uptrending and downtrending players."""
    all_data = []

    # Collect all player data from session state
    for key, data in st.session_state.data_ranges.items():
        data_with_range = data.copy()
        data_with_range['Time Range'] = key
        all_data.append(data_with_range)

    combined_data = pd.concat(all_data)

    if not combined_data.empty:
        metrics = ['FP/G']  # Define metrics of interest
        trend_data = combined_data[combined_data['Time Range'].isin(TIME_RANGE_ORDER)]

        # Calculate average performance per player
        trend_averages = (trend_data
                          .groupby(['Player', 'Time Range', 'Fantasy_Manager'])[metrics]
                          .mean()
                          .unstack(level='Time Range'))

        trend_averages.columns = trend_averages.columns.map(lambda x: x[1])  # Flatten MultiIndex
        
        # Prompt user for trend calculation method
        user_choice = st.radio("Choose a method for trend calculation", ('Standard', 'Exponential', 'Custom'))

        if user_choice == 'Standard':
            trend_averages = standard_weighted_trend(trend_averages)
        elif user_choice == 'Exponential':
            trend_averages = exponential_weighted_trend(trend_averages)
        elif user_choice == 'Custom':
            weight_14d = st.number_input("Enter weight for 14 Days", value=1.0)
            weight_60d = st.number_input("Enter weight for 60 Days", value=1.0)
            weight_ytd = st.number_input("Enter weight for YTD", value=1.5)
            trend_averages = custom_weighted_trend(trend_averages, weight_14d, weight_60d, weight_ytd)

        # Reorder the columns according to TIME_RANGE_ORDER
        ordered_columns = TIME_RANGE_ORDER + ['Trend']
        trend_averages = trend_averages[ordered_columns]  # Select and order columns

        # Define rigid YTD performance intervals
        bins = range(0, int(trend_averages['YTD'].max()) + 10, 10)
        labels = [f"{i}-{i+9}" for i in bins[:-1]]
        trend_averages['YTD Group'] = pd.cut(trend_averages['YTD'], bins=bins, labels=labels, right=False)

        # Identify injured players (having 0 in any Time Range)
        injured_players = trend_averages[(trend_averages[TIME_RANGE_ORDER] == 0).any(axis=1)]
        healthy_players = trend_averages[(trend_averages[TIME_RANGE_ORDER] != 0).all(axis=1)]

        # Identify uptrending and downtrending players from healthy players
        uptrending_players = healthy_players[healthy_players['Trend'] > 0]
        downtrending_players = healthy_players[healthy_players['Trend'] < 0]

        # Sort by YTD Group and then by Trend
        uptrending_players = uptrending_players.sort_values(by=['YTD Group', 'Trend'], ascending=[True, False])
        downtrending_players = downtrending_players.sort_values(by=['YTD Group', 'Trend'])

        # Display results for uptrending and downtrending players side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uptrending Players, Sell High")
            for group in labels[::-1]:  # Descending order
                group_uptrending = uptrending_players[uptrending_players['YTD Group'] == group]
                if not group_uptrending.empty:
                    st.write(f"**YTD FP/G = {group}:**")
                    st.dataframe(group_uptrending.drop(columns=['YTD Group']))

        with col2:
            st.subheader("Downtrending Players, Buy Low")
            for group in labels[::-1]:  # Descending order
                group_downtrending = downtrending_players[downtrending_players['YTD Group'] == group]
                if not group_downtrending.empty:
                    st.write(f"**YTD FP/G = {group}:**")
                    st.dataframe(group_downtrending.drop(columns=['YTD Group']))
            
            # Display injured players at the bottom
            st.subheader("Injured Players")
            if not injured_players.empty:
                st.dataframe(injured_players.drop(columns=['YTD Group']))
            else:
                st.write("No injured players identified.")
    else:
        st.error("No player data available for trend analysis.")


def standard_weighted_trend(trend_averages, weight_14d=1, weight_60d=1, weight_ytd=1.5):
    trend_averages['14 Days Diff'] = trend_averages['14 Days'] - trend_averages['7 Days']
    trend_averages['60 Days Diff'] = trend_averages['60 Days'] - trend_averages['30 Days']
    trend_averages['YTD Diff'] = trend_averages['YTD'] - trend_averages['60 Days']
    
    trend_averages['Trend'] = (
        (weight_14d * trend_averages['14 Days Diff']) +
        (weight_60d * trend_averages['60 Days Diff']) +
        (weight_ytd * trend_averages['YTD Diff'])
    ).round(2)

    return trend_averages


def exponential_weighted_trend(trend_averages):
    trend_averages['14 Days Diff'] = trend_averages['14 Days'] - trend_averages['7 Days']
    trend_averages['60 Days Diff'] = trend_averages['60 Days'] - trend_averages['30 Days']
    trend_averages['YTD Diff'] = trend_averages['YTD'] - trend_averages['60 Days']
    
    # Applying exponential weights (decay factor)
    weight_14d = 0.6
    weight_60d = 0.3
    weight_ytd = 1.0
    
    trend_averages['Trend'] = (
        (weight_14d * trend_averages['14 Days Diff']) +
        (weight_60d * trend_averages['60 Days Diff']) +
        (weight_ytd * trend_averages['YTD Diff'])
    ).round(2)

    return trend_averages


def custom_weighted_trend(trend_averages, weight_14d, weight_60d, weight_ytd):
    trend_averages['14 Days Diff'] = trend_averages['14 Days'] - trend_averages['7 Days']
    trend_averages['60 Days Diff'] = trend_averages['60 Days'] - trend_averages['30 Days']
    trend_averages['YTD Diff'] = trend_averages['YTD'] - trend_averages['60 Days']
    
    trend_averages['Trend'] = (
        (weight_14d * trend_averages['14 Days Diff']) +
        (weight_60d * trend_averages['60 Days Diff']) +
        (weight_ytd * trend_averages['YTD Diff'])
    ).round(2)

    return trend_averages


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

    st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; border: 1px solid #4a4a4a; margin: 1rem 0;'>
            <h3 style='margin-top: 0;'>🔄 Trade Comparison Analysis</h3>
            <p>Detailed comparison of potential trade targets based on multiple performance metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Performance Charts", "📋 Detailed Metrics"])

    with tab1:
        st.subheader("Top Trade Matches")
        summary_data = [{
            'Player': target['name'],
            'Team': target['manager'],
            'Match Score': target['match_score'],
            'FP/G': target['metrics']['FP/G'],
            'Recent Form': target['metrics']['Recent FP/G'],
            'Games Played': target['metrics']['Games Played']
        } for target in trade_targets[:5]]  # Show top 5 matches

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df.style
            .background_gradient(subset=['Match Score'], cmap='RdYlGn')
            .format({
                'Match Score': '{:.1f}',
                'FP/G': '{:.2f}',
                'Recent Form': '{:.2f}',
                'Games Played': '{:.1f}'
            })
        )

    with tab2:
        st.subheader("Performance Comparison")
        metrics_comparison = []

        for target in trade_targets[:3]:  # Compare top 3 targets
            metrics_comparison.extend([
                {'Player': selected_player, 'Metric': 'FP/G', 'Value': player_stats['FP/G'].mean()},
                {'Player': target['name'], 'Metric': 'FP/G', 'Value': target['metrics']['FP/G']},
                {'Player': selected_player, 'Metric': 'Recent', 'Value': player_stats[player_stats['Time Range'] == '7 Days']['FP/G'].mean()},
                {'Player': target['name'], 'Metric': 'Recent', 'Value': target['metrics']['Recent FP/G']},
                {'Player': selected_player, 'Metric': 'Consistency', 'Value': player_stats['FP/G'].std()},
                {'Player': target['name'], 'Metric': 'Consistency', 'Value': target['metrics']['Consistency']}
            ])

        chart_df = pd.DataFrame(metrics_comparison)
        fig = px.bar(chart_df, x='Metric', y='Value', color='Player', barmode='group',
                      title='Performance Metrics Comparison', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Detailed Metrics Analysis")
        for target in trade_targets:
            with st.expander(f"📊 {target['name']} (Match Score: {target['match_score']})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### Performance Metrics")
                    for metric, value in target['metrics'].items():
                        st.metric(metric, f"{value:,.2f}",
                                   delta=f"{value - player_stats['FP/G'].mean():+.2f}" if metric == 'FP/G' else None)

                with col2:
                    st.markdown("##### Match Quality Breakdown")
                    for aspect, score in target['match_details'].items():
                        quality = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                        st.write(f"{quality} {aspect.replace('_', ' ').title()}: {score:.1f}%")


def display_fantasy_managers_teams(current_data):
    """Display metrics for each Fantasy Manager's team and allow selection of top N players."""
    df = current_data.reset_index()
    fantasy_managers = df['Fantasy_Manager'].unique()
    selected_managers = st.multiselect("Select Fantasy Managers:", fantasy_managers, default=fantasy_managers.tolist())

    # Process data for each manager
    all_stats = {}
    for manager in selected_managers:
        team_players = df[df['Fantasy_Manager'] == manager]['Player'].unique()
        stats = {}
        for player in team_players:
            stats_df, _ = calculate_player_stats(st.session_state.data_ranges, player, ['FP/G'])
            stats[player] = stats_df if not stats_df.empty else None
        
        all_stats[manager] = pd.concat([v for v in stats.values() if v is not None], ignore_index=True)

    if selected_managers:
        tab1, tab2 = st.tabs(["Team Details", "Rankings & Trade Analysis"])

        with tab1:
            st.title("Fantasy Managers' Teams")
            st.write("Red and Green highlight indicates high and low performance for each player.")

            for manager in selected_managers:
                stats_df = all_stats.get(manager)
                if stats_df is not None and not stats_df.empty:
                    with st.expander(f"{manager}'s Players", expanded=True):
                        team_metrics = stats_df.groupby('Time Range')['FP/G'].agg(['mean', 'median', 'std']).round(2)
                        team_metrics.columns = ['Mean FP/G', 'Median FP/G', 'Std Dev']
                        st.write("Team Metrics by Time Range:")
                        st.table(team_metrics)

                        player_performance = stats_df.pivot(index='Player', columns='Time Range', values='FP/G')
                        styled_stats = player_performance.style\
                            .highlight_max(axis=1, color='green')\
                            .highlight_min(axis=1, color='red')\
                            .format("{:.2f}")

                        st.write("Player Performance Breakdown:")
                        st.dataframe(styled_stats)

        with tab2:
            display_team_rankings(all_stats)