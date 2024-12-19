"""
Player Data Display Module
Handles all player data visualization and comparison functionality.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define constants for time range ordering
TIME_RANGE_ORDER = ['7 Days', '14 Days', '30 Days', '60 Days', 'YTD']

def get_ordered_time_ranges(ascending=True):
    """Get time ranges in the correct order."""
    return TIME_RANGE_ORDER if ascending else TIME_RANGE_ORDER[::-1]

def get_available_time_ranges(data_ranges):
    """Get the time ranges that are actually present in the data."""
    return sorted(data_ranges.keys(), 
                 key=lambda x: TIME_RANGE_ORDER.index(x) if x in TIME_RANGE_ORDER else len(TIME_RANGE_ORDER))

def calculate_player_stats(data_ranges, player, metrics):
    """
    Calculate player statistics across different time ranges.
    Returns both the detailed stats DataFrame and a DataFrame with standard deviations.
    """
    stats = []
    std_devs = {}
    
    # First, collect all values for each metric to calculate overall std dev
    metric_values = {metric: [] for metric in metrics}
    for range_name, df in data_ranges.items():
        df_reset = df.reset_index()
        player_data = df_reset[df_reset['Player'] == player]
        if not player_data.empty:
            for metric in metrics:
                metric_values[metric].append(player_data[metric].iloc[0])
    
    # Calculate overall std dev for each metric
    for metric in metrics:
        std_devs[metric] = np.std(metric_values[metric]) if metric_values[metric] else 0
    
    # Now collect the regular stats
    for range_name, df in data_ranges.items():
        df_reset = df.reset_index()
        player_data = df_reset[df_reset['Player'] == player]
        if not player_data.empty:
            row_data = {'Player': player, 'Time Range': range_name}
            for metric in metrics:
                row_data[metric] = player_data[metric].iloc[0]
            stats.append(row_data)
    
    stats_df = pd.DataFrame(stats)
    # Sort by the predefined time range order
    stats_df['Time Range'] = pd.Categorical(stats_df['Time Range'], TIME_RANGE_ORDER)
    stats_df = stats_df.sort_values('Time Range')
    
    # Create std dev DataFrame
    std_dev_df = pd.DataFrame([{
        'Player': player,
        **{f'{metric}_STD': std_devs[metric] for metric in metrics}
    }])
    
    return stats_df, std_dev_df

def display_player_comparison(data_ranges, selected_players, metrics):
    """
    Display a comprehensive comparison table for selected players.
    Shows metrics and their standard deviations across different time ranges.
    """
    if not selected_players or not metrics:
        return
    
    # Store DataFrames for later use
    if 'comparison_data' not in st.session_state:
        st.session_state.comparison_data = {}
    
    # Create comparison tables for each metric
    for metric in metrics:
        st.subheader(f"{metric} Comparison")
        
        # Collect data for all players
        all_stats = []
        all_std_devs = []
        
        for player in selected_players:
            stats_df, std_dev_df = calculate_player_stats(data_ranges, player, [metric])
            all_stats.append(stats_df)
            all_std_devs.append(std_dev_df)
        
        if all_stats:
            # Combine all player stats
            combined_stats = pd.concat(all_stats)
            combined_std_devs = pd.concat(all_std_devs)
            
            # Create the main comparison table
            pivot_table = combined_stats.pivot(
                index='Player',
                columns='Time Range',
                values=metric
            )
            
            # Add the STD_DEV column
            pivot_table[f'STD_DEV'] = combined_std_devs[f'{metric}_STD'].values
            
            # Ensure columns are in the correct order
            time_range_cols = [col for col in TIME_RANGE_ORDER if col in pivot_table.columns]
            pivot_table = pivot_table[time_range_cols + ['STD_DEV']]
            
            # Store the DataFrame for later use
            st.session_state.comparison_data[metric] = pivot_table
            
            # Apply styling
            styled_table = pivot_table.style\
                .highlight_max(subset=time_range_cols, color='green')\
                .highlight_min(subset=time_range_cols, color='red')\
                .format("{:.2f}")\
                .set_properties(**{
                    #'color': 'black',
                    'border': '1px solid #ddd'
                })
            
            st.dataframe(styled_table)

def calculate_team_metrics(data_ranges, players, metrics, n_best=None):
    """Calculate combined metrics for a team's top N players."""
    all_stats = []
    all_std_devs = []
    
    for player in players:
        stats_df, std_dev_df = calculate_player_stats(data_ranges, player, metrics)
        all_stats.append(stats_df)
        all_std_devs.append(std_dev_df)
    
    if not all_stats:
        return None
        
    # Combine all player stats
    combined_stats = pd.concat(all_stats)
    combined_std_devs = pd.concat(all_std_devs)
    
    # Get actual time ranges from the data
    available_ranges = get_available_time_ranges(data_ranges)
    
    # Create pivot tables for each metric
    metric_tables = {}
    for metric in metrics:
        pivot = combined_stats.pivot(
            index='Player',
            columns='Time Range',
            values=metric
        )
        
        # Sort players by their average performance in this metric
        pivot['Avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('Avg', ascending=False)
        
        if n_best:
            pivot = pivot.head(n_best)
            
        # Calculate team totals and averages for available time ranges
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
    
    # Reset index to access Player column
    df = current_data.reset_index()
    
    fantasy_managers = df['Fantasy_Manager'].unique()
    fantasy_manager = st.selectbox("Select Fantasy Manager's Team:", fantasy_managers)
    
    if fantasy_manager:
        team_players = df[df['Fantasy_Manager'] == fantasy_manager]['Player'].unique().tolist()
        
        # Available metrics with defaults
        available_metrics = ['FPts', 'FP/G', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP']
        comparison_metrics = st.multiselect(
            "Select Stats to Compare:",
            available_metrics,
            default=['FP/G', 'FPts']
        )
        
        # Add N-best selector
        n_best = st.slider("Show Top N Players:", min_value=1, max_value=len(team_players), value=5)
        
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
                        # Drop the Avg column for display and only include available ranges
                        display_df = data['individual'][available_ranges].copy()
                        styled_table = display_df.style\
                            .highlight_max(axis=0, color='green')\
                            .highlight_min(axis=0, color='red')\
                            .format("{:.2f}")\
                            .set_properties(**{'border': '1px solid #ddd'})
                        st.dataframe(styled_table)
                    
                    with col2:
                        st.write("Team Statistics")
                        team_stats = pd.DataFrame({
                            'Total': data['totals'],
                            'Average': data['averages'],
                            'Std Dev': data['std_devs']
                        })
                        st.dataframe(team_stats.style.format("{:.2f}"))
        
        # Multi-select for individual player comparison
        st.subheader("Individual Player Comparison")
        comparison_players = st.multiselect(
            "Select Players to Compare:",
            team_players
            
        )
        
        if comparison_players and comparison_metrics:
            display_player_comparison(data_ranges, comparison_players, comparison_metrics)

def display_player_trends(player, current_data):
    """Display comprehensive trend analysis for a selected player."""
    all_data = []

    for key, data in st.session_state.data_ranges.items():
        data_with_player = data.reset_index()
        player_data = data_with_player[data_with_player['Player'] == player]
        if not player_data.empty:
            player_data['Time Range'] = key
            all_data.append(player_data)
    
    if all_data:
        combined_data = pd.concat(all_data)
        
        st.subheader(f'Performance Trends for {player}')
        
        # Time range sorting
        x_axis_sort_order = st.selectbox("Select Time Range Order:", ["Ascending", "Descending"])
        combined_data = combined_data.sort_values(by='Time Range', 
            key=lambda x: pd.Categorical(x, TIME_RANGE_ORDER),
            ascending=x_axis_sort_order == "Ascending")
        
        # Store the sorted data for later use
        if 'trend_data' not in st.session_state:
            st.session_state.trend_data = {}
        st.session_state.trend_data[player] = combined_data

        # Metric selection with defaults
        available_metrics = ['FPts', 'FP/G', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP']
        y_axis_metrics = st.multiselect("Select Metrics to Plot:", available_metrics, default=['FP/G', 'FPts'])

        if y_axis_metrics:
            # Create plot with custom styling
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color_map = {'FP/G': 'blue', 'FPts': 'orange', 'GP': 'green'}

            for y_metric in y_axis_metrics:
                color = color_map.get(y_metric, None)
                ax1.plot(combined_data['Time Range'], combined_data[y_metric], 
                        marker='o', label=y_metric, color=color)

            ax1.set_xlabel('Time Range')
            ax1.set_ylabel('Value')
            ax1.set_title(f'{player} Metrics by Time Range')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.warning(f"No historical data available for {player}.")

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
                st.warning("Player not found.")
        else:
            st.dataframe(combined_data)
    else:
        st.warning("No data available to display.")

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
    st.title("Team Rankings & Trade Analysis")
    
    if not all_stats:
        st.warning("No team data available for analysis.")
        return
    
    # Team selection
    selected_managers = st.multiselect("Select your team:", options=list(all_stats.keys()))
    
    if not selected_managers:
        st.warning("Please select at least one team.")
        return
    
    # Calculate overall team metrics
    team_rankings = []
    for manager, stats_df in all_stats.items():
        avg_fpg = stats_df['FP/G'].mean()
        std_dev = stats_df['FP/G'].std()
        consistency = std_dev / avg_fpg if avg_fpg > 0 else float('inf')
        
        team_rankings.append({
            'Manager': manager,
            'Average FP/G': round(avg_fpg, 2),
            'Std Dev': round(std_dev, 2),
            'Consistency Score': round(consistency, 3),
            'Team Size': stats_df['Player'].nunique()
        })
    
    rankings_df = pd.DataFrame(team_rankings)
    
    # Display team rankings
    st.subheader("Team Rankings")
    st.dataframe(
        rankings_df.sort_values('Average FP/G', ascending=False)
        .style.highlight_max(subset=['Average FP/G'], color='green')
        .highlight_min(subset=['Consistency Score'], color='red')
        .format({
            'Average FP/G': '{:.2f}',
            'Std Dev': '{:.2f}',
            'Consistency Score': '{:.3f}'
        })
    )
    
    # Detailed metrics for selected teams
    for manager in selected_managers:
        stats_df = all_stats.get(manager)
        st.subheader(f"{manager}'s Team Metrics")
        
        if stats_df is not None and not stats_df.empty:
            team_metrics = stats_df.groupby('Time Range')['FP/G'].agg(['mean', 'median', 'std']).round(2)
            team_metrics.columns = ['Mean FP/G', 'Median FP/G', 'Std Dev']
            st.dataframe(team_metrics)
        else:
            st.write("No data available.")
    
    # Trade Analysis Section
    st.subheader("Trade Opportunities")
    trade_suggestions = generate_trade_opportunities(all_stats, selected_managers)
    st.dataframe(trade_suggestions)

def generate_trade_opportunities(all_stats, selected_managers):
    trade_suggestions = []

    for manager in selected_managers:
        players = all_stats[manager]['Player'].unique()
        player_stats = all_stats[manager]

        for player in players:
            current_fpg = player_stats[player_stats['Player'] == player]['FP/G'].mean()

            # Get potential trade targets based on performance
            targets = get_possible_trade_targets(player, all_stats, selected_managers, current_fpg)
            
            trade_suggestions.append({
                'Player Offered': player,
                'Potential Trade Targets': ', '.join(targets)
            })

    return pd.DataFrame(trade_suggestions)

def get_possible_trade_targets(player, all_stats, selected_managers, current_fpg):
    targets = []
    
    for manager in selected_managers:
        if player in all_stats[manager]['Player'].values:
            continue  # Skip if the player is in the same team

        # Fetch players from opposing teams
        opposing_players = all_stats[manager]['Player']

        # Suggest players who have better or comparable FP/G
        for target in opposing_players:
            target_fpg = all_stats[manager][all_stats[manager]['Player'] == target]['FP/G'].mean()
            
            # Implement your trade ratio or selection criteria
            if target_fpg >= 0.9 * current_fpg:  # Example condition: target must be at least 90% of the offered player's performance
                targets.append(target)

    return list(set(targets))  # Return unique targets
# Sample usage
# display_team_rankings(your_all_stats_data) 

def display_fantasy_managers_teams(current_data):
    """Display metrics for each Fantasy Manager's team and allow selection of top N players."""
    
    df = current_data.reset_index()
    
    # Get unique fantasy managers
    fantasy_managers = df['Fantasy_Manager'].unique()
    
    # Auto-select all fantasy managers by default
    default_selection = fantasy_managers.tolist()
    selected_managers = st.multiselect("Select Fantasy Managers:", fantasy_managers, default=default_selection)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Team Details", "Rankings & Trade Analysis"])
    
    if selected_managers:
        # Filter data once for all selected managers
        filtered_df = df[df['Fantasy_Manager'].isin(selected_managers)]
        
        # Process data for each manager
        all_stats = {}
        for manager in selected_managers:
            team_players = filtered_df[filtered_df['Fantasy_Manager'] == manager]['Player'].unique()
            stats = {}
            
            # Calculate stats for each player
            for player in team_players:
                player_stats = calculate_player_stats(st.session_state.data_ranges, player, ['FP/G'])
                if not player_stats[0].empty:
                    stats[player] = player_stats[0]
            
            if stats:
                all_stats[manager] = pd.concat(stats.values()).reset_index(drop=True)
        
        with tab1:
            st.title("Fantasy Managers' Teams")
            # Define constant number of columns (3 per row)
            num_columns = 3
            
            # Iterate over selected teams in groups of num_columns
            for start in range(0, len(selected_managers), num_columns):
                end = start + num_columns
                cols = st.columns(num_columns)

                for i, manager in enumerate(selected_managers[start:end]):
                    stats_df = all_stats.get(manager)
                    
                    with cols[i]:
                        st.write(f"### {manager}'s Players")
                        
                        if stats_df is None or stats_df.empty:
                            st.write("No data available.")
                            continue

                        # Calculate team-wide metrics for each time range
                        team_metrics = stats_df.groupby('Time Range')['FP/G'].agg(['mean', 'median', 'std']).round(2)
                        team_metrics.columns = ['Mean FP/G', 'Median FP/G', 'Std Dev']
                        st.write("Team Metrics by Time Range:")
                        st.dataframe(team_metrics)

                        # Create player performance table
                        player_performance = stats_df.pivot(index='Player', columns='Time Range', values='FP/G')
                        
                        # Calculate player-specific metrics
                        player_stats = []
                        for player in player_performance.index:
                            player_data = stats_df[stats_df['Player'] == player]
                            mean_fpg = player_data['FP/G'].mean()
                            std_fpg = player_data['FP/G'].std()
                            player_stats.append({
                                'Player': player,
                                'Average FP/G': round(mean_fpg, 2),
                                'Std Dev': round(std_fpg, 2)
                            })
                        
                        player_metrics = pd.DataFrame(player_stats).set_index('Player')
                        
                        # Combine performance and metrics
                        combined_stats = pd.concat([player_performance, player_metrics], axis=1)
                        
                        # Style the combined table
                        styled_stats = combined_stats.style\
                            .highlight_max(axis=1, subset=player_performance.columns, color='green')\
                            .highlight_min(axis=1, subset=player_performance.columns, color='red')\
                            .format("{:.2f}")
                        
                        st.write("Player Performance Breakdown:")
                        st.dataframe(styled_stats)
        
        with tab2:
            display_team_rankings(all_stats)
