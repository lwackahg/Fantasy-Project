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