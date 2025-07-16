"""
Business logic for the team scouting feature.
"""

import pandas as pd
import numpy as np

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
    
    std_dev_df = pd.DataFrame([{
        'Player': player, 
        **{f'{metric}_STD': std_devs[metric] for metric in metrics}
    }])
    
    return stats_df, std_dev_df

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

    metric_tables = {}
    for metric in metrics:
        pivot = combined_stats.pivot(index='Player', columns='Time Range', values=metric)
        if n_best:
            top_players = pivot.mean(axis=1).nlargest(n_best).index
            pivot = pivot.loc[top_players]

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
