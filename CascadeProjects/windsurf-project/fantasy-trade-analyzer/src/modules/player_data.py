"""
Player Data Module
Handles core player data processing and calculations.
"""
import pandas as pd
import numpy as np

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
    stats = []
    metric_values = {metric: [] for metric in metrics}

    for range_name, df in data_ranges.items():
        player_data = df[df['Player'] == player]
        if not player_data.empty:
            for metric in metrics:
                value = player_data[metric].iloc[0]
                metric_values[metric].append(value)
                stats.append({**{'Player': player, 'Time Range': range_name}, **{metric: value}})

    std_devs = {metric: np.std(metric_values[metric]) if metric_values[metric] else 0 for metric in metrics}
    stats_df = pd.DataFrame(stats)
    stats_df['Time Range'] = pd.Categorical(stats_df['Time Range'], TIME_RANGE_ORDER)
    stats_df.sort_values('Time Range', inplace=True)

    std_dev_df = pd.DataFrame([{'Player': player, **{f'{metric}_STD': std_devs[metric] for metric in metrics}}])

    return stats_df, std_dev_df

def calculate_team_metrics(data_ranges, players, metrics, n_best=None):
    """Calculate combined metrics for a team's top N players."""
    all_stats = []
    all_std_devs = []

    for player in players:
        stats_df, std_dev_df = calculate_player_stats(data_ranges, player, metrics)
        all_stats.append(stats_df)
        all_std_devs.append(std_dev_df)

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

def gather_player_data(data_ranges):
    """Collect and combine all player data from different time ranges."""
    all_data = [data.assign(Time_Range=key) for key, data in data_ranges.items()]
    return pd.concat(all_data).reset_index(drop=True)
