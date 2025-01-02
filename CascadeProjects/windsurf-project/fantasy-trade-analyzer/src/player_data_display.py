"""
Player Data Display Module
Handles all player data visualization and comparison functionality.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


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
            default=['FP/G']
        )
        
        # Add N-best selector
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

    # Collect player data from session state
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
        sort_order = st.selectbox("Select Time Range Order:", ["Ascending", "Descending"])
        combined_data = combined_data.sort_values(by='Time Range', 
            key=lambda x: pd.Categorical(x, TIME_RANGE_ORDER), 
            ascending=sort_order == "Descending")

        # Store the sorted data for later use in session state
        st.session_state.trend_data = {
            "player": player,
            "data": combined_data
        }

        # Metric selection with defaults
        available_metrics = ['FPts', 'FP/G', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'GP']
        y_axis_metrics = st.multiselect("Select Metrics to Plot:", available_metrics, default=['FP/G'])

        if y_axis_metrics:
            # Create a new DataFrame for charting
            plot_data = combined_data[['Time Range'] + y_axis_metrics]
            plot_data.set_index('Time Range', inplace=True)

            # Create a Plotly line chart
            fig = px.line(plot_data, x=plot_data.index, y=y_axis_metrics, markers=True, title=f'Performance Trends for {player}')
            fig.update_traces(marker=dict(size=8))  # Customize marker size

            # Show the Plotly chart in Streamlit
            st.plotly_chart(fig)

            st.write(f"Current Metrics for {player}:")
            st.dataframe(plot_data)

        else:
            st.warning("Please select at least one metric to plot.")

    else:
        st.error(f"No historical data available for {player}.")

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
        if avg_fpg > 0:
            consistency = (stats_df['FP/G'].std() / avg_fpg) if avg_fpg else float('inf')
        else:
            consistency = float('inf')
        return {'Average FP/G': round(avg_fpg, 2), 'Std Dev': round(stats_df['FP/G'].std(), 2), 
                'Consistency Score': round(consistency, 3), 'Team Size': stats_df['Player'].nunique()}

    team_rankings = pd.DataFrame(
        [{**{'Manager': manager}, **calculate_team_metrics(stats_df)} for manager, stats_df in all_stats.items() if not stats_df.empty]
    )
    
    st.subheader("Team Rankings. Average across all time ranges") 
    styled_rankings = team_rankings.style\
        .format({'Average FP/G': '{:.2f}', 'Std Dev': '{:.2f}', 'Consistency Score': '{:.3f}', 'Team Size': '{:d}'})\
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
                        st.table(team_metrics.style.highlight_max(axis=0, color='green').highlight_min(axis=0, color='red').format('{:.2f}')) 
                    else: 
                        st.write("No data available.") 
     
    # Assume all_stats and selected_managers are already defined
    st.subheader("Trade Opportunities")
    trade_data = generate_trade_opportunities(all_stats, selected_managers)
    st.write("First Select the Maximum FP/G Difference you are willing to accept")
    st.write("Then Select a Player to Analyze Opportunities")
    
    # Get max difference from user
    max_difference = get_user_max_difference()
    
    # Get all available players
    all_players = []
    for manager, stats in all_stats.items():
        if not stats.empty:
            all_players.extend(stats['Player'].unique())
    all_players = sorted(list(set(all_players)))  # Remove duplicates and sort
    
    # Player selection
    selected_player = st.selectbox(
        "Select a Player to Analyze Trade Opportunities",
        options=all_players,
        help="Choose a player to find potential trade targets"
    )
    
    if selected_player:
        # Find player's team and current FP/G
        player_team = None
        player_stats = None
        current_fpg = None
        
        for manager, stats in all_stats.items():
            if selected_player in stats['Player'].values:
                player_team = manager
                player_stats = stats[stats['Player'] == selected_player]
                current_fpg = player_stats['FP/G'].mean()
                break
        
        if player_team and current_fpg:
            st.write(f"Current Team: **{player_team}**")
            st.write(f"Current FP/G: **{current_fpg:.2f}**")
            
            # Get trade targets
            trade_targets = get_possible_trade_targets(
                selected_player,
                all_stats,
                [m for m in all_stats.keys() if m != player_team],  # Exclude player's current team
                current_fpg
            )
            
            # Display trade comparison
            if trade_targets:
                display_trade_comparison(selected_player, player_stats, trade_targets)
            else:
                st.warning("No suitable trade targets found within the specified criteria.")
        else:
            st.error("Could not find player's current team or statistics.")

def get_user_max_difference():
    """Get user-specified maximum acceptable FP/G difference with improved UI."""
    st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; border: 1px solid #4a4a4a; margin: 0.5rem 0;'>
            <h4 style='margin-top: 0;'>🎯 Trade Match Settings</h4>
            <p style='margin-bottom: 0.5rem;'>Configure how closely matched players should be for trade suggestions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        max_difference = st.slider(
            "Maximum FP/G Difference:",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help="Players with FP/G differences within this range will be considered as potential trade matches"
        )
    
    with col2:
        st.metric(
            "Match Threshold",
            f"±{max_difference:.1f} FP/G",
            delta=None,
            help="Current maximum difference allowed between players"
        )
    
    with col3:
        match_quality = "Strict" if max_difference <= 2 else "Balanced" if max_difference <= 5 else "Flexible"
        match_color = "#00ff00" if match_quality == "Strict" else "#ffbb00" if match_quality == "Balanced" else "#ff7700"
        st.markdown(f"""
            <div style='padding: 0.5rem; border-radius: 0.3rem; background-color: {match_color}20; border: 1px solid {match_color}; text-align: center;'>
                <span style='color: {match_color}; font-weight: bold;'>{match_quality}</span>
                <br/>
                <small>Match Type</small>
            </div>
        """, unsafe_allow_html=True)
    
    return max_difference

def generate_trade_opportunities(all_stats, selected_managers):
    """
    Generate enhanced trade suggestions based on comprehensive player performance analysis.
    
    Args:
        all_stats (dict): Dictionary of team stats
        selected_managers (list): List of selected manager names
        
    Returns:
        list: List of trade suggestions with format (player_name, targets, trade_details)
    """
    trade_suggestions = []

    for manager in selected_managers:
        if manager not in all_stats or all_stats[manager].empty:
            continue
            
        manager_stats = all_stats[manager]
        players = manager_stats['Player'].unique()

        for player in players:
            # Get comprehensive player stats
            player_stats = manager_stats[manager_stats['Player'] == player]
            
            # Calculate advanced player metrics
            player_metrics = {
                'current_fpg': player_stats['FP/G'].mean(),
                'recent_fpg': player_stats[player_stats['Time Range'] == '7 Days']['FP/G'].mean(),
                'trend_fpg': player_stats[player_stats['Time Range'] == '30 Days']['FP/G'].mean(),
                'games_played': len(player_stats),  # Count of records instead of GP
                'consistency': player_stats['FP/G'].std() if len(player_stats) > 1 else 0,
                'recent_games': len(player_stats[player_stats['Time Range'] == '7 Days'])
            }
            
            if pd.isna(player_metrics['current_fpg']):
                continue

            # Find trade targets with enhanced matching
            targets = []
            trade_details = []
            
            for other_manager, other_stats in all_stats.items():
                if other_manager == manager or other_stats.empty:
                    continue
                    
                for target in other_stats['Player'].unique():
                    target_stats = other_stats[other_stats['Player'] == target]
                    
                    # Calculate comprehensive target metrics
                    target_metrics = {
                        'current_fpg': target_stats['FP/G'].mean(),
                        'recent_fpg': target_stats[target_stats['Time Range'] == '7 Days']['FP/G'].mean(),
                        'trend_fpg': target_stats[target_stats['Time Range'] == '30 Days']['FP/G'].mean(),
                        'games_played': len(target_stats),  # Count of records instead of GP
                        'consistency': target_stats['FP/G'].std() if len(target_stats) > 1 else 0,
                        'recent_games': len(target_stats[target_stats['Time Range'] == '7 Days'])
                    }
                    
                    if pd.isna(target_metrics['current_fpg']):
                        continue
                    
                    # Calculate match scores for different criteria
                    fpg_ratio = target_metrics['current_fpg'] / player_metrics['current_fpg']
                    consistency_diff = abs(target_metrics['consistency'] - player_metrics['consistency'])
                    games_diff = abs(target_metrics['games_played'] - player_metrics['games_played'])
                    recent_form_diff = abs(target_metrics['recent_fpg'] - player_metrics['recent_fpg'])
                    
                    # Score the match quality (0-100)
                    match_scores = {
                        'fpg_score': 100 * (1 - abs(1 - fpg_ratio)),
                        'consistency_score': 100 * (1 - min(consistency_diff / 5, 1)),
                        'games_score': 100 * (1 - min(games_diff / 10, 1)),
                        'form_score': 100 * (1 - min(recent_form_diff / 5, 1))
                    }
                    
                    # Calculate weighted overall match score
                    overall_score = (
                        0.4 * match_scores['fpg_score'] +
                        0.25 * match_scores['consistency_score'] +
                        0.2 * match_scores['games_score'] +
                        0.15 * match_scores['form_score']
                    )
                    
                    # Add target if overall score is good enough (>70)
                    if overall_score > 70:
                        targets.append(target)
                        trade_details.append((
                            target,
                            {
                                'FP/G': target_metrics['current_fpg'],
                                'Recent FP/G': target_metrics['recent_fpg'],
                                'Trend FP/G': target_metrics['trend_fpg'],
                                'Games Played': target_metrics['games_played'],
                                'Recent Games': target_metrics['recent_games'],
                                'Consistency': target_metrics['consistency'],
                                'Match Score': round(overall_score, 1)
                            }
                        ))

            if targets:
                # Sort trade details by match score
                trade_details.sort(key=lambda x: x[1]['Match Score'], reverse=True)
                name = f"📈 {player} ➜ Trade Suggestions"
                trade_suggestions.append((name, targets, trade_details))

    return trade_suggestions

def get_possible_trade_targets(player, all_stats, selected_managers, current_fpg):
    """
    Find possible trade targets for a given player with comprehensive analysis.
    
    Args:
        player (str): Player name to find trades for
        all_stats (dict): Dictionary of team stats
        selected_managers (list): List of selected manager names
        current_fpg (float): Current player's FP/G
        
    Returns:
        list: List of potential trade targets with detailed metrics
    """
    if pd.isna(current_fpg):
        return []

    # Get player's team and comprehensive stats
    player_team = next((team for team, stats in all_stats.items() 
                       if player in stats['Player'].values), None)
    if not player_team:
        return []

    player_stats = all_stats[player_team][all_stats[player_team]['Player'] == player]
    
    # Calculate comprehensive player metrics
    player_metrics = {
        'fpg': current_fpg,
        'recent_fpg': player_stats[player_stats['Time Range'] == '7 Days']['FP/G'].mean(),
        'trend_fpg': player_stats[player_stats['Time Range'] == '30 Days']['FP/G'].mean(),
        'games_played': len(player_stats),  # Count of records instead of GP
        'consistency': player_stats['FP/G'].std() if len(player_stats) > 1 else 0,
        'recent_games': len(player_stats[player_stats['Time Range'] == '7 Days']),
        'ytd_total': player_stats[player_stats['Time Range'] == 'YTD']['FP/G'].sum()
    }

    targets = []
    for manager in selected_managers:
        if manager not in all_stats or all_stats[manager].empty or manager == player_team:
            continue

        for target in all_stats[manager]['Player'].unique():
            target_stats = all_stats[manager][all_stats[manager]['Player'] == target]
            
            # Calculate comprehensive target metrics
            target_metrics = {
                'fpg': target_stats['FP/G'].mean(),
                'recent_fpg': target_stats[target_stats['Time Range'] == '7 Days']['FP/G'].mean(),
                'trend_fpg': target_stats[target_stats['Time Range'] == '30 Days']['FP/G'].mean(),
                'games_played': len(target_stats),  # Count of records instead of GP
                'consistency': target_stats['FP/G'].std() if len(target_stats) > 1 else 0,
                'recent_games': len(target_stats[target_stats['Time Range'] == '7 Days']),
                'ytd_total': target_stats[target_stats['Time Range'] == 'YTD']['FP/G'].sum()
            }
            
            if pd.isna(target_metrics['fpg']):
                continue

            # Calculate match scores
            match_scores = {
                'fpg_match': 100 * (1 - abs(1 - target_metrics['fpg'] / player_metrics['fpg'])),
                'recent_form': 100 * (1 - min(abs(target_metrics['recent_fpg'] - player_metrics['recent_fpg']) / 5, 1)),
                'consistency': 100 * (1 - min(abs(target_metrics['consistency'] - player_metrics['consistency']) / 3, 1)),
                'games_played': 100 * (1 - min(abs(target_metrics['games_played'] - player_metrics['games_played']) / 5, 1)),
                'trend': 100 * (1 - min(abs(target_metrics['trend_fpg'] - player_metrics['trend_fpg']) / 5, 1))
            }

            # Calculate weighted overall match score
            overall_score = (
                0.35 * match_scores['fpg_match'] +
                0.25 * match_scores['recent_form'] +
                0.20 * match_scores['consistency'] +
                0.10 * match_scores['games_played'] +
                0.10 * match_scores['trend']
            )

            if overall_score >= 75:  # Higher threshold for direct matches
                targets.append({
                    'name': target,
                    'manager': manager,
                    'match_score': round(overall_score, 1),
                    'metrics': {
                        'FP/G': round(target_metrics['fpg'], 2),
                        'Recent FP/G': round(target_metrics['recent_fpg'], 2),
                        'Trend FP/G': round(target_metrics['trend_fpg'], 2),
                        'Games Played': round(target_metrics['games_played'], 1),
                        'Consistency': round(target_metrics['consistency'], 2),
                        'Recent Games': int(target_metrics['recent_games']),
                        'YTD Total': round(target_metrics['ytd_total'], 0)
                    },
                    'match_details': {k: round(v, 1) for k, v in match_scores.items()}
                })

    # Sort targets by match score
    return sorted(targets, key=lambda x: x['match_score'], reverse=True)

def display_trade_comparison(selected_player, player_stats, trade_targets):
    """
    Display enhanced trade comparison visualization with detailed metrics and charts.
    
    Args:
        selected_player (str): Name of the selected player
        player_stats (pd.DataFrame): Stats for the selected player
        trade_targets (list): List of potential trade targets with their metrics
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
        # Display top matches in a clean table
        st.subheader("Top Trade Matches")
        
        # Create a summary DataFrame
        summary_data = []
        for target in trade_targets[:5]:  # Show top 5 matches
            summary_data.append({
                'Player': target['name'],
                'Team': target['manager'],
                'Match Score': target['match_score'],
                'FP/G': target['metrics']['FP/G'],
                'Recent Form': target['metrics']['Recent FP/G'],
                'Games Played': target['metrics']['Games Played']
            })
        
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
        
        # Create metrics comparison chart
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
        
        # Create a grouped bar chart using Plotly
        fig = px.bar(
            chart_df,
            x='Metric',
            y='Value',
            color='Player',
            barmode='group',
            title='Performance Metrics Comparison',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Detailed Metrics Analysis")
        
        # Display detailed metrics for each target
        for target in trade_targets:
            with st.expander(f"📊 {target['name']} (Match Score: {target['match_score']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Performance Metrics")
                    for metric, value in target['metrics'].items():
                        st.metric(
                            metric,
                            f"{value:,.2f}",
                            delta=f"{value - player_stats['FP/G'].mean():+.2f}" if metric == 'FP/G' else None
                        )
                
                with col2:
                    st.markdown("##### Match Quality Breakdown")
                    for aspect, score in target['match_details'].items():
                        quality = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                        st.write(f"{quality} {aspect.replace('_', ' ').title()}: {score:.1f}%")

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
            st.write("Red and Green highlight indicates high and low performance for each player.")

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
                        st.table(team_metrics)

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

def display_trade_comparison(selected_player, player_stats, trade_targets):
    """
    Display enhanced trade comparison visualization with detailed metrics and charts.
    
    Args:
        selected_player (str): Name of the selected player
        player_stats (pd.DataFrame): Stats for the selected player
        trade_targets (list): List of potential trade targets with their metrics
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
        # Display top matches in a clean table
        st.subheader("Top Trade Matches")
        
        # Create a summary DataFrame
        summary_data = []
        for target in trade_targets[:5]:  # Show top 5 matches
            summary_data.append({
                'Player': target['name'],
                'Team': target['manager'],
                'Match Score': target['match_score'],
                'FP/G': target['metrics']['FP/G'],
                'Recent Form': target['metrics']['Recent FP/G'],
                'Games Played': target['metrics']['Games Played']
            })
        
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
        
        # Create metrics comparison chart
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
        
        # Create a grouped bar chart using Plotly
        fig = px.bar(
            chart_df,
            x='Metric',
            y='Value',
            color='Player',
            barmode='group',
            title='Performance Metrics Comparison',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Detailed Metrics Analysis")
        
        # Display detailed metrics for each target
        for target in trade_targets:
            with st.expander(f"📊 {target['name']} (Match Score: {target['match_score']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Performance Metrics")
                    for metric, value in target['metrics'].items():
                        st.metric(
                            metric,
                            f"{value:,.2f}",
                            delta=f"{value - player_stats['FP/G'].mean():+.2f}" if metric == 'FP/G' else None
                        )
                
                with col2:
                    st.markdown("##### Match Quality Breakdown")
                    for aspect, score in target['match_details'].items():
                        quality = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                        st.write(f"{quality} {aspect.replace('_', ' ').title()}: {score:.1f}%")
