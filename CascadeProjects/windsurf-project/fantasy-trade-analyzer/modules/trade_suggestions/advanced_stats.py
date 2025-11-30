"""
Advanced Statistics Module

PCA, clustering, Monte Carlo simulation, and percentile ranking
for enhanced player analysis and trade evaluation.

Based on 903_Euclidean_Distance_Implementation.md Section 11
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter
from scipy import stats

from modules.trade_suggestions.player_similarity import (
    create_player_vector,
    DIMENSION_NAMES,
    FPG_COLUMNS,
    _get_fpg_value,
    _get_fpg_column,
)


# =============================================================================
# Principal Component Analysis (PCA)
# =============================================================================

def compute_pca_projection(
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    n_components: int = 2,
) -> Tuple[np.ndarray, object, List[str], np.ndarray]:
    """
    Project all players into 2D/3D space using PCA.
    
    Uses a simple SVD-based PCA implementation to avoid sklearn dependency.
    
    Args:
        all_players_df: DataFrame with all players
        league_means: League-wide means for normalization
        league_stds: League-wide standard deviations
        n_components: Number of principal components (2 or 3)
        
    Returns:
        Tuple of:
        - projected_coords: (n_players, n_components) array
        - explained_variance_ratio: variance explained by each component
        - player_names: list of player names in same order
        - components: principal component vectors
    """
    vectors = []
    player_names = []
    
    for _, row in all_players_df.iterrows():
        vector = create_player_vector(row, league_means, league_stds)
        vectors.append(vector)
        player_names.append(row['Player'])
    
    X = np.array(vectors)
    
    # Center the data (already z-scored, but ensure mean=0)
    X_centered = X - X.mean(axis=0)
    
    # SVD for PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Explained variance ratio
    explained_variance = (S ** 2) / (len(X) - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance
    
    # Project onto principal components
    projected = X_centered @ Vt.T[:, :n_components]
    
    return projected, explained_variance_ratio[:n_components], player_names, Vt[:n_components]


def get_pca_explained_variance(explained_variance_ratio: np.ndarray) -> Dict[str, float]:
    """Get variance explained by each principal component."""
    return {
        f"PC{i+1}": float(var) 
        for i, var in enumerate(explained_variance_ratio)
    }


def interpret_pca_components(components: np.ndarray) -> List[Dict[str, float]]:
    """
    Interpret what each principal component represents.
    
    Returns loadings for each dimension on each component.
    """
    interpretations = []
    
    for i, component in enumerate(components):
        loadings = {
            DIMENSION_NAMES[j]: float(component[j])
            for j in range(len(DIMENSION_NAMES))
        }
        interpretations.append({
            'component': f'PC{i+1}',
            'loadings': loadings,
            'dominant_feature': DIMENSION_NAMES[np.argmax(np.abs(component))],
        })
    
    return interpretations


# =============================================================================
# K-Means Clustering
# =============================================================================

def cluster_players(
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    n_clusters: int = 5,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, Dict[int, str], np.ndarray]:
    """
    Cluster players into archetypes using K-Means.
    
    Simple K-Means implementation to avoid sklearn dependency.
    
    Args:
        all_players_df: DataFrame with all players
        league_means: League-wide means
        league_stds: League-wide standard deviations
        n_clusters: Number of clusters
        max_iterations: Maximum iterations for convergence
        
    Returns:
        Tuple of:
        - cluster_labels: array of cluster assignments
        - cluster_names: dict mapping cluster_id to descriptive name
        - centroids: cluster center coordinates
    """
    vectors = []
    for _, row in all_players_df.iterrows():
        vectors.append(create_player_vector(row, league_means, league_stds))
    
    X = np.array(vectors)
    n_samples = len(X)
    
    # Initialize centroids randomly
    np.random.seed(42)
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[indices].copy()
    
    labels = np.zeros(n_samples, dtype=int)
    
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        new_labels = np.zeros(n_samples, dtype=int)
        for i, point in enumerate(X):
            distances = [np.linalg.norm(point - c) for c in centroids]
            new_labels[i] = np.argmin(distances)
        
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        
        # Update centroids
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
    
    # Name clusters based on centroid characteristics
    cluster_names = {}
    for i in range(n_clusters):
        centroid = centroids[i]
        # Interpret based on dimension order: [Mean FPts, Availability, -CV%, Value]
        fpg_z, avail_z, consistency_z, value_z = centroid
        
        if fpg_z > 1.0 and consistency_z > 0.5:
            cluster_names[i] = "Elite Studs"
        elif fpg_z > 0.5 and avail_z > 0.5:
            cluster_names[i] = "Reliable Starters"
        elif fpg_z > 0 and consistency_z < -0.5:
            cluster_names[i] = "Boom/Bust"
        elif avail_z > 1.0:
            cluster_names[i] = "Iron Men"
        elif fpg_z < -0.5:
            cluster_names[i] = "Streamers"
        else:
            cluster_names[i] = f"Role Players"
    
    return labels, cluster_names, centroids


def analyze_team_composition(
    team_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_names: Dict[int, str],
    all_players_df: pd.DataFrame,
) -> Dict[str, int]:
    """
    Analyze a team's archetype composition.
    
    Returns count of players in each archetype.
    """
    team_players = set(team_df['Player'].values)
    
    composition = Counter()
    for i, (_, row) in enumerate(all_players_df.iterrows()):
        if row['Player'] in team_players:
            cluster_id = cluster_labels[i]
            archetype = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            composition[archetype] += 1
    
    return dict(composition)


def get_cluster_summary(
    cluster_labels: np.ndarray,
    cluster_names: Dict[int, str],
    all_players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get summary statistics for each cluster.
    """
    summaries = []
    fpg_col = _get_fpg_column(all_players_df)
    
    for cluster_id, name in cluster_names.items():
        mask = cluster_labels == cluster_id
        cluster_players = all_players_df.iloc[mask]
        
        if len(cluster_players) == 0:
            continue
        
        avg_fpg = cluster_players[fpg_col].mean() if fpg_col in cluster_players.columns else 0
        avg_cv = cluster_players['CV%'].mean() if 'CV%' in cluster_players.columns else 0
        avg_value = cluster_players['Value'].mean() if 'Value' in cluster_players.columns else 0
        
        summaries.append({
            'Archetype': name,
            'Count': len(cluster_players),
            'Avg FP/G': avg_fpg if not pd.isna(avg_fpg) else 0,
            'Avg CV%': avg_cv if not pd.isna(avg_cv) else 0,
            'Avg Value': avg_value if not pd.isna(avg_value) else 0,
        })
    
    return pd.DataFrame(summaries)


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def simulate_weekly_outcome(
    team_df: pd.DataFrame,
    n_simulations: int = 1000,
    games_target: int = 25,
) -> Dict[str, float]:
    """
    Monte Carlo simulation of weekly team performance.
    
    Uses CV% to model game-to-game variance.
    
    Args:
        team_df: DataFrame with team roster
        n_simulations: Number of simulations to run
        games_target: Target games per week (default 25)
        
    Returns:
        Dict with distribution statistics (mean, std, percentiles)
    """
    weekly_totals = []
    
    # Find the correct FP/G column
    fpg_col = _get_fpg_column(team_df)
    
    for _ in range(n_simulations):
        weekly_fp = 0.0
        games_used = 0
        
        # Sort by FP/G descending (use best players first)
        sorted_team = team_df.sort_values(fpg_col, ascending=False)
        
        for _, player in sorted_team.iterrows():
            if games_used >= games_target:
                break
            
            fpg = player.get(fpg_col, 50)
            if pd.isna(fpg):
                fpg = 50
            cv_pct = player.get('CV%', 30)
            if pd.isna(cv_pct):
                cv_pct = 30
            std_dev = fpg * (cv_pct / 100)
            
            # Assume ~3.5 games per player per week
            player_games = min(4, games_target - games_used)
            
            for _ in range(player_games):
                # Sample from normal distribution
                game_fp = max(0, random.gauss(fpg, std_dev))
                weekly_fp += game_fp
                games_used += 1
                
                if games_used >= games_target:
                    break
        
        weekly_totals.append(weekly_fp)
    
    totals = np.array(weekly_totals)
    
    return {
        'mean': float(np.mean(totals)),
        'std': float(np.std(totals)),
        'p10': float(np.percentile(totals, 10)),  # Floor
        'p25': float(np.percentile(totals, 25)),
        'p50': float(np.percentile(totals, 50)),  # Median
        'p75': float(np.percentile(totals, 75)),
        'p90': float(np.percentile(totals, 90)),  # Ceiling
    }


def compare_trade_outcomes(
    your_team_before: pd.DataFrame,
    your_team_after: pd.DataFrame,
    n_simulations: int = 1000,
) -> Dict[str, any]:
    """
    Compare simulated outcomes before/after a trade.
    
    Returns probability metrics for trade evaluation.
    """
    before_stats = simulate_weekly_outcome(your_team_before, n_simulations)
    after_stats = simulate_weekly_outcome(your_team_after, n_simulations)
    
    return {
        'mean_change': after_stats['mean'] - before_stats['mean'],
        'floor_change': after_stats['p10'] - before_stats['p10'],
        'ceiling_change': after_stats['p90'] - before_stats['p90'],
        'variance_change': after_stats['std'] - before_stats['std'],
        'median_change': after_stats['p50'] - before_stats['p50'],
        'before': before_stats,
        'after': after_stats,
    }


def simulate_head_to_head(
    team_a_df: pd.DataFrame,
    team_b_df: pd.DataFrame,
    n_simulations: int = 1000,
    games_target: int = 25,
) -> Dict[str, float]:
    """
    Simulate head-to-head matchups between two teams.
    
    Returns win probability for team A.
    """
    team_a_wins = 0
    team_b_wins = 0
    ties = 0
    
    # Find correct FP/G columns
    fpg_col_a = _get_fpg_column(team_a_df)
    fpg_col_b = _get_fpg_column(team_b_df)
    
    for _ in range(n_simulations):
        # Simulate one week for each team
        team_a_fp = 0.0
        team_b_fp = 0.0
        
        # Team A
        games_used = 0
        for _, player in team_a_df.sort_values(fpg_col_a, ascending=False).iterrows():
            if games_used >= games_target:
                break
            fpg = player.get(fpg_col_a, 50)
            if pd.isna(fpg):
                fpg = 50
            cv_pct = player.get('CV%', 30)
            if pd.isna(cv_pct):
                cv_pct = 30
            std_dev = fpg * (cv_pct / 100)
            player_games = min(4, games_target - games_used)
            for _ in range(player_games):
                team_a_fp += max(0, random.gauss(fpg, std_dev))
                games_used += 1
                if games_used >= games_target:
                    break
        
        # Team B
        games_used = 0
        for _, player in team_b_df.sort_values(fpg_col_b, ascending=False).iterrows():
            if games_used >= games_target:
                break
            fpg = player.get(fpg_col_b, 50)
            if pd.isna(fpg):
                fpg = 50
            cv_pct = player.get('CV%', 30)
            if pd.isna(cv_pct):
                cv_pct = 30
            std_dev = fpg * (cv_pct / 100)
            player_games = min(4, games_target - games_used)
            for _ in range(player_games):
                team_b_fp += max(0, random.gauss(fpg, std_dev))
                games_used += 1
                if games_used >= games_target:
                    break
        
        if team_a_fp > team_b_fp:
            team_a_wins += 1
        elif team_b_fp > team_a_fp:
            team_b_wins += 1
        else:
            ties += 1
    
    return {
        'team_a_win_pct': team_a_wins / n_simulations * 100,
        'team_b_win_pct': team_b_wins / n_simulations * 100,
        'tie_pct': ties / n_simulations * 100,
    }


# =============================================================================
# Percentile Ranking
# =============================================================================

def calculate_percentile_ranks(
    all_players_df: pd.DataFrame,
    columns: List[str] = None,
) -> pd.DataFrame:
    """
    Add percentile rank columns for key metrics.
    
    Percentiles are more intuitive than z-scores for users.
    
    Args:
        all_players_df: DataFrame with all players
        columns: List of columns to calculate percentiles for
        
    Returns:
        DataFrame with added percentile columns
    """
    if columns is None:
        columns = ['Mean FPts', 'CV%', 'Value']
    
    df = all_players_df.copy()
    fpg_col = _get_fpg_column(df)
    
    for col in columns:
        # Handle FP/G column name variations
        actual_col = fpg_col if col == 'Mean FPts' else col
        
        if actual_col not in df.columns:
            continue
        
        values = df[actual_col].dropna()
        
        # For CV%, lower is better, so invert
        if col == 'CV%':
            df[f'{col}_pctl'] = df[actual_col].apply(
                lambda x: 100 - stats.percentileofscore(values, x, kind='rank') if pd.notna(x) else 50
            ).round(1)
        else:
            df[f'{col}_pctl'] = df[actual_col].apply(
                lambda x: stats.percentileofscore(values, x, kind='rank') if pd.notna(x) else 50
            ).round(1)
    
    return df


def get_player_percentile_profile(
    player_row: pd.Series,
    all_players_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Get percentile ranks for a single player across all dimensions.
    """
    profile = {}
    fpg_col = _get_fpg_column(all_players_df)
    
    for col in DIMENSION_NAMES:
        # Handle FP/G column name variations
        actual_col = fpg_col if col == 'Mean FPts' else col
        
        if actual_col not in all_players_df.columns:
            profile[col] = 50.0
            continue
        
        values = all_players_df[actual_col].dropna()
        
        # Get player value using appropriate method
        if col == 'Mean FPts':
            player_val = _get_fpg_value(player_row, None)
        else:
            player_val = player_row.get(col) if col in player_row.index else None
        
        if player_val is None or pd.isna(player_val):
            profile[col] = 50.0
        elif col == 'CV%':
            # Lower CV% is better
            profile[col] = 100 - stats.percentileofscore(values, player_val, kind='rank')
        else:
            profile[col] = stats.percentileofscore(values, player_val, kind='rank')
    
    return profile


# =============================================================================
# Trend Analysis
# =============================================================================

def calculate_momentum_score(
    recent_games: List[float],
    window_size: int = 5,
) -> float:
    """
    Calculate momentum score based on recent game performance.
    
    Positive = trending up, Negative = trending down.
    
    Args:
        recent_games: List of recent game fantasy points (newest first)
        window_size: Number of games to consider
        
    Returns:
        Momentum score (-100 to 100)
    """
    if len(recent_games) < 2:
        return 0.0
    
    games = recent_games[:window_size]
    
    # Simple linear regression slope
    x = np.arange(len(games))
    y = np.array(games[::-1])  # Reverse so oldest is first
    
    if len(x) < 2:
        return 0.0
    
    # Calculate slope
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    
    # Normalize to -100 to 100 scale
    # Assume typical FP/G is ~70, so slope of 5 per game is significant
    momentum = np.clip(slope * 20, -100, 100)
    
    return float(momentum)
