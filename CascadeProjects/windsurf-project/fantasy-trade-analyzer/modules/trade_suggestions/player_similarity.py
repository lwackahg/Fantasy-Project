"""
Player Similarity Module

Core functions for calculating player similarity using Euclidean distance
and related metrics. Integrates with existing trade suggestion engine.

Based on 903_Euclidean_Distance_Implementation.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from modules.trade_suggestions.trade_suggestions_config import (
    MIN_GAMES_REQUIRED,
    AVG_GAMES_PER_PLAYER,
    ROSTER_SIZE,
)

# =============================================================================
# Configuration
# =============================================================================

# Dimension weights reflecting league philosophy (FP/G efficiency > availability > consistency)
DIMENSION_WEIGHTS = {
    'Mean FPts': 2.0,      # Primary driver of value
    'AvailabilityRatio': 1.0,  # Important for hitting 25 games
    'CV%': 0.5,            # Consistency modifier
    'Value': 1.5,          # Composite score
}

DIMENSION_NAMES = ['Mean FPts', 'AvailabilityRatio', 'CV%', 'Value']

# Column name mappings (handle both naming conventions)
FPG_COLUMNS = ['Mean FPts', 'FP/G', 'FPts/G', 'Avg FPts']  # Priority order


def _get_fpg_column(df: pd.DataFrame) -> str:
    """Find the FP/G column name in the DataFrame."""
    for col in FPG_COLUMNS:
        if col in df.columns:
            return col
    return 'Mean FPts'  # Default fallback

# Position slot configuration
POSITION_SLOTS = {'G': 3, 'F': 3, 'C': 2, 'Flex': 2}


# =============================================================================
# League Statistics
# =============================================================================

def calculate_league_stats(all_players_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate league-wide means and standard deviations for normalization.
    
    Args:
        all_players_df: DataFrame with all players in the league
        
    Returns:
        Tuple of (means_dict, stds_dict) for each dimension
    """
    means = {}
    stds = {}
    
    for col in DIMENSION_NAMES:
        # Handle FP/G column name variations
        if col == 'Mean FPts':
            # Try multiple column names for FP/G
            fpg_col = None
            for candidate in FPG_COLUMNS:
                if candidate in all_players_df.columns:
                    fpg_col = candidate
                    break
            
            if fpg_col:
                means[col] = float(all_players_df[fpg_col].mean())
                stds[col] = float(all_players_df[fpg_col].std())
                if stds[col] == 0:
                    stds[col] = 1.0
            else:
                means[col] = 70.0
                stds[col] = 25.0
        elif col in all_players_df.columns:
            means[col] = float(all_players_df[col].mean())
            stds[col] = float(all_players_df[col].std())
            # Prevent division by zero
            if stds[col] == 0:
                stds[col] = 1.0
        else:
            # Default values if column missing
            if col == 'AvailabilityRatio':
                means[col] = 0.75
                stds[col] = 0.15
            elif col == 'CV%':
                means[col] = 30.0
                stds[col] = 10.0
            elif col == 'Value':
                means[col] = 50.0
                stds[col] = 20.0
    
    return means, stds


# =============================================================================
# Player Vector Creation
# =============================================================================

def _get_fpg_value(player_row: pd.Series, default: float = 70.0) -> float:
    """Get FP/G value from player row, handling different column names."""
    for col in FPG_COLUMNS:
        if col in player_row.index:
            val = player_row.get(col)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                result = float(val)
                # Return 0 for negative values (data anomaly)
                return max(0.0, result) if result < 0 else result
    return default


def create_player_vector(
    player_row: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> np.ndarray:
    """
    Create a standardized (z-score) vector for a player.
    
    Dimensions:
    - Mean FPts: Fantasy points per game
    - AvailabilityRatio: Games played ratio (durability)
    - CV%: Coefficient of variation (inverted - lower is better)
    - Value: Composite value score
    
    Args:
        player_row: Series containing player data
        league_means: Dict of league-wide means per dimension
        league_stds: Dict of league-wide standard deviations per dimension
        
    Returns:
        numpy array of z-scores [fpg_z, avail_z, consistency_z, value_z]
    """
    # Extract raw values with defaults (handle multiple column name conventions)
    fpg = _get_fpg_value(player_row, league_means.get('Mean FPts', 70))
    availability = player_row.get('AvailabilityRatio', league_means.get('AvailabilityRatio', 0.75))
    cv_pct = player_row.get('CV%', league_means.get('CV%', 30))
    value = player_row.get('Value', league_means.get('Value', 50))
    
    # Z-score standardization
    fpg_z = (fpg - league_means['Mean FPts']) / league_stds['Mean FPts']
    avail_z = (availability - league_means['AvailabilityRatio']) / league_stds['AvailabilityRatio']
    # Invert CV% so higher = better (more consistent)
    cv_z = -(cv_pct - league_means['CV%']) / league_stds['CV%']
    value_z = (value - league_means['Value']) / league_stds['Value']
    
    return np.array([fpg_z, avail_z, cv_z, value_z])


def create_player_vector_raw(player_row: pd.Series) -> np.ndarray:
    """
    Create a raw (non-standardized) vector for a player.
    Useful for display purposes.
    """
    return np.array([
        _get_fpg_value(player_row, 0),
        player_row.get('AvailabilityRatio', 0.75),
        player_row.get('CV%', 30),
        player_row.get('Value', 50),
    ])


# =============================================================================
# Distance Calculations
# =============================================================================

def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two player vectors.
    
    Args:
        vector_a: First player's standardized vector
        vector_b: Second player's standardized vector
        
    Returns:
        Euclidean distance (lower = more similar)
    """
    return float(np.linalg.norm(vector_a - vector_b))


def weighted_euclidean_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    weights: np.ndarray = None,
) -> float:
    """
    Weighted Euclidean distance.
    
    Weights reflect league philosophy:
    - FP/G efficiency > Value > Availability > Consistency
    
    Args:
        vector_a: First player's standardized vector
        vector_b: Second player's standardized vector
        weights: Optional weight array (uses DIMENSION_WEIGHTS if None)
        
    Returns:
        Weighted Euclidean distance
    """
    if weights is None:
        weights = get_dimension_weights()
    
    diff = vector_a - vector_b
    weighted_diff = diff * np.sqrt(weights)
    return float(np.linalg.norm(weighted_diff))


def get_dimension_weights() -> np.ndarray:
    """Return weight array matching vector dimension order."""
    return np.array([
        DIMENSION_WEIGHTS['Mean FPts'],
        DIMENSION_WEIGHTS['AvailabilityRatio'],
        DIMENSION_WEIGHTS['CV%'],
        DIMENSION_WEIGHTS['Value'],
    ])


# =============================================================================
# Similarity Scoring
# =============================================================================

def player_similarity_score(
    distance: float,
    max_distance: float = 8.0,
) -> float:
    """
    Convert distance to a 0-100 similarity score.
    
    Args:
        distance: Euclidean distance between players
        max_distance: Maximum expected distance (for scaling)
        
    Returns:
        Similarity score (0-100, higher = more similar)
    """
    similarity = max(0, 1 - distance / max_distance) * 100
    return round(similarity, 1)


def calculate_similarity(
    player_a: pd.Series,
    player_b: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    use_weights: bool = True,
) -> Tuple[float, float]:
    """
    Calculate similarity between two players.
    
    Args:
        player_a: First player's data
        player_b: Second player's data
        league_means: League-wide means
        league_stds: League-wide standard deviations
        use_weights: Whether to use weighted distance
        
    Returns:
        Tuple of (similarity_score, distance)
    """
    vec_a = create_player_vector(player_a, league_means, league_stds)
    vec_b = create_player_vector(player_b, league_means, league_stds)
    
    if use_weights:
        distance = weighted_euclidean_distance(vec_a, vec_b)
    else:
        distance = euclidean_distance(vec_a, vec_b)
    
    similarity = player_similarity_score(distance)
    
    return similarity, distance


# =============================================================================
# Cosine Similarity
# =============================================================================

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Cosine similarity between two player vectors.
    
    Returns value between -1 and 1:
    - 1 = identical direction (same profile shape)
    - 0 = orthogonal (completely different profiles)
    - -1 = opposite (inverse profiles)
    
    Useful for finding players with similar "roles" regardless of production level.
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


# =============================================================================
# Player Search Functions
# =============================================================================

def find_similar_players(
    target_player: str,
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    n: int = 5,
    exclude_same_team: bool = True,
    position_filter: str = None,
    min_fpg: float = None,
) -> List[Tuple[str, float, pd.Series]]:
    """
    Find N most similar players using KNN approach.
    
    Args:
        target_player: Name of the player to find matches for
        all_players_df: DataFrame with all players
        league_means: League-wide means for normalization
        league_stds: League-wide standard deviations
        n: Number of results to return
        exclude_same_team: Whether to exclude players on same team
        position_filter: Optional position to filter by (e.g., 'G', 'F', 'C')
        min_fpg: Optional minimum FP/G threshold
        
    Returns:
        List of (player_name, similarity_score, player_data) tuples
    """
    if target_player not in all_players_df['Player'].values:
        return []
    
    target_row = all_players_df[all_players_df['Player'] == target_player].iloc[0]
    target_vector = create_player_vector(target_row, league_means, league_stds)
    target_team = target_row.get('Status', '')  # Team code in combined_data
    
    similarities = []
    
    for _, row in all_players_df.iterrows():
        if row['Player'] == target_player:
            continue
        if exclude_same_team and row.get('Status') == target_team:
            continue
        if position_filter:
            row_pos = str(row.get('Position', '')).upper()
            if position_filter.upper() not in row_pos:
                continue
        if min_fpg is not None and _get_fpg_value(row, 0) < min_fpg:
            continue
        
        player_vector = create_player_vector(row, league_means, league_stds)
        distance = weighted_euclidean_distance(target_vector, player_vector)
        similarity = player_similarity_score(distance)
        
        similarities.append((row['Player'], similarity, row))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]


def find_role_matches(
    target_player: str,
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    min_fpg: float = 50.0,
    n: int = 5,
) -> List[Tuple[str, float, float]]:
    """
    Find players with similar profiles but potentially different production.
    
    Useful for identifying:
    - Breakout candidates (similar role, lower current FP/G)
    - Buy-low targets (similar role, temporarily depressed stats)
    
    Args:
        target_player: Name of the player to find matches for
        all_players_df: DataFrame with all players
        league_means: League-wide means
        league_stds: League-wide standard deviations
        min_fpg: Minimum FP/G to consider
        n: Number of results
        
    Returns:
        List of (player_name, cosine_similarity, fpg_difference) tuples
    """
    if target_player not in all_players_df['Player'].values:
        return []
    
    target_row = all_players_df[all_players_df['Player'] == target_player].iloc[0]
    target_vector = create_player_vector(target_row, league_means, league_stds)
    target_fpg = _get_fpg_value(target_row, 0)
    
    matches = []
    
    for _, row in all_players_df.iterrows():
        if row['Player'] == target_player:
            continue
        if _get_fpg_value(row, 0) < min_fpg:
            continue
        
        player_vector = create_player_vector(row, league_means, league_stds)
        cos_sim = cosine_similarity(target_vector, player_vector)
        fpg_diff = _get_fpg_value(row, 0) - target_fpg
        
        matches.append((row['Player'], cos_sim, fpg_diff))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:n]


# =============================================================================
# Roster Analysis
# =============================================================================

def roster_centroid(
    roster_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> np.ndarray:
    """
    Calculate the centroid (average vector) of a roster.
    
    Useful for comparing team profiles and tracking how trades
    shift a team's overall composition.
    """
    if roster_df.empty:
        return np.zeros(4)
    
    vectors = []
    for _, row in roster_df.iterrows():
        vectors.append(create_player_vector(row, league_means, league_stds))
    
    return np.mean(vectors, axis=0)


def core_roster_centroid(
    roster_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    core_size: int = 8,
) -> np.ndarray:
    """
    Calculate centroid of just the core players (top N by FP/G).
    
    More representative of actual weekly performance since only
    top 7-8 players typically contribute to the 25-game minimum.
    """
    if roster_df.empty:
        return np.zeros(4)
    
    # Sort by FP/G and take top core_size
    fpg_col = _get_fpg_column(roster_df)
    sorted_roster = roster_df.sort_values(fpg_col, ascending=False)
    core_players = sorted_roster.head(core_size)
    
    return roster_centroid(core_players, league_means, league_stds)


def roster_distance(
    roster_a_df: pd.DataFrame,
    roster_b_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> float:
    """
    Calculate distance between two rosters based on their centroids.
    
    Useful for comparing team compositions or tracking how a trade
    changes a team's profile.
    """
    centroid_a = roster_centroid(roster_a_df, league_means, league_stds)
    centroid_b = roster_centroid(roster_b_df, league_means, league_stds)
    
    return euclidean_distance(centroid_a, centroid_b)


# =============================================================================
# Trade Evaluation Helpers
# =============================================================================

def trade_package_similarity(
    give_players: List[pd.Series],
    get_players: List[pd.Series],
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> Dict[str, float]:
    """
    Compare trade packages using vector analysis.
    
    Returns metrics about how similar the packages are and
    which dimensions favor which side.
    """
    if not give_players or not get_players:
        return {'similarity': 0, 'distance': float('inf')}
    
    # Calculate package centroids
    give_vectors = [create_player_vector(p, league_means, league_stds) for p in give_players]
    get_vectors = [create_player_vector(p, league_means, league_stds) for p in get_players]
    
    give_centroid = np.mean(give_vectors, axis=0)
    get_centroid = np.mean(get_vectors, axis=0)
    
    distance = weighted_euclidean_distance(give_centroid, get_centroid)
    similarity = player_similarity_score(distance)
    
    # Dimension-by-dimension comparison
    diff = get_centroid - give_centroid
    dimension_changes = {
        'fpg_change': float(diff[0]),
        'availability_change': float(diff[1]),
        'consistency_change': float(diff[2]),
        'value_change': float(diff[3]),
    }
    
    return {
        'similarity': similarity,
        'distance': distance,
        'give_centroid': give_centroid.tolist(),
        'get_centroid': get_centroid.tolist(),
        **dimension_changes,
    }


def find_replacement_candidates(
    player_to_replace: str,
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    target_teams: List[str] = None,
    n: int = 5,
) -> List[Tuple[str, float, pd.Series]]:
    """
    Find players who could replace a given player in trades.
    
    Filters to players on specified teams (trade targets) and
    returns most similar options.
    
    Args:
        player_to_replace: Name of player to find replacements for
        all_players_df: DataFrame with all players
        league_means: League-wide means
        league_stds: League-wide standard deviations
        target_teams: List of team codes to search (None = all teams)
        n: Number of results
        
    Returns:
        List of (player_name, similarity_score, player_data) tuples
    """
    if player_to_replace not in all_players_df['Player'].values:
        return []
    
    target_row = all_players_df[all_players_df['Player'] == player_to_replace].iloc[0]
    target_vector = create_player_vector(target_row, league_means, league_stds)
    target_team = target_row.get('Status', '')
    target_pos = str(target_row.get('Position', '')).upper()
    
    candidates = []
    
    for _, row in all_players_df.iterrows():
        if row['Player'] == player_to_replace:
            continue
        
        # Must be on a different team
        if row.get('Status') == target_team:
            continue
        
        # Filter to target teams if specified
        if target_teams and row.get('Status') not in target_teams:
            continue
        
        # Prefer same position
        row_pos = str(row.get('Position', '')).upper()
        position_match = any(p in row_pos for p in target_pos.split('/'))
        
        player_vector = create_player_vector(row, league_means, league_stds)
        distance = weighted_euclidean_distance(target_vector, player_vector)
        
        # Slight bonus for position match
        if position_match:
            distance *= 0.9
        
        similarity = player_similarity_score(distance)
        candidates.append((row['Player'], similarity, row))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]
