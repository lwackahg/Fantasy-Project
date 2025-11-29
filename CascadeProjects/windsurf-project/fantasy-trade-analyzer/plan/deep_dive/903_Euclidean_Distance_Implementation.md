# 903 – Euclidean Distance Implementation Plan

_Applying multi-dimensional distance metrics to improve trade analysis and player evaluation._

This document tracks the implementation of Euclidean distance and related vector-space mathematics into the fantasy trade analyzer, following the strategic framework established in:

- `900_League_Constitution.md` (rules)
- `901_League_Understanding.md` (practical strategy)
- `902_League_Advanced_Strategy_and_Modeling.md` (modeling framework)

---

## 1. The Core Insight

**The 25-game minimum with PPG penalties transforms the problem from volume-space to efficiency-space.**

This is the fundamental insight that justifies the entire Euclidean distance approach:

- Traditional fantasy: maximize total points (volume)
- This league: maximize FP/G efficiency around a fixed 25-game target
- Implication: player comparisons must be **multi-dimensional**, not just FP/G ratios

### The 7-8 Player Core Reality

From 902's modeling:
```
CoreSize ≈ MinGames ÷ G_avg
CoreSize ≈ 25 ÷ 3.5 ≈ 7.14 → 7-8 core players
```

**Key implication**: Euclidean distance should weight impact on your top 7-8 players much more heavily than changes to spots 9-10 (flex/streaming).

---

## 2. Implementation Phases

### Phase 1: Basic Euclidean Distance (Weeks 1-2)

**Goal**: Establish the foundation with simple normalized player vectors.

**Target Files**:
- New: `modules/player_similarity.py`
- Modify: `modules/trade_suggestions.py`

**Core Implementation**:

```python
import numpy as np
from typing import Dict, List, Optional
import pandas as pd

# Category weights based on league scoring (Article V)
SCORING_WEIGHTS = {
    'PTS': 2.0,
    'REB': 2.0,
    'AST': 3.0,
    '3PTM': 3.0,
    'STL': 6.0,  # High value
    'BLK': 4.0,
    'OREB': 1.0,
    '2D': 4.0,
    'TO': -2.0,
    'FG_MISS': -2.0,
    'FT_MISS': -1.0,
}

def standardize(value: float, mean: float, std: float) -> float:
    """Z-score standardization."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def create_player_vector(
    player_stats: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float]
) -> np.ndarray:
    """
    Create a standardized player vector for distance calculations.
    
    Dimensions:
    - FP/G (primary)
    - GP_rate (availability)
    - Per-game category rates (PTS, REB, AST, STL, BLK, etc.)
    - CV% (consistency, inverted so lower = better)
    """
    vector = np.array([
        standardize(player_stats.get('FP/G', 0), league_means['FP/G'], league_stds['FP/G']),
        standardize(player_stats.get('GP_rate', 0.8), league_means['GP_rate'], league_stds['GP_rate']),
        standardize(player_stats.get('PTS_per_game', 0), league_means['PTS'], league_stds['PTS']),
        standardize(player_stats.get('REB_per_game', 0), league_means['REB'], league_stds['REB']),
        standardize(player_stats.get('AST_per_game', 0), league_means['AST'], league_stds['AST']),
        standardize(player_stats.get('STL_per_game', 0), league_means['STL'], league_stds['STL']),
        standardize(player_stats.get('BLK_per_game', 0), league_means['BLK'], league_stds['BLK']),
        # Invert CV% so lower distance = more similar consistency
        -standardize(player_stats.get('CV%', 30), league_means['CV%'], league_stds['CV%']),
    ])
    return vector


def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calculate Euclidean distance between two player vectors."""
    return np.linalg.norm(vector_a - vector_b)


def player_similarity_score(distance: float, max_distance: float = 5.0) -> float:
    """
    Convert distance to a 0-100 similarity score.
    
    0 distance = 100% similar
    max_distance = 0% similar
    """
    similarity = max(0, 1 - (distance / max_distance))
    return round(similarity * 100, 1)
```

**Deliverables**:
- [ ] `create_player_vector()` function
- [ ] `euclidean_distance()` function
- [ ] `find_similar_players()` function
- [ ] League-wide standardization utilities
- [ ] Unit tests

---

### Phase 2: Weighted Distance & Correlation Awareness (Weeks 3-4)

**Goal**: Add category weights and correlation-aware comparisons (Mahalanobis distance).

**Why Mahalanobis?**
- Euclidean treats all dimensions as independent
- But PTS and FG_MISS are correlated (high scorers miss more shots)
- Mahalanobis accounts for these correlations

**Core Implementation**:

```python
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

def weighted_euclidean_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Weighted Euclidean distance.
    
    Weights should reflect category importance in league scoring.
    """
    diff = vector_a - vector_b
    weighted_diff = diff * np.sqrt(weights)
    return np.linalg.norm(weighted_diff)


def calculate_covariance_matrix(player_vectors: np.ndarray) -> np.ndarray:
    """
    Calculate the covariance matrix from all player vectors.
    
    Used for Mahalanobis distance to account for stat correlations.
    """
    return np.cov(player_vectors.T)


def mahalanobis_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """
    Mahalanobis distance - accounts for correlations between stats.
    
    More sophisticated than Euclidean, captures that:
    - High scorers tend to have more FG misses
    - Bigs tend to have more blocks AND rebounds
    - Guards tend to have more assists AND steals
    """
    try:
        cov_inv = inv(cov_matrix)
        return mahalanobis(vector_a, vector_b, cov_inv)
    except np.linalg.LinAlgError:
        # Fall back to Euclidean if covariance matrix is singular
        return euclidean_distance(vector_a, vector_b)
```

**Deliverables**:
- [ ] `weighted_euclidean_distance()` function
- [ ] `calculate_covariance_matrix()` from league data
- [ ] `mahalanobis_distance()` function
- [ ] Integration with `trade_suggestions.py`

---

### Phase 3: Position & Schedule Awareness (Weeks 5-6)

**Goal**: Add positional adjustments and basic schedule density analysis.

**Why Position Matters**:
- Guards vs Bigs have fundamentally different stat profiles
- A "similar" player should be positionally compatible
- Comparing a guard to a center in raw Euclidean space is misleading

**Core Implementation**:

```python
# Position-specific baseline profiles
POSITION_PROFILES = {
    'G': {'AST': 'high', 'STL': 'high', 'REB': 'low', 'BLK': 'low'},
    'F': {'AST': 'medium', 'STL': 'medium', 'REB': 'medium', 'BLK': 'medium'},
    'C': {'AST': 'low', 'STL': 'low', 'REB': 'high', 'BLK': 'high'},
}

def position_adjusted_distance(
    player_a: pd.Series,
    player_b: pd.Series,
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    position_penalty: float = 0.3
) -> float:
    """
    Calculate distance with position adjustment.
    
    Same-position players get raw distance.
    Different-position players get a penalty multiplier.
    """
    base_distance = euclidean_distance(vector_a, vector_b)
    
    pos_a = player_a.get('Position', 'F')
    pos_b = player_b.get('Position', 'F')
    
    if pos_a == pos_b:
        return base_distance
    elif set([pos_a, pos_b]) == {'G', 'F'} or set([pos_a, pos_b]) == {'F', 'C'}:
        # Adjacent positions: small penalty
        return base_distance * (1 + position_penalty * 0.5)
    else:
        # G vs C: full penalty
        return base_distance * (1 + position_penalty)


def schedule_fit_score(
    player_schedule: List[int],
    roster_schedule: List[int],
    target_games: int = 25
) -> float:
    """
    Calculate how well a player's schedule fits with the roster.
    
    Returns 0-1 score where 1 = perfect fit.
    """
    combined = [r + p for r, p in zip(roster_schedule, player_schedule)]
    
    # Penalize weeks that go over target
    overage_penalty = sum(max(0, week - target_games) for week in combined)
    
    # Penalize weeks that fall short
    shortage_penalty = sum(max(0, target_games - week) for week in combined)
    
    total_penalty = overage_penalty + shortage_penalty
    max_penalty = target_games * len(combined)  # Theoretical max
    
    return 1 - (total_penalty / max_penalty)
```

**Deliverables**:
- [ ] Position-adjusted distance function
- [ ] Schedule fit scoring
- [ ] Integration with roster analysis

---

## 3. Trade Package Equivalence

**The Problem**: Current N-for-M trade validation uses simple FP/G ratios.

From 902 Section 7.3:
```
minimum_return = superstar_fpg + worst_player_fpg
```

This is one-dimensional. A 2-for-1 trade might look "fair" in total FP/G but have massive **profile distance**.

**Solution**: Use Euclidean distance to validate package equivalence.

```python
def package_profile_distance(
    outgoing_players: List[pd.Series],
    incoming_players: List[pd.Series],
    league_means: Dict[str, float],
    league_stds: Dict[str, float]
) -> float:
    """
    Calculate Euclidean distance between combined package profiles.
    
    Lower distance = more equivalent packages.
    """
    # Sum the profile vectors for each package
    outgoing_vectors = [create_player_vector(p, league_means, league_stds) for p in outgoing_players]
    incoming_vectors = [create_player_vector(p, league_means, league_stds) for p in incoming_players]
    
    outgoing_profile = np.sum(outgoing_vectors, axis=0)
    incoming_profile = np.sum(incoming_vectors, axis=0)
    
    return euclidean_distance(outgoing_profile, incoming_profile)


def validate_trade_by_profile(
    outgoing: List[pd.Series],
    incoming: List[pd.Series],
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    max_profile_distance: float = 2.0
) -> Dict:
    """
    Validate a trade based on profile equivalence.
    
    Returns:
        dict with 'is_fair', 'distance', 'recommendation'
    """
    distance = package_profile_distance(outgoing, incoming, league_means, league_stds)
    
    if distance <= max_profile_distance * 0.5:
        recommendation = "Excellent match - packages are very similar"
        is_fair = True
    elif distance <= max_profile_distance:
        recommendation = "Acceptable match - some profile differences"
        is_fair = True
    elif distance <= max_profile_distance * 1.5:
        recommendation = "Marginal match - significant profile differences"
        is_fair = False
    else:
        recommendation = "Poor match - packages are fundamentally different"
        is_fair = False
    
    return {
        'is_fair': is_fair,
        'distance': round(distance, 2),
        'recommendation': recommendation
    }
```

---

## 4. Roster Balance Analysis

**Goal**: Measure how balanced a roster is in category-space.

```python
def roster_centroid(
    roster_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float]
) -> np.ndarray:
    """
    Calculate the average profile (centroid) for a roster.
    """
    vectors = [
        create_player_vector(row, league_means, league_stds)
        for _, row in roster_df.iterrows()
    ]
    return np.mean(vectors, axis=0)


def roster_balance_score(
    roster_df: pd.DataFrame,
    league_centroid: np.ndarray,
    league_means: Dict[str, float],
    league_stds: Dict[str, float]
) -> float:
    """
    Measure how balanced a roster is relative to league average.
    
    Low distance from league centroid = well-balanced roster.
    High distance = extreme strengths/weaknesses.
    """
    roster_profile = roster_centroid(roster_df, league_means, league_stds)
    return euclidean_distance(roster_profile, league_centroid)


def identify_roster_weaknesses(
    roster_df: pd.DataFrame,
    league_centroid: np.ndarray,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    category_names: List[str]
) -> List[Dict]:
    """
    Identify which categories a roster is weak in.
    
    Returns list of categories where roster is >1 std below league average.
    """
    roster_profile = roster_centroid(roster_df, league_means, league_stds)
    
    weaknesses = []
    for i, cat in enumerate(category_names):
        diff = roster_profile[i] - league_centroid[i]
        if diff < -1.0:  # More than 1 std below average
            weaknesses.append({
                'category': cat,
                'deficit': round(diff, 2),
                'severity': 'critical' if diff < -1.5 else 'moderate'
            })
    
    return sorted(weaknesses, key=lambda x: x['deficit'])
```

---

## 5. Simplified Two-Tier Distance (Recommended Approach)

Per the feedback, a simplified two-tier metric captures 90% of the value:

```python
def smart_distance(
    player_a: pd.Series,
    player_b: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    context: str = 'regular'
) -> float:
    """
    Two-tier distance metric:
    - Tier 1: Core stats that directly impact FP/G
    - Tier 2: Meta stats (consistency, availability, schedule)
    
    Context-aware weighting for regular season vs playoffs.
    """
    # Tier 1: Core production stats
    CORE_STATS = ['FP/G', 'PTS_per_game', 'REB_per_game', 'AST_per_game', 'STL_per_game', 'BLK_per_game']
    CORE_WEIGHTS = np.array([2.0, 1.0, 1.0, 1.5, 3.0, 2.0])  # Based on scoring values
    
    core_a = np.array([player_a.get(s, 0) for s in CORE_STATS])
    core_b = np.array([player_b.get(s, 0) for s in CORE_STATS])
    
    # Standardize
    core_a_std = np.array([standardize(core_a[i], league_means[s], league_stds[s]) for i, s in enumerate(CORE_STATS)])
    core_b_std = np.array([standardize(core_b[i], league_means[s], league_stds[s]) for i, s in enumerate(CORE_STATS)])
    
    core_distance = weighted_euclidean_distance(core_a_std, core_b_std, CORE_WEIGHTS)
    
    # Tier 2: Meta stats
    meta_a = np.array([
        player_a.get('CV%', 30) / 100,
        player_a.get('GP_rate', 0.8),
        player_a.get('schedule_fit', 0.5)
    ])
    meta_b = np.array([
        player_b.get('CV%', 30) / 100,
        player_b.get('GP_rate', 0.8),
        player_b.get('schedule_fit', 0.5)
    ])
    
    meta_distance = euclidean_distance(meta_a, meta_b)
    
    # Context-aware combination
    if context == 'playoffs':
        # Playoffs: consistency and availability matter more
        return 0.6 * core_distance + 0.4 * meta_distance
    else:
        # Regular season: production matters more
        return 0.7 * core_distance + 0.3 * meta_distance
```

---

## 6. Data Requirements

### Required Data Sources

| Data | Source | Status |
|------|--------|--------|
| Player FP/G | `combined_data` | ✅ Available |
| Player GP | `combined_data` | ✅ Available |
| Player CV% | `consistency_integration.py` | ✅ Available |
| Per-game category stats | Game logs (`data/player_game_log_cache/`) | ✅ Available (needs aggregation) |
| League-wide means/stds | Calculated from rosters | 🔄 To implement |
| Schedule data | Legacy module (separate update) | 🔜 Deferred |

### Data Extraction Tasks

1. **Aggregate per-game category stats from game logs**
   - Location: `data/player_game_log_cache/`
   - Data already exists in game logs, just need to calculate per-game averages
   - Categories: PTS, REB, AST, STL, BLK, TO, FG_MISS, FT_MISS

2. **Calculate league-wide standardization parameters**
   - Mean and std for each stat across all rostered players
   - Update weekly as data changes

3. **Build correlation matrix**
   - For Mahalanobis distance
   - Can be calculated once per season

> **Note**: Schedule data integration is handled by a separate legacy module and will be updated independently. Phase 3 schedule awareness is deferred until that module is ready.

---

## 7. Questions to Resolve

### From Counterpart Feedback

1. **Data availability**: Do we have historical game logs to calculate correlation matrices and entropy weights?
   - **Answer**: Yes, game logs are cached in `data/player_game_log_cache/`. Need to extract category-level stats.

2. **Computational budget**: How real-time does this need to be?
   - **Answer**: Trade suggestions run on-demand, not real-time. Can afford ~5-10 seconds of computation.

3. **User interface**: How to present multi-dimensional similarity?
   - **Proposal**: Single "similarity score" (0-100) with drill-down to category breakdown.

### Implementation Questions

4. **Core size weighting**: Should distance calculations weight core players (1-7) differently than flex (8-10)?
   - **Proposal**: Yes, use weighted average where core players have 2x weight.

5. **Position handling**: Separate distance calculations per position, or unified with penalty?
   - **Proposal**: Unified with penalty (simpler, still effective).

6. **Caching strategy**: How to cache player vectors and league stats?
   - **Proposal**: Cache league stats per session, player vectors on first calculation.

---

## 8. What We're Skipping (Initially)

Per counterpart feedback, these advanced concepts are deferred:

| Concept | Why Skip | Revisit When |
|---------|----------|--------------|
| Riemannian Geometry | Computational overhead, implementation complexity | After Phase 3 if needed |
| Quantum Superposition | Hard to validate empirically | Never (probably) |
| Full Phase Space | Velocity components can be approximated with trend analysis | After basic implementation works |
| Dynamic Time Warping | Expensive, overkill for this use case | If schedule analysis becomes critical |

---

## 9. Success Metrics

### Phase 1 Success Criteria
- [ ] Can calculate similarity score between any two players
- [ ] `find_similar_players(player_name, n=5)` returns sensible results
- [ ] Unit tests pass for all distance functions

### Phase 2 Success Criteria
- [ ] Weighted distance produces different rankings than unweighted
- [ ] Mahalanobis distance handles correlated stats correctly
- [ ] Integration with trade suggestions improves replacement matching

### Phase 3 Success Criteria
- [ ] Position-adjusted distance produces position-appropriate matches
- [ ] Schedule fit scoring identifies good/bad schedule combinations
- [ ] Roster balance analysis identifies real weaknesses

### Overall Success
- [ ] Trade suggestions are more "realistic" (fewer obviously bad trades)
- [ ] Users report better player replacement recommendations
- [ ] Package equivalence validation catches lopsided N-for-M trades

---

## 10. Next Steps

1. **Create `modules/player_similarity.py`** with Phase 1 functions
2. **Extract per-game category stats** from game log cache
3. **Calculate league-wide standardization parameters**
4. **Write unit tests** for distance functions
5. **Integrate with `trade_suggestions.py`** for replacement matching

---

## Appendix: Mathematical Reference

### Euclidean Distance
```
d(a, b) = √(Σ(aᵢ - bᵢ)²)
```

### Weighted Euclidean Distance
```
d(a, b) = √(Σwᵢ(aᵢ - bᵢ)²)
```

### Mahalanobis Distance
```
d(a, b) = √((a - b)ᵀ Σ⁻¹ (a - b))
```
Where Σ is the covariance matrix.

### Z-Score Standardization
```
z = (x - μ) / σ
```

### Similarity Score (from distance)
```
similarity = max(0, 1 - d/d_max) × 100
```

---

_Last updated: November 29, 2025_
