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

**Target Files** (aligned with existing codebase):
- New: `modules/trade_suggestions/player_similarity.py`
- Modify: `modules/trade_suggestions/trade_suggestions_core.py`
- Leverage: `modules/trade_analysis/consistency_integration.py` (existing CV% data)
- Leverage: `modules/player_value/logic.py` (existing value scoring)

**Data Sources Already Available**:
- `Mean FPts` (FP/G) — from `combined_data` DataFrame
- `CV%`, `Boom%`, `Bust%` — from `consistency_integration.py`
- `AvailabilityRatio` — from `player_value/logic.py`
- `Value` — from `trade_suggestions_core.py` calculations

**Core Implementation**:

```python
import numpy as np
from typing import Dict, List, Optional
import pandas as pd

# Import existing consistency utilities
from modules.trade_analysis.consistency_integration import (
    load_player_consistency,
    build_league_consistency_index,
)
from modules.trade_suggestions.trade_suggestions_config import (
    ROSTER_SIZE,
    MIN_GAMES_REQUIRED,
    AVG_GAMES_PER_PLAYER,
)


def standardize(value: float, mean: float, std: float) -> float:
    """Z-score standardization."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def create_player_vector(
    player_row: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float]
) -> np.ndarray:
    """
    Create a standardized player vector for distance calculations.
    
    Uses actual column names from the codebase:
    - 'Mean FPts' (FP/G from combined_data)
    - 'CV%' (from consistency_integration)
    - 'Value' (from trade_suggestions_core)
    - 'AvailabilityRatio' (from player_value/logic)
    """
    vector = np.array([
        # Primary: FP/G efficiency (most important in this league)
        standardize(
            player_row.get('Mean FPts', 0),
            league_means.get('Mean FPts', 50),
            league_stds.get('Mean FPts', 20)
        ),
        # Availability (games played rate)
        standardize(
            player_row.get('AvailabilityRatio', 0.8),
            league_means.get('AvailabilityRatio', 0.75),
            league_stds.get('AvailabilityRatio', 0.15)
        ),
        # Consistency (CV% inverted so lower = better)
        -standardize(
            player_row.get('CV%', 30),
            league_means.get('CV%', 30),
            league_stds.get('CV%', 10)
        ),
        # Composite value score
        standardize(
            player_row.get('Value', 50),
            league_means.get('Value', 50),
            league_stds.get('Value', 20)
        ),
    ])
    return vector


def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calculate Euclidean distance between two player vectors."""
    return float(np.linalg.norm(vector_a - vector_b))


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
- [ ] `create_player_vector()` function using existing data columns
- [ ] `euclidean_distance()` function
- [ ] `find_similar_players()` function
- [ ] `calculate_league_stats()` to compute means/stds from roster data
- [ ] Unit tests in `tests/test_player_similarity.py`

---

### Phase 2: Weighted Distance & Core-Aware Comparisons (Weeks 3-4)

**Goal**: Add league-specific weights and core-size awareness.

**Why Core-Weighted?**
- From 902: only top 7-8 players drive weekly scoring
- Changes to core spots matter more than flex/streaming slots
- Integrate with existing `_get_core_size()` and `_calculate_core_value()` from `trade_suggestions_core.py`

**Core Implementation**:

```python
from modules.trade_suggestions.trade_suggestions_core import (
    _get_core_size,
    _calculate_core_value,
)
from modules.trade_suggestions.trade_suggestions_config import (
    MIN_GAMES_REQUIRED,
    AVG_GAMES_PER_PLAYER,
)

# Weights aligned with league philosophy from 901/902
# FP/G is king, availability enables it, consistency reduces variance
DIMENSION_WEIGHTS = {
    'Mean FPts': 2.0,       # Primary driver (FP/G efficiency)
    'AvailabilityRatio': 1.0,  # Enables hitting 25-game minimum
    'CV%': 0.5,             # Consistency modifier
    'Value': 1.5,           # Composite value (includes scarcity)
}


def weighted_euclidean_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Weighted Euclidean distance.
    
    Weights reflect league philosophy:
    - FP/G efficiency > availability > consistency
    """
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


def core_weighted_similarity(
    player_a: pd.Series,
    player_b: pd.Series,
    player_a_rank: int,
    player_b_rank: int,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    core_size: int = None,
) -> float:
    """
    Calculate similarity with core-position weighting.
    
    Players in core positions (1-7) get higher weight than flex (8-10).
    This aligns with 902's insight that only top 7-8 players matter.
    """
    if core_size is None:
        core_size = _get_core_size(MIN_GAMES_REQUIRED, AVG_GAMES_PER_PLAYER)
    
    vector_a = create_player_vector(player_a, league_means, league_stds)
    vector_b = create_player_vector(player_b, league_means, league_stds)
    
    base_weights = get_dimension_weights()
    
    # Apply core multiplier: core players get 2x weight
    core_mult_a = 2.0 if player_a_rank <= core_size else 1.0
    core_mult_b = 2.0 if player_b_rank <= core_size else 1.0
    avg_core_mult = (core_mult_a + core_mult_b) / 2
    
    adjusted_weights = base_weights * avg_core_mult
    
    distance = weighted_euclidean_distance(vector_a, vector_b, adjusted_weights)
    return player_similarity_score(distance)
```

**Deliverables**:
- [ ] `weighted_euclidean_distance()` function
- [ ] `get_dimension_weights()` aligned with league scoring
- [ ] `core_weighted_similarity()` integrating with existing core logic
- [ ] Integration with `trade_suggestions_core.py`

---

### Phase 3: Position & Roster Context Awareness (Weeks 5-6)

**Goal**: Add positional adjustments and roster-context awareness.

**Why Position Matters**:
- League uses 3G / 3F / 2C / 2Flex daily lineups
- Position eligibility affects trade feasibility
- Similar FP/G players at different positions have different roster value

**Integration with Existing Data**:
- Position data comes from `combined_data['Position']` column
- Roster context from `trade_suggestions_core._calculate_floor_impact()`

**Core Implementation**:

```python
from modules.trade_suggestions.trade_suggestions_core import (
    _calculate_floor_impact,
    _simulate_core_value_gain,
)

# Position slots from league rules (Article IV)
POSITION_SLOTS = {
    'G': 3,   # 3 Guard slots
    'F': 3,   # 3 Forward slots  
    'C': 2,   # 2 Center slots
    'Flex': 2,  # 2 Flex slots (any position)
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
    
    # Use 'Position' column from combined_data
    pos_a = str(player_a.get('Position', 'F')).upper()
    pos_b = str(player_b.get('Position', 'F')).upper()
    
    # Normalize multi-position eligibility (e.g., "G/F" -> primary)
    pos_a = pos_a.split('/')[0] if '/' in pos_a else pos_a
    pos_b = pos_b.split('/')[0] if '/' in pos_b else pos_b
    
    if pos_a == pos_b:
        return base_distance
    elif set([pos_a, pos_b]) == {'G', 'F'} or set([pos_a, pos_b]) == {'F', 'C'}:
        # Adjacent positions: small penalty
        return base_distance * (1 + position_penalty * 0.5)
    else:
        # G vs C: full penalty
        return base_distance * (1 + position_penalty)


def roster_context_similarity(
    player: pd.Series,
    your_team: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> Dict[str, float]:
    """
    Evaluate how a player fits into a specific roster context.
    
    Returns dict with:
    - 'core_fit': How well player fits into core (top 7-8)
    - 'floor_impact': Effect on roster floor (bottom 2-3)
    - 'position_fit': Whether roster needs this position
    """
    from modules.trade_suggestions.trade_suggestions_core import _get_core_size
    
    core_size = _get_core_size()
    
    # Simulate adding this player
    player_fpg = player.get('Mean FPts', 0)
    team_sorted = your_team.sort_values('Mean FPts', ascending=False)
    
    # Would this player make the core?
    if len(team_sorted) >= core_size:
        core_threshold = team_sorted.iloc[core_size - 1]['Mean FPts']
        core_fit = 1.0 if player_fpg > core_threshold else 0.5
    else:
        core_fit = 1.0  # Team has room in core
    
    # Position need analysis
    pos = str(player.get('Position', 'F')).upper().split('/')[0]
    pos_counts = your_team['Position'].apply(
        lambda x: str(x).upper().split('/')[0]
    ).value_counts().to_dict()
    
    current_count = pos_counts.get(pos, 0)
    max_slots = POSITION_SLOTS.get(pos, 2)
    position_fit = 1.0 if current_count < max_slots else 0.7
    
    # Floor impact (using existing function)
    floor_impact = _calculate_floor_impact(
        your_team,
        [],  # Not giving anyone
        [player],  # Adding this player
        floor_size=2
    )
    
    return {
        'core_fit': core_fit,
        'floor_impact': floor_impact,
        'position_fit': position_fit,
    }
```

**Deliverables**:
- [ ] Position-adjusted distance function
- [ ] `roster_context_similarity()` using existing core/floor logic
- [ ] Integration with `trade_suggestions_core.py`

---

## 3. Trade Package Equivalence

**The Problem**: Current N-for-M trade validation uses FP/G ratios and core impact.

From 902 Section 7.3:
```
minimum_return = superstar_fpg + worst_player_fpg
```

**Existing Validation** (in `trade_suggestions_realism.py`):
- `_check_opponent_core_avg_drop()` — ensures opponent's core FP/G doesn't drop too much
- `MAX_OPP_CORE_AVG_DROP` — configurable threshold (default 1.5 FP/G)

**Enhancement**: Add profile distance as a secondary validation layer.

```python
from modules.trade_suggestions.trade_suggestions_core import (
    _check_opponent_core_avg_drop,
    _simulate_core_value_gain,
    _determine_trade_reasoning,
)
from modules.trade_suggestions.trade_suggestions_config import (
    MAX_OPP_CORE_AVG_DROP,
)


def package_profile_distance(
    outgoing_players: List[pd.Series],
    incoming_players: List[pd.Series],
    league_means: Dict[str, float],
    league_stds: Dict[str, float]
) -> float:
    """
    Calculate Euclidean distance between combined package profiles.
    
    Lower distance = more equivalent packages.
    Complements existing core-drop validation.
    """
    outgoing_vectors = [create_player_vector(p, league_means, league_stds) for p in outgoing_players]
    incoming_vectors = [create_player_vector(p, league_means, league_stds) for p in incoming_players]
    
    outgoing_profile = np.sum(outgoing_vectors, axis=0)
    incoming_profile = np.sum(incoming_vectors, axis=0)
    
    return euclidean_distance(outgoing_profile, incoming_profile)


def validate_trade_by_profile(
    outgoing: List[pd.Series],
    incoming: List[pd.Series],
    your_team: pd.DataFrame,
    opp_team: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    core_size: int,
) -> Dict:
    """
    Validate a trade using both existing core logic AND profile distance.
    
    Integrates with existing validation from trade_suggestions_core.py.
    
    Returns:
        dict with 'is_fair', 'profile_distance', 'core_gain', 'reasoning'
    """
    # Profile distance (new)
    profile_dist = package_profile_distance(outgoing, incoming, league_means, league_stds)
    
    # Core value gain (existing logic)
    baseline_core = sum(
        your_team.nlargest(core_size, 'Mean FPts')['Value']
    ) if not your_team.empty else 0
    
    core_gain = _simulate_core_value_gain(
        your_team, outgoing, incoming, core_size, baseline_core
    )
    
    # Floor impact (existing logic)
    from modules.trade_suggestions.trade_suggestions_core import _calculate_floor_impact
    floor_delta = _calculate_floor_impact(your_team, outgoing, incoming)
    
    # Trade reasoning (existing logic)
    reasoning = _determine_trade_reasoning(core_gain, floor_delta)
    
    # Combined fairness check
    max_profile_distance = 3.0  # Tunable threshold
    is_fair = profile_dist <= max_profile_distance and core_gain >= -10.0
    
    return {
        'is_fair': is_fair,
        'profile_distance': round(profile_dist, 2),
        'core_gain': round(core_gain, 1),
        'floor_delta': round(floor_delta, 1),
        'reasoning': reasoning,
    }
```

---

## 4. Roster Balance Analysis

**Goal**: Measure roster balance using existing value infrastructure.

**Integration with Existing Code**:
- `player_value/logic.py` already computes `ProductionScore`, `ConsistencyScore`, `AvailabilityScore`
- `trade_suggestions_core.py` has `_calculate_core_value()` and `_calculate_floor_impact()`

```python
from modules.player_value.logic import build_player_value_profiles
from modules.trade_suggestions.trade_suggestions_core import (
    _get_core_size,
    _calculate_core_value,
)

# Dimension names matching our vector structure
DIMENSION_NAMES = ['Mean FPts', 'AvailabilityRatio', 'CV%', 'Value']


def roster_centroid(
    roster_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float]
) -> np.ndarray:
    """
    Calculate the average profile (centroid) for a roster.
    
    Uses 'Player' column to match with combined_data structure.
    """
    if roster_df.empty or 'Player' not in roster_df.columns:
        return np.zeros(len(DIMENSION_NAMES))
    
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
) -> List[Dict]:
    """
    Identify which dimensions a roster is weak in.
    
    Returns list of dimensions where roster is >1 std below league average.
    """
    roster_profile = roster_centroid(roster_df, league_means, league_stds)
    
    weaknesses = []
    for i, dim in enumerate(DIMENSION_NAMES):
        diff = roster_profile[i] - league_centroid[i]
        
        # For CV%, negative diff is actually good (lower variance)
        if dim == 'CV%':
            diff = -diff  # Invert for interpretation
        
        if diff < -1.0:  # More than 1 std below average
            weaknesses.append({
                'dimension': dim,
                'deficit': round(diff, 2),
                'severity': 'critical' if diff < -1.5 else 'moderate'
            })
    
    return sorted(weaknesses, key=lambda x: x['deficit'])


def calculate_league_stats(all_rosters: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate league-wide means and standard deviations for normalization.
    
    Args:
        all_rosters: Combined DataFrame of all rostered players
        
    Returns:
        Tuple of (means_dict, stds_dict)
    """
    means = {}
    stds = {}
    
    for col in DIMENSION_NAMES:
        if col in all_rosters.columns:
            means[col] = float(all_rosters[col].mean())
            stds[col] = float(all_rosters[col].std())
        else:
            # Defaults if column missing
            defaults = {
                'Mean FPts': (50.0, 20.0),
                'AvailabilityRatio': (0.75, 0.15),
                'CV%': (30.0, 10.0),
                'Value': (50.0, 20.0),
            }
            means[col], stds[col] = defaults.get(col, (0.0, 1.0))
    
    return means, stds
```

---

## 5. Simplified Two-Tier Distance (Recommended Approach)

A simplified two-tier metric using existing data columns captures 90% of the value:

```python
def smart_distance(
    player_a: pd.Series,
    player_b: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    context: str = 'regular'
) -> float:
    """
    Two-tier distance metric using existing codebase columns:
    - Tier 1: Production (Mean FPts, Value)
    - Tier 2: Reliability (CV%, AvailabilityRatio)
    
    Context-aware weighting for regular season vs playoffs.
    Aligns with 901/902 philosophy: FP/G efficiency is king.
    """
    # Tier 1: Production stats (from combined_data)
    PRODUCTION_COLS = ['Mean FPts', 'Value']
    PRODUCTION_WEIGHTS = np.array([2.0, 1.5])  # FP/G weighted higher
    
    prod_a = np.array([player_a.get(s, 0) for s in PRODUCTION_COLS])
    prod_b = np.array([player_b.get(s, 0) for s in PRODUCTION_COLS])
    
    # Standardize
    prod_a_std = np.array([
        standardize(prod_a[i], league_means.get(s, 50), league_stds.get(s, 20))
        for i, s in enumerate(PRODUCTION_COLS)
    ])
    prod_b_std = np.array([
        standardize(prod_b[i], league_means.get(s, 50), league_stds.get(s, 20))
        for i, s in enumerate(PRODUCTION_COLS)
    ])
    
    prod_distance = weighted_euclidean_distance(prod_a_std, prod_b_std, PRODUCTION_WEIGHTS)
    
    # Tier 2: Reliability stats (from consistency_integration + player_value)
    # Note: CV% is inverted (lower is better)
    reliability_a = np.array([
        -player_a.get('CV%', 30) / 100,  # Inverted
        player_a.get('AvailabilityRatio', 0.8),
    ])
    reliability_b = np.array([
        -player_b.get('CV%', 30) / 100,
        player_b.get('AvailabilityRatio', 0.8),
    ])
    
    reliability_distance = euclidean_distance(reliability_a, reliability_b)
    
    # Context-aware combination (from 901 strategy)
    if context == 'playoffs':
        # Playoffs: consistency and availability matter more
        # "Priority shifts heavily to FP/G and consistency" - 901 Section 5.3
        return 0.6 * prod_distance + 0.4 * reliability_distance
    else:
        # Regular season: production matters more
        return 0.7 * prod_distance + 0.3 * reliability_distance
```

---

## 6. Data Requirements

### Required Data Sources (Mapped to Existing Codebase)

| Data | Source | Column Name | Status |
|------|--------|-------------|--------|
| Player FP/G | `combined_data` DataFrame | `Mean FPts` | ✅ Available |
| Player GP | `combined_data` DataFrame | `GP` | ✅ Available |
| Player Value | `trade_suggestions_core.py` | `Value` | ✅ Available |
| Player CV% | `consistency_integration.py` | `CV%` | ✅ Available |
| Boom/Bust rates | `consistency_integration.py` | `Boom%`, `Bust%` | ✅ Available |
| Availability | `player_value/logic.py` | `AvailabilityRatio` | ✅ Available |
| Position | `combined_data` DataFrame | `Position` | ✅ Available |
| League-wide stats | To calculate from rosters | N/A | 🔄 To implement |

### Existing Functions to Leverage

| Function | Module | Purpose |
|----------|--------|--------|
| `load_player_consistency()` | `consistency_integration.py` | Get CV%, boom/bust for a player |
| `build_league_consistency_index()` | `consistency_integration.py` | Batch load all player consistency |
| `enrich_roster_with_consistency()` | `consistency_integration.py` | Add CV% columns to roster DataFrame |
| `build_player_value_profiles()` | `player_value/logic.py` | Get availability, production scores |
| `_get_core_size()` | `trade_suggestions_core.py` | Calculate core size (7-8 players) |
| `_calculate_core_value()` | `trade_suggestions_core.py` | Sum value of top N players |
| `_calculate_floor_impact()` | `trade_suggestions_core.py` | Measure floor change from trade |

### Implementation Tasks

1. **Create `calculate_league_stats()` function**
   - Compute mean/std for `Mean FPts`, `CV%`, `AvailabilityRatio`, `Value`
   - Use all rostered players from `combined_data`
   - Cache per session

2. **Create `create_player_vector()` function**
   - Use existing column names from codebase
   - Standardize using league stats

3. **Integrate with `trade_suggestions_core.py`**
   - Add similarity scoring to replacement matching
   - Use existing core/floor logic

---

## 7. Questions to Resolve

### From Counterpart Feedback

1. **Data availability**: Do we have the data we need?
   - **Answer**: Yes. All required data exists in the codebase:
     - `Mean FPts` in `combined_data`
     - `CV%` via `consistency_integration.py`
     - `AvailabilityRatio` via `player_value/logic.py`
     - `Value` via `trade_suggestions_core.py`

2. **Computational budget**: How real-time does this need to be?
   - **Answer**: Trade suggestions run on-demand. Current engine has `MAX_COMPLEXITY_OPS = 200_000` cap. Similarity calculations are O(1) per player pair, so no concern.

3. **User interface**: How to present multi-dimensional similarity?
   - **Proposal**: Single "similarity score" (0-100) in trade suggestion output. Existing UI in `trade_suggestions_ui_tab.py` can display it.

### Implementation Questions

4. **Core size weighting**: Should distance calculations weight core players (1-7) differently than flex (8-10)?
   - **Answer**: Yes. Use `_get_core_size()` from `trade_suggestions_core.py` (returns 7-8 based on `MIN_GAMES_REQUIRED / AVG_GAMES_PER_PLAYER`). Core players get 2x weight.

5. **Position handling**: Separate distance calculations per position, or unified with penalty?
   - **Answer**: Unified with penalty. Position data from `combined_data['Position']` column. Penalty of 0.3 for non-adjacent positions.

6. **Caching strategy**: How to cache player vectors and league stats?
   - **Answer**: Use existing patterns:
     - `@lru_cache` decorator (see `consistency_integration.py` line 17)
     - Session-level caching via Streamlit `st.session_state`

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

1. **Create `modules/trade_suggestions/player_similarity.py`** with Phase 1 functions
   - `create_player_vector()` using existing column names
   - `euclidean_distance()` and `player_similarity_score()`
   - `calculate_league_stats()` for normalization

2. **Integrate with existing consistency data**
   - Use `build_league_consistency_index()` from `consistency_integration.py`
   - Enrich player vectors with CV%, Boom%, Bust%

3. **Add `find_similar_players()` function**
   - Input: player name, roster DataFrame, n (number of results)
   - Output: List of similar players with similarity scores

4. **Write unit tests** in `tests/test_player_similarity.py`
   - Test vector creation with mock data
   - Test distance calculations
   - Test similarity scoring

5. **Integrate with `trade_suggestions_core.py`**
   - Add similarity scoring to replacement matching
   - Use in `_simulate_core_value_gain()` for better replacements

---

## 11. Advanced Statistical Methods & Algorithms

Beyond basic Euclidean distance, these additional techniques provide competitive advantages:

### 11.1. K-Nearest Neighbors (KNN) for Player Replacement

**Purpose**: Find the N most similar players to a given player for trade targeting.

```python
from typing import List, Tuple
from functools import lru_cache

def find_similar_players(
    target_player: str,
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    n: int = 5,
    exclude_same_team: bool = True,
    position_filter: str = None,
) -> List[Tuple[str, float, pd.Series]]:
    """
    Find N most similar players using KNN approach.
    
    Returns list of (player_name, similarity_score, player_data) tuples.
    Integrates with existing roster data structure.
    """
    if target_player not in all_players_df['Player'].values:
        return []
    
    target_row = all_players_df[all_players_df['Player'] == target_player].iloc[0]
    target_vector = create_player_vector(target_row, league_means, league_stds)
    target_team = target_row.get('Status', '')  # Team code in combined_data
    target_pos = str(target_row.get('Position', '')).upper()
    
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
        
        player_vector = create_player_vector(row, league_means, league_stds)
        distance = euclidean_distance(target_vector, player_vector)
        similarity = player_similarity_score(distance)
        
        similarities.append((row['Player'], similarity, row))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]
```

### 11.2. Cosine Similarity for Profile Matching

**Purpose**: Compare player "shapes" regardless of magnitude (useful for role-based matching).

```python
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
    
    Returns: (player_name, cosine_similarity, fpg_difference)
    """
    target_row = all_players_df[all_players_df['Player'] == target_player].iloc[0]
    target_vector = create_player_vector(target_row, league_means, league_stds)
    target_fpg = target_row.get('Mean FPts', 0)
    
    matches = []
    
    for _, row in all_players_df.iterrows():
        if row['Player'] == target_player:
            continue
        if row.get('Mean FPts', 0) < min_fpg:
            continue
        
        player_vector = create_player_vector(row, league_means, league_stds)
        cos_sim = cosine_similarity(target_vector, player_vector)
        fpg_diff = row.get('Mean FPts', 0) - target_fpg
        
        matches.append((row['Player'], cos_sim, fpg_diff))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:n]
```

### 11.3. Principal Component Analysis (PCA) for Dimensionality Reduction

**Purpose**: Reduce player vectors to 2-3 dimensions for visualization while preserving variance.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_pca_projection(
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    n_components: int = 2,
) -> Tuple[np.ndarray, PCA, List[str]]:
    """
    Project all players into 2D/3D space using PCA.
    
    Returns:
        - projected_coords: (n_players, n_components) array
        - pca_model: fitted PCA for inverse transform
        - player_names: list of player names in same order
    """
    vectors = []
    player_names = []
    
    for _, row in all_players_df.iterrows():
        vector = create_player_vector(row, league_means, league_stds)
        vectors.append(vector)
        player_names.append(row['Player'])
    
    X = np.array(vectors)
    
    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(X_scaled)
    
    return projected, pca, player_names


def get_pca_explained_variance(pca: PCA) -> Dict[str, float]:
    """Get variance explained by each principal component."""
    return {
        f"PC{i+1}": float(var) 
        for i, var in enumerate(pca.explained_variance_ratio_)
    }
```

### 11.4. Clustering for Team Archetype Analysis

**Purpose**: Group players/teams into strategic archetypes.

```python
from sklearn.cluster import KMeans
from collections import Counter

def cluster_players(
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    n_clusters: int = 5,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Cluster players into archetypes using K-Means.
    
    Returns:
        - cluster_labels: array of cluster assignments
        - cluster_names: dict mapping cluster_id to descriptive name
    """
    vectors = []
    for _, row in all_players_df.iterrows():
        vectors.append(create_player_vector(row, league_means, league_stds))
    
    X = np.array(vectors)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Name clusters based on centroid characteristics
    cluster_names = {}
    for i in range(n_clusters):
        centroid = kmeans.cluster_centers_[i]
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
            cluster_names[i] = f"Cluster {i}"
    
    return labels, cluster_names


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
```

### 11.5. Monte Carlo Simulation for Trade Outcome Projection

**Purpose**: Simulate weekly outcomes to estimate win probability changes.

```python
import random

def simulate_weekly_outcome(
    team_df: pd.DataFrame,
    n_simulations: int = 1000,
    games_target: int = 25,
) -> Dict[str, float]:
    """
    Monte Carlo simulation of weekly team performance.
    
    Uses CV% to model game-to-game variance.
    Returns distribution statistics.
    """
    weekly_totals = []
    
    for _ in range(n_simulations):
        weekly_fp = 0.0
        games_used = 0
        
        # Sort by FP/G descending (use best players first)
        sorted_team = team_df.sort_values('Mean FPts', ascending=False)
        
        for _, player in sorted_team.iterrows():
            if games_used >= games_target:
                break
            
            fpg = player.get('Mean FPts', 50)
            cv_pct = player.get('CV%', 30)
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
        'p50': float(np.percentile(totals, 50)),  # Median
        'p90': float(np.percentile(totals, 90)),  # Ceiling
    }


def compare_trade_outcomes(
    your_team_before: pd.DataFrame,
    your_team_after: pd.DataFrame,
    n_simulations: int = 1000,
) -> Dict[str, float]:
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
        'before': before_stats,
        'after': after_stats,
    }
```

### 11.6. Percentile Rank System

**Purpose**: Express player value as league-wide percentile for intuitive comparison.

```python
from scipy import stats

def calculate_percentile_ranks(
    all_players_df: pd.DataFrame,
    columns: List[str] = None,
) -> pd.DataFrame:
    """
    Add percentile rank columns for key metrics.
    
    Percentiles are more intuitive than z-scores for users.
    """
    if columns is None:
        columns = ['Mean FPts', 'CV%', 'Value']
    
    df = all_players_df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # For CV%, lower is better, so invert
        if col == 'CV%':
            df[f'{col}_pctl'] = 100 - stats.percentileofscore(
                df[col].dropna(), df[col], kind='rank'
            ).round(1)
        else:
            df[f'{col}_pctl'] = df[col].apply(
                lambda x: stats.percentileofscore(df[col].dropna(), x, kind='rank')
            ).round(1)
    
    return df
```

---

## 12. Streamlit Visualization Components

These visualization components integrate with the existing UI patterns in `trade_suggestions_ui_tab.py`.

### 12.1. Player Similarity Radar Chart

```python
import plotly.graph_objects as go

def render_player_comparison_radar(
    player_a: pd.Series,
    player_b: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> go.Figure:
    """
    Radar chart comparing two players across all dimensions.
    
    Uses Plotly for interactive visualization.
    """
    dimensions = ['Mean FPts', 'Availability', 'Consistency', 'Value']
    
    # Normalize to 0-100 scale for display
    def normalize_for_display(val, mean, std, invert=False):
        z = (val - mean) / std if std > 0 else 0
        # Convert z-score to 0-100 (z of -2 = 0, z of +2 = 100)
        normalized = 50 + (z * 25)
        if invert:
            normalized = 100 - normalized
        return max(0, min(100, normalized))
    
    values_a = [
        normalize_for_display(player_a.get('Mean FPts', 50), league_means['Mean FPts'], league_stds['Mean FPts']),
        normalize_for_display(player_a.get('AvailabilityRatio', 0.75) * 100, 75, 15),
        normalize_for_display(player_a.get('CV%', 30), league_means['CV%'], league_stds['CV%'], invert=True),
        normalize_for_display(player_a.get('Value', 50), league_means['Value'], league_stds['Value']),
    ]
    
    values_b = [
        normalize_for_display(player_b.get('Mean FPts', 50), league_means['Mean FPts'], league_stds['Mean FPts']),
        normalize_for_display(player_b.get('AvailabilityRatio', 0.75) * 100, 75, 15),
        normalize_for_display(player_b.get('CV%', 30), league_means['CV%'], league_stds['CV%'], invert=True),
        normalize_for_display(player_b.get('Value', 50), league_means['Value'], league_stds['Value']),
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_a + [values_a[0]],  # Close the polygon
        theta=dimensions + [dimensions[0]],
        fill='toself',
        name=player_a.get('Player', 'Player A'),
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.3)',
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=values_b + [values_b[0]],
        theta=dimensions + [dimensions[0]],
        fill='toself',
        name=player_b.get('Player', 'Player B'),
        line_color='#2196F3',
        fillcolor='rgba(33, 150, 243, 0.3)',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=True,
        title="Player Profile Comparison",
        height=400,
    )
    
    return fig


def display_player_comparison(player_a: pd.Series, player_b: pd.Series, 
                               league_means: Dict, league_stds: Dict):
    """Streamlit component for player comparison."""
    st.subheader("📊 Player Profile Comparison")
    
    # Calculate similarity
    vec_a = create_player_vector(player_a, league_means, league_stds)
    vec_b = create_player_vector(player_b, league_means, league_stds)
    distance = euclidean_distance(vec_a, vec_b)
    similarity = player_similarity_score(distance)
    
    # Display similarity score
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Similarity Score", f"{similarity:.1f}%")
    with col2:
        st.metric("Euclidean Distance", f"{distance:.2f}")
    with col3:
        cos_sim = cosine_similarity(vec_a, vec_b)
        st.metric("Profile Match", f"{cos_sim*100:.1f}%")
    
    # Radar chart
    fig = render_player_comparison_radar(player_a, player_b, league_means, league_stds)
    st.plotly_chart(fig, use_container_width=True)
```

### 12.2. League-Wide Player Map (2D PCA Scatter)

```python
import plotly.express as px

def render_league_player_map(
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    highlight_players: List[str] = None,
    color_by: str = 'Position',
) -> go.Figure:
    """
    2D scatter plot of all players using PCA projection.
    
    Allows visual identification of similar players and team compositions.
    """
    # Compute PCA projection
    projected, pca, player_names = compute_pca_projection(
        all_players_df, league_means, league_stds, n_components=2
    )
    
    # Build DataFrame for plotting
    plot_df = pd.DataFrame({
        'Player': player_names,
        'PC1': projected[:, 0],
        'PC2': projected[:, 1],
        'FP/G': all_players_df['Mean FPts'].values,
        'Position': all_players_df['Position'].values,
        'Team': all_players_df['Status'].values,
    })
    
    # Add highlight column
    if highlight_players:
        plot_df['Highlight'] = plot_df['Player'].isin(highlight_players)
    else:
        plot_df['Highlight'] = False
    
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color=color_by,
        size='FP/G',
        hover_name='Player',
        hover_data=['FP/G', 'Team'],
        title='League Player Map (PCA Projection)',
    )
    
    # Highlight specific players
    if highlight_players:
        highlight_df = plot_df[plot_df['Highlight']]
        fig.add_trace(go.Scatter(
            x=highlight_df['PC1'],
            y=highlight_df['PC2'],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='star'),
            text=highlight_df['Player'],
            textposition='top center',
            name='Selected Players',
        ))
    
    # Add explained variance to axis labels
    var_explained = get_pca_explained_variance(pca)
    fig.update_layout(
        xaxis_title=f"PC1 ({var_explained['PC1']*100:.1f}% variance)",
        yaxis_title=f"PC2 ({var_explained['PC2']*100:.1f}% variance)",
        height=600,
    )
    
    return fig


def display_league_map(all_players_df: pd.DataFrame, league_means: Dict, 
                        league_stds: Dict, your_team: str = None):
    """Streamlit component for league player map."""
    st.subheader("🗺️ League Player Map")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        color_by = st.selectbox(
            "Color by",
            ['Position', 'Team'],
            key='player_map_color'
        )
        
        highlight_team = st.selectbox(
            "Highlight team",
            ['None'] + list(all_players_df['Status'].unique()),
            key='player_map_highlight'
        )
        
        highlight_players = []
        if highlight_team != 'None':
            highlight_players = all_players_df[
                all_players_df['Status'] == highlight_team
            ]['Player'].tolist()
    
    with col2:
        fig = render_league_player_map(
            all_players_df, league_means, league_stds,
            highlight_players=highlight_players,
            color_by=color_by,
        )
        st.plotly_chart(fig, use_container_width=True)
```

### 12.3. Trade Impact Visualization

```python
def render_trade_vector_change(
    your_team_before: pd.DataFrame,
    your_team_after: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> go.Figure:
    """
    Visualize how a trade moves your team's centroid in vector space.
    """
    centroid_before = roster_centroid(your_team_before, league_means, league_stds)
    centroid_after = roster_centroid(your_team_after, league_means, league_stds)
    
    dimensions = ['FP/G', 'Availability', 'Consistency', 'Value']
    
    fig = go.Figure()
    
    # Before (baseline)
    fig.add_trace(go.Bar(
        name='Before Trade',
        x=dimensions,
        y=centroid_before,
        marker_color='#FF9800',
    ))
    
    # After
    fig.add_trace(go.Bar(
        name='After Trade',
        x=dimensions,
        y=centroid_after,
        marker_color='#4CAF50',
    ))
    
    fig.update_layout(
        barmode='group',
        title='Team Profile Change (Standardized)',
        yaxis_title='Z-Score',
        height=400,
    )
    
    return fig


def render_monte_carlo_distribution(
    before_stats: Dict[str, float],
    after_stats: Dict[str, float],
) -> go.Figure:
    """
    Visualize simulated weekly outcome distributions before/after trade.
    """
    fig = go.Figure()
    
    # Create distribution curves (approximated as normal)
    x_range = np.linspace(
        min(before_stats['p10'], after_stats['p10']) - 50,
        max(before_stats['p90'], after_stats['p90']) + 50,
        100
    )
    
    before_curve = stats.norm.pdf(x_range, before_stats['mean'], before_stats['std'])
    after_curve = stats.norm.pdf(x_range, after_stats['mean'], after_stats['std'])
    
    fig.add_trace(go.Scatter(
        x=x_range, y=before_curve,
        mode='lines', fill='tozeroy',
        name='Before Trade',
        line_color='#FF9800',
        fillcolor='rgba(255, 152, 0, 0.3)',
    ))
    
    fig.add_trace(go.Scatter(
        x=x_range, y=after_curve,
        mode='lines', fill='tozeroy',
        name='After Trade',
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.3)',
    ))
    
    # Add vertical lines for means
    fig.add_vline(x=before_stats['mean'], line_dash='dash', line_color='#FF9800')
    fig.add_vline(x=after_stats['mean'], line_dash='dash', line_color='#4CAF50')
    
    fig.update_layout(
        title='Projected Weekly FP Distribution',
        xaxis_title='Weekly Fantasy Points',
        yaxis_title='Probability Density',
        height=400,
    )
    
    return fig


def display_trade_simulation(your_team_before: pd.DataFrame, 
                              your_team_after: pd.DataFrame,
                              league_means: Dict, league_stds: Dict):
    """Streamlit component for trade simulation results."""
    st.subheader("🎲 Trade Outcome Simulation")
    
    with st.spinner("Running Monte Carlo simulation..."):
        comparison = compare_trade_outcomes(your_team_before, your_team_after)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Expected Weekly FP",
            f"{comparison['after']['mean']:.0f}",
            f"{comparison['mean_change']:+.0f}"
        )
    with col2:
        st.metric(
            "Floor (10th pctl)",
            f"{comparison['after']['p10']:.0f}",
            f"{comparison['floor_change']:+.0f}"
        )
    with col3:
        st.metric(
            "Ceiling (90th pctl)",
            f"{comparison['after']['p90']:.0f}",
            f"{comparison['ceiling_change']:+.0f}"
        )
    with col4:
        variance_emoji = "📉" if comparison['variance_change'] < 0 else "📈"
        st.metric(
            "Variance Change",
            f"{variance_emoji}",
            f"{comparison['variance_change']:+.1f}"
        )
    
    # Distribution chart
    fig = render_monte_carlo_distribution(comparison['before'], comparison['after'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Team profile change
    fig2 = render_trade_vector_change(
        your_team_before, your_team_after, league_means, league_stds
    )
    st.plotly_chart(fig2, use_container_width=True)
```

### 12.4. Similar Players Finder UI

```python
def display_similar_players_finder(all_players_df: pd.DataFrame,
                                    league_means: Dict, league_stds: Dict):
    """Streamlit component for finding similar players."""
    st.subheader("🔍 Find Similar Players")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        target_player = st.selectbox(
            "Select a player",
            all_players_df['Player'].tolist(),
            key='similar_player_target'
        )
        
        n_results = st.slider("Number of results", 3, 15, 5)
        
        exclude_same_team = st.checkbox("Exclude same team", value=True)
        
        position_options = ['Any'] + list(all_players_df['Position'].unique())
        position_filter = st.selectbox("Position filter", position_options)
        if position_filter == 'Any':
            position_filter = None
    
    with col2:
        if target_player:
            similar = find_similar_players(
                target_player, all_players_df, league_means, league_stds,
                n=n_results, exclude_same_team=exclude_same_team,
                position_filter=position_filter
            )
            
            if similar:
                results_df = pd.DataFrame([
                    {
                        'Player': name,
                        'Similarity': f"{score:.1f}%",
                        'FP/G': f"{row.get('Mean FPts', 0):.1f}",
                        'CV%': f"{row.get('CV%', 0):.1f}",
                        'Team': row.get('Status', ''),
                    }
                    for name, score, row in similar
                ])
                
                st.dataframe(results_df, hide_index=True, use_container_width=True)
                
                # Quick comparison with top match
                if len(similar) > 0:
                    top_match_name, top_score, top_row = similar[0]
                    target_row = all_players_df[
                        all_players_df['Player'] == target_player
                    ].iloc[0]
                    
                    with st.expander(f"Compare with {top_match_name} ({top_score:.1f}% similar)"):
                        fig = render_player_comparison_radar(
                            target_row, top_row, league_means, league_stds
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No similar players found with current filters.")
```

### 12.5. Team Archetype Analysis UI

```python
def display_team_archetype_analysis(all_players_df: pd.DataFrame,
                                     your_team_df: pd.DataFrame,
                                     league_means: Dict, league_stds: Dict):
    """Streamlit component for team archetype breakdown."""
    st.subheader("🏀 Team Archetype Analysis")
    
    # Cluster all players
    labels, cluster_names = cluster_players(
        all_players_df, league_means, league_stds, n_clusters=5
    )
    
    # Analyze your team
    composition = analyze_team_composition(
        your_team_df, labels, cluster_names, all_players_df
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of team composition
        fig = go.Figure(data=[go.Pie(
            labels=list(composition.keys()),
            values=list(composition.values()),
            hole=0.4,
        )])
        fig.update_layout(title="Your Team Composition", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recommendations based on composition
        st.markdown("#### 💡 Roster Insights")
        
        elite_count = composition.get('Elite Studs', 0)
        reliable_count = composition.get('Reliable Starters', 0)
        boom_bust_count = composition.get('Boom/Bust', 0)
        
        if elite_count == 0:
            st.warning("⚠️ No elite studs - consider consolidation trades")
        elif elite_count >= 3:
            st.success("✅ Strong elite core")
        
        if boom_bust_count >= 3:
            st.warning("⚠️ High volatility roster - may want more consistency")
        
        if reliable_count >= 5:
            st.success("✅ Good reliability base for hitting 25 games")
        elif reliable_count < 3:
            st.info("ℹ️ Consider adding more reliable starters")
```

---

## 13. Integration Plan for Visualizations

### New Files to Create

| File | Purpose |
|------|---------|
| `modules/trade_suggestions/player_similarity.py` | Core similarity functions |
| `modules/trade_suggestions/advanced_stats.py` | PCA, clustering, Monte Carlo |
| `modules/trade_suggestions/similarity_viz.py` | Plotly visualization functions |
| `pages/Player_Similarity.py` | New Streamlit page for similarity tools |

### Integration with Existing UI

1. **Add to `trade_suggestions_ui_tab.py`**:
   - Similarity score in trade suggestion cards
   - "Compare Players" button that opens radar chart
   - Monte Carlo simulation toggle for detailed analysis

2. **Add to `trade_analysis/ui.py`**:
   - Team archetype breakdown in trade analysis
   - Vector change visualization for proposed trades

3. **New Streamlit Page** (`pages/Player_Similarity.py`):
   - Similar players finder
   - League player map (PCA scatter)
   - Team composition analysis

### Session State Keys

```python
# Add to session_state initialization
if 'league_stats' not in st.session_state:
    st.session_state.league_stats = None  # (means, stds) tuple

if 'pca_projection' not in st.session_state:
    st.session_state.pca_projection = None  # Cached PCA results

if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None  # Cached cluster assignments
```

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

---

## Revision Notes

**v2 (Nov 29, 2025)**: Aligned with actual codebase structure:
- Updated target files to use correct module paths (`modules/trade_suggestions/`, `modules/trade_analysis/`)
- Changed column names to match existing data (`Mean FPts`, `CV%`, `Value`, `AvailabilityRatio`)
- Added references to existing functions (`_get_core_size()`, `_calculate_core_value()`, `_calculate_floor_impact()`)
- Mapped data sources to actual files (`consistency_integration.py`, `player_value/logic.py`)
- Removed references to non-existent per-game category stats (PTS, REB, AST per game)
- Added integration points with existing trade suggestion engine

**v3 (Nov 29, 2025)**: Added advanced statistical methods and visualizations:
- **Section 11**: Advanced algorithms (KNN, Cosine Similarity, PCA, K-Means Clustering, Monte Carlo Simulation, Percentile Ranks)
- **Section 12**: Streamlit visualization components using Plotly (Radar charts, PCA scatter plots, Monte Carlo distributions, Team archetype pie charts)
- **Section 13**: Integration plan for new files and UI components
- All visualizations follow existing patterns from `trade_suggestions_ui_tab.py`
- Added session state keys for caching expensive computations

**v4 (Nov 29, 2025)**: IMPLEMENTATION COMPLETE ✅
- Created `modules/trade_suggestions/player_similarity.py` - Core similarity functions
- Created `modules/trade_suggestions/advanced_stats.py` - PCA, clustering, Monte Carlo
- Created `modules/trade_suggestions/similarity_viz.py` - Plotly visualization components
- Created `pages/5_Player_Similarity.py` - New Streamlit page with 5 tabs
- Integrated similarity analysis into `trade_suggestions_ui_tab.py` (new expander section)
- Integrated Monte Carlo simulation into `trade_analysis/ui.py`
- Updated `__init__.py` to export all new functions
