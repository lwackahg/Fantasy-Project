from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TeamStrategy:
    """
    Defines a strategic approach for building a fantasy team.

    Attributes:
        name (str): The name of the strategy (e.g., 'balanced', 'stars_scrubs').
        category_weights (Dict[str, float]): Weights for each statistical category to prioritize.
        position_priority (Dict[str, float]): Weights to prioritize certain positions.
        tier_distribution (Dict[int, float]): The ideal percentage of budget to spend on players in each tier.
    """
    name: str
    category_weights: Dict[str, float] = field(default_factory=dict)
    position_priority: Dict[str, float] = field(default_factory=dict)
    tier_distribution: Dict[int, float] = field(default_factory=dict)

# --- Default Strategies ---

BALANCED_STRATEGY = TeamStrategy(
    name='balanced',
    category_weights={'PTS': 1.0, 'REB': 1.0, 'AST': 1.0, 'STL': 1.0, 'BLK': 1.0, '3PM': 1.0, 'FG%': 1.0, 'FT%': 1.0},
    position_priority={'G': 1.0, 'F': 1.0, 'C': 1.0, 'Flx': 1.0, 'Bench': 1.0},
    tier_distribution={1: 0.35, 2: 0.30, 3: 0.20, 4: 0.10, 5: 0.05}
)

STARS_AND_SCRUBS_STRATEGY = TeamStrategy(
    name='stars_scrubs',
    category_weights={'PTS': 1.2, 'REB': 0.9, 'AST': 1.1, 'STL': 0.8, 'BLK': 0.8, '3PM': 1.0, 'FG%': 0.9, 'FT%': 0.9},
    position_priority={'G': 1.2, 'F': 1.1, 'C': 0.9, 'Flx': 0.9, 'Bench': 0.8},
    tier_distribution={1: 0.60, 2: 0.25, 3: 0.10, 4: 0.03, 5: 0.02}
)

# A dictionary to hold all available strategies
STRATEGIES = {
    'balanced': BALANCED_STRATEGY,
    'stars_scrubs': STARS_AND_SCRUBS_STRATEGY
}
