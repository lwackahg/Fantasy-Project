"""
Business logic for the player data feature.
"""

import pandas as pd

def merge_with_draft_results(player_data: pd.DataFrame, draft_results: pd.DataFrame) -> pd.DataFrame:
    """Merges player data with draft results."""
    if draft_results is None or draft_results.empty:
        return player_data
    return player_data.merge(draft_results[['Player', 'Bid', 'Pick']], on='Player', how='left')
