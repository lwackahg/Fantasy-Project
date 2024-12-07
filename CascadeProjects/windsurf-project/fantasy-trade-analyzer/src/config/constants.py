"""Application-wide constants for the Fantasy Basketball Trade Analyzer."""

from typing import Dict, List

# Application Configuration
PAGE_TITLE: str = "Fantasy Basketball Trade Analyzer"
PAGE_ICON: str = "🏀"
LAYOUT: str = "wide"
INITIAL_SIDEBAR_STATE: str = "expanded"

# Data Processing
REQUIRED_COLUMNS: List[str] = ['Player', 'Team', 'FP/G', 'Status']
NUMERIC_COLUMNS: List[str] = [
    'FPts', 'FP/G', 'PTS', 'OREB', 'DREB', 'REB', 
    'AST', 'STL', 'BLK', 'TOV'
]

# Analysis Weights
TIME_RANGES: Dict[str, float] = {
    '7 Days': 0.4,   # Most recent data weighted highest
    '14 Days': 0.3,
    '30 Days': 0.2,
    '60 Days': 0.1   # Oldest data weighted lowest
}

# UI Colors
ERROR_COLOR: str = "#e74c3c"     # Red for errors and negative trends
SUCCESS_COLOR: str = "#2ecc71"   # Green for success and positive trends
WARNING_COLOR: str = "#f39c12"   # Yellow for warnings and neutral trends
NEUTRAL_COLOR: str = "#95a5a6"   # Gray for neutral or inactive states
