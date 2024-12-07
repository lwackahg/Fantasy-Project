"""Module for shared utilities and constants used across the application."""

from typing import List, Dict, Any

# Time range constants
TIME_RANGES = ['60 Days', '30 Days', '14 Days', '7 Days']

# Statistical metrics
DEFAULT_METRICS = ['FP/G', 'FPts']
ADVANCED_METRICS = ['mean_fpg', 'median_fpg', 'std_fpg', 'total_fpts', 'avg_gp']

def format_stats_for_display(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format statistical values for display.
    
    Args:
        stats: Dictionary of statistics
        
    Returns:
        Dictionary with formatted statistics
    """
    formatted_stats = {}
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            formatted_stats[key] = f"{value:.2f}" if isinstance(value, float) else str(value)
        else:
            formatted_stats[key] = str(value)
    return formatted_stats

def validate_required_columns(df: 'pd.DataFrame', required_cols: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: pandas DataFrame to validate
        required_cols: List of required column names
        
    Returns:
        True if all required columns are present, False otherwise
    """
    return all(col in df.columns for col in required_cols)

def get_time_range_weight(time_range: str) -> float:
    """
    Get the weight for a specific time range in statistical calculations.
    
    Args:
        time_range: Time range string (e.g., '7 Days', '14 Days')
        
    Returns:
        Weight value between 0 and 1
    """
    weights = {
        '7 Days': 0.4,
        '14 Days': 0.3,
        '30 Days': 0.2,
        '60 Days': 0.1
    }
    return weights.get(time_range, 0.0)
