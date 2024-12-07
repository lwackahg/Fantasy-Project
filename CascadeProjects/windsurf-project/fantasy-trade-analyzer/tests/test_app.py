"""Unit tests for the Fantasy Trade Analyzer app functionality."""

import pytest
import pandas as pd
import streamlit as st
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import (
    init_session_state,
    load_csv_file,
    extract_days_from_filename,
    calculate_team_stats,
    calculate_player_value,
    get_trend_color,
    get_fairness_color,
    SUCCESS_COLOR,
    WARNING_COLOR,
    ERROR_COLOR
)

@pytest.fixture
def sample_player_data():
    """Fixture providing sample player data for testing."""
    return pd.DataFrame({
        'Player': ['Player A', 'Player B'],
        'Team': ['Team1', 'Team2'],
        'FP/G': [25.5, 30.2],
        'GP': [10, 12],
        'MIN': [32.5, 35.1],
        'FPts': [255.0, 362.4]  # Added FPts column
    })

@pytest.fixture
def mock_session_state():
    """Fixture providing a mocked Streamlit session state."""
    with patch.object(st, 'session_state', create=True) as mock_state:
        mock_state.data_ranges = {}
        mock_state.trade_history = []
        mock_state.debug_mode = False
        yield mock_state

def test_init_session_state(mock_session_state):
    """Test session state initialization."""
    init_session_state()
    
    assert hasattr(st.session_state, 'data_ranges')
    assert hasattr(st.session_state, 'trade_history')
    assert hasattr(st.session_state, 'debug_mode')
    assert isinstance(st.session_state.trade_history, list)
    assert isinstance(st.session_state.data_ranges, dict)
    assert isinstance(st.session_state.debug_mode, bool)

def test_load_csv_file_valid():
    """Test loading a valid CSV file."""
    # Create a temporary CSV file for testing
    test_data = pd.DataFrame({
        'Player': ['Test Player'],
        'Team': ['Test Team'],
        'FP/G': [25.0]
    })
    test_file = Path('test_data.csv')
    test_data.to_csv(test_file, index=False)
    
    try:
        result = load_csv_file(test_file)
        assert isinstance(result, pd.DataFrame)
        assert 'Player' in result.columns
        assert len(result) > 0
    finally:
        # Clean up the test file
        test_file.unlink()

def test_load_csv_file_invalid():
    """Test loading an invalid CSV file."""
    invalid_file = Path('nonexistent.csv')
    result = load_csv_file(invalid_file)
    assert result is None

def test_extract_days_from_filename():
    """Test extracting days from filename."""
    test_cases = [
        ('player_stats_(7).csv', '7 Days'),
        ('stats_(14).csv', '14 Days'),
        ('data_(30).csv', '30 Days'),
        ('export_(60).csv', '60 Days'),
        ('invalid_file.csv', None)
    ]
    
    for filename, expected in test_cases:
        result = extract_days_from_filename(filename)
        assert result == expected

@patch('streamlit.session_state', create=True)
def test_calculate_team_stats(mock_session_state, sample_player_data):
    """Test team statistics calculation."""
    mock_session_state.debug_mode = False
    stats = calculate_team_stats(sample_player_data)
    
    assert isinstance(stats, dict)
    assert 'mean_fpg' in stats
    assert 'total_fpts' in stats
    assert stats['mean_fpg'] > 0
    assert stats['total_fpts'] > 0

def test_calculate_player_value():
    """Test player value calculation."""
    player_data = {
        '7 Days': pd.DataFrame({'FP/G': [25.0], 'GP': [5]}),
        '14 Days': pd.DataFrame({'FP/G': [24.0], 'GP': [10]}),
        '30 Days': pd.DataFrame({'FP/G': [23.0], 'GP': [20]})
    }
    
    value = calculate_player_value(player_data)
    assert isinstance(value, float)
    assert value > 0

def test_get_trend_color():
    """Test trend color generation."""
    test_cases = [
        (5.0, True, SUCCESS_COLOR),   # Positive change, positive is good
        (-3.0, True, ERROR_COLOR),    # Negative change, positive is good
        (2.0, False, ERROR_COLOR),    # Positive change, positive is bad
        (-4.0, False, SUCCESS_COLOR)  # Negative change, positive is bad
    ]
    
    for value, is_positive_good, expected_color in test_cases:
        color = get_trend_color(value, is_positive_good)
        assert color == expected_color

def test_get_fairness_color():
    """Test fairness color generation."""
    test_cases = [
        (0.9, SUCCESS_COLOR),    # High fairness - green
        (0.6, WARNING_COLOR),    # Medium fairness - yellow
        (0.3, ERROR_COLOR)       # Low fairness - red
    ]
    
    for score, expected_color in test_cases:
        color = get_fairness_color(score)
        assert color == expected_color
