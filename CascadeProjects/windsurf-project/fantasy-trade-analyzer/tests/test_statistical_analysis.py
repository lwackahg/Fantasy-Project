"""Tests for statistical analysis module"""
import pytest
import pandas as pd
import streamlit as st
from unittest.mock import patch
from src.statistical_analysis import analyze_team_stats, analyze_player_stats

# Mock visualization functions
def mock_plot_performance_trends(*args, **kwargs):
    return {}

def mock_display_stats_table(*args, **kwargs):
    pass

@pytest.fixture(autouse=True)
def mock_visualization():
    """Mock visualization functions"""
    with patch('src.statistical_analysis.plot_performance_trends', mock_plot_performance_trends), \
         patch('src.statistical_analysis.display_stats_table', mock_display_stats_table):
        yield

@pytest.fixture
def mock_data():
    """Create mock data for testing"""
    data = pd.DataFrame({
        'Player': ['Player1', 'Player2', 'Player3', 'Player4'],
        'Status': ['Team A', 'Team A', 'Team B', 'Team B'],
        'FP/G': [20.5, 15.3, 18.7, 12.1],
        'FPts': [205, 153, 187, 121],
        'GP': [10, 10, 10, 10],
        'Position': ['F', 'D', 'F', 'D']
    })
    return data

@pytest.fixture
def setup_session_state(mock_data):
    """Setup session state with mock data"""
    if not hasattr(st, 'session_state'):
        st.session_state = {}
    
    st.session_state.data_ranges = {
        '30 Days': mock_data.copy(),
        '14 Days': mock_data.copy(),
        '7 Days': mock_data.copy()
    }

def test_analyze_team_stats_empty_data():
    """Test team stats analysis with empty data"""
    if not hasattr(st, 'session_state'):
        st.session_state = {}
    st.session_state.data_ranges = {}
    
    # Should not raise an error, just return None
    result = analyze_team_stats('Team A')
    assert result is None

def test_analyze_team_stats_basic(setup_session_state):
    """Test basic team stats analysis"""
    # Create a placeholder for st.write and st.plotly_chart
    def mock_write(*args, **kwargs):
        pass
    
    def mock_plotly_chart(*args, **kwargs):
        pass
    
    st.write = mock_write
    st.plotly_chart = mock_plotly_chart
    
    # Test should not raise any errors
    analyze_team_stats('Team A', n_top_players=2)
    
    # Verify team stats calculations
    team_data = st.session_state.data_ranges['30 Days'][
        st.session_state.data_ranges['30 Days']['Status'] == 'Team A'
    ]
    assert len(team_data) == 2
    assert team_data['FP/G'].mean() == pytest.approx(17.9, 0.1)
    assert team_data['FPts'].sum() == 358

def test_analyze_player_stats_empty_data():
    """Test player stats analysis with empty data"""
    if not hasattr(st, 'session_state'):
        st.session_state = {}
    st.session_state.data_ranges = {}
    
    # Should not raise an error, just return None
    result = analyze_player_stats('Player1')
    assert result is None

def test_analyze_player_stats_basic(setup_session_state):
    """Test basic player stats analysis"""
    # Create placeholders for streamlit components
    def mock_write(*args, **kwargs):
        pass
    
    def mock_plotly_chart(*args, **kwargs):
        pass
    
    def mock_checkbox(*args, **kwargs):
        return True
    
    def mock_multiselect(*args, **kwargs):
        return []
    
    def mock_columns(*args):
        class MockColumn:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return [MockColumn(), MockColumn()]
    
    st.write = mock_write
    st.plotly_chart = mock_plotly_chart
    st.checkbox = mock_checkbox
    st.multiselect = mock_multiselect
    st.columns = mock_columns
    
    # Test should not raise any errors
    analyze_player_stats('Player1')
    
    # Verify player data
    player_data = st.session_state.data_ranges['30 Days'][
        st.session_state.data_ranges['30 Days']['Player'] == 'Player1'
    ]
    assert len(player_data) == 1
    assert player_data['FP/G'].iloc[0] == 20.5
    assert player_data['FPts'].iloc[0] == 205

def test_analyze_team_stats_nonexistent_team(setup_session_state):
    """Test team stats analysis with non-existent team"""
    def mock_write(*args, **kwargs):
        pass
    
    def mock_plotly_chart(*args, **kwargs):
        pass
    
    st.write = mock_write
    st.plotly_chart = mock_plotly_chart
    
    # Test should not raise any errors
    analyze_team_stats('Nonexistent Team')
    
    # Verify that no data is found for the team
    team_data = st.session_state.data_ranges['30 Days'][
        st.session_state.data_ranges['30 Days']['Status'] == 'Nonexistent Team'
    ]
    assert len(team_data) == 0

def test_analyze_player_stats_nonexistent_player(setup_session_state):
    """Test player stats analysis with non-existent player"""
    def mock_write(*args, **kwargs):
        pass
    
    def mock_plotly_chart(*args, **kwargs):
        pass
    
    def mock_checkbox(*args, **kwargs):
        return True
    
    def mock_multiselect(*args, **kwargs):
        return []
    
    def mock_columns(*args):
        class MockColumn:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return [MockColumn(), MockColumn()]
    
    st.write = mock_write
    st.plotly_chart = mock_plotly_chart
    st.checkbox = mock_checkbox
    st.multiselect = mock_multiselect
    st.columns = mock_columns
    
    # Test should not raise any errors
    analyze_player_stats('Nonexistent Player')
    
    # Verify that no data is found for the player
    player_data = st.session_state.data_ranges['30 Days'][
        st.session_state.data_ranges['30 Days']['Player'] == 'Nonexistent Player'
    ]
    assert len(player_data) == 0
