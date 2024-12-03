import pytest
import pandas as pd
import os
from src.data_import import DataImporter
from src.team_mappings import TEAM_MAPPINGS

@pytest.fixture
def data_importer():
    return DataImporter()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Player': ['Player1', 'Player2', 'Player3'],
        'Position': ['G,Flx', 'F,C,Flx', 'G,F,Flx'],
        'Status': ['Sar', '15', 'DBD'],
        'FPts': [100.5, 90.2, 85.7],
        'FP/G': [10.05, 9.02, 8.57],
        'Extra Column': ['A', 'B', 'C']  # Additional column to test flexibility
    })

def test_extract_history_length(data_importer):
    """Test extraction of history length from filename"""
    # Test valid filename
    assert data_importer.extract_history_length("Fantrax-Players-Team-(7).csv") == 7
    assert data_importer.extract_history_length("Fantrax-Players-Something-(60).csv") == 60
    
    # Test invalid filenames
    assert data_importer.extract_history_length("invalid.csv") is None
    assert data_importer.extract_history_length("Fantrax-Players-Team.csv") is None

def test_team_mapping(data_importer):
    """Test that team abbreviations map to correct full names"""
    for abbrev, full_name in TEAM_MAPPINGS.items():
        assert data_importer.get_full_team_name(abbrev) == full_name

def test_invalid_team_mapping(data_importer):
    """Test handling of invalid team abbreviation"""
    invalid_abbrev = 'INVALID'
    assert data_importer.get_full_team_name(invalid_abbrev) == invalid_abbrev

def test_csv_required_columns(data_importer, tmp_path):
    """Test validation of required columns"""
    # Create CSV with missing columns
    invalid_df = pd.DataFrame({
        'Player': ['Player1'],
        'Position': ['G,Flx']  # Missing required columns
    })
    csv_path = tmp_path / "invalid_test.csv"
    invalid_df.to_csv(csv_path, index=False)
    
    # Should return empty DataFrame due to missing columns
    result = data_importer.import_csv(str(csv_path))
    assert result.empty

def test_csv_import_with_valid_data(data_importer, sample_data, tmp_path):
    """Test successful CSV import with valid data"""
    # Create test CSV file with history length in name
    csv_path = tmp_path / "Fantrax-Players-Test-(7).csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Import and verify
    result = data_importer.import_csv(str(csv_path))
    
    assert not result.empty
    assert 'TeamName' in result.columns
    assert len(result) == len(sample_data)
    assert 'Extra Column' in result.columns  # Verify additional columns are preserved
    assert data_importer.get_history_length() == 7  # Verify history length extraction
    
    # Verify team name mapping
    for idx, row in result.iterrows():
        expected_team_name = TEAM_MAPPINGS.get(row['Status'], row['Status'])
        assert row['TeamName'] == expected_team_name

def test_data_format_validation(data_importer, tmp_path):
    """Test validation of data formats"""
    # Create CSV with invalid data types
    invalid_data = pd.DataFrame({
        'Player': ['Player1'],
        'Position': ['G,Flx'],
        'Status': ['Sar'],
        'FPts': ['invalid'],  # Should be numeric
        'FP/G': [10.5]
    })
    csv_path = tmp_path / "format_test.csv"
    invalid_data.to_csv(csv_path, index=False)
    
    # Should handle invalid data gracefully
    result = data_importer.import_csv(str(csv_path))
    assert not result.empty  # Should still import but might have conversion warnings

def test_file_not_found(data_importer):
    """Test handling of non-existent file"""
    result = data_importer.import_csv("nonexistent_file.csv")
    assert result.empty

def test_column_filtering(data_importer, sample_data, tmp_path):
    """Test column filtering functionality"""
    csv_path = tmp_path / "filter_test.csv"
    sample_data.to_csv(csv_path, index=False)
    
    data_importer.import_csv(str(csv_path))
    filtered = data_importer.filter_columns(['Player', 'Status'])
    
    assert list(filtered.columns) == ['Player', 'Status']
    assert len(filtered) == len(sample_data)

def test_get_available_columns(data_importer, sample_data, tmp_path):
    """Test getting available columns"""
    csv_path = tmp_path / "columns_test.csv"
    sample_data.to_csv(csv_path, index=False)
    
    data_importer.import_csv(str(csv_path))
    columns = data_importer.get_available_columns()
    
    assert set(columns) == set(sample_data.columns) | {'TeamName'}  # Include TeamName column

def test_data_preview(data_importer, sample_data, tmp_path, capsys):
    """Test data preview functionality"""
    csv_path = tmp_path / "Fantrax-Players-Test-(60).csv"
    sample_data.to_csv(csv_path, index=False)
    
    data_importer.import_csv(str(csv_path))
    preview = data_importer.preview_data(rows=2)
    
    # Check preview content
    assert len(preview) == 2
    assert all(col in preview.columns for col in sample_data.columns)
    
    # Check printed output
    captured = capsys.readouterr()
    output = captured.out
    
    assert "Data Preview Information:" in output
    assert "History Length: 60 games" in output
    assert f"Total Rows: {len(sample_data)}" in output
    assert f"Total Columns: {len(sample_data.columns) + 1}" in output  # +1 for TeamName
    assert "Position Distribution:" in output
    assert "Team Rosters:" in output

def test_get_data_by_history(data_importer, sample_data, tmp_path):
    """Test getting data by history length"""
    csv_path = tmp_path / "Fantrax-Players-Test-(7).csv"
    sample_data.to_csv(csv_path, index=False)
    
    data_importer.import_csv(str(csv_path))
    history_data = data_importer.get_data_by_history()
    
    assert 7 in history_data
    assert isinstance(history_data[7], pd.DataFrame)
    assert len(history_data[7]) == len(sample_data)
