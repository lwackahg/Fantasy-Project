import re
import pandas as pd
import os
from typing import Optional, Dict, Any, List, Tuple
from team_mappings import TEAM_MAPPINGS

class DataImporter:
    """
    Handles data import from CSV files and API sources
    """
    def __init__(self, data_source: str = 'csv'):
        """
        Initialize data importer with specified source
        
        :param data_source: 'csv' or 'api'
        """
        self.data_source = data_source
        self.data = None
        self.team_mappings = TEAM_MAPPINGS
        self.required_columns = {'Player', 'Position', 'Status', 'FPts', 'FP/G'}
        self.history_length = None

    def extract_history_length(self, file_path: str) -> int:
        """
        Extract history length from filename
        Only cares about the (X) at the end, where X is the number of days
        Examples that would work:
        - any-name-(30).csv
        - stats-(7).csv
        - Fantrax-Players-(14).csv
        - MyData - (60).csv
        
        :param file_path: Path to the CSV file
        :return: History length in games, or None if not found
        """
        filename = os.path.basename(file_path)
        # Look for any number in parentheses right before .csv
        match = re.search(r'\((\d+)\)\.csv$', filename)
        if match:
            return int(match.group(1))
        return None

    def get_full_team_name(self, abbreviation: str) -> str:
        """
        Get full team name from abbreviation
        
        :param abbreviation: Team abbreviation
        :return: Full team name
        """
        return self.team_mappings.get(abbreviation, abbreviation)

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame contains required columns
        
        :param df: DataFrame to validate
        :return: True if valid, False otherwise
        """
        missing_columns = self.required_columns - set(df.columns)
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        return True

    def import_csv(self, file_path: str) -> pd.DataFrame:
        """
        Import data from CSV file with validation
        
        :param file_path: Path to CSV file
        :return: Validated DataFrame
        """
        try:
            # Extract history length from filename
            self.history_length = self.extract_history_length(file_path)
            
            # Read CSV with all columns
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Validate required columns
            if not self.validate_columns(df):
                return pd.DataFrame()
            
            # Add full team names based on Status column
            df['TeamName'] = df['Status'].map(self.get_full_team_name)
            
            self.data = df
            return df
        
        except Exception as e:
            print(f"CSV Import Error: {str(e)}")
            return pd.DataFrame()

    def preview_data(self, rows: int = 5) -> pd.DataFrame:
        """
        Get a preview of the imported data
        
        :param rows: Number of rows to preview (default: 5)
        :return: DataFrame with preview rows
        """
        if self.data is None:
            return pd.DataFrame()
        
        preview = self.data.head(rows)
        
        # Print some basic information
        print("\nData Preview Information:")
        print(f"History Length: {self.history_length} games")
        print(f"Total Rows: {len(self.data)}")
        print(f"Total Columns: {len(self.data.columns)}")
        
        # Print column information
        print("\nColumns:")
        for col in self.data.columns:
            # Get column data type and number of non-null values
            dtype = self.data[col].dtype
            non_null = self.data[col].count()
            print(f"- {col} (Type: {dtype}, Non-null: {non_null})")
        
        # Print unique teams and their player counts
        print("\nTeam Rosters:")
        team_counts = self.data.groupby(['Status', 'TeamName']).size().reset_index(name='Players')
        for _, row in team_counts.iterrows():
            print(f"- {row['Status']}: {row['TeamName']} ({row['Players']} players)")
        
        # Print position distribution
        print("\nPosition Distribution:")
        pos_counts = self.data['Position'].value_counts()
        for pos, count in pos_counts.items():
            print(f"- {pos}: {count} players")
        
        return preview

    def get_data_by_history(self) -> Dict[int, pd.DataFrame]:
        """
        Get dictionary of DataFrames grouped by history length
        
        :return: Dictionary with history length as key and DataFrame as value
        """
        if self.data is None:
            return {}
        
        return {
            self.history_length: self.data
        }

    def get_available_columns(self) -> List[str]:
        """
        Get list of all available columns in the data
        
        :return: List of column names
        """
        if self.data is None:
            return []
        return list(self.data.columns)

    def get_history_length(self) -> Optional[int]:
        """
        Get the history length of the current dataset
        
        :return: History length in games, or None if not set
        """
        return self.history_length

    def fetch_api_data(self, api_key: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        Fetch data from Fantrax API
        
        :param api_key: Fantrax API key
        :param params: Additional API parameters
        :return: DataFrame with player statistics
        """
        base_url = "https://api.fantrax.com/v1/players"
        
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['players'])
            
            # Add full team names if Status column exists
            if 'Status' in df.columns:
                df['TeamName'] = df['Status'].map(self.get_full_team_name)
            
            self.data = df
            return df
        
        except requests.RequestException as e:
            print(f"API Fetch Error: {str(e)}")
            return pd.DataFrame()

    def filter_columns(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter DataFrame to include only specified columns
        
        :param columns: List of columns to keep
        :return: Filtered DataFrame
        """
        if self.data is None:
            return pd.DataFrame()
        
        if columns:
            valid_columns = [col for col in columns if col in self.data.columns]
            return self.data[valid_columns]
        
        return self.data
