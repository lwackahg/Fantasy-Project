"""Module for handling data import from various sources"""
import re
import pandas as pd
import os
import logging
from typing import Optional, Dict, Any, List, Tuple

class DataImporter:
    """
    Handles data import from CSV files and API sources
    """
    def __init__(self):
        """Initialize data importer"""
        self.data_ranges: Dict[str, pd.DataFrame] = {}
        self.required_columns = {'Player', 'Team', 'Position', 'Status', 'FPts', 'FP/G'}
        
    def extract_history_length(self, file_path: str) -> Optional[int]:
        """
        Extract history length from filename.
        
        Supported formats:
        - any-name-(30).csv
        - stats-(7).csv
        - Fantrax-Players-(14).csv
        - MyData - (60).csv
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            int: History length in days, or None if not found
        """
        filename = os.path.basename(file_path)
        match = re.search(r'\((\d+)\)\.csv$', filename)
        if match:
            days = int(match.group(1))
            if days in [7, 14, 30, 60]:  # Only accept standard time ranges
                return days
        return None
        
    def import_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Import and validate a single CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with imported data, or None if validation fails
            
        Raises:
            ValueError: If file is missing required columns
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Clean and preprocess data
            df = self._preprocess_dataframe(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error importing {file_path}: {str(e)}")
            return None
            
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the imported DataFrame.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            Cleaned and preprocessed DataFrame
        """
        # Remove any completely empty rows or columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Convert numeric columns
        numeric_cols = ['FPts', 'FP/G', 'MIN']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Fill NaN values
        df = df.fillna({
            'FPts': 0,
            'FP/G': 0,
            'MIN': 0,
            'Status': 'Unknown'
        })
        
        return df
        
    def import_multiple_files(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Import multiple CSV files and organize them by time range.
        
        Args:
            file_paths: List of CSV file paths to import
            
        Returns:
            Dictionary mapping time ranges to DataFrames
        """
        data_ranges = {}
        
        for file_path in file_paths:
            history_length = self.extract_history_length(file_path)
            if history_length is None:
                logging.warning(f"Could not extract history length from {file_path}")
                continue
                
            df = self.import_csv(file_path)
            if df is not None:
                range_key = f"{history_length} Days"
                if range_key in data_ranges:
                    # If we already have data for this range, merge it
                    data_ranges[range_key] = pd.concat([data_ranges[range_key], df]).drop_duplicates()
                else:
                    data_ranges[range_key] = df
                    
        self.data_ranges = data_ranges
        return data_ranges
        
    def get_data_range(self, time_range: str) -> Optional[pd.DataFrame]:
        """
        Get data for a specific time range.
        
        Args:
            time_range: Time range string (e.g., '7 Days', '14 Days')
            
        Returns:
            DataFrame for the specified range, or None if not available
        """
        return self.data_ranges.get(time_range)
        
    def get_available_ranges(self) -> List[str]:
        """Get list of available time ranges"""
        return sorted(self.data_ranges.keys(), key=lambda x: int(x.split()[0]))
        
    def validate_data(self) -> bool:
        """
        Validate that we have data for all required time ranges.
        
        Returns:
            True if all required ranges are present, False otherwise
        """
        return all(f"{days} Days" in self.data_ranges for days in [7, 14, 30, 60])
