"""Fantasy Basketball Trade Analyzer Application.

This module serves as the main entry point for the Fantasy Basketball Trade Analyzer.
It provides a comprehensive interface for analyzing fantasy basketball trades and team performance.

Key Features:
- Trade Analysis: Evaluate the fairness and impact of player trades
- Team Statistics: Analyze team performance across multiple metrics
- Player Performance: Track individual player trends and statistics
- League Analytics: View league-wide statistics and rankings

The application uses Streamlit for the user interface and integrates with various
data sources to provide up-to-date player statistics and analysis.

Dependencies:
    - streamlit: Web application framework
    - pandas: Data manipulation and analysis
    - plotly: Interactive data visualization
    - numpy: Numerical computations
"""

# Standard library imports
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from collections import Counter

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np

# Local application imports
from data_import import DataImporter
from trade_analysis import TradeAnalyzer
from statistical_analysis import StatisticalAnalyzer
from config.constants import REQUIRED_COLUMNS, NUMERIC_COLUMNS
from ui.trade_page import render_trade_page
from ui.team_stats_page import render_team_stats_page
from ui.player_analysis_page import render_player_analysis_page

# Application Configuration Constants
PAGE_TITLE: str = "Fantasy Basketball Trade Analyzer"
PAGE_ICON: str = "🏀"
LAYOUT: str = "wide"
INITIAL_SIDEBAR_STATE: str = "expanded"

# Data Processing Constants
TIME_RANGES: Dict[str, float] = {
    '7 Days': 0.4,   # Most recent data weighted highest
    '14 Days': 0.3,
    '30 Days': 0.2,
    '60 Days': 0.1   # Oldest data weighted lowest
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Streamlit session state variables for application state management.
def init_session_state():
    """
    Initialize session state variables for the application.
    
    This function sets up the initial state for the application, including:
    - Data storage for player statistics across different time ranges
    - Trade history tracking
    - Debug mode settings
    - Analysis tool instances
    
    The session state persists across reruns of the app, ensuring data consistency
    during user interactions.
    """
    if 'data_importer' not in st.session_state:
        st.session_state.data_importer = DataImporter()
    if 'data_ranges' not in st.session_state:
        st.session_state.data_ranges = {}
    if 'current_range' not in st.session_state:
        st.session_state.current_range = None
    if 'stats_analyzer' not in st.session_state:
        st.session_state.stats_analyzer = StatisticalAnalyzer()
    if 'trade_analyzer' not in st.session_state:
        st.session_state.trade_analyzer = TradeAnalyzer()  # Initialize TradeAnalyzer   
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []

def handle_file_upload():
    """Handle file upload and data processing"""
    uploaded_files = st.file_uploader(
        "Upload your player statistics CSV files",
        type='csv',
        accept_multiple_files=True,
        help="Upload CSV files with player statistics. Files should be named with the format: 'name-(days).csv'"
    )
    
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            # Save uploaded files temporarily
            file_paths = []
            for file in uploaded_files:
                temp_path = Path(file.name)
                temp_path.write_bytes(file.getvalue())
                file_paths.append(str(temp_path))
                
            try:
                # Import the files
                data_ranges = st.session_state.data_importer.import_multiple_files(file_paths)
                
                if data_ranges:
                    st.session_state.data_ranges = data_ranges
                    st.success(f"Successfully loaded data for {len(data_ranges)} time ranges!")
                    
                    # Initialize visualizer with the first available range
                    first_range = next(iter(data_ranges.values()))
                    # st.session_state.visualizer = DataVisualizer(first_range)
                    
                    if st.session_state.debug_mode:
                        st.write("### Loaded Data Ranges")
                        for range_key, data in data_ranges.items():
                            st.write(f"- {range_key}: {len(data)} players")
                    
                else:
                    st.error("No valid data was found in the uploaded files.")
                    
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                logging.error(f"File processing error: {str(e)}")
                
            finally:
                # Clean up temporary files
                for file_path in file_paths:
                    try:
                        Path(file_path).unlink()
                    except Exception as e:
                        logging.error(f"Error cleaning up {file_path}: {str(e)}")

def select_time_range():
    """Allow user to select time range for analysis"""
    if st.session_state.data_ranges:
        available_ranges = st.session_state.data_importer.get_available_ranges()
        selected_range = st.selectbox(
            "Select Time Range for Analysis",
            available_ranges,
            index=0 if available_ranges else None,
            help="Choose the time range for player statistics analysis"
        )
        
        if selected_range and selected_range != st.session_state.current_range:
            st.session_state.current_range = selected_range
            data = st.session_state.data_ranges[selected_range]
            # st.session_state.visualizer = DataVisualizer(data)
            
            # Display data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Players", len(data))
            with col2:
                st.metric("Teams", data['Team'].nunique())
            with col3:
                st.metric("Avg FP/G", f"{data['FP/G'].mean():.1f}")
            
            if st.session_state.debug_mode:
                with st.expander("View Data Details"):
                    st.dataframe(data.describe())

def main():
    """Main application entry point."""
    # Configure page
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    st.sidebar.title("Fantasy Trade Analyzer")
    handle_file_upload()
    select_time_range()
    
    # Main content
    if not st.session_state.data_ranges:
        st.info("👋 Welcome! Please upload your player statistics CSV files to begin.")
        return
    
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs([
        "Trade Analysis",
        "Team Statistics",
        "Player Analysis"
    ])
    
    # Render appropriate page based on selected tab
    with tab1:
        render_trade_page()
    
    with tab2:
        render_team_stats_page()
    
    with tab3:
        render_player_analysis_page()

if __name__ == "__main__":
    main()
