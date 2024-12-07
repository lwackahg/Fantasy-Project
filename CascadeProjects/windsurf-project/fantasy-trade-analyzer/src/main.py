import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Import custom modules
from data_import import DataImporter
from trade_analysis import TradeAnalyzer
from statistical_analysis import StatisticalAnalyzer
from visualization import DataVisualizer

class FantasyTradeAnalyzer:
    """
    Main application class for Fantasy Trade Analyzer
    """
    def __init__(self, data_source='csv'):
        """
        Initialize the Fantasy Trade Analyzer
        
        :param data_source: Source of data ('csv' or 'api')
        """
        load_dotenv()  # Load environment variables
        
        self.data_source = data_source
        self.data_importer = DataImporter(data_source)
        self.data = None
        
  

    def load_data(self, file_path=None):
        """
        Load data from specified source
        
        :param file_path: Path to CSV file (optional)
        """
        if self.data_source == 'csv':
            if not file_path:
                raise ValueError("CSV file path is required")
            self.data = self.data_importer.import_csv(file_path)
        

        if self.data is None or self.data.empty:
            raise ValueError("Failed to load data")

    def analyze_trade(self, trade_proposal):
        """
        Analyze a proposed trade
        
        :param trade_proposal: Dictionary of teams and traded players
        :return: Trade fairness analysis
        """
        trade_analyzer = TradeAnalyzer(self.data)
        return trade_analyzer.evaluate_trade_fairness(trade_proposal)

    def generate_visualizations(self, visualization_type='team_comparison', **kwargs):
        """
        Generate different types of visualizations
        
        :param visualization_type: Type of visualization to generate
        :param kwargs: Additional parameters for visualization
        :return: Matplotlib figure
        """
        visualizer = DataVisualizer(self.data)
        
        if visualization_type == 'team_comparison':
            return visualizer.team_comparison_bar(**kwargs)
        
        elif visualization_type == 'player_performance':
            return visualizer.player_performance_boxplot(**kwargs)
        
        elif visualization_type == 'player_trend':
            return visualizer.player_trend_line(**kwargs)

def main():
    """
    Main entry point for the application
    """
    try:
        # Example usage
        analyzer = FantasyTradeAnalyzer(data_source='csv')
        analyzer.load_data('path/to/your/fantasy_data.csv')
        
        # Example trade proposal
        trade_proposal = {
            'team1': ['Player A', 'Player B'],
            'team2': ['Player C', 'Player D']
        }
        
        trade_analysis = analyzer.analyze_trade(trade_proposal)
        print("Trade Fairness Analysis:", trade_analysis)
        
        # Generate team comparison visualization
        team_comparison_plot = analyzer.generate_visualizations(
            visualization_type='team_comparison', 
            teams=['Team 1', 'Team 2', 'Team 3']
        )
        team_comparison_plot.show()
    
    except Exception as e:
        print(f"Error in Fantasy Trade Analyzer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
