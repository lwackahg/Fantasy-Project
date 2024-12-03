# Fantasy Trade Analyzer

A comprehensive tool for analyzing fantasy basketball trades, featuring data import, statistical analysis, and trade evaluation.

## Features

### Core Features
- [x] CSV data import from Fantrax exports
- [x] Team roster tracking and analysis
- [x] Interactive trade evaluation
- [x] Visualization of trade impacts
- [x] Multi-team trade support (2-5 teams)
- [x] Multiple time range analysis (7, 14, 30, 60 days)
- [x] Before/After trade comparison

### Trade Analysis Features
- Fantasy Points Analysis
  - Per game (FP/G) analysis
  - Total points (FPts) comparison
  - Mean, median, and standard deviation calculations
- Position Impact Analysis
  - Position distribution before/after trade
  - Position scarcity adjustments
- Team Statistics
  - Roster size changes
  - Overall team performance metrics
  - Statistical balance analysis

## Setup Instructions

### Prerequisites
- Python 3.9+
- Required packages (installed automatically):
  - streamlit
  - pandas
  - plotly
  - numpy

### Installation

1. Clone or download the repository
2. Navigate to the project directory
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Using the Batch File
1. Double-click `run_app.bat` in the project directory
2. The application will open in your default web browser

#### Option 2: Using Command Line
1. Open a terminal/command prompt
2. Navigate to the project directory
3. Run:
```bash
python -m streamlit run src/app.py
```

## Data Files

The analyzer expects Fantrax export files in CSV format. 

### File Naming Convention
- Files should end with (X).csv where X is the number of days
- Examples:
  - `any-name-(7).csv` for 7-day stats
  - `stats-(14).csv` for 14-day stats
  - `data-(30).csv` for 30-day stats
  - `export-(60).csv` for 60-day stats

### Required Data Format
The CSV files must contain the following required columns:
- `Player`: Player name
- `Team`: Team abbreviation (will be converted to full name)
- `FP/G`: Fantasy points per game

Additional recommended columns for full functionality:
- `FPts`: Total fantasy points
- `MIN`: Minutes played
- `PTS`: Points scored
- `REB`: Total rebounds
- `AST`: Assists
- `STL`: Steals
- `BLK`: Blocks
- `TOV`: Turnovers

### Data Directory
Place your CSV files in the `data/` directory of the project. The analyzer will automatically detect and load files with the correct naming format.

### Exporting Data from Fantrax
1. Go to your Fantrax league
2. Navigate to the Players page
3. Click on "Download to CSV"
4. Save the file with the appropriate naming convention in the data directory
5. Repeat for different time ranges (7, 14, 30, 60 days)

### Common Data Issues
If you encounter issues loading data:
1. Check that your CSV files follow the correct naming convention with (X) days
2. Verify that all required columns are present
3. Ensure numeric columns contain valid numbers
4. Make sure the files are placed in the `data/` directory

## Usage

1. Select number of teams involved (2-5)
2. Choose teams from the dropdowns
3. Select players to trade from each team
4. Choose which players each team receives
5. Click "Analyze Trade" to see:
   - Trade fairness score
   - Before/After comparisons
   - Position impact
   - Statistical changes

## Project Structure

```
fantasy-trade-analyzer/
├── src/
│   ├── app.py             # Main Streamlit application
│   ├── data_import.py     # Data import and validation
│   └── trade_analysis.py  # Trade evaluation logic
├── data/                  # Place CSV files here
└── run_app.bat           # Easy launch script
```

## Known Issues & Solutions

If you encounter numeric conversion errors:
- Ensure your CSV files contain valid numeric data
- The app will handle invalid data gracefully
- Check the console for specific error messages

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## Recent Updates

#### Enhanced Trade Analysis Display
- Tabbed interface for multi-team trade analysis
- Expandable sections for detailed statistics
- Color-coded trade fairness indicators:
  - 🟢 Green (≥80%): Very fair trade
  - 🟡 Yellow (≥60%): Moderately fair trade
  - 🟠 Orange (≥40%): Slightly unfair trade
  - 🔴 Red (<40%): Very unfair trade
- Improved player statistics visualization
- Net value change calculation and display

#### Statistical Improvements
- Top X players analysis for consistent team evaluation
- Before/After trade statistics across multiple time ranges
- Enhanced statistical calculations:
  - Mean and median FP/G
  - Standard deviation for team balance (Green = more balanced, Red = less balanced)
  - Games played (GP) tracking
  - Total fantasy points analysis

#### User Interface Enhancements
- Dark theme for better readability
- Organized layout with tabs and expanders
- Interactive player selection interface
- Improved error handling and data validation
- Clear visual indicators for trade impact

### Upcoming Features
- Player trend analysis and visualization
- Advanced statistical metrics
- Trade suggestion engine
- Custom scoring system support
- Historical trade tracking
