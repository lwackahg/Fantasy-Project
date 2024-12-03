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

### New Features
- **Trade Impact**: Now prioritized at the top of the trade analysis page for immediate insights.
- **League Statistics Tab**: Provides comprehensive league-wide metrics and FP/G distribution.
- **API Data Fetching**: The application can now fetch player game data directly from the Fantrax API, enhancing the analysis with real-time data.

### Layout Improvements
- Organized layout with tabs and expanders
- Interactive player selection interface
- Improved error handling and data validation
- Clear visual indicators for trade impact

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

## Fantrax API Integration

To enable data fetching from the Fantrax API, ensure you have set up your API key in the `.env` file located in the project directory. The key should be set as follows:

```
FANTRAX_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual API key.

## Accessing Private League Data

To access private league data, follow these steps:

1. **Automate Login with Selenium**:
   - Run the `fantrax_login.py` script to open a Chrome browser and log in to Fantrax. This will save your session cookies to a file.
   - Ensure you have `selenium` and `webdriver-manager` installed.
   ```bash
   pip install selenium webdriver-manager
   ```

2. **Use Cookies with Fantrax API**:
   - Run the `fantrax_api_access.py` script to load the saved cookies and connect to the Fantrax API.
   - Update the script with your league ID to access specific league data.

These steps allow you to authenticate and access private league data using the Fantrax API.

## Connecting to a Private League

To connect to a private league or access specific pages in a public league that are not public, follow these steps:

### Step 1: Install Required Packages
Ensure you have `selenium` and `webdriver-manager` installed:
```bash
pip install selenium
pip install webdriver-manager
```

### Step 2: Automate Login and Save Cookies
Use the following script to log in to Fantrax and save your session cookies:
```python
import pickle
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

service = Service(ChromeDriverManager().install())

options = Options()
options.add_argument("--window-size=1920,1600")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36")

with webdriver.Chrome(service=service, options=options) as driver:
    driver.get("https://www.fantrax.com/login")
    time.sleep(30)  # Wait for the user to log in
    pickle.dump(driver.get_cookies(), open("fantraxloggedin.cookie", "wb"))
```

### Step 3: Use Cookies with Fantrax API
Load the saved cookies into a session and use it with the Fantrax API:
```python
import pickle
from fantraxapi import FantraxAPI
from requests import Session

session = Session()

with open("fantraxloggedin.cookie", "rb") as f:
    for cookie in pickle.load(f):
        session.cookies.set(cookie["name"], cookie["value"])

league_id = "96igs4677sgjk7ol"

api = FantraxAPI(league_id, session=session)

print(api.trade_block())  # Access the private Trade Block page
```

This process allows you to authenticate with Fantrax and access private league data using the Fantrax API.

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
