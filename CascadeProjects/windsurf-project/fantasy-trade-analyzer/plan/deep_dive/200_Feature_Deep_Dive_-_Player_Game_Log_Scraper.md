# Feature Deep Dive: Player Game Log Scraper

## Overview

The **Player Game Log Scraper** is a Selenium-based tool that extracts detailed game-by-game statistics for individual players from Fantrax. It provides comprehensive variability analysis to help evaluate player consistency, identify boom/bust tendencies, and make informed trade decisions.

**Key Features:**
- Scrapes all game-by-game stats from Fantrax player pages
- Calculates variability metrics (CV%, std dev, range, boom/bust rates)
- Caches results locally for fast repeated access
- Bulk scraping for all rostered players in one session
- Visual analysis with charts and downloadable CSV exports

---

## Architecture

### Module Structure

```
modules/player_game_log_scraper/
├── __init__.py
├── logic.py                # Scraping, caching, and calculation logic (~1100 lines)
├── db_store.py             # SQLite database storage for game logs (~300 lines)
├── ui.py                   # Main UI entry point (~580 lines)
├── ui_components.py        # Visualization components (~290 lines)
├── ui_league_overview.py   # League overview interface (~700 lines)
├── ui_fantasy_teams.py     # Fantasy team roster views (~700 lines)
├── ui_team_rosters.py      # Team roster display (~300 lines)
└── ui_viewer.py            # Public viewer for cached data (~600 lines)

pages/
├── 2_Player_Consistency.py # Public viewer page
└── 6_Admin_Tools.py        # Consolidated admin page with password protection
    └── Tab: Player Game Logs
```

**Note:** The UI has been significantly expanded with new views for fantasy teams, team rosters, and a public viewer. The module now supports both JSON cache and SQLite database storage.

### Data Flow

```
User Input (Player Selection)
    ↓
Check Cache (data/player_game_log_cache/)
    ↓
[Cache Hit] → Load JSON → Display Results
    ↓
[Cache Miss or Force Refresh]
    ↓
Selenium Login → Navigate to Player Page
    ↓
Parse Game Log Table (BeautifulSoup)
    ↓
Convert to DataFrame → Calculate Metrics
    ↓
Save to Cache → Display Results
```

---

## Core Components

### 1. Web Scraping (`scrape_player_game_log`)

**Target URL:**
```
https://www.fantrax.com/player/{player_code}/{league_id}
```

**Scraping Strategy:**
- Uses Selenium WebDriver with ChromeDriver
- Logs in once, reuses session for multiple players
- Targets the **second large table** on the page with class:
  ```
  i-table minimal-scrollbar i-table--standard i-table--outside-sticky
  ```
- Extracts all columns from the header row
- Parses all data rows into a structured DataFrame

**Columns Extracted:**
- Date, Team, Opp, Score, FPts, MIN
- FGM, FGA, FG%, 3PTM, 3PTA, 3PT%
- FTM, FTA, FT%, REB, AST, ST, BLK, TO, PF, PTS

**Error Handling:**
- Gracefully handles missing tables
- Tries multiple selectors for player name extraction
- Returns empty DataFrame on failure with error message

### 2. Caching System

**Cache Location:**
```
data/player_game_log_cache/player_game_log_{player_code}_{league_id}.json
```

**Cache Structure:**
```json
{
  "player_name": "Shai Gilgeous-Alexander",
  "player_code": "04qro",
  "league_id": "ifa1anexmdgtlk9s",
  "scraped_at": "2024-11-09 12:00:00",
  "game_log": [
    {
      "Date": "Nov 8",
      "Team": "OKC",
      "Opp": "@GSW",
      "Score": "W 105-101",
      "FPts": 89.0,
      "MIN": "35:28",
      ...
    }
  ]
}
```

**Cache Behavior:**
- Individual scrapes check cache first (unless Force Refresh)
- Bulk scrapes always overwrite cache
- Cache persists until manually cleared or force refreshed

### 3. Variability Metrics (`calculate_variability_stats`)

**Calculated Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Mean FPts** | Average of all games | Expected value per game |
| **Median FPts** | Middle value | Less affected by outliers |
| **Std Dev** | Standard deviation | Absolute variability |
| **Min/Max** | Lowest/highest game | Performance range |
| **Range** | Max - Min | Total scoring spread |
| **CV %** | (Std Dev / Mean) × 100 | Relative consistency |
| **Boom Games** | Games > Mean + 1 SD | High-ceiling performances |
| **Bust Games** | Games < Mean - 1 SD | Low-floor performances |
| **Boom Rate %** | Boom games / Total games | Frequency of elite games |
| **Bust Rate %** | Bust games / Total games | Frequency of poor games |

**CV% Interpretation:**
- **< 20%**: Very consistent (e.g., elite centers)
- **20-30%**: Moderate variability (most players)
- **> 30%**: Volatile (high risk/reward)

### 4. Player Selection (`get_available_players_from_csv`)

**Data Source:**
- Reads from most recent `Fantrax-Players-*-(YTD).csv` file
- Filters to only rostered players (`Status != 'FA'`)
- Returns ~188 players (league-specific)

**Why This Approach:**
- ✅ Only scrapes relevant players (on teams)
- ✅ Automatically updates when new CSV downloaded
- ✅ No manual player ID management needed

### 5. Bulk Scraping (`bulk_scrape_all_players`)

**Process:**
1. Load all rostered players from CSV
2. Start single Selenium session
3. Login to Fantrax once
4. Loop through all players with 2-second delays
5. Scrape each player's game log
6. Save to cache
7. Track successes/failures
8. Close driver

**Performance:**
- ~2 seconds per player
- 188 players = ~6 minutes total
- Single login = efficient, no spam

**Progress Tracking:**
- Real-time progress bar in UI
- Shows current player being scraped
- Summary of successes/failures at end

---

## User Interface

### Page Access

**Location:** Admin Tools (Page 6) → Player Game Logs Tab
- Password protected via `check_password()`
- Consolidated with other admin tools (Downloader, Weekly Standings, Standings Adjuster)

### Two Main Tabs

#### 🔍 Individual Player Analysis

**Player Selection:**
- Dropdown with all rostered players (Status != FA)
- Player code display
- Manual code input fallback

**Controls:**
- League ID input (defaults to env variable)
- Force refresh checkbox (unique key: `player_game_log_force_refresh`)
- Clear cache button
- Bulk scrape all button
- Get game log button

**Results Display:**
1. **Variability Metrics** (with tooltips)
   - Mean, Median, Std Dev
   - Min, Max, Range
   - CV% with interpretation
   
2. **Elite Player Context** (if Mean > 80 FPts/G)
   - Explains variability in context of elite production
   
3. **Boom/Bust Analysis**
   - Boom/bust game counts and rates
   - Visual metrics
   
4. **Visual Analysis** (4 sub-tabs)
   - FPts Trend: Line chart with boom/bust zones
   - Distribution: Histogram with skewness/kurtosis
   - Boom/Bust Zones: Scatter plot with categorization
   - Category Breakdown: Bar chart of PTS/REB/AST/etc.
   
5. **Full Game Log Table**
   - Sortable, filterable DataFrame
   - Prioritized columns (FPts, Date, Team, Opp, Score)
   
6. **CSV Download Button**

#### 📊 League Overview

**Summary Metrics:**
- Total players cached
- Average CV% across league
- Count of consistent players (CV < 20%)
- Count of volatile players (CV > 30%)

**Consistency Table:**
- All cached players with variability metrics
- Built-in column filtering (⋮ menu on headers)
- Consistency indicator column (🟢/🟡/🔴)
- Sortable by any metric
- CSV download

**Visualizations** (3 sub-tabs):
- CV% Distribution: Histogram with consistency thresholds
- Consistency vs Production: Scatter plot with quadrants
- Boom/Bust Analysis: Bubble chart (size = FPts, color = CV%)

### Tooltips & Help Text

**Metric Tooltips:**
- **Mean FPts**: "Average fantasy points per game"
- **Median FPts**: "Middle value - less affected by outliers"
- **Std Dev**: "Measures spread of scores. Higher = more variable"
- **Range**: "Difference between max and min"
- **CV%**: "Coefficient of Variation. Lower % = more consistent"

**Elite Player Context:**
> "⭐ **Elite Player Context:** With a 116.5 FPts/G average, even 'low' games of 70 FPts are excellent. The variability metrics show consistency *relative to this player's elite production*, not absolute fantasy value."

---

## Integration Points

### Environment Variables (fantrax.env)

```env
FANTRAX_USERNAME=your_email@example.com
FANTRAX_PASSWORD=your_password
FANTRAX_DEFAULT_LEAGUE_ID=ifa1anexmdgtlk9s
```

### Shared Utilities

**From `modules/weekly_standings_analyzer/logic.py`:**
- `get_chrome_driver()` - WebDriver setup
- `login_to_fantrax()` - Authentication

**From `data_loader.py`:**
- Uses same data directory structure
- Compatible with existing CSV files

---

## Use Cases

### 1. Trade Evaluation
**Scenario:** Evaluating a trade offer for a high-variance player

**Workflow:**
1. Select player from dropdown
2. Review CV% and boom/bust rates
3. Compare to your team's needs (consistency vs. upside)
4. Check recent game trends in full log
5. Download CSV for deeper analysis

**Decision Factors:**
- Low CV% (< 20%) = Safe, reliable floor
- High boom rate = High ceiling for playoffs
- High bust rate = Risky for weekly matchups

### 2. Weekly Lineup Decisions
**Scenario:** Deciding between two players for a flex spot

**Workflow:**
1. Scrape both players
2. Compare recent 5-game averages
3. Check opponent matchup history
4. Evaluate consistency needs for the week

### 3. League-Wide Analysis
**Scenario:** Identifying buy-low/sell-high candidates

**Workflow:**
1. Run bulk scrape on Sunday night
2. Export all player CSVs
3. Analyze CV% distribution across league
4. Target consistent players from struggling teams
5. Sell volatile players after boom weeks

### 4. Playoff Preparation
**Scenario:** Building a high-ceiling playoff roster

**Workflow:**
1. Bulk scrape all rostered players
2. Sort by boom rate %
3. Identify high-upside players
4. Trade for players with 30%+ boom rates
5. Accept higher variance for championship upside

---

## Technical Details

### Numeric Conversion

**Challenge:** Fantrax uses commas and percentages in HTML

**Solution:**
```python
def convert_to_numeric(value):
    if pd.isna(value) or value == '':
        return None
    if isinstance(value, str):
        value = value.replace(',', '')
        if '%' in value:
            return float(value.replace('%', '')) / 100
    return float(value)
```

### Player Name Extraction

**Challenge:** Fantrax page structure varies

**Solution:** Try multiple selectors in order:
1. `h1.player-name`
2. `h1` (any)
3. `div.player-header__name`

Falls back to "Unknown Player" if all fail.

### Table Selection

**Challenge:** Multiple tables on page with same class

**Solution:** Target the **second occurrence** of the large table class:
```python
tables = soup.find_all('table', class_='i-table minimal-scrollbar i-table--standard i-table--outside-sticky')
if len(tables) >= 2:
    game_log_table = tables[1]  # Second table
```

---

## Performance Considerations

### Caching Strategy
- **Individual scrapes:** Check cache first (instant)
- **Bulk scrapes:** Always fresh (6 min for 188 players)
- **Weekly workflow:** Bulk scrape Sunday, cache all week

### Rate Limiting
- 2-second delay between players in bulk scrape
- Prevents Fantrax rate limiting
- Single login session reduces load

### Memory Usage
- JSON cache files: ~10-50 KB per player
- 188 players = ~5 MB total cache
- Minimal memory footprint

---

## Future Enhancements

### Potential Features
1. **Incremental Updates**
   - Only scrape new games since last cache
   - Faster weekly updates
   
2. **Advanced Visualizations**
   - Line charts showing FPts trends over time
   - Heatmaps for consistency by week
   - Scatter plots for boom/bust distribution
   
3. **Comparative Analysis**
   - Side-by-side player comparisons
   - League-wide consistency rankings
   - Position-based variability benchmarks
   
4. **Trade Integration**
   - Auto-populate trade analyzer with variability data
   - Risk-adjusted trade valuations
   - Consistency-based trade recommendations
   
5. **Predictive Models**
   - Forecast future performance based on trends
   - Identify regression candidates
   - Schedule-adjusted projections

---

## Troubleshooting

### Common Issues

**Issue:** "No players found to scrape"
- **Cause:** No Fantrax-Players CSV in data folder
- **Fix:** Download YTD data from Downloader page

**Issue:** "Unknown Player" in results
- **Cause:** Fantrax page structure changed
- **Fix:** Update player name selectors in logic.py

**Issue:** Scraper times out
- **Cause:** Slow internet or Fantrax server issues
- **Fix:** Increase timeout in Selenium driver config

**Issue:** Bulk scrape fails mid-way
- **Cause:** Connection lost or rate limited
- **Fix:** Check completed players in cache, restart from failures

---

## Code Examples

### Individual Player Scrape
```python
from modules.player_game_log_scraper.logic import scrape_player_game_log

df, player_name, from_cache = scrape_player_game_log(
    player_code="04qro",
    league_id="ifa1anexmdgtlk9s",
    username="user@example.com",
    password="password",
    force_refresh=False
)
```

### Calculate Variability
```python
from modules.player_game_log_scraper.logic import calculate_variability_stats

stats = calculate_variability_stats(df)
print(f"CV%: {stats['cv_percent']:.1f}%")
print(f"Boom Rate: {stats['boom_rate']:.1f}%")
```

### Bulk Scrape
```python
from modules.player_game_log_scraper.logic import bulk_scrape_all_players

result = bulk_scrape_all_players(
    league_id="ifa1anexmdgtlk9s",
    username="user@example.com",
    password="password",
    player_dict=None,  # Auto-loads from CSV
    progress_callback=lambda c, t, n: print(f"{c}/{t}: {n}")
)

print(f"Success: {result['success_count']}/{result['total']}")
```

---

## Related Documentation

- [**180: Trade Analysis**](./180_Feature_Deep_Dive_-_Trade_Analysis.md) - Integration target
- [**190: Standings Tools**](./190_Feature_Deep_Dive_-_Standings_Tools.md) - Similar scraping pattern
- [**140: Fantrax Downloader**](./140_Feature_Deep_Dive_-_Downloader.md) - CSV data source
- [**035: Data Dictionary**](./035_Data_Dictionary.md) - Data schemas

---

## Code Organization

### Modular Refactoring (Nov 2025)

The UI has been significantly expanded from the original 3-file structure to 7 focused modules:

**logic.py** (~1100 lines) - Core scraping and caching
- `get_chrome_driver()` - Selenium WebDriver setup with optimized settings
- `login_to_fantrax()` - Authentication handling
- `scrape_player_game_log()` - Legacy scraper
- `get_player_game_log_full()` - Full season scraper with season support
- `bulk_scrape_all_players_full()` - Multi-season bulk scraping
- `build_league_cache_index()` - League-level cache index generation
- `load_league_cache_index()` - Index loading with rebuild option
- `calculate_variability_stats()` - CV%, boom/bust calculations

**db_store.py** (~300 lines) - SQLite database storage
- `get_db_path()` - Database path resolution
- `init_db()` - Schema initialization
- `upsert_game_log()` - Insert/update game logs
- `get_player_game_log()` - Retrieve player data
- `get_all_players_for_league()` - List all cached players
- Alternative to JSON cache for larger datasets

**ui.py** (~580 lines) - Main entry point
- `show_player_game_log_scraper()` - Main function with tab layout
- Season selector with multi-season support (2018-19 to 2025-26)
- Individual player analysis with variability metrics
- Bulk scrape controls with progress tracking
- Cache management (clear, refresh)

**ui_components.py** (~290 lines) - Visualization components
- `display_variability_metrics()` - Metric cards with tooltips
- `display_boom_bust_analysis()` - Boom/bust rates and counts
- `display_fpts_trend_chart()` - FPts trend line chart
- `display_distribution_chart()` - Histogram with skewness/kurtosis
- `display_boom_bust_zones_chart()` - Scatter plot with categorization
- `display_category_breakdown()` - Category bar chart (PTS/REB/AST/etc.)

**ui_league_overview.py** (~700 lines) - League overview
- `show_league_overview()` - Main overview with multiple tabs
- Season filtering and comparison
- Consistency table with sorting and filtering
- CV% distribution histogram
- Consistency vs Production scatter plot
- Boom/bust bubble chart
- Team-level aggregations

**ui_fantasy_teams.py** (~700 lines) - Fantasy team views
- `_build_fantasy_team_view()` - Team roster builder from cache
- Team-by-team roster display with consistency metrics
- Aggregated team statistics
- Trade target identification
- Integration with Trade Suggestions engine

**ui_team_rosters.py** (~300 lines) - Team roster display
- `show_team_rosters()` - Team roster interface
- Player cards with key metrics
- Roster depth analysis
- Position-based grouping

**ui_viewer.py** (~600 lines) - Public viewer
- `show_player_consistency_viewer()` - Public-facing viewer
- No authentication required
- Read-only access to cached data
- Filtering and search capabilities

**Benefits:**
- Single Responsibility Principle - each file has one purpose
- Easier to maintain and debug
- Reusable components across multiple pages
- Better testability
- Cleaner imports and dependencies
- Separation of admin (scraping) and public (viewing) functionality

---

## Changelog

**v2.0 (Nov 2025)**
- Expanded UI to 7 modular files (logic, db_store, ui, ui_components, ui_league_overview, ui_fantasy_teams, ui_team_rosters, ui_viewer)
- Added SQLite database storage option (`db_store.py`)
- Multi-season support (2018-19 to 2025-26)
- League cache index for faster lookups (`build_league_cache_index()`)
- Fantasy team views with roster aggregations
- Public viewer page (2_Player_Consistency.py)
- Integration with Trade Suggestions engine
- Optimized Chrome driver with eager loading strategy
- Parallel scraping support with concurrent.futures
- Season-aware cache file naming: `player_game_log_full_{code}_{league}_{season}.json`

**v1.1 (Nov 2024)**
- Refactored UI into modular components (3 files)
- Moved to Admin Tools page with password protection
- Added League Overview tab with filtering
- Fixed duplicate checkbox IDs with unique keys
- Added consistency indicator column (🟢/🟡/🔴)
- Improved table filtering with native Streamlit controls

**v1.0 (Nov 2024)**
- Initial implementation
- Basic scraping and caching
- Variability metrics calculation
- Bulk scraping functionality
- Streamlit UI with tooltips and context
- Added new Streamlit page for the tool
- Merged player IDs from multiple season files for completeness
