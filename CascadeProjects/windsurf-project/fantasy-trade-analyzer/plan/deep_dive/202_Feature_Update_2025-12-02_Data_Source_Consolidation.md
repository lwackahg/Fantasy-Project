# Feature Update: Data Source Consolidation (2025-12-02)

## Overview
This update focuses on refactoring the data access layer for player game logs to eliminate code duplication, improve maintainability, and ensure a "Single Source of Truth" for data loading.

## Key Changes

### 1. Centralized Data Loading in `modules/player_game_log_scraper/logic.py`
- **New Function:** `get_player_code_by_name(league_id, player_name)`
  - Resolves player names to IDs using the cached league index.
- **New Function:** `load_cached_player_log(player_code, league_id, season)`
  - Unified loader that checks the database first, then falls back to the JSON cache.
  - Handles season-specific logic transparently.
- **New Function:** `get_latest_available_season(league_id, player_code)`
  - Helper to find the most recent season for a player if none is specified.

### 2. Streamlined Consistency Integration
- `modules/trade_analysis/consistency_integration.py` was refactored to remove manual file globbing and JSON parsing.
- It now delegates all data loading to the new centralized functions in the scraper module.
- `_load_raw_player_data` simply calls `load_cached_player_log`.

### 3. Optimized UI Viewers
- **Individual Player Analysis (`ui_viewer.py`)**:
  - Now uses `load_league_cache_index` to populate the player list, avoiding expensive directory scans.
  - Uses `load_cached_player_log` for data retrieval.
- **League Overview (`ui_league_overview.py`)**:
  - Similarly updated to use the index for discovering available seasons and cache files.

### 4. Historical Trade Analyzer
- `modules/historical_trade_analyzer/logic.py` was significantly reduced in size (~50 lines removed).
- Redundant logic for building player indices and parsing files was replaced with calls to the centralized scraper logic.
- Retains necessary date parsing logic specific to historical analysis while reusing the core data loader.

## Impact
- **Performance:** Faster initial loads for UI components by using the index instead of scanning thousands of JSON files.
- **Maintainability:** Changes to the cache structure or file naming conventions now only need to be handled in one place (`logic.py`).
- **Consistency:** All modules (Trade Analysis, Historical Analysis, Consistency Viewers) now are guaranteed to load data in the exact same way.
