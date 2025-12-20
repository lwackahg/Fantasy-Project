# Fantasy Website Review and Improvement Plan

This document outlines the key improvements, bug fixes, and future plans for the Fantasy Basketball Trade Analyzer application.

---

## Session: July 5th, 2025 - Major Stability Fixes & Schedule Analysis Overhaul

This session focused on resolving critical stability issues that were making the application unusable and significantly enhancing the Schedule Analysis feature.

### 1. Schedule Analysis Enhancement

The Schedule Analysis page was completely overhauled to provide a more powerful and automated user experience.

*   **What was done:**
    *   The core logic (`calculate_all_schedule_swaps`) was refactored to not only swap schedules between two teams but to calculate the ripple effect on the **entire league's standings**.
    *   A new, comprehensive UI (`display_all_swaps_analysis`) was created to display a filterable table of every possible schedule swap.
    *   For any selected swap, the UI now shows a detailed, color-coded breakdown of how every team's position in the standings changes, immediately highlighting the biggest winners and losers.
    *   The old, manual "pick two teams to swap" interface was completely replaced with this new automated analysis.
*   **How it was done:**
    *   The logic in `modules/schedule_analysis/logic.py` was updated to iterate through every team pair, perform the swap, and then recalculate and compare the full league standings before and after.
    *   A new display function was added to `modules/schedule_analysis/ui.py` that uses Streamlit tabs and dataframes to present the complex data in an intuitive way.
    *   The `4_Schedule_Analysis.py` page was updated to call these new logic and UI functions.

### 2. Multi-League Data Loading Overhaul

A major usability flaw was identified and corrected. The previous data loading system was rigid, only allowing a single "default" dataset and preventing users from managing or switching between different leagues. This was completely overhauled.

*   **What was done:**
    *   The downloader was fixed to save files with unique, league-specific names.
    *   The data loader UI in the sidebar was completely replaced. The complex file-picker was removed in favor of a simple, clean dropdown menu that automatically detects and lists all available league datasets.
    *   The application now intelligently auto-loads the best available dataset on startup and correctly pre-selects that league in the dropdown menu.
*   **How it was done:**
    *   The filename generation in `modules/fantrax_downloader/logic.py` was corrected to include the league name.
    *   The UI in `modules/data_loader_ui/ui.py` was refactored to scan the data directory, group files by league, and present these leagues in a `st.selectbox`.
    *   The startup logic in `app.py` was updated to use the new loader functions, storing the name of the loaded league in `st.session_state` to ensure the UI is always synchronized.

### 3. UI/UX Enhancements

*   **Streamlined Data Downloading:** The downloader functionality was moved from its own dedicated page directly onto the main application page, accessible via an expander. This simplifies navigation and makes the UI more intuitive, rendering the separate `Downloader` page redundant.

### 4. Critical Bug Fixes & Stability Improvements

A series of critical bugs related to data loading and UI state management were identified and resolved.

*   **What was done:**
    *   Fixed the "Please load schedule data from the main page" error by making the default data loading process robust.
    *   Resolved the "No Fantrax player CSV files found" error by fixing a bug in the file discovery logic.
    *   Eliminated a confusing UI bug where the sidebar showed an incorrect data loading status, contradicting the main page.
    *   Fixed multiple syntax and import errors that were crashing the Schedule Analysis page.
    *   **Fixed Critical Data Integrity Bug:** Resolved an issue where top players were being dropped from non-default CSVs. The CSV parser was updated to correctly handle numbers with comma thousands separators, preventing player data from being discarded during the cleaning process.
*   **How it was done:**
    *   **Robust Data Loading:** The `load_default_data` function in `app.py` was overhauled. It now attempts to load both player and schedule data, provides specific warnings for each if they fail, and only sets the `st.session_state.data_loaded` flag if **all** data is loaded successfully. This prevents the app from getting into a "stuck" state.
    *   **File Discovery Fix:** The regular expression in `_get_league_name_from_filename` (`modules/data_loader_ui/ui.py`) was corrected to handle filenames that do not contain a league name, ensuring all default player CSVs are found.
    *   **UI State Synchronization:** The sidebar UI in `modules/sidebar/ui.py` was refactored. It no longer performs its own file checks and instead relies on the central `st.session_state.data_loaded` flag, ensuring the UI is always consistent.

### 3. Future Improvements

*   **Unit Testing:** Add unit tests for the new `calculate_all_schedule_swaps` logic to ensure its accuracy and prevent future regressions.
*   **Performance:** For very large leagues, the all-swaps calculation could be slow. Investigate caching strategies or performance optimizations if needed.
*   **UI/UX Refinements:** Continue to gather feedback on the new Schedule Analysis page and make iterative improvements to the display and filtering options.

This document outlines the plan for reviewing and improving the Fantasy Website, a Streamlit-based application for fantasy basketball analysis.

## 1. Code Quality Issues

### Critical Issues

- **Lack of Test Coverage:** The `tests/` directory contains mostly placeholder files with no meaningful tests. This is a critical risk for any refactoring effort.

### Code Organization

- **Large Files:** Several files are very large (e.g., `schedule_display.py`, `trade_analysis.py`, `player_data_display.py`), suggesting they contain too much logic and should be broken down.
- **Mixed Concerns:** Files likely mix data processing, business logic, and UI code, which should be separated for better maintainability.
- **Monolithic UI Functions:** The files `src/schedule_display.py`, `src/trade_analysis.py`, and `src/player_data_display.py` contain large, monolithic functions (e.g., `display_schedule_swap`, `display_trade_results`, `display_team_scouting`) that are responsible for the entire UI rendering of their respective features. These functions mix concerns by handling UI layout, state management, data processing, and visualization, which makes them difficult to read, maintain, and test.

### Error Handling

- The application relies heavily on Streamlit's default error reporting. There is little to no custom error handling for issues like missing data, failed calculations, or invalid user input. This can lead to ungraceful failures and a poor user experience.

## 2. Streamlit-Specific Improvements

### Underutilized Features

*To be populated after code review.*

### UI/UX Enhancements

- **Componentization:** The complex UI sections within `schedule_display.py`, like the swap results display, can be broken into smaller, reusable functions or components. For example, the detailed metric display for a swapped team could be a self-contained function `display_team_impact_metrics(team_data, original_stats, new_stats)`.
- **Clearer Visualizations:** The Plotly chart generation logic is embedded directly within the display functions in both `schedule_display.py` and `trade_analysis.py`. Encapsulating this logic in separate, reusable functions (e.g., `create_win_change_barchart(comparison_df)` and `create_performance_line_chart(metric_data)`) would improve readability and code organization.

### Performance Optimizations

- **Caching:** Review the use of `@st.cache_data` and `@st.cache_resource` to ensure data loading and expensive computations are optimized.
- **Session State:** Analyze `st.session_state` usage for efficiency and to prevent unnecessary re-renders.

## 3. Testing Strategy

### Current Coverage

- The current test coverage is critically low. Most files in the `tests/` directory are placeholders that only test for successful module imports (e.g., `test_app.py`, `test_data_loader.py`).
- The `test_schedule_analysis.py` file contains a single, basic test for the `swap_team_schedules` function, but it does not cover edge cases or validate the correctness of the output stats.
- The `test_trade_options.py` file seems to have more substance, but overall, the project lacks a testing foundation.

### Critical Areas for More Tests

- **Business Logic:** The first priority is to add comprehensive unit tests for all functions that will be moved into the `modules/*/logic.py` files. This includes all calculations from `schedule_analysis.py`, `trade_analysis.py`, and the data processing functions in `player_data_display.py`.
- **Data Loading:** The logic in `data_loader.py` is critical and must have robust test coverage, including tests for handling malformed CSV files or missing data.
- **UI Components:** While end-to-end UI testing is complex with Streamlit, we can test the individual UI component functions that will be created in `modules/*/ui.py`. We can check if they render correctly given specific inputs by mocking Streamlit functions.

### Test Organization

- **Mirror `src` structure:** The `tests/` directory should mirror the proposed `src` structure (e.g., `tests/modules/schedule_analysis/test_logic.py`).
- **Use Pytest fixtures:** Use `pytest` fixtures to create reusable test data (e.g., sample DataFrames) to make tests cleaner and more maintainable.
- **Adopt a clear naming convention:** Test functions should be named descriptively (e.g., `test_swap_team_schedules_with_empty_dataframe`).

## 4. Modularity Plan

### Proposed Module Structure

A more modular structure would separate concerns more cleanly.

- **`src/`**
  - `app.py`: Main application entry point.
  - `config.py`: Application configuration.
  - `data_loader.py`: Data loading and initial processing.
  - `session_manager.py`: Session state initialization.
  - **`pages/`**: Each file represents a Streamlit page.
    - `schedule_analysis_page.py`: UI for the schedule analysis feature.
    - `trade_analyzer_page.py`: UI for the trade analyzer.
    - `player_data_page.py`: UI for player data exploration.
  - **`modules/`**: Contains the business logic and UI components for each feature.
    - **`schedule_analysis/`**
      - `logic.py`: Core functions like `swap_team_schedules` and `calculate_all_schedule_swaps` (moved from `src/schedule_analysis.py`).
      - `ui.py`: UI components like `render_swap_selection`, `render_swap_results_table`, `create_win_change_barchart` (extracted from `src/schedule_display.py`).
    - **`trade_analysis/`**
      - `logic.py`: Core trade evaluation logic (from `src/trade_analysis.py`).
      - `ui.py`: UI components for displaying trade options and results.
    - **`player_data/`**
      - `logic.py`: Player data processing functions.
      - `ui.py`: UI components for displaying player stats and charts.

### Breaking Down Large Files

- **`schedule_display.py`:** This file should be deprecated. Its contents will be split into `pages/schedule_analysis_page.py` (the page layout) and `modules/schedule_analysis/ui.py` (reusable UI components). The `display_schedule_swap` function will be broken into smaller functions within `ui.py`.
- **`trade_analysis.py` & `trade_options.py`:** The logic from these files should be consolidated into `modules/trade_analysis/logic.py`. The UI-related parts should move to `pages/trade_analyzer_page.py` and `modules/trade_analysis/ui.py`.
- **`player_data_display.py`:** Logic moves to `modules/player_data/logic.py` and UI to `pages/player_data_page.py` and `modules/player_data/ui.py`.
- **`schedule_analysis.py`**: This file's logic will be moved to `modules/schedule_analysis/logic.py`.

### Dependency Management

- Review `requirements.txt` to ensure all dependencies are listed and up-to-date.
