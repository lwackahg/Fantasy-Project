# Deep Dive: Team Analyzer Feature

This document provides a detailed breakdown of the Team Analyzer feature, which allows users to assess the categorical strengths and weaknesses of any team in the league.

---

### 1. Architecture

The feature follows the application's standard modular design:

- **Page Entry Point (`pages/5_Team_Analyzer.py`)**: A simple script that serves as the entry point for the page. It ensures data is loaded before calling the main UI display function.

- **UI Module (`modules/team_analyzer/ui.py`)**: This module renders the user interface. It includes:
  - A dropdown to select a team to analyze.
  - A dropdown to select the time range for the analysis (e.g., `YTD`, `30 Days`).
  - A Plotly radar chart that provides a powerful visualization of the selected team's ranks across all statistical categories.
  - An expandable section containing a DataFrame of the full league rankings for detailed comparison.

- **Logic Module (`modules/team_analyzer/logic.py`)**: This module contains the functions that perform the statistical calculations.

---

### 2. User Flow

1.  The user navigates to the **Team Analyzer** page from the sidebar.
2.  The user selects a team and a time range from the dropdown menus.
3.  The application automatically calculates and displays the analysis.
4.  The user can interpret the radar chart to quickly identify strong and weak categories (a rank of 1 is best and is on the outer edge of the chart).
5.  The user can expand the table to see the exact rankings for all teams in the league.

---

### 3. Core Logic

The analysis is performed by two main functions:

1.  **`calculate_team_stats()`**: This function takes the combined player data and a selected time range. It then:
    - Filters the data to include only the relevant time period.
    - Groups the data by `Fantasy_Manager`.
    - Calculates the sum for each of the core statistical categories (`PTS`, `REB`, `AST`, `ST`, `BLK`, `3PTM`, `TO`).
    - **Inverts Turnovers**: It multiplies the `TO` category by -1. In fantasy basketball, fewer turnovers are better. This inversion ensures that when ranks are calculated, a lower turnover total results in a better (higher) rank, consistent with all other categories.

2.  **`calculate_league_ranks()`**: This function takes the output from `calculate_team_stats()` and uses the powerful `pandas.DataFrame.rank()` method. It ranks each team against the others for every statistical category, with `ascending=False` ensuring that higher statistical totals receive a better rank. The result is a DataFrame where each cell contains a team's league-wide rank for that category.
