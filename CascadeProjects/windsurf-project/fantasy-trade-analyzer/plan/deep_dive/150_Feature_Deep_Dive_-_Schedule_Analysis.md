# Deep Dive: Schedule Analysis Feature

This document provides a detailed overview of the Schedule Analysis feature, a powerful tool for analyzing schedule luck and its impact on league standings.

---

### 1. Purpose

The Schedule Analysis feature allows users to move beyond simple win/loss records and explore the element of luck in their fantasy league. It answers the question, "What would the standings look like if Team A had played Team B's schedule?" By simulating schedule swaps, users can identify which teams have had lucky or unlucky matchups and how dramatically the league landscape could change under different circumstances.

---

### 2. Architecture

The feature is built on a three-part structure:

- **Page Entry Point (`pages/4_Schedule_Analysis.py`)**: This script sets up the main page layout, handles top-level data filtering for teams and scoring periods, and orchestrates the calls to the UI and logic modules.

- **UI Module (`ui/schedule_analysis_ui.py`)**: This module is responsible for rendering all the user-facing components, including:
  - Current league standings and team performance stats.
  - Interactive selectors for performing a **Manual Schedule Swap** between two teams.
  - Detailed metrics and charts showing the impact of a manual swap on the selected teams' records and standings.
  - A comprehensive, filterable table for the **All Possible Swaps** analysis, highlighting the most impactful swaps and their ripple effects across the league.
  - Raw schedule data presented in both list and table views.

- **Logic Module (`logic/schedule_analysis.py`)**: This module contains the computational engine of the feature. It handles the complex calculations required for the analysis.

---

### 3. Key Features & Logic

#### a. Team Stats Calculation

- **`calculate_team_stats()`**: This function processes the raw schedule data to calculate key performance metrics for each team, including Wins, Losses, Ties, Points For, Points Against, and Win Percentage. This forms the baseline for all comparisons.

#### b. Manual Schedule Swap

- **`swap_team_schedules()`**: This is the core simulation function. When a user selects two teams to swap, this function meticulously reconstructs the entire league schedule. It correctly handles three scenarios for each matchup in the season:
  1.  Matchups involving **Team 1** (but not Team 2) are given to **Team 2** (with Team 2's original score for that week).
  2.  Matchups involving **Team 2** (but not Team 1) are given to **Team 1** (with Team 1's original score for that week).
  3.  Matchups where **Team 1 and Team 2 play each other** have their roles and scores reversed.
- After rebuilding the schedule, it calls `calculate_team_stats()` again to generate the new standings.

#### c. All Possible Swaps Analysis

- **`calculate_all_schedule_swaps()`**: This is a computationally intensive function that is cached with `@st.cache_data` to ensure performance. It systematically iterates through every possible unique pair of teams in the league.
- For each pair, it runs the `swap_team_schedules()` simulation.
- It then calculates the change in standings position for **every team in the league** as a result of the swap.
- The results are compiled into a summary DataFrame that identifies the direct impact on the two swapped teams, as well as the biggest beneficiaries ("Biggest Winner") and victims ("Biggest Loser") among the other teams.
- An overall "Impact" score is calculated for each swap to easily sort and find the most league-altering scenarios.
