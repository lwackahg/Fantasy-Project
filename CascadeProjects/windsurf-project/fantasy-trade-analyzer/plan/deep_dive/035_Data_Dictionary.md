# Data Dictionary

This document serves as a guide to the primary data sources used in the Fantasy Trade Analyzer. It explains the purpose and schema of the key CSV files and the DataFrames that are loaded into the application's session state.

---

## 1. Player Statistics (`Fantrax-Players-*.csv`)

-   **Purpose**: These files contain the core player performance data for different time ranges (e.g., Last 7 Days, Last 15 Days, Year-to-Date). They are the foundation for nearly all analysis in the application.
-   **Loaded Into**: `st.session_state['player_data']` (a dictionary of DataFrames, keyed by time range).
-   **Key Columns**:
    -   `Player`: The player's full name (Primary Key).
    -   `Team`: The player's NBA team.
    -   `Pos`: The player's eligible fantasy positions.
    -   `Status`: The player's current status (e.g., FA, On Team).
    -   `FP/G`: Fantasy Points Per Game.
    -   `FPts`: Total Fantasy Points.
    -   `% Owned`: The percentage of leagues on Fantrax where the player is owned.
    -   *Various Stat Columns*: `FG%`, `FT%`, `3PTM`, `PTS`, `REB`, `AST`, `ST`, `BLK`, `TO`.

---

## 2. League Schedule (`schedule.csv`)

-   **Purpose**: This file contains the complete head-to-head matchup schedule for the entire season. It is essential for the Schedule Swap Analysis feature.
-   **Loaded Into**: `st.session_state['schedule_data']`.
-   **Key Columns**:
    -   `Period`: The scoring period or week number.
    -   `Team 1`: The name of the first team in the matchup.
    -   `Team 2`: The name of the second team in the matchup.
    -   `Score 1`: The fantasy score for Team 1.
    -   `Score 2`: The fantasy score for Team 2.
    -   `Winner`: The calculated winner of the matchup (1, 2, or 0 for a tie).

---

## 3. Draft Results (`Fantrax-Draft-Results-*.csv`)

-   **Purpose**: This file contains the results of the league's auction or snake draft. It is used to enrich the player data with draft-day cost information.
-   **Loaded Into**: `st.session_state['draft_results']` (if the file exists).
-   **Key Columns**:
    -   `Player`: The player's full name (Primary Key).
    -   `Team`: The fantasy team that drafted the player.
    -   `Bid`: The auction price paid for the player.
    -   `Pick`: The overall pick number (for snake drafts).

---

## 4. Player Projections (`player_projections.csv`)

-   **Purpose**: This is a generated file used exclusively by the Auction Draft Tool. It contains forward-looking projections for player performance, which are used to calculate baseline auction values.
-   **Loaded Into**: `st.session_state['pps_df']`.
-   **Key Columns**:
    -   `Player`: The player's full name (Primary Key).
    -   `PPS`: The calculated Player Power Score, a composite value blending historical performance and injury risk.
    -   `Pos`: The player's primary position.
    -   `Team`: The player's NBA team.
    -   `MarketValue`: The player's average auction price from previous seasons.
