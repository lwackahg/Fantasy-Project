# UI Components (`logic/ui_components.py`)

This document provides a detailed breakdown of the UI components that constitute the Fantasy Trade Analyzer's auction tool interface. The components are built using Streamlit and are designed to be modular and state-driven.

---

### Phase Badge and Roster Enforcement (UI Updates)

- **Phase Badge (Early/Mid/Late)**: Near the `On the Clock` header on the draft page, a small caption displays the current draft phase as `Early`, `Mid`, or `Late`, computed from league budget depletion. This mirrors the phase concept used in optional in-draft models.
- **Roster Capacity Enforcement**: The draft form now filters the `Assign Position` options to only show positions with remaining capacity for the selected team. Flex is offered if available; Bench is only shown when `Bench > 0`. Server-side validation in `drafting_callback()` prevents assigning beyond capacity and auto-falls back to Flex when a core slot is full and Flex capacity remains.

### 1. `render_setup_page()`

**Purpose**: To render the initial configuration page where users set up the draft parameters before it begins.

**Key Features**:
- **Settings Management**: Allows users to import and export their draft settings as a JSON file. This is useful for saving and loading complex configurations.
- **League Setup**: Basic league parameters such as the number of teams, budget per team, roster spots, and the number of games in a season.
- **Roster Composition**: Defines the number of players required for each position (G, F, C, Flex, Bench). It includes validation to ensure the composition matches the total roster spots.
- **Valuation Models**: Users can select which base value and in-draft scarcity models they want to use for player valuation.
- **Adjustments**: Provides UI for entering a list of injured players and for defining the PPS percentile cutoffs for player tiers.
- **Trend Weights**: Allows customization of the weights applied to the last four seasons of player data when calculating the Player Power Score (PPS).

---

### 2. `render_draft_board()`

**Purpose**: To display the main draft board of available players in a rich, interactive grid.

**Key Features**:
- **Interactive Grid**: Uses the `st_aggrid` library to create a sortable, filterable, and customizable grid.
- **Dynamic Columns**: The grid dynamically adds or removes columns for individual valuation models based on how many are selected in the setup.
- **Custom Styling**: Applies conditional formatting to cells to improve readability:
  - **Tier**: Color-coded backgrounds for each player tier.
  - **Confidence**: Color-coded backgrounds (green, gold, red) to indicate the confidence level in the player's valuation.
- **Pinned Columns**: The player's name is pinned to the left for easy reference while scrolling horizontally.

---

### 3. `render_team_rosters()`

**Purpose**: To provide a compact, at-a-glance view of every team's roster and budget.

**Key Features**:
- **Columnar Layout**: Displays each team in its own column for a clean, organized view.
- **Real-time Updates**: Shows the remaining budget and the list of drafted players for each team, which updates instantly after each draft pick.

---

### 4. `render_player_analysis()`

**Purpose**: To render a detailed analysis section for a player selected from the draft board.

**Key Features**:
- **Draft Form**: Contains the form to draft the selected player, including selecting the drafting team and setting the draft price.
- **Valuation Details**: Breaks down the player's valuation, showing the average base and adjusted values, as well as the individual values from each selected model.
- **Key Metrics**: Displays important metrics like Tier, VORP (Value Over Replacement Player), and Market Value.
- **Historical Performance**: Shows a table with the player's games played (GP) and fantasy points per game (FP/G) for the last four seasons.

---

### Injury Annotations (UI Enhancements)

The auction tool surfaces injury information across the UI to improve draft-time awareness:

- **Player Select and Labels**: Injured players are labeled with "(INJ)" via a `DisplayName` field during selection and in tables.
- **Player Analysis Banner**: When a selected player has `IsInjured` set, a red banner shows the injury duration from `InjuryStatus` (e.g., "Half Season", "Full Season").
- **All Player Values Table**: Adds an `Injury` column and lightly tints rows for injured players for quick scanning.
- **Nomination Targets**: Full-season injured players are excluded from the "Top 5 Nomination Targets"; other injuries display inline (e.g., "Inj: Half Season").

Inputs for injury status are parsed from the setup textarea and stored as a session-level `injury_map`. These annotations are computed in the page layer and do not alter the valuation engine's core logic.
- **Team Optimizer Integration**: Includes a button to run the `team_optimizer` for the user's selected team. It calculates and displays the optimal roster that can be built with the remaining budget and roster spots.

---

### 5. `render_draft_summary()`

**Purpose**: To display a summary of all players who have been drafted so far.

**Key Features**:
- **Simple Table**: Renders a clean table showing the drafted players, the team that drafted them, and their draft price.
- **Value Comparison**: Includes columns for the player's calculated `BaseValue` and `AdjValue` alongside their `DraftPrice` to easily see if a player was a good value pick.
