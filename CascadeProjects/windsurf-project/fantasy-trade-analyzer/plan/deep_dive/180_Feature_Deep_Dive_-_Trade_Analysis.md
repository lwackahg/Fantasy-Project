# Deep Dive: Trade Analysis

**File Path:** `pages/1_Trade_Analysis.py`

## 1. Purpose and Overview

The Trade Analysis feature is a core component of the application, designed to provide a deep, data-driven evaluation of fantasy basketball trades. It allows users to simulate a trade involving two or more teams and see a detailed breakdown of the statistical impact on each team's roster. The analysis compares key performance metrics before and after the proposed trade across multiple time ranges (e.g., YTD, Last 60 Days, Last 14 Days), helping users make informed trading decisions.

---

## 2. Architecture and Core Components

The feature's architecture is modular, separating the main page, UI rendering, and business logic into distinct components.

```mermaid
graph TD
    subgraph "Trade Analysis Architecture"
        A[1_Trade_Analysis.py] --> B[modules/trade_analysis/ui.py];
        B -- Calls --> C[modules/trade_analysis/logic.py];

        subgraph "UI Layer"
            B -- Renders --> D[Team/Player Selection & Results Visualization];
        end

        subgraph "Logic Layer"
            C -- Contains --> E(TradeAnalyzer Class);
            E -- Performs --> F[Before & After Roster Analysis];
        end
    end
```

-   **`pages/1_Trade_Analysis.py`**: A lightweight entry point that sets the page configuration and calls the main UI display function.
-   **`modules/trade_analysis/ui.py`**: Manages the entire user interface. It's responsible for rendering the team and player selection widgets and for displaying the final analysis in a structured format with tabs, tables, and Plotly charts.
-   **`modules/trade_analysis/logic.py`**: Contains the `TradeAnalyzer` class, which is the engine of the feature. It handles all the data manipulation and statistical calculations required to evaluate a trade.

---

## 3. Core Logic: The `TradeAnalyzer` Class

The analysis is performed by the `evaluate_trade_fairness` method within the `TradeAnalyzer` class. Here is a step-by-step breakdown of its process:

1.  **Identify Players**: For each team involved in the trade, the method identifies the `outgoing_players` (those being traded away) and `incoming_players` (those being received).
2.  **Iterate Through Time Ranges**: The analysis is performed independently for each time range ('YTD', '60 Days', '30 Days', '14 Days', '7 Days').
3.  **Construct Pre-Trade Roster**: For a given time range, it takes the team's current roster and filters it down to the top `n` players based on Fantasy Points per Game (FP/G). This represents the team's core contributors.
4.  **Construct Post-Trade Roster**: It creates a hypothetical new roster by:
    -   Removing the `outgoing_players` from the team's data.
    -   Finding the `incoming_players`' data from the global dataset.
    -   Adding the `incoming_players` to the team.
    -   It then filters this new roster down to the top `n` players by FP/G.
5.  **Calculate Metrics**: For both the pre-trade and post-trade rosters, it calculates a dictionary of key performance metrics:
    -   `mean_fpg`: Mean Fantasy Points per Game
    -   `median_fpg`: Median Fantasy Points per Game
    -   `std_dev`: Standard Deviation of FP/G (a measure of consistency)
    -   `total_fpts`: Sum of all Fantasy Points
    -   `avg_gp`: Mean Games Played
6.  **Store Results**: All pre-trade and post-trade rosters and metrics are stored in a nested dictionary, which is returned to the UI layer for display.

---

## 4. UI and Visualization

The `ui.py` module takes the complex dictionary of results from the logic layer and presents it in an intuitive way:

-   **Team Tabs**: The results for each team are separated into their own tabs for clarity.
-   **Trade Overview**: A high-level summary shows which players are being received and traded away.
-   **Metrics Table**: A powerful table shows the Before -> After comparison for each key metric, with color-coding (green for improvement, red for decline) to quickly highlight the impact.
-   **Performance Charts**: Plotly line charts visualize the trend of key metrics (like FP/G) across all time ranges, making it easy to see if a trade provides a short-term gain but a long-term loss.
-   **Roster Details**: A side-by-side view of the pre-trade and post-trade rosters, with outgoing players highlighted in red and incoming players highlighted in green.

## 5. Trade History & Snapshot Replay

### 5.1 History Storage

The Trade Analyzer persists trade history to JSON under:

- `data/trade_history/league_<league_id>_trades.json` 

Each entry has a compact schema:

```jsonc
{
  "trade_teams": {
    "TeamA": { "Player 1": "TeamB", "Player 2": "TeamC" },
    "TeamB": { "Player 3": "TeamA" }
  },
  "summary": "Human‑readable multi‑line summary of the trade by team/time range",
  "label": "Optional user‑defined label",
  "date": "2025-11-10",          // ISO date, empty for non‑historical
  "num_players": 10,             // Top-N players included in the analysis
  "source": "live | historical", // See below
  "season": "2025-26",           // Present for historical entries
  "rosters_by_team": {           // Present for historical entries
    "TeamA": ["Player 1", "..."],
    "TeamB": ["Player 3", "..."]
  },
  "league_id": "ifa1anexmdgtlk9s"
}
```

Notes:

- **Live** trades (`source` absent or != `"historical"`) are computed against the current CSV‑driven `combined_data`.
- **Historical** trades (`source == "historical"`) originate from the Historical Trade Analyzer and include enough metadata to rebuild a full snapshot from game‑log caches.

### 5.2 Trade History UI

The main Trade Analysis page renders a collapsible:

- **"Trade Analysis History"** expander:
  - Lists entries (most recent first).
  - Displays `date` + `label` with a `(historical snapshot)` badge when `source == "historical"`.
  - Shows the stored `summary` text for quick scanning.
  - Provides a **"View details"** button for each entry.

Internally, the button simply captures the selected entry; the replay itself is rendered outside the narrow column to preserve full‑width layout.

### 5.3 Replay Behavior (_replay_trade_from_history)

Replays are handled by `modules.trade_analysis.ui._replay_trade_from_history(entry)`:

```python
def _replay_trade_from_history(entry: Dict[str, Any]) -> None:
    """Recompute and display a read-only view of a cached trade history entry.

    For historical entries with full context, this rebuilds the original
    game-log snapshot. For other entries, it recomputes the analysis using
    the current combined_data.
    """
```

**Historical entries (`source == "historical"`):**

1. Extract `season`, `date`, `league_id`, and `rosters_by_team` from the entry.
2. Parse `date` into a `trade_dt`.
3. Call `build_historical_combined_data(trade_dt, league_id, season, rosters_by_team)` from `modules.historical_trade_analyzer.logic` to reconstruct `combined_data` as of the trade date.
4. Instantiate a fresh `TradeAnalyzer(snapshot_df)` and call `evaluate_trade_fairness(trade_teams, num_players)`.
5. Annotate each team’s results with:
   - `season` 
   - `trade_date` (string)
   - `league_id` 
6. Pass the results to `display_trade_results`, yielding the **full normal Trade Analysis UI in a read‑only mode**.

**Non‑historical / legacy entries:**

- If `combined_data` is loaded:
  - Update the existing `st.session_state.trade_analyzer` with the current data.
  - Re‑run `evaluate_trade_fairness(trade_teams, num_players)`.
  - Display via `display_trade_results`.
- This shows how the trade looks **under current data**, not as a historical snapshot.

### 5.4 Limitations

- Historical replay is exact only for entries that came from the Historical Trade Analyzer (i.e. have `season`, `date`, `rosters_by_team`, `league_id`).
- Older entries without this context cannot be reconstructed as true snapshots; they are recomputed using whatever `combined_data` is currently loaded.
- To keep history JSON compact, **the full `analysis_results` dict is not persisted**; instead, the snapshot is deterministically rebuilt from the game‑log caches.
