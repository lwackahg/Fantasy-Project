# Feature Update — 2025-11-12

## Scope
- Scraper optimizations for season traversal and caching.
- Season-aware UI across League Overview, Fantasy Teams, and NBA Team Rosters.
- Standardization of cache format and season parsing.

---

## Scraper: Smart Season Skipping and Caching
- Added consecutive empty-season tracking per player.
  - Counter increments when a valid season has a table header but 0 game rows (`status: no_games_played`).
  - Counter resets to 0 immediately when any games are found and saved (`status: success`).
  - If `consecutive_empty_seasons >= 3`, remaining seasons for that player are skipped.
- Games tab navigation improved:
  - Open Games (Fntsy) once per player; switch seasons via the dropdown only.
  - Track current season on page to avoid redundant switches.
- Caching strategy:
  - Always scrape the current season (2025-26) even if a cache exists.
  - Skip past seasons if cache present unless `force_refresh=True`.
  - Standard cache schema: `data/player_game_log_cache/player_game_log_full_{player_code}_{league_id}_{season}.json` where `season` is `YYYY_YY`.
  - Empty seasons are written with `{ status: "no_games_played", data: [] }` to prevent re-scrapes.
- Error handling
  - Player-level navigation exceptions mark all seasons as failed for that player iteration.
  - Season-level parsing exceptions append to `failed_items` with season context.

### Affected Code
- modules/player_game_log_scraper/logic.py
  - `bulk_scrape_all_players_full(...)`
  - Empty season handling writes `status: no_games_played` and increments empty counter.
  - Successful save resets empty counter.

---

## UI: Season-Aware Views
- Main Player Consistency page
  - Added global Season selector affecting:
    - League Overview
    - Fantasy Teams
    - NBA Team Rosters
  - Individual Player Analysis remains multi-season capable.
- League Overview
  - Filters cache files by the selected season.
  - Header reflects the active season.
- Fantasy Teams
  - Filters by selected season.
  - Header reflects the active season.
- NBA Team Rosters
  - Filters by selected season.
  - Header reflects the active season.

### Affected Code
- modules/player_game_log_scraper/ui_viewer.py
  - Added season extraction from cache filenames and a shared season selector.
  - Passes season-filtered cache file set to subviews.
- modules/player_game_log_scraper/ui_league_overview.py
  - Added in-file season selector and filtering (also receives from main viewer when embedded).
- modules/player_game_log_scraper/ui_fantasy_teams.py
  - Accepts `selected_season`; uses season-filtered caches.
- modules/player_game_log_scraper/ui_team_rosters.py
  - Accepts `selected_season`; uses season-filtered caches.

---

## Data Format Standardization
- Filenames: `player_game_log_full_{code}_{league}_{YYYY_YY}.json`.
- Keys:
  - `player_name`, `player_code`, `league_id`, `season` (e.g., `2024-25`),
  - `data` as list of game rows,
  - `status`: `success` or `no_games_played`.
- UI season parsing unified to take last two parts of filename, convert `_` to `-` for display.

### League Cache Index
- New per-league summary file: `player_game_log_index_{league_id}.json`.
- Structure:
  - `league_id`: Fantrax league ID.
  - `generated_at`: UTC timestamp when the index was built.
  - `players`: mapping of `player_code -> { player_name, seasons }`.
  - Each `seasons[season]` entry contains:
    - `status`: `success` or `no_games_played`.
    - `games`: number of games in that cached season.
    - `cache_file`: filename of the underlying per-player-season JSON.
    - `last_modified`: filesystem mtime of that JSON.
- Built and refreshed via:
  - `build_league_cache_index(league_id)`
  - Loaded with `load_league_cache_index(league_id, rebuild_if_missing=True)`.
- Used by the Player Consistency viewer to:
  - Discover available seasons without globbing every file.
  - Quickly find the relevant cache files for a selected season.

---

## Player Value Analyzer (Cross-Season Value & Durability)

- New backend module: `modules/player_value/logic.py`
  - Core function: `build_player_value_profiles(league_id, seasons=None, min_games_per_season=20, min_seasons=1, max_games_per_season=82) -> pd.DataFrame`.
  - Uses the **league cache index** + per-player-season JSONs to compute per-season stats and aggregate into a multi-season profile per player.
  - Per-season profile (from raw `game_log` via `calculate_variability_stats` + `calculate_multi_range_stats`):
    - `games_played`
    - `mean_fpts`, `median_fpts`, `std_dev`, `cv_pct`
    - `boom_rate`, `bust_rate`
    - `min_fpts`, `max_fpts`
    - `last15_mean_fpts`
  - Aggregated cross-season fields per player:
    - `SeasonsIncluded`, `TotalGames`, `AvgGamesPerSeason`
    - `AvgMeanFPts`, `AvgCV%`, `AvgBoomRate%`, `AvgBustRate%`
    - `AvailabilityRatio` = `AvgGamesPerSeason / max_games_per_season` (clamped to 1.0).
  - Derived scores (normalized 0–100):
    - `ProductionScore` (higher `AvgMeanFPts` is better)
    - `ConsistencyScore` (lower `AvgCV%` is better)
    - `AvailabilityScore` (from `AvailabilityRatio`)
    - `PlayoffReliabilityScore` (currently aliased to `AvailabilityScore`).
  - Composite `ValueScore`:
    - `0.40 * ProductionScore + 0.25 * ConsistencyScore + 0.20 * AvailabilityScore + 0.15 * PlayoffReliabilityScore`.
  - `DurabilityTier` buckets based on `AvailabilityRatio`:
    - `Ironman` (>= 0.90), `Reliable` (>= 0.75), `Fragile` (>= 0.60), `Landmine` (< 0.60).

### New Page: `pages/9_Player_Value_Analyzer.py`

- Purpose: cross-player **ranking** and **visualization** of multi-season value and durability.
- Inputs:
  - League ID (text input, defaulting from env `FANTRAX_DEFAULT_LEAGUE_ID`).
  - Seasons to include (multi-select populated from league cache index).
  - `min_games_per_season`, `min_seasons` filters.
- Behavior:
  - Calls `build_player_value_profiles(...)` under a spinner and bails out gracefully if empty.
  - Displays a **rankings table** with key columns:
    - `Player`, `ValueScore`, `ProductionScore`, `ConsistencyScore`, `AvailabilityScore`, `PlayoffReliabilityScore`,
      `DurabilityTier`, `SeasonsIncluded`, `AvgMeanFPts`, `AvgCV%`, `AvgGamesPerSeason`.
  - Uses Streamlit main layout (no sidebar filters) with:
    - "Filters" section at the top.
    - Main table below.
  - Visualization: **Value vs Production (Trade Target Map)**
    - Plotly scatter:
      - X = `AvgMeanFPts` (Avg FPts/G).
      - Y = `ValueScore`.
      - Color = `DurabilityTier`.
      - Size = `AvgGamesPerSeason`.
    - Designed to highlight:
      - Top-right, `Ironman` / `Reliable` = prime acquisition targets.
      - High production but low value/durability = potential sell-highs.
    - Implemented **without** `trendline="ols"` to avoid a hard dependency on `statsmodels`.
  - CSV export: full value profile table as `player_value_rankings_{league_id}.csv`.

### Integration: Trade Analysis

- File: `modules/trade_analysis/logic.py`
  - `TradeAnalyzer.evaluate_trade_fairness(...)` now:
    - Pulls `league_id` from `st.session_state` (falling back to `FANTRAX_DEFAULT_LEAGUE_ID`).
    - Builds `value_profiles_df = build_player_value_profiles(league_id)` once per call (guarded with try/except).
    - For each time range (`YTD`, `60 Days`, `30 Days`, `14 Days`, `7 Days`):
      - Builds pre- and post-trade top-N rosters (existing behavior).
      - Enriches both with consistency metrics via `enrich_roster_with_consistency` (existing behavior).
      - **New:** joins `value_profiles_df[['Player', 'ValueScore']]` onto each roster to compute:
        - `pre_trade_value_scores[time_range]` and `post_trade_value_scores[time_range]`:
          - `total_value_score` (sum of `ValueScore` for that roster).
          - `avg_value_score` (mean of `ValueScore`).
      - `value_changes[time_range]` now includes:
        - `value_score_change` = post `total_value_score` − pre `total_value_score` (when available).
    - `analysis_results[team]` extended with:
      - `pre_trade_value_scores`, `post_trade_value_scores`, `value_changes` (including `value_score_change`).

- File: `modules/trade_analysis/ui.py`
  - `_display_trade_metrics_table(...)` consumes the new fields:
    - Looks up `pre_trade_value_scores` / `post_trade_value_scores` per time range.
    - Adds a **"Total ValueScore"** column to the metrics table when data exists:
      - Renders as `pre → post` with green/red coloring based on improvement.
  - Net effect: Trade Analysis now shows **production, consistency, and multi-season value/durability impact** side by side for each team and time range.

### Integration: Player Consistency Viewer

- File: `modules/player_game_log_scraper/ui_viewer.py`
  - Imports `build_player_value_profiles`.
  - Inside `show_player_consistency_viewer`:
    - After loading the league cache index:
      - Builds `value_profiles_df = build_player_value_profiles(league_id)` once (try/except guarded).
  - The **Individual Player Analysis** tab now receives `value_profiles_df`:
    - `show_individual_player_viewer(league_id, cache_files, value_profiles_df)`.
  - Inside `show_individual_player_viewer`:
    - After the user selects `selected_player` / `player_code`, the UI attempts to locate the player in `value_profiles_df`:
      - Matches on `player_code` first, falls back to `Player` name.
    - If found, renders a **"📈 Multi-Season Value Profile"** panel above the single-season / multi-season controls with:
      - `ValueScore`, `ProductionScore`, `ConsistencyScore`, `AvailabilityScore`, `DurabilityTier`.
      - `SeasonsIncluded`, `AvgGamesPerSeason`, `AvgMeanFPts`.
    - This ties the value/durability model directly into the per-player consistency UI without changing existing variability charts.

### Follow-up UI Refinements — 2025-11-14

- **Unified Player Value & Consistency hub**
  - `pages/9_Player_Value_Analyzer.py` was promoted to a combined hub:
    - Page title: **"Player Value & Consistency"**.
    - Two main tabs sharing a single league ID input:
      - **📈 Player Value Rankings** — cross-player multi-season value view (existing value analyzer behavior).
      - **📊 Player Consistency Browser** — embeds the existing Player Consistency viewer.
  - The consistency tab calls `show_player_consistency_viewer(initial_league_id=league_id)` so the league field is pre-populated from the hub.

- **Consistency viewer API tweak**
  - `show_player_consistency_viewer` now accepts an optional `initial_league_id` parameter.
  - Behavior is unchanged when called without arguments (e.g., from `2_Player_Consistency.py`).

- **CV% tiers and metric explanations**
  - Consistency tiers were relaxed to better match real fantasy distributions:
    - `< 25%` → `🟢 Very Consistent`.
    - `25–40%` → `🟡 Solid / Moderate`.
    - `> 40%` → `🔴 Volatile / Boom-Bust`.
  - `display_variability_metrics` now:
    - Shows CV% with **two decimal places**.
    - Updates the help text to describe the new thresholds and emphasize that lower CV% is better.
  - The Multi-Season Value Profile block in `show_individual_player_viewer` adds a short caption explaining:
    - How ValueScore blends Production, Consistency, and Availability.
    - That Consistency rewards lower CV% and Durability is derived from games played tiers (Ironman/Reliable/Fragile/Landmine).
    - The updated CV tiers (`<25%`, `25–40%`, `>40%`).

- **Rounding / presentation cleanup**
  - Multi-season value metrics (ValueScore, Production, Consistency, Availability, Avg FP/G, Avg GP/Season) are now rendered with **at most 2 decimal places**.
  - The Player Value Rankings table in `9_Player_Value_Analyzer.py` rounds key numeric columns to two decimals before display.
  - Single-season Visual Analysis adds an explanatory caption clarifying:
    - **FPts Trend**: line chart with shaded boom/bust bands around the mean.
    - **Boom/Bust Zones**: per-game classification into Boom/Normal/Bust using the same ±1 standard deviation thresholds, plus a summary table.

#### Additional Refinements — YoY, CV Tiers, and Navigation

- **YoY comparison folded into the Value & Consistency hub**
  - The standalone **YoY Comparison** page has been merged into a third tab on `9_Player_Value_Analyzer.py`:
    - **📈 Player Value Rankings** (multi-season value view).
    - **📊 Player Consistency Browser** (embedded consistency viewer).
    - **📊 YoY Trends** (historical FP/G year-over-year comparison).
  - The YoY tab reuses `load_and_compare_seasons` and mirrors the original summary:
    - Total players, Improvers, Decliners, Breakouts (>20%), plus the full FP/G and YoY% table.

- **Global CV% tiers used across views**
  - Introduced shared constants and tier helper in `modules/trade_analysis/consistency_integration.py`:
    - `CONSISTENCY_VERY_MAX_CV = 25.0`, `CONSISTENCY_MODERATE_MAX_CV = 40.0`.
    - `get_consistency_tier(cv_percent) -> "🟢 Very Consistent" | "🟡 Solid / Moderate" | "🔴 Volatile / Boom-Bust"`.
  - Fantasy Teams / Team Rosters (`ui_fantasy_teams.py`, `ui_team_rosters.py`) now:
    - Use the shared constants for team-level CV breakdowns and scatter thresholds.
    - Use `get_consistency_tier` for per-player consistency labels in roster tables.
  - League Overview (`ui_league_overview.py`) uses the same thresholds for:
    - Summary metrics, consistency filters, color indicators, and CV% histogram v-lines.
  - Trade Suggestions and Trade Analysis UIs reference the updated tiers in help text and risk labels so all consistency messaging is aligned.

- **YoY performance pre-warming in the hub**
  - Added `_get_yoy_context_for_league(league_id)` and `_get_yoy_comparison_cached(...)` helpers in `9_Player_Value_Analyzer.py`:
    - Context helper locates historical YTD CSVs for the current league using `FANTRAX_LEAGUE_IDS/NAMES` mapping and returns `(league_name_sanitized, seasons)` when at least two seasons are present.
    - Comparison helper wraps `load_and_compare_seasons` in `@st.cache_data` for fast reuse.
  - While viewing **📈 Player Value Rankings**, the page now **warms the YoY cache in the background**:
    - If YoY context is available, the comparison is computed and cached once.
    - The **📊 YoY Trends** tab then reads from the cache, dramatically reducing first-render latency.

- **Legacy page redirects for clean navigation**
  - To avoid duplicate heavy UIs and speed up navigation:
    - `pages/2_Player_Consistency.py` and `pages/8_YoY_Comparison.py` were converted into thin wrappers.
    - They now only import Streamlit, show a short "view moved" message, and expose a `st.page_link("pages/9_Player_Value_Analyzer.py", ...)` into the hub, followed by `st.stop()`.
  - This preserves existing sidebar entries and URLs for backwards compatibility, but ensures all users land on the consolidated **Player Value & Consistency** hub experience.

---

## Pending Work
- Current Season Performance Summary (Fantasy Teams)
  - Add per-team current season summary: avg FPts, CV%, last-10 trend, boom/bust.
- Calculation Optimizations
  - Avoid repeated JSON loads via session caches.
  - Vectorize stat calculations; reuse shared helpers.
- Player Details Season Selector
  - Add a per-player season dropdown within details panel.
- Testing
  - End-to-end: verify season scoping in all three tabs.
  - Scraper: verify skip after 3 empty seasons and force_refresh behavior.

---

## Operational Notes
- Headless mode stability tweaks previously added were removed per current working baseline; revisit if re-enabling headless.
- Current season constant currently treated as `2025-26`; ensure it is updated seasonally or computed.
