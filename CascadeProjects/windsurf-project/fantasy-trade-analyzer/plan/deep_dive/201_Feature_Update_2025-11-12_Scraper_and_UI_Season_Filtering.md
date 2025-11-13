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
