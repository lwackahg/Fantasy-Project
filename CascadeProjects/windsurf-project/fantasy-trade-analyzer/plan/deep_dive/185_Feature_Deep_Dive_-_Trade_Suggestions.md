# Deep Dive: Trade Suggestions Engine

**Core files (current):**
- `modules/trade_suggestions.py` – main orchestrator (value model, league context, public API).
- `modules/trade_suggestions_config.py` – constants, trade-balance presets.
- `modules/trade_suggestions_core.py` – core value / floor impact / realism caps.
- `modules/trade_suggestions_realism.py` – realism and fairness checks.
- `modules/trade_suggestions_search.py` – pattern-specific search helpers.
- `modules/trade_suggestions_ui_tab.py` – embedded UI tab.
- `pages/5_Trade_Suggestions.py` – legacy full-page UI (still functional, may be simplified to host the tab).

---

## 1. Purpose and Evolution

The Trade Suggestions engine scans league rosters to surface realistic trade ideas that improve a manager's roster while remaining plausible for opponents. It combines an exponential value system with layered realism heuristics, consistency metrics, and intuitive Streamlit visualizations. The November 2025 update focused on curbing unrealistic proposals by integrating CV% (consistency) checks, tightening imbalance thresholds, and validating individual player matchups.

---

## 2. Architecture Overview (post-refactor)

```mermaid
graph TD
    UI[Trade Suggestions UI (page/tab)] -->|config, click| FS[find_trade_suggestions]
    FS --> VC[calculate_league_scarcity_context]
    VC --> CT[trade_suggestions_core._update_realism_caps_from_league]
    FS --> PV[calculate_player_value]
    FS --> PS[_find_* pattern helpers (trade_suggestions_search)]
    PS --> RL[_is_realistic_trade & ratio guards]
    FS --> OUT[(suggestion dicts)]
```

- **UI layer**
  - `modules.trade_suggestions_ui_tab.display_trade_suggestions_tab` (embedded tab) and `pages/5_Trade_Suggestions.py` (page) drive configuration and rendering.
  - They call `find_trade_suggestions(...)` and then apply a **UI-level opponent FP-loss filter** plus presentation logic.

- **Engine orchestrator (`modules.trade_suggestions.py`)**
  - Computes league scarcity context and percentile-based tiers.
  - Calls `_update_realism_caps_from_league(...)` to derive opponent-loss caps from league stats and Trade Balance.
  - Computes player values using `calculate_player_value`.
  - Delegates to `_find_*` pattern helpers in `trade_suggestions_search`.
  - Sorts, deduplicates, and buckets suggestions before returning them.

- **Realism and core helpers**
  - `trade_suggestions_core` handles:
    - Core-size/value calculation.
    - Simulating core FP changes for both sides.
    - Floor impact.
    - League-driven caps (`MAX_OPP_CORE_AVG_DROP`, `MAX_OPP_WEEKLY_LOSS`, equal-count caps).
  - `trade_suggestions_realism` encapsulates `_is_realistic_trade` and supporting checks (FP/G ratios, tier protection, CV trade-off, etc.).
  - `trade_suggestions_search` owns the pattern-specific combinatorics and calls into both core and realism helpers.

---

## 3. League-Aware Value Model (current behavior)

`calculate_player_value()` is intentionally **FP/G-centric** with modest adjustments so numbers stay interpretable and consistent with the league constitution docs (901/902).

1. **Production (base curve)**
   - `calculate_exponential_value(fpts)` applies a mild exponential curve with exponent ≈ **1.3** and a scaling factor anchored around 50 FP/G.
   - This makes stars worth more than a simple linear FP/G model (superstar premium) without blowing up values.

2. **Consistency modifier (CV%)**
   - Players with **very high volatility** (CV% > ~35) get a small penalty (down to about −10%).
   - Stable players are not explicitly rewarded; the philosophy is “FP/G first, consistency as a tiebreaker.”

3. **League scarcity context (percentile- and position-based)**
   - `calculate_league_scarcity_context` computes:
     - Replacement level (top ~85% of rostered players).
     - Percentile-based **tiers**: elite, star, quality, starter, streamer.
     - Position scarcity based on how many players at each position are rostered.
   - `calculate_player_value` then applies light multipliers:
     - Top-percentile players get a small boost; deep-bench players a small trim.
     - Scarce positions get a slight bump; over-supplied ones a slight reduction.
   - Multipliers are intentionally narrow (around ±5–10%) so FP/G remains the primary driver.

There is **no separate trend-analysis layer** in the current implementation; any future trend-based boosts will need to be added explicitly if we decide they’re useful.

---

## 4. Realism System (league-driven & slider-aware)

`_is_realistic_trade()` and its helpers enforce several layers of realism so suggested trades stay plausible for real managers.

### 4.1 League-driven caps (opponent protection)

- `trade_suggestions_core._update_realism_caps_from_league` uses:
  - League tiers (`elite`, `star`, `quality`, `starter`, `streamer`).
  - League-wide average CV%.
  - `TRADE_BALANCE_LEVEL` (1–50 slider) from UI.
- From these it derives:
  - `MAX_OPP_CORE_AVG_DROP` – maximum allowed drop in an opponent’s core FP/G average.
  - `EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS` / `MAX_OPP_WEEKLY_LOSS` – weekly core FP loss caps.
- Pattern helpers in `trade_suggestions_search` enforce:
  - `opp_core_gain >= -MAX_OPP_WEEKLY_LOSS` for all trades.
  - `opp_core_avg_drop <= MAX_OPP_CORE_AVG_DROP` for certain heavier consolidations (e.g., 3‑for‑1, 3‑for‑2).

### 4.2 Decomposed realism checks

`trade_suggestions_realism` breaks realism into focused helpers, including:

- **FP/G ratio checks** – average and total FP/G comparisons tuned for consolidations, expansions, and equal-count patterns.
- **Package quality checks** – ensure you’re not offering “one good piece plus scrubs” for a star, or accepting wildly inferior pieces.
- **Tier protection** – tier-aware rules so elite and star players only move for appropriately strong returns.
- **Consistency (CV%) trade-off** – prevents trading steadier production for volatility without adequate FP/G compensation.
- **Best-player and matchup checks** – compare top players on each side and per-slot matchups so each piece has a roughly comparable counterpart.

At the top level, `_is_realistic_trade` orchestrates these checks and returns a single boolean. Trade Balance and league tiers influence the internal thresholds, but the details are encapsulated in the helpers rather than hard-coded here.

### 4.3 Trade Balance slider behavior

- The UI exposes **Trade Balance (1 = super strict, 50 = very loose)**.
- `set_trade_balance_preset(level)` stores `TRADE_BALANCE_LEVEL` and adjusts equal-count caps.
- `_update_realism_caps_from_league` then scales opponent-loss and ratio caps using this level.
- Inside `_is_realistic_trade`, extremely loose settings (e.g. `TRADE_BALANCE_LEVEL >= 40` and non‑equal patterns) can bypass some of the strictest realism gates, turning the engine into an exploratory “idea generator” while still respecting opponent-loss caps.

In practice:

- Lower levels (~1–5) behave conservatively and surface trades that look fair in a human league.
- Higher levels (up to 50) allow more lopsided consolidations and depth plays, especially when combined with a loose UI filter for opponent core FP change.

---

## 5. UI Enhancements & Deep Dive

`_display_trade_suggestion()` in `trade_suggestions_ui_tab.py` provides comprehensive trade analysis:

### 5.1 Core Display Components

- **Side-by-side tables** with FP/G and CV% per player
- **Weekly Core FP Impact chart** (bar chart comparing your change vs opponent change)
- **Trade verdict badges**: Excellent, Strong, Decent, Marginal, Trade-Off, Core FP Loss
- **Opponent acceptance indicators**: Win-win, negotiate carefully, may not accept

### 5.2 Trade Rationale ("Why this trade works")

Dynamic bullet points explaining:
- Core upgrade/downgrade with FP/G and weekly FP changes
- Trade pattern context (consolidation vs depth play)
- Package FP/G comparison
- Consistency impact (risk reduction vs higher variance)

### 5.3 Deep Dive Expander (compact)

Three-column metrics display:
- Your Avg FP/G, Your Avg CV%
- Their Avg FP/G, Their Avg CV%
- FP/G Change, CV% Change
- Risk level assessment (Low/Moderate/High)

### 5.4 Recent Form Analysis

Compares YTD vs Last 7/14/30 performance:
- Identifies players with significant recent FP/G swings (±3 FP/G)
- Flags trending up (📈) or down (📉) players
- Helps identify buy-low/sell-high opportunities

### 5.5 Talking Points for Opponent

AI-generated negotiation suggestions:
- Opponent core FP impact framing
- Package FP/G advantage from their perspective
- Consistency trade-offs
- Pattern-specific selling angles (depth vs consolidation)
- **YoY Integration**: Highlights sell-high candidates (YoY upswing) and buy-low candidates (YoY decline) using historical data

### 5.6 Trade Framework Analysis Expander

Multi-angle strategic breakdown:

**Package-level comparison:**
- FP/G advantage/disadvantage assessment
- Consistency upgrade/downgrade analysis

**Roster construction impact:**
- Core upgrade magnitude (major/solid/modest)
- Consolidation vs depth trade implications

**Strategic trade-offs:**
- Star-for-depth vs depth-for-star bets
- Risk profile shifts
- Opponent sacrifice vs win-win structure

**Bottom line verdict:**
- Strong trade, Strategic overpay, Modest upgrade, or Marginal trade

### 5.7 Roster Snapshot Expander

Before/after roster comparison (top 10 players):
- Your team: before and after with Trade Status column (IN/OUT)
- Opponent team: before and after with Trade Status column
- Sorted by Mean FPts or FP/G

### 5.8 Full Trade Analysis Integration

Button to run complete Trade Analysis tool:
- Converts suggestion to `trade_teams` format
- Calls `run_trade_analysis()` with 7/14/30/60d time ranges
- Displays full `display_trade_results()` output

---

## 6. YoY Data Integration

The Trade Suggestions UI integrates Year-over-Year comparison data:

```python
from modules.historical_ytd_downloader.logic import load_and_compare_seasons, get_available_seasons
```

**Usage in talking points:**
- Identifies players with +5 FP/G YoY improvement (sell-high candidates you're offering)
- Identifies players with -5 FP/G YoY decline (buy-low candidates you're receiving)
- Provides specific FP/G trajectory data for negotiation leverage

---

## 7. Workflow Summary (engine + UI)

1. **Load league data and rosters** from cached game logs / DB.
2. **UI collects configuration:**
   - Your team, trade patterns, Min Value Gain.
   - Trade Balance level (1–50).
   - Optional target/excluded teams and players.
   - Opponent min weekly core FP change (UI-level filter).
3. **Engine call – `find_trade_suggestions(...)`:**
   - Filters rosters by games played (GP share of league max).
   - Computes league scarcity context and percentile-based tiers.
   - Updates realism caps from league stats and Trade Balance.
   - Computes player values (FP/G-centric with light scarcity/CV adjustments).
   - Estimates search complexity and prunes the heaviest patterns if necessary.
   - Computes your and opponents’ core values.
   - Calls `_find_*` pattern helpers to enumerate candidate trades, applying early realism checks and opponent-loss caps.
   - Sorts, deduplicates, buckets by team/pattern, and returns top `max_suggestions`.
4. **UI post-filter:**
   - Filters suggestions by `opp_core_gain >= realism_min_opp_core` (e.g., −150 FP/week for very loose exploration).
   - Displays metrics, rationale, and roster snapshots for each suggestion.

---

## 7. Future Enhancements

- **Dynamic Threshold Tuning** – Allow power users to adjust realism caps or CV% sensitivity.
- **Positional Needs Analysis** – Highlight trades that rebalance positional scarcity.
- **Schedule Integration** – Consider future strength-of-schedule when ranking trades.
- **Scenario Saving** – Let users bookmark interesting trades to revisit later.
