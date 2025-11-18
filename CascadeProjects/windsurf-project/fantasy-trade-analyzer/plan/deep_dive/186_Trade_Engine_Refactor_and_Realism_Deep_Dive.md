# Trade Engine Refactor & Realism Deep Dive

This document aggregates medium/long-term improvement ideas for the trade suggestion engine so we can track them and gradually implement them. It is meant to replace scattered notes across older planning docs.

---

## 1. Architectural & Structural Refactor

### 1.1 Break up `trade_suggestions.py` (monolith)

**Goal:** Improve maintainability and make it safer to tune realism logic and performance.

**Current split (implemented Nov 2025):**

- `modules.trade_suggestions_config`
  - League- and engine-level constants (e.g. `ROSTER_SIZE`, `MAX_COMBINATIONS_PER_PATTERN`).
  - Trade-balance configuration (`TRADE_BALANCE_LEVEL`, `set_trade_balance_preset`).

- `modules.trade_suggestions_core`
  - Core-size and core-value helpers.
  - Floor-impact calculation.
  - League-driven realism caps (`_update_realism_caps_from_league`, `MAX_OPP_CORE_AVG_DROP`, `MAX_OPP_WEEKLY_LOSS`, equal-count caps).

- `modules.trade_suggestions_realism`
  - `_is_realistic_trade` and its decomposed helpers for FP/G ratios, consolidation/expansion quality, tier protections, CV trade-offs, and matchups.

- `modules.trade_suggestions_search`
  - All pattern-specific search helpers (`_find_1_for_1_trades`, `_find_2_for_1_trades`, …, `_find_3_for_3_trades`).
  - Early realism checks and cheap filters before expensive core simulations.

- `modules.trade_suggestions` (thinner orchestrator)
  - `calculate_exponential_value`, `calculate_player_value`, `calculate_league_scarcity_context`.
  - `find_trade_suggestions` orchestrating scarcity context, realism-cap updates, and delegating to `_find_*` search helpers.

- Facades for smaller import surfaces:
  - `modules.trade_suggestions_value` – re-exports value/league-context helpers.
  - `modules.trade_suggestions_impact` – re-exports core-impact helpers.

This achieves the intent of the original refactor proposal while keeping public imports stable for the rest of the app.

### 1.2 Generic N-for-M pattern finder (future)

Originally the engine lived entirely in `trade_suggestions.py` with 9 near-duplicate pattern helpers:

- `_find_1_for_1_trades`, `_find_2_for_1_trades`, `_find_2_for_2_trades`, `_find_3_for_1_trades`, `_find_3_for_2_trades`,
- `_find_1_for_2_trades`, `_find_1_for_3_trades`, `_find_2_for_3_trades`, `_find_3_for_3_trades`.

These helpers have now been **moved into `modules.trade_suggestions_search`** and share more common structure (early filters, realism checks, suggestion payload shape), but they are still implemented as separate functions. This keeps pattern-specific tuning easy to reason about while we continue to refine realism.

The idea of a fully generic N-for-M dispatcher that takes `(num_you_give, num_they_give)` plus a small `PatternConfig` remains a **future refactor** and should only be attempted once:

- We have good test coverage for each pattern’s current behavior, and
- We are confident we won’t lose pattern-specific nuance (e.g., consolidation vs depth heuristics).

---

## 2. Value Model & Realism

### 2.1 Exponential value exponent alignment

The UI documentation previously described a strong exponential curve (e.g. `value = (FPts ** 1.8) * 0.35`), but the current engine uses a much flatter exponent (`1.05`) to keep things FP/G-centric.

**Idea:** Move toward a steeper exponent over time, but do it incrementally and in sync with realism thresholds:

- Step 1: pick a moderate exponent (e.g. 1.3–1.4) and adjust the scaling factor so typical FP/G ranges map to reasonable values.
- Step 2: re-tune consolidation/expansion thresholds and realism caps so the engine still behaves sensibly.
- Step 3: either update the UI explanation to match the chosen exponent or continue stepping toward the documented 1.8 curve if it feels good for the league.

This should be done with test cases and concrete examples (Jokic, Luka, mid-tier stars, depth pieces) so that consolidation trades “feel” right.

### 2.2 League-driven realism caps (current work, further tuning)

We now have a league-aware realism system that:

- Uses `league_tiers['quality']` and `league_tiers['star']` to derive an FP/G gap (`fp_unit`).
- Uses `league_avg_cv` to adjust tolerance based on how swingy the league is.
- Uses `TRADE_BALANCE_LEVEL` (1–10 slider) to scale strictness around that baseline.

From these, we derive:

- `MAX_OPP_CORE_AVG_DROP` (master core FP/G drop cap).
- `EQUAL_COUNT_MAX_OPP_WEEKLY_LOSS` and `MAX_OPP_WEEKLY_LOSS` as weekly loss caps.
- `EQUAL_COUNT_MAX_AVG_FPTS_RATIO` and `EQUAL_COUNT_MAX_TOTAL_FPTS_RATIO` as equal-count FP/G ratio limits.

**Future tuning ideas:**

- Calibrate the multipliers (`0.3` of tier gap, strictness range `0.7–1.3`, etc.) by comparing against real league trades.
- Optionally surface these derived caps in a developer/debug section of the UI for transparency.
- Consider using more of the percentile lookup (e.g., percentile-based protections instead of just tier thresholds) inside `_is_realistic_trade`.

### 2.3 `_is_realistic_trade` decomposition

The realism function should be split into smaller, named checks:

- `_check_avg_fpts_ratio(...)`
- `_check_total_fpts_ratio(...)`
- `_check_consolidation_quality(...)`
- `_check_expansion_quality(...)`
- `_check_cv_tradeoff(...)`
- `_check_tier_protection_elite_star(...)`
- `_check_best_player_matchups(...)`

The top-level `_is_realistic_trade` then becomes a readable sequence of boolean guards instead of a 300+ line god function.

---

## 3. Performance Improvements

### 3.1 Replace `iterrows()` in hot loops

The pattern search functions currently use nested `iterrows()` inside combinatorial loops. This is slow.

**Planned change:**

- Convert DataFrames to lists once per team:
  - `your_rows = list(your_team.itertuples(index=False))` or `to_dict('records')`.
  - `their_rows = list(other_team.itertuples(index=False))`.
- Run `combinations()` over these lists instead of `iterrows()`.

This keeps the logic identical but reduces overhead per combination.

### 3.2 Smarter pruning / branch-and-bound (future)

Potential future improvements:

- Use simple upper bounds on possible core gain from a partial package to prune branches early.
- Run a “beam search” that focuses on top candidate players instead of full combinatorial enumeration within each pattern.

These are optional and should be introduced only if performance is still a bottleneck after cheaper wins.

---

## 4. UX & Presentation Improvements

### 4.1 Better progress feedback during analysis

Using the existing `estimate_trade_search_complexity` output:

- Show the estimated total combinations before search.
- Optionally show per-opponent counts.
- Use an `st.empty()` placeholder or progress bar to update status as each team is processed.

This makes long runs feel less opaque.

### 4.2 Group suggestions by opponent team

Instead of a flat list of expanders:

- Group `suggestions` by `team`.
- Optionally use `st.tabs` or sections per opponent.
- Within each tab/section, show that team’s best trades sorted by `value_gain`.

This aligns with how users think: “What can I offer Team X?”

---

## 5. Testing & Robustness

### 5.1 Unit tests for realism and value logic

Introduce `pytest` tests for:

- `calculate_exponential_value` across FP/G ranges.
- Core gain simulation and weekly scaling.
- Realism checks for:
  - obvious fleeces (should be rejected),
  - fair consolidations (should be accepted),
  - depth trades,
  - elite/star protections.

These tests are critical before large refactors (generic pattern finder, module split, exponent changes).

### 5.2 Structured suggestion objects

Longer-term, consider:

- Using `dataclasses` or `pydantic` models for the suggestion payload instead of raw dicts.
- This would make downstream usage safer (no typos in keys, better editor support) and easier to refactor.

---

## 6. Data Layer Future Work

This is lower priority than trade logic, but worth keeping in mind:

- Move away from scattered JSON files toward a more robust store (e.g. SQLite or DuckDB) for player and league data.
- Benefits:
  - Easier cross-player / cross-league queries.
  - More consistent schema.
  - Better performance for analytical queries.

Implementation would involve:

- A small data access layer that abstracts whether we read from JSON or DB.
- Gradual migration of loaders (e.g. `_build_fantasy_team_view`) to query the DB.

---

## 7. Integration with Existing Deep Dives

This doc should eventually tie into or be referenced from:

- `plan/deep_dive/902_League_Advanced_Strategy_and_Modeling.md` (advanced strategy & model rationale).
- Any future deep dives specifically about trade realism presets, social acceptability of trades, and balance tuning.

Older plan docs (`IMPLEMENTATION_SUMMARY.md`, `IMPROVEMENTS.md`, etc.) can be mined for still-relevant items and then retired once their contents are captured here and in the 90x-series deep dives.
