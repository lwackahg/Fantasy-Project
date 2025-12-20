# Auction Draft Tool: Session State Fix & Removal of Drafted Players

## Summary
- Implemented a safe pattern to clear `player_on_the_block` selectbox without triggering Streamlit session_state mutation errors.
- Ensured drafted players are removed from `available_players` immediately after drafting, so they disappear from the dropdown and grids.
- Restored early return on insufficient budget.
- Added consistent entries to `st.session_state.drafted_players` used by summaries and recalculation.

## Files Changed
- `pages/2_Auction_Draft_Tool.py`

## Details
- Added `_reset_player_on_block` flag in session state during initialization to coordinate resets before widget instantiation.
- Updated `clear_player_on_block()` to set the flag instead of directly mutating `st.session_state['player_on_the_block']`.
- At the start of the active draft branch, processed the flag to set `player_on_the_block = None` before creating the selectbox.
- Called `st.rerun()` after a successful draft to ensure the reset takes effect before widgets render.
- On drafting:
  - Restored `return` on budget failure.
  - Appended a normalized record to `st.session_state.drafted_players` with keys: `PlayerName`, `Team`, `DraftPrice`, `Position`, `BaseValue`, `AdjValue`.
  - Filtered `st.session_state.available_players` to drop the drafted `PlayerName`.

## Rationale
- Streamlit prohibits changing a widget’s key-backed session state after the widget is created. Using a flag + early processing avoids exceptions.
- Removing the drafted player from `available_players` guarantees they won’t appear in `recalculated_df` or the selectbox.

## Additional Changes (2025-08-07 Evening)
- UI: Draft Price input now caps at the selected team’s remaining budget and displays that maximum inline. Default value is clamped into the allowed range.
- UX: "Detailed Model Values" expander is open by default to keep model transparency visible.
- Bot: Nomination target prices are now rounded up to whole dollars (ceiling) for cleaner bidding guidance. Added detailed explanation metadata per recommendation (weights used, component scores, and target math) for a "Why this target?" expander in the UI.
- Display: Nomination list shows whole-dollar "Worth" and "Target" values and includes a per-item explanation expander.

### 1-per-Spot Max Bid Rule (2025-08-08)
- Rule: Max bid = current team budget − $1 × remaining roster spots after this pick.
- Implemented in `logic/ui_components.py` (form input cap and caption) and `pages/2_Auction_Draft_Tool.py` (`drafting_callback` server-side check with friendly error).

### Same-Tier Scarcity (2025-08-08)
- Scarcity counts now use the player's OWN tier rather than hard-coding Tier 1.
- Bot details return keys: `position`, `same_tier_count`, `tier`, and `global_same_tier_total`.
- UI caption updated in `pages/2_Auction_Draft_Tool.py` to show: `Tier {tier} left: {same_tier_count}`.
- Explanations in `logic/smart_auction_bot.py` use tier-aware wording and avoid misleading "only 0" messages at draft start.
- Position parsing normalized to handle commas or slashes (e.g., `G,Flx` or `F/C`).
- Flex needs are considered when counting needy teams (teams can still bid via `Flx`).

## Verification Steps
1. Start draft and draft a player.
2. Confirm: dropdown no longer shows the drafted player.
3. Confirm: player appears in the team roster and draft history.
4. Confirm: no Streamlit session_state mutation error occurs when clearing selection.

## Change (2025-08-12): Avg Scarcity % Calculation
- Avg Scarcity % is now computed only from the scarcity premiums of the currently selected scarcity models (excluding "No Scarcity Adjustment").
- Applied in both the main "All Player Values" table and the drafted players' cached values expander to keep them consistent.
- Rationale: prevents mixing inactive or unselected models and keeps the metric aligned with the models the user chose.
- Files: `pages/2_Auction_Draft_Tool.py`
- Field: stored as `AvgScarcityPremiumPct` (raw percent value), labeled as "Scarcity %" in the UI.
