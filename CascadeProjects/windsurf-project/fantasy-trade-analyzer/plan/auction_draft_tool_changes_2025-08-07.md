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

## Verification Steps
1. Start draft and draft a player.
2. Confirm: dropdown no longer shows the drafted player.
3. Confirm: player appears in the team roster and draft history.
4. Confirm: no Streamlit session_state mutation error occurs when clearing selection.
