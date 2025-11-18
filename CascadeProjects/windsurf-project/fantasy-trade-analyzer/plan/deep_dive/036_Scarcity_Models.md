# Deep Dive: Advanced Scarcity Models

This document details the logic and strategic purpose of each scarcity model used in the Auction Draft Tool. These models dynamically adjust a player's value (`AdjValue`) during the draft based on changing market conditions.

All scarcity models work by calculating a `scarcity_premium`, which is a multiplier applied to a player's preliminary value. A premium > 1.0 increases the player's value, while a premium < 1.0 decreases it.

---

### 1. No Scarcity Adjustment

*   **Logic**: This is the baseline model. The `scarcity_premium` is always **1.0**.
*   **Purpose**: To provide an unadjusted view of player value, relying solely on the pre-draft `BaseValue` calculations. It serves as a control against which other models can be compared.

---

### 2. Tier-Based Scarcity

*   **Logic**: Calculates a premium based on the percentage of players already drafted from a specific tier. The premium increases as more players from that tier are taken.
    *   `tier_drafted_ratio = 1 - (current_players_in_tier / initial_players_in_tier)`
    *   `premium = 1 + (tier_drafted_ratio * 0.25)`
*   **Purpose**: To increase the value of the remaining players in a rapidly depleting tier. If all the "Tier 1" players are being drafted quickly, the value of the last few remaining "Tier 1" players goes up. The maximum premium is capped at 25%.

---

### 3. Positional Scarcity

*   **Logic**: Similar to Tier-Based Scarcity, but operates on player positions (G, F, C, etc.). The premium increases as more players from a specific position are drafted.
    *   `pos_drafted_ratio = 1 - (current_players_at_pos / initial_players_at_pos)`
    *   `premium = 1 + (pos_drafted_ratio * 0.30)`
*   **Purpose**: To increase the value of players at a scarce position. If there is a run on Centers, the value of the remaining Centers will rise. The premium is slightly higher (30%) than tier scarcity, reflecting the hard constraints of roster construction.

---

### 4. Combined Scarcity (Positional + Tier)

*   **Logic**: This model calculates both the `tier_premium` and the `pos_premium` and then takes a weighted average of the two (50/50).
    *   `premium = (tier_premium * 0.5) + (pos_premium * 0.5)`
*   **Purpose**: To provide a balanced view of scarcity that considers both a player's talent level (tier) and their role (position). This prevents overvaluing a low-tier player at a scarce position or a high-tier player at a plentiful one.

---

### 5. Contrarian Fade

*   **Logic**: This model *discounts* the value of players from tiers that are still plentiful. It identifies tiers with the highest remaining player counts and applies a small discount.
    *   It calculates a `tier_remaining_ratio` and normalizes it to create a discount factor.
    *   `premium = 1 - (normalized_ratio * 0.15)`
*   **Purpose**: To implement a contrarian strategy. If everyone is ignoring "Tier 3" players, this model suggests that you can get them at a slight discount, preventing you from overpaying for a player just because they belong to a certain tier. The maximum discount is 15%.

---

### 6. Roster Slot Demand

*   **Logic**: This model calculates the total number of unfilled roster spots for each position across all teams in the league. It then creates a premium based on the relative demand for each position.
    *   `demand[pos] = sum(required_pos - filled_pos)` for all teams.
    *   `demand_factor = demand[pos] / total_demand`
    *   `premium = 1 + (demand_factor * 0.35)`
*   **Purpose**: To precisely value players based on the league's collective need. If many teams still need to fill their "Guard" slots, the value of available Guards will increase significantly. This is a more direct measure of demand than `Positional Scarcity`. The max premium is 35%.

---

### 7. Opponent Budget Targeting

*   **Logic**: A strategic model that adjusts player value based on the financial state of the league.
    *   It calculates the league's `budget_depletion_ratio` (how much money has been spent).
    *   It defines "luxury" players (Tiers 1-2) and "value" players (Tiers 3-4).
    *   **Early in the draft** (budgets are full), it applies a premium to **luxury players** to encourage nominating them to drain opponent budgets.
    *   **Late in the draft** (budgets are low), it applies a premium to **value players**, identifying them as more likely to be secured at a good price.
*   **Purpose**: To provide nomination advice that is sensitive to the financial context of the draft. It helps the user decide whether to nominate a star player to force opponents to spend big or a mid-tier player who might be a bargain.
