# 902 – League Advanced Strategy & Modeling

_Deep math / modeling notes for how this league actually functions, used to guide tool design._

This document assumes familiarity with:

- `900_League_Constitution.md` (rules)
- `901_League_Understanding.md` (practical strategy)

Here we formalize a few concepts:

- Expected games and core size per week
- Effective team FP and FP/G under the min-games + penalty rules
- How to think about players as random variables (mean, variance, availability)
- How this should drive trade evaluation metrics and heuristics

---

## 1. Weekly Games & Core Size Model

### 1.1. Definitions

- `R` = roster size (total players on team)
- `G_avg` = average real NBA games per player per fantasy week
- `MinGames` = league minimum games requirement
- `CoreSize` = number of roster spots that effectively drive most of your counted games

Currently:

- `R = 10`
- `G_avg ≈ 3.5`
- `MinGames = 25` (subject to change by amendment)

### 1.2. Expected Total Games & Minimum

Theoretical maximum team games in a week:

- `Games_max = R × G_avg`
- With current values: `Games_max ≈ 10 × 3.5 = 35`.

The minimum is calibrated using a 71.4% utilization factor:

- `MinGames ≈ Games_max × 0.714`

This generalizes to any roster/MinGames configuration.

### 1.3. Core Size Approximation

We define **CoreSize** as the number of roster spots needed to generate MinGames worth of games if each core player hits G_avg games:

- `CoreSize ≈ MinGames ÷ G_avg`

Examples:

- `MinGames = 25` → `CoreSize ≈ 25 ÷ 3.5 ≈ 7.14` → 7–8 core players
- `MinGames = 30` → `CoreSize ≈ 30 ÷ 3.5 ≈ 8.57` → 8–9 core players
- `MinGames = 35` → `CoreSize = 35 ÷ 3.5 = 10` → all 10 players are core

Interpretation:

- At current settings, **~7–8 players drive most of your counted games**.
- The remaining slots are primarily **flex/streaming** levers.
- If MinGames increases, more of the roster becomes "core" and pure depth/availability gains value.

### 1.4. Historical League Variants

In past seasons this league has used different roster sizes (for example, 12-player rosters), different manager counts, and different lineup templates (such as 4G / 3F / 2C / 1 Flex) that implied a higher `MinGames` (often around 30 instead of 25).

The model above already covers these eras, because everything is parameterized by `R`, `G_avg`, and `MinGames`:

- Larger rosters or more starting spots increase `Games_max` and typically push `MinGames` higher.
- Higher `MinGames` increases `CoreSize`, so more of the roster becomes "core" and depth/availability are worth more.
- Smaller leagues or fewer managers generally mean a stronger waiver pool, which should be reflected in the assumed waiver FP/G baseline used later in the trade model.

---

## 2. Effective Team FP & FP/G Under League Rules

### 2.1. Baseline Model (No Penalties)

Let each player `i` be characterized by:

- `μ_i` = FP/G (mean fantasy points per game)
- `g_i` = expected games played in a given week

If there were **no minimums or penalties**, total expected fantasy points:

- `TeamFP_total = Σ_i (μ_i × g_i)`

Team FP/G (if we just divide by total games):

- `TeamPPG = TeamFP_total ÷ Σ_i g_i`

### 2.2. Incorporating MinGames & Overages

The league rules adjust outcomes based on MinGames:

- If `Games_total >= MinGames` and you do not exceed by much:
  - Winner is essentially the team with **higher TeamFP_total**, but structurally this is very close to **higher TeamPPG** times MinGames.
- If `Games_total < MinGames` and opponent is ≥ MinGames:
  - Auto-loss, regardless of efficiency.
- If `Games_total > MinGames`:
  - Penalty = `Over × TeamPPG`, where `Over = Games_total − MinGames`.

Net effective score when over:

- `Games_total = MinGames + Over`
- `TeamFP_total_raw = TeamPPG × (MinGames + Over)`
- `Penalty = Over × TeamPPG`
- `TeamFP_effective = TeamFP_total_raw − Penalty = TeamPPG × MinGames`

So, **once you cross MinGames**, your effective score is:

> `TeamFP_effective ≈ TeamPPG × MinGames`

independent of how many extra games you play.

This is the core reason the league behaves like a **TeamPPG efficiency contest** around a fixed MinGames.

### 2.3. Implications for Roster Construction

Given the above:

- Your main objective is to **maximize TeamPPG** while:
  - Avoiding `Games_total < MinGames` (auto-loss).
  - Avoiding relying on very low FP/G games to reach MinGames.
- Once you are reliably above MinGames:
  - Adding more low FP/G games does not increase `TeamFP_effective`.
  - It may **lower TeamPPG**, making the penalty formula worse.

**Floor risk warning**: The penalty math assumes your extra games are roughly at your team's average PPG. But a player with a floor of 25 FP/G can drag TeamPPG down faster than the model suggests. If your 9th or 10th man regularly posts 20-25 FP games, those games actively hurt you even when you're above MinGames—they dilute your PPG, which then becomes the penalty rate for *all* your overage games. The trade engine should weight the lowest end of your roster explicitly, not just the core.

Thus, in modeling terms:

- Core players are evaluated primarily by **μ_i (FP/G)** and **risk/variance**.
- Non-core slots are evaluated by **how efficiently they help you stay above MinGames** without killing TeamPPG.
- **Floor players** (roster spots 9-10) should be evaluated by their *downside* FP/G, not their average—a player who averages 45 but floors at 20 is more dangerous than one who averages 42 but floors at 35.

---

## 3. Player as Random Variable: Mean, Volatility, Availability

For modeling and tool design, treat each player as a random variable:

- `μ_i` = mean FP/G
- `σ_i` = standard deviation of FP/G (volatility)
- `CV_i = σ_i ÷ μ_i` = coefficient of variation
- `p_i` = probability player actually plays in a given scheduled game (availability/injury risk)

### 3.1. Effective Games & Availability

Let `g_sched_i` be the number of NBA games scheduled for player `i` that week, and `p_i` the chance he plays each:

- `E[g_i] = g_sched_i × p_i`

**Critical note**: `p_i` is not static. It drifts as:

- Roles shift (a player moving to the bench has different rest patterns than a starter)
- Rotations settle (early-season minutes volatility stabilizes by December)
- Nagging injuries grow stale (a "day-to-day" tag in week 1 means something different than the same tag in week 12)
- Load management patterns emerge (stars on playoff teams rest more in March)

Treating availability as a **weekly estimate** rather than a season-long average makes the risk metric more predictive. The tool should ideally re-estimate `p_i` each week based on recent news, not just historical GP%.

We can approximate `p_i` via:

- Historical games played rate (baseline)
- Current injury flags and IL usage (adjustment)
- Recent games played in last 2-3 weeks (trend)
- Manual league knowledge (override for edge cases)

High-level modeling:

- High `μ_i` and high `p_i` → cornerstone studs.
- High `μ_i` but low `p_i` → high-upside, high-risk pieces.
- Moderate `μ_i` and high `p_i` → volume stabilizers.

### 3.2. Volatility and Matchup Shape

- Lower `CV_i` → more stable weekly output.
- Higher `CV_i` → more boom/bust.

Given MinGames is fixed, volatility impacts:

- How often you spike above an opponent’s score.
- How often you crater below, even with good average FP/G.

For trade modeling, volatility can be folded in as:

- A mild discount for extremely high CV (boom/bust bench pieces).
- A small bonus for very low CV ("metronome" players) when your roster is otherwise volatile.

---

## 4. Conceptual Trade Evaluation Model

This section is not code, but a conceptual blueprint for the trade suggestion engine.

### 4.1. Team-Level Impact Rather Than Package Sums

Instead of purely summing per-player values of the trade packages, we should approximate **team-level impact**:

1. Estimate your roster **before** the trade.
2. Estimate your roster **after** the trade.
3. For each scenario, compute:
   - `CoreSize` based on current MinGames.
   - `CorePPG_before` / `CorePPG_after` from your top `CoreSize` players by μ_i.
   - A simple availability risk metric (e.g., expected total games vs MinGames).
4. Define trade value gain along lines of:

   - `ΔCorePPG = CorePPG_after − CorePPG_before`
   - `ΔCoreFP = ΔCorePPG × MinGames` (approximate weekly FP change)

Then adjust for:

- Change in availability risk (how close you are to falling under MinGames).
- Change in volatility profile.

### 4.2. Role of Availability/Durability in the Metric

Availability should:

- **Heavily penalize** trades that make it likely you fall below MinGames.
- **Mildly penalize** trades that increase the amount of risky streaming needed.
- **Reward** trades that keep or improve CorePPG while:
  - Increasing expected total games enough to comfortably clear MinGames.

But crucially:

- Availability is a **modifier around FP/G**, not the primary objective.
- Pure volume that manifests only as extra games **beyond MinGames** should have sharply diminished value.

### 4.3. Role of Streaming Capacity

Teams gain value from having:

- A strong core that generates high CorePPG.
- A few flexible slots where **smart, news-driven streaming** can exploit schedule and role changes.

Modeling this:

- We can treat flex/streamer slots as having a **baseline FP/G**, plus potential upside from good streaming.
- Trades that free up a flex slot by consolidating 2-for-1 into a stud:
  - Increase CorePPG.
  - Increase the number of slots where streaming skill can be applied.

Thus, trade suggestions can flag:

- "Consolidation + streaming upside" scenarios.
- "You are too thin; this trade forces you into desperate low-FP/G streams" scenarios.

---

## 5. Example Heuristics Derived from the Model

The following are high-level rules the tool can encode as heuristics:

1. **Core FP/G priority**
   - Prefer trades where `ΔCorePPG` is significantly positive (after accounting for availability).

2. **MinGames safety**
   - Avoid trades that drop expected total games so low that you need multiple low-FP/G streamers to hit MinGames.

3. **Avoid pure volume traps**
   - Two players with modest FP/G that simply add theoretical extra games beyond MinGames should be discounted relative to a single higher-FP/G player.

4. **Consolidate when you have depth**
   - If you comfortably exceed MinGames and have multiple mid-tier players on your bench, look for 2-for-1 or 3-for-1 trades into higher μ_i studs.

5. **Stabilize when you’re fragile**
   - If you are frequently scrambling to hit MinGames, a trade that slightly lowers CorePPG but significantly improves availability can be justified.

6. **Context-aware streaming**
   - Treat a freed roster spot as potential value: if your team is already managed aggressively (lots of schedule/rules knowledge), consolidation can be worth more than the model alone would show.

These ideas can be further tuned with historical league data once available.

---

## 6. Future Modeling Directions

Potential extensions:

- Incorporate real schedule data (back-to-backs, 3-in-4s) to estimate per-week `g_sched_i`.
- Estimate `p_i` (availability) using past seasons and injury tags, updated weekly.
- Simulate weekly outcomes under different trade scenarios to approximate win probability change, not just FP/G change.
- Build playoff-specific models where upcoming schedule and rest patterns are weighted more than season-long averages.

### 6.1. Schedule Cluster Analysis (Playoff Weeks)

A player's utility in playoff weeks depends heavily on **schedule shape**, not just total games. Consider:

- **Player A**: 4-3-4 split across three playoff weeks (11 games)
- **Player B**: 3-3-2 split across three playoff weeks (8 games)

Even if both players have identical FP/G, Player A provides ~37% more playoff volume. In an efficiency league, this matters because:

1. More games from high-FP/G players = more chances to hit MinGames with quality
2. Fewer forced streams in critical weeks
3. Schedule clustering (multiple games in a single week) is more valuable than spread-out games

The trade engine should eventually incorporate **projected playoff schedule density** as a modifier, especially for trades made after Week 12 when playoff seeding becomes clearer.

### 6.2. Distribution-Based Simulation (Monte Carlo)

The current model produces point estimates (`ΔCorePPG`). The natural evolution is to convert this into a **distribution**:

- Each player has `μ_i` (mean FP/G) and `σ_i` (standard deviation)
- Simulate 10,000+ weekly outcomes by sampling from each player's distribution
- Produce a **win probability shift** instead of a raw FP/G delta

This enables:

- Risk-adjusted trade grades ("This trade raises your ceiling but lowers your floor")
- Matchup-specific analysis ("Against Team X's high-variance roster, you want stability")
- Playoff survival probability modeling

The infrastructure for this exists once we have reliable `σ_i` estimates per player.

This doc should serve as the conceptual foundation for tuning the trade suggestions engine, risk ratings, and any future "league meta" insights.

---

## 7. Advanced Trade Evaluation: The Consolidation vs Deconstruction Framework

This section formalizes the nuanced decision-making process for N-for-M trades, particularly consolidation (trading up) and deconstruction (trading down).

### 7.1. The Core Question

Given a roster of 10 players where only the top ~7-8 contribute meaningfully to weekly scoring, how do we evaluate whether trading N players for M players improves championship odds?

The naive approach (summing package FP/G) fails because it ignores:
1. **Core displacement**: Only your top 7-8 players' FP/G matters
2. **Waiver wire replacement**: Open roster spots get filled from a barren waiver pool (~40 FP/G)
3. **Superstar premium**: Elite players have non-linear ceiling value
4. **Floor stability**: Bottom 2-3 players determine your flexibility buffer

### 7.2. The Mathematical Model

For any trade, calculate:

```
roster_before = current 10 players
roster_after = (roster_before - players_out) + players_in

# If consolidating (giving more than getting), fill with waiver
if len(players_in) < len(players_out):
    roster_after += [40 FP/G] * (len(players_out) - len(players_in))

# Core impact (primary driver)
core_fpg_before = avg(top_7(roster_before))
core_fpg_after = avg(top_7(roster_after))
core_delta = core_fpg_after - core_fpg_before

# Floor impact (secondary driver)
floor_fpg_before = avg(bottom_2(roster_before))
floor_fpg_after = avg(bottom_2(roster_after))
floor_delta = floor_fpg_after - floor_fpg_before

# Superstar premium (non-linear adjustment)
# Elite players (80+ FP/G) have ceiling value beyond their average
superstar_premium = calculate_ceiling_value(players_out, players_in)

# Final trade value
trade_value = (core_delta * 25) + (floor_delta * 10) + superstar_premium
```

### 7.3. The Break-Even Point for Consolidation

When trading a superstar for multiple players (deconstruction), the break-even point is:

```
minimum_return = superstar_fpg + worst_player_fpg
```

**Key framing**: This is a **displacement model**, not an additive rotation. You are not adding players to your roster—you are replacing specific slots. The superstar vacates a core slot; your worst player gets dropped entirely. The incoming players must fill *both* holes.

Roster math in fantasy confuses people because they think in terms of "I'm getting two players, so I'm adding value." No. You're *replacing* two slots that previously held (a) your best player and (b) your worst player. The incoming package must exceed the sum of what those slots were producing, or you've lost ground.

This represents the minimum combined FP/G needed to maintain roster value, since:
- You lose the superstar (core slot now empty)
- You drop your worst player to make room (floor slot now empty)
- The incoming players must replace both slots, not "add to" your roster

**Example**: Trading a 125 FP/G superstar when your worst player is 39 FP/G:
- Break-even: 125 + 39 = 164 FP/G
- Two 82 FP/G players = 164 FP/G (exact break-even)
- Two 75 FP/G players = 150 FP/G (loss of 14 FP/G, likely bad trade)
- Two 85 FP/G players = 170 FP/G (gain of 6 FP/G, likely good trade)

### 7.4. Dynamic Adjustments to Break-Even

The raw break-even is a starting point. Apply these adjustments:

**A. Superstar Premium (+5-10%)**
Elite players (80+ FP/G) have irreplaceable ceiling value for playoff weeks. Add 5-10% to the break-even when trading them away.

```
adjusted_breakeven = raw_breakeven * 1.05  # to 1.10
```

**B. Structural Desperation (-5-10%)**
If your roster has severe depth issues (3+ players below 45 FP/G), you may accept slightly below break-even to fix structural flaws.

```
if severe_depth_issues:
    adjusted_breakeven = raw_breakeven * 0.95
```

**C. Timing Context**
- **Early season (Weeks 1-10)**: Prioritize consistency and depth. Accept trades closer to raw break-even.
- **Late season (Weeks 15+)**: Prioritize ceiling. Demand premium above break-even.

### 7.5. The Reverse: When to Consolidate (Trade Up)

For teams with good depth but no superstar, consolidation is optimal when:

1. **You have roster redundancy**: 3+ players in the 55-65 FP/G range
2. **You're hitting 25 games easily**: Expected weekly games > 28
3. **You lack ceiling**: Your best player is below 75 FP/G

The consolidation creates value by:
- Upgrading a core spot (direct FP/G gain)
- Converting a rigid roster spot into a flexible streaming slot
- Simplifying daily lineup decisions (reducing error risk)

**Example**: Trading two 70 FP/G players for one 85 FP/G player:
- Package loss: 140 - 85 = 55 FP/G
- But you add a 40 FP/G waiver player
- Net roster loss: 55 - 40 = 15 FP/G
- Core gain: 85 replaces your 7th-best player (say, 60 FP/G) = +25 FP/G
- **Net impact: +10 FP/G to core, plus strategic flexibility**

### 7.6. Implementation in Trade Engine

The trade suggestion engine should:

1. **Calculate break-even dynamically** for each team's roster structure
2. **Apply contextual adjustments** based on roster health and season timing
3. **Flag trades as**:
   - "Elite" (>10% above break-even)
   - "Strong" (5-10% above break-even)
   - "Fair" (±5% of break-even)
   - "Marginal" (5-10% below break-even, only if structural benefit)
   - "Bad" (>10% below break-even)
4. **Explain the strategic value** beyond raw FP/G (ceiling, floor, flexibility)

This framework ensures trades are evaluated holistically, not just on package totals.
