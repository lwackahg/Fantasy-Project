# 904 – Auction Draft Value Theory

_How to think about auction dollars, FP/G-to-price mappings, break-even bids, budget allocation, and steal/overpay detection in a min-games efficiency league._

This document assumes familiarity with:

- `900_League_Constitution.md` (rules)
- `901_League_Understanding.md` (practical strategy — especially the FP/G efficiency thesis)
- `902_League_Advanced_Strategy_and_Modeling.md` (team-level math, CoreSize, effective FP)

Where `901` explains _why_ FP/G is the primary currency and `902` formalizes it, this document asks: **Given that FP/G is king, how much should each unit of FP/G cost in auction dollars?**

---

## 1. The Central Idea

In a min-games efficiency league:

> `TeamFP_effective ≈ TeamPPG × MinGames`

Your weekly score is determined by **how good your games are**, not how many you play. The auction draft is where you purchase those games. Every dollar you spend either raises or lowers your team's eventual PPG.

The goal of auction theory in this format is to answer:

1. **What is a player's FP/G worth in dollars?**
2. **At what price does a player become an overpay (hurts the rest of your roster)?**
3. **At what price is a player a steal (you should always buy)?**
4. **How should you allocate $200 across your roster to maximize TeamPPG?**

---

## 2. The Player Share Formula

Every player contributes a measurable share of your team's weekly PPG. This is the foundation of all auction value math.

### 2.1. Definition

```
Player's Share of TeamPPG = (Player_FP/G × G_avg) / MinGames
```

Where:
- `G_avg` = average games per player per week (≈ 3.5)
- `MinGames` = league minimum games requirement

This tells you: **Of your team's total PPG, how much does this one player account for?**

### 2.2. Applied to Real Players (25/26 Season YTD)

Using `G_avg = 3.5`:

| Player | FP/G | Share @ 25 min | Share @ 30 min | Share Δ |
|---|---:|---:|---:|---:|
| Nikola Jokic | 112.6 | 15.76 | 13.14 | -2.62 |
| Luka Doncic | 98.4 | 13.77 | 11.48 | -2.30 |
| Victor Wembanyama | 88.5 | 12.39 | 10.33 | -2.06 |
| Shai Gilgeous-Alexander | 85.0 | 11.90 | 9.92 | -1.98 |
| Giannis Antetokounmpo | 82.9 | 11.61 | 9.67 | -1.94 |
| Jalen Johnson | 81.8 | 11.45 | 9.54 | -1.91 |
| Cade Cunningham | 80.5 | 11.27 | 9.39 | -1.88 |
| Tyrese Maxey | 80.1 | 11.21 | 9.35 | -1.87 |
| Kawhi Leonard | 77.6 | 10.86 | 9.05 | -1.81 |
| Jamal Murray | 75.0 | 10.50 | 8.75 | -1.75 |
| Donovan Mitchell | 74.9 | 10.49 | 8.74 | -1.75 |
| James Harden | 73.9 | 10.35 | 8.62 | -1.73 |
| Anthony Edwards | 71.9 | 10.07 | 8.39 | -1.68 |
| Josh Giddey | 71.9 | 10.06 | 8.39 | -1.67 |
| Alperen Sengun | 71.6 | 10.03 | 8.36 | -1.67 |
| KAT | 70.9 | 9.93 | 8.27 | -1.65 |
| Scottie Barnes | 69.0 | 9.66 | 8.05 | -1.61 |
| Kevin Durant | 69.0 | 9.66 | 8.05 | -1.61 |
| LeBron James | 68.5 | 9.59 | 7.99 | -1.60 |
| Jalen Brunson | 66.1 | 9.26 | 7.71 | -1.54 |
| Bam Adebayo | 66.6 | 9.32 | 7.77 | -1.55 |
| LaMelo Ball | 65.6 | 9.18 | 7.65 | -1.53 |
| Avg Starter (~65) | 65.0 | 9.10 | 7.58 | -1.52 |
| Avg Rotation (~55) | 55.0 | 7.70 | 6.42 | -1.28 |
| Avg Streamer (~45) | 45.0 | 6.30 | 5.25 | -1.05 |
| Waiver Floor (~40) | 40.0 | 5.60 | 4.67 | -0.93 |

### 2.3. Key Insight: The Share Dilution Effect

As MinGames increases, every player's share of TeamPPG decreases. But **elite players lose more absolute share** than average ones.

- Jokic loses 2.62 FP/G of share going from 25→30 min
- A 45 FP/G streamer loses only 1.05

This means: **The higher the minimum games, the less a single superstar can carry.** The gap between "Jokic + trash" and "balanced roster" widens as MinGames increases, because Jokic's share shrinks while the trash players' share stays relatively constant — they still drag down the average, they just do it over more counted games.

---

## 3. The FP/G Tier System and Auction Price Mapping

### 3.1. FP/G Tiers (Based on 25/26 YTD Data)

| Tier | FP/G Range | # of Players | Archetype | Draft $ Range (Historical) |
|---|---:|---:|---|---:|
| **S** | 100+ | 2 | Transcendent (Jokic, Luka) | $90-190 |
| **A** | 80-100 | 5-7 | Elite (Wemby, SGA, Giannis, JJ, Maxey, Cade) | $50-115 |
| **B** | 70-80 | 7-10 | Star (Murray, Mitchell, Harden, Kawhi, Ant, Sengun) | $25-65 |
| **C** | 60-70 | ~15 | Solid Starter (Barnes, KD, KAT, LaMelo, LeBron, Brunson, Bam) | $15-55 |
| **D** | 50-60 | ~25 | Rotation (D. White, Dyson, Fox, Pritchard, Bane, Vucevic) | $5-30 |
| **E** | 40-50 | ~30+ | Fringe/Streamer (DeMar, DiVincenzo, Gobert, Ayton) | $1-20 |
| **F** | <40 | deep | Waiver wire | $1 |

Note: Historical draft prices are pulled from the 25/26 auction. Prices vary wildly based on league hype, who's nominating, and draft energy. The ranges above capture realistic spreads, not outliers (e.g., Jokic at $190 was an anomaly; $115-130 is the normal ceiling).

### 3.2. The FP/G-to-Dollar Conversion

The fundamental question: **How many FP/G does $1 buy?**

Total league budget: `16 teams × $200 = $3,200`
Total roster spots: `16 × R` (where R = roster size)

If every dollar perfectly tracked FP/G, we could calculate:

```
Total League FP/G Pool = Σ (FP/G of all rostered players)
Dollar per FP/G = $3,200 / Total League FP/G Pool
```

But auction markets don't work this way. Elite players command a **non-linear premium** because:
- There's only one Jokic
- Multiple teams compete for scarce elite FP/G
- The marginal value of going from 55→85 FP/G in a roster spot is much higher than 40→55

### 3.3. Empirical Price-per-FP/G from 25/26 Draft

Using actual 25/26 draft results and end-of-season FP/G:

| Player | Draft $ | Actual FP/G | $/FP/G |
|---|---:|---:|---:|
| Jokic | $190 | 112.6 | $1.69 |
| Luka | $114 | 98.4 | $1.16 |
| Wemby | $111 | 88.5 | $1.25 |
| SGA | $96 | 85.0 | $1.13 |
| Giannis | $94 | 82.9 | $1.13 |
| Sabonis | $90 | (not top — poor value) | — |
| Sengun | $65 | 71.6 | $0.91 |
| AD | $65 | (injury-limited) | — |
| Cade | $59 | 80.5 | $0.73 |
| Ant Edwards | $58 | 71.9 | $0.81 |
| Harden | $56 | 73.9 | $0.76 |
| Curry | $55 | 70.9 (43 GP) | $0.78 |
| LeBron | $53 | 68.5 (60 GP) | $0.77 |
| Booker | $53 | 61.9 | $0.86 |
| Barnes | $50 | 69.0 | $0.72 |
| KD | $50 | 69.0 | $0.72 |
| Brunson | $45 | 66.1 | $0.68 |
| Giddey | $46 | 71.9 | $0.64 |
| Maxey | $45 | 80.1 | $0.56 |
| Bam | $44 | 66.6 | $0.66 |
| Mitchell | $41 | 74.9 | $0.55 |
| Jalen Johnson | $40 | 81.8 | $0.49 |
| Siakam | $40 | 61.4 | $0.65 |
| Murray | $40 | 75.0 | $0.53 |
| Fox | $39 | 57.6 | $0.68 |
| LaMelo | $39 | 65.6 | $0.59 |
| Jalen Williams | $36 | (mid) | — |
| Chet | $36 | 61.6 | $0.58 |
| DeMar | $30 | 47.9 | $0.63 |
| Derrick White | $26 | 57.8 | $0.45 |
| Josh Hart | $24 | 54.2 | $0.44 |
| Embiid | $22 | 70.1 (38 GP) | $0.31 |
| Dyson | $21 | 58.3 | $0.36 |
| Ingram | $18 | 55.4 | $0.33 |
| Cooper Flagg | $16 | 61.1 | $0.26 |
| Pritchard | $17 | 53.8 | $0.32 |

### 3.4. The $/FP/G Curve

The data reveals a clear pattern:

| $/FP/G | Price Range | What It Means |
|---:|---|---|
| **$1.50+** | $130+ | Severe overpay (Jokic $190 territory) |
| **$1.00-1.50** | $80-130 | Premium price — only justified for S-tier or bidding war |
| **$0.70-1.00** | $40-70 | Fair market — typical for A/B tier players |
| **$0.50-0.70** | $20-45 | Good value — solid starters at reasonable cost |
| **$0.30-0.50** | $10-25 | Strong value zone — mid-tier starters, breakout candidates |
| **<$0.30** | $1-15 | Steal territory — if FP/G holds, massive surplus value |

**The best draft outcomes come from buying as much FP/G as possible in the $0.30-0.70 range** and avoiding the $1.00+ range unless it's a true tier-S player at a "reasonable" premium.

---

## 4. Break-Even Bid Theory

This is the core analytical framework: **At what price does buying a specific player break even vs. the alternative use of those dollars?**

### 4.1. The Concept

When you spend $X on a player, you're not just buying that player. You're also **reducing the budget available for every other roster slot**. The break-even price is the point where the FP/G you gain from this player is exactly offset by the FP/G you lose from cheaper remaining picks.

### 4.2. The Formula

```
Break-Even Price = Budget - (Remaining_Slots × Cost_Per_Slot_to_Achieve_Target_PPG)
```

More precisely, for a target TeamPPG:

```
Target_TeamPPG = (Player_FP/G × G_avg + Remaining_FP/G_Total × G_avg_remaining) / MinGames
```

Solving for what the remaining roster needs to average:

```
Required_Remaining_FP/G = (Target_TeamPPG × MinGames - Player_FP/G × G_avg) / (Remaining_Games)
```

Where:
- `Remaining_Games = MinGames - G_avg` (games from the one player)
- Or more generally: `Remaining_Games = (R - 1) × G_avg` scaled to MinGames

Then:

```
Cost_to_Fill_Remaining = Required_Remaining_FP/G mapped to $/FP/G curve × (R - 1) slots
Break_Even_Price = $200 - Cost_to_Fill_Remaining
```

### 4.3. Break-Even Bids for Key Players

We define the **target TeamPPG** as the "competitive threshold" — the average PPG of a well-built balanced team.

#### At 25-Game Minimum (10-man roster, competitive TeamPPG ≈ 61)

For each player, we ask: "If I buy this player, what do my other 9 guys need to average, and what does that cost?"

```
Required_Remaining_FP/G = (61.0 × 25 - Player_FP/G × 3.5) / 21.5
```

| Player | FP/G | Required Rest FP/G | $/player for Rest | Total Rest Cost | **Break-Even Bid** |
|---|---:|---:|---:|---:|---:|
| **Jokic** | 112.6 | 52.6 | ~$13 | ~$117 | **~$83** |
| **Luka** | 98.4 | 54.9 | ~$14 | ~$126 | **~$74** |
| **Wemby** | 88.5 | 56.5 | ~$15 | ~$135 | **~$65** |
| **SGA** | 85.0 | 57.1 | ~$16 | ~$144 | **~$56** |
| **Giannis** | 82.9 | 57.4 | ~$16 | ~$144 | **~$56** |
| **Maxey** | 80.1 | 57.9 | ~$16 | ~$144 | **~$56** |
| **Murray** | 75.0 | 58.7 | ~$17 | ~$153 | **~$47** |
| **D. Mitchell** | 74.9 | 58.7 | ~$17 | ~$153 | **~$47** |
| **Harden** | 73.9 | 58.9 | ~$17 | ~$153 | **~$47** |
| **Sengun** | 71.6 | 59.3 | ~$17 | ~$153 | **~$47** |
| **KAT** | 70.9 | 59.4 | ~$18 | ~$162 | **~$38** |
| **Barnes** | 69.0 | 59.7 | ~$18 | ~$162 | **~$38** |
| **KD** | 69.0 | 59.7 | ~$18 | ~$162 | **~$38** |
| **LeBron** | 68.5 | 59.8 | ~$18 | ~$162 | **~$38** |
| **Brunson** | 66.1 | 60.2 | ~$18 | ~$162 | **~$38** |

#### At 30-Game Minimum (11-man roster, competitive TeamPPG ≈ 57.5)

```
Required_Remaining_FP/G = (57.5 × 30 - Player_FP/G × 3.5) / 26.5
```

| Player | FP/G | Required Rest FP/G | $/player for Rest | Total Rest Cost | **Break-Even Bid** |
|---|---:|---:|---:|---:|---:|
| **Jokic** | 112.6 | 50.2 | ~$11 | ~$110 | **~$90** |
| **Luka** | 98.4 | 52.1 | ~$12 | ~$120 | **~$80** |
| **Wemby** | 88.5 | 53.4 | ~$13 | ~$130 | **~$70** |
| **SGA** | 85.0 | 53.9 | ~$13 | ~$130 | **~$70** |
| **Giannis** | 82.9 | 54.2 | ~$13 | ~$130 | **~$70** |
| **Maxey** | 80.1 | 54.5 | ~$14 | ~$140 | **~$60** |
| **Murray** | 75.0 | 55.2 | ~$14 | ~$140 | **~$60** |
| **D. Mitchell** | 74.9 | 55.2 | ~$14 | ~$140 | **~$60** |
| **Harden** | 73.9 | 55.4 | ~$14 | ~$140 | **~$60** |
| **Sengun** | 71.6 | 55.7 | ~$15 | ~$150 | **~$50** |
| **KAT** | 70.9 | 55.8 | ~$15 | ~$150 | **~$50** |
| **Barnes** | 69.0 | 56.0 | ~$15 | ~$150 | **~$50** |
| **KD** | 69.0 | 56.0 | ~$15 | ~$150 | **~$50** |
| **LeBron** | 68.5 | 56.1 | ~$15 | ~$150 | **~$50** |
| **Brunson** | 66.1 | 56.4 | ~$16 | ~$160 | **~$40** |

### 4.4. Interpreting the Break-Even Table

**Jokic at $130 (realistic market price):**
- Break-even at 25-min: ~$83. Overpay by **~$47**.
- Break-even at 30-min: ~$90. Overpay by **~$40**.
- That $40-47 overpay directly reduces your remaining roster's average FP/G by ~$4-5 per player, which translates to roughly 3-4 FP/G per remaining slot. Across 9-10 slots, that's a **team-level PPG loss of ~2-3 FP/G**.

**Jalen Johnson at $40 (actual draft price):**
- 81.8 FP/G at $40 → $/FP/G = $0.49
- Break-even at 25-min: ~$56. **$16 under break-even = steal.**
- Break-even at 30-min: ~$65. **$25 under break-even = massive steal.**

**Maxey at $45 (actual draft price):**
- 80.1 FP/G at $45 → $/FP/G = $0.56
- Break-even at 25-min: ~$56. **$11 under = solid value.**
- Break-even at 30-min: ~$60. **$15 under = great value.**

**Sabonis at $90 (actual draft price):**
- Actual FP/G: unclear from top data, likely ~62-65
- Break-even for a 63 FP/G player at 25-min: ~$38
- **Overpaid by ~$52.** This is the type of pick that quietly destroys a roster.

---

## 5. Budget Allocation Strategies

### 5.1. The Three Archetypes

Based on the math above, there are three fundamentally different ways to allocate $200:

#### Stars & Scrubs (Top-Heavy)

Spend big on 1-2 elite players, fill the rest with minimum bids.

```
Example (25-min, 10-man):
  Star 1:     $85  → ~90 FP/G
  Star 2:     $50  → ~75 FP/G
  Remaining:  $65 / 8 = ~$8/player → ~48-52 FP/G avg

  TeamPPG ≈ ((90 + 75) × 3.5 + 8 × 50 × 3.5) / 25
          ≈ (577.5 + 1,400) / 25
          ≈ 79.1 ... wait, let me be precise:

  Total FP across 25 games:
    Star 1: 90 × 3.5 = 315
    Star 2: 75 × 3.5 = 262.5
    8 others: avg 50 × 3.5 × (games used to hit 25) 
    
  Simplified: TeamPPG = weighted avg of all players' FP/G
    = (90 + 75 + 50×8) / 10 = 56.5
  TeamFP_effective = 56.5 × 25 = 1,413
```

#### Balanced Build (Spread)

Distribute budget more evenly across all slots.

```
Example (25-min, 10-man):
  Top pick:   $45  → ~75 FP/G
  Pick 2:     $35  → ~70 FP/G
  Pick 3:     $28  → ~65 FP/G
  Pick 4-5:   $22 each → ~60 FP/G
  Pick 6-7:   $15 each → ~55 FP/G
  Pick 8-10:  $6 each  → ~48 FP/G
  Total: $200

  TeamPPG = (75 + 70 + 65 + 60 + 60 + 55 + 55 + 48 + 48 + 48) / 10 = 58.4
  TeamFP_effective = 58.4 × 25 = 1,460
```

#### Mid-Heavy (Value Zone)

Target the $0.30-0.70 $/FP/G sweet spot. Load up on 4-5 mid-tier players who return the most FP/G per dollar.

```
Example (25-min, 10-man):
  Pick 1:     $55  → ~80 FP/G (A-tier value buy)
  Pick 2:     $40  → ~75 FP/G (B-tier value)
  Pick 3:     $30  → ~68 FP/G
  Pick 4:     $25  → ~64 FP/G
  Pick 5:     $18  → ~58 FP/G
  Pick 6:     $14  → ~55 FP/G
  Pick 7:     $10  → ~52 FP/G
  Pick 8-10:  $8 total → ~44 FP/G
  Total: $200

  TeamPPG = (80 + 75 + 68 + 64 + 58 + 55 + 52 + 44 + 44 + 44) / 10 = 58.4
  TeamFP_effective = 58.4 × 25 = 1,460
```

### 5.2. Comparison of Archetypes

| Archetype | Team PPG (25-min) | Team PPG (30-min) | Ceiling | Floor | Best When |
|---|---:|---:|---|---|---|
| Stars & Scrubs | ~56-57 | ~52-53 | High (star booms) | Low (star misses game) | Low MinGames, deep waiver wire |
| Balanced | ~58-59 | ~56-57 | Moderate | Moderate-High | Higher MinGames, thin waivers |
| Mid-Heavy | ~58-60 | ~56-58 | Moderate-High | Moderate-High | You find 2-3 value picks in the $/FP/G sweet spot |

**Key finding: Stars & Scrubs is the weakest archetype in this league format**, especially as MinGames increases. Balanced and Mid-Heavy converge in output but Mid-Heavy has slightly more upside because it targets the value sweet spot aggressively.

### 5.3. The Optimal Budget Curve

Based on the break-even analysis, the ideal allocation follows this rough curve:

| Pick # | Budget % | Dollar Range | Target FP/G | Target $/FP/G |
|---:|---:|---:|---:|---:|
| 1 | 22-28% | $44-56 | 75-85 | $0.55-0.70 |
| 2 | 17-23% | $34-46 | 68-78 | $0.50-0.60 |
| 3 | 12-17% | $24-34 | 62-70 | $0.38-0.50 |
| 4 | 8-13% | $16-26 | 58-65 | $0.30-0.42 |
| 5 | 6-10% | $12-20 | 54-60 | $0.22-0.35 |
| 6 | 4-7% | $8-14 | 50-56 | $0.16-0.25 |
| 7 | 2-5% | $4-10 | 46-52 | $0.10-0.20 |
| 8-R | 1-3% each | $1-6 | 40-48 | $0.03-0.15 |

This curve keeps your $/FP/G in the efficient range across all picks. It naturally produces a Balanced or Mid-Heavy build.

---

## 6. Steal and Overpay Detection

### 6.1. The Framework

During a live auction, you need fast heuristics to decide: **Is this player going too cheap? Too expensive? Should I bid?**

Define:

```
Surplus Value = Break_Even_Price - Current_Bid
```

- `Surplus > 0` → the player is **below break-even** (potential steal)
- `Surplus < 0` → the player is **above break-even** (overpay territory)
- `Surplus ≈ 0` → fair price

### 6.2. Steal Thresholds

| Surplus | Signal | Action |
|---:|---|---|
| **+$20 or more** | Major steal | Buy aggressively — this player returns far more FP/G per dollar than any alternative use of the money |
| **+$10 to +$19** | Solid value | Strong buy if the player fills a roster need |
| **+$1 to +$9** | Fair-to-good | Buy if you need the position; don't chase if you don't |
| **$0 to -$9** | Slight overpay | Acceptable only for scarce positions or if you have excess budget |
| **-$10 to -$19** | Overpay | Let someone else have them unless it's your last core slot |
| **-$20 or worse** | Severe overpay | Never. The budget damage to your remaining picks isn't recoverable |

### 6.3. Real Examples from 25/26 Draft

Using 25-min break-evens:

| Player | Draft $ | Break-Even | Surplus | Verdict |
|---|---:|---:|---:|---|
| **Jalen Johnson** | $40 | ~$56 | **+$16** | Solid steal |
| **Maxey** | $45 | ~$56 | **+$11** | Good value |
| **Donovan Mitchell** | $41 | ~$47 | **+$6** | Fair-to-good |
| **Murray** | $40 | ~$47 | **+$7** | Fair-to-good |
| **Cade Cunningham** | $59 | ~$56 | **-$3** | Slight overpay (acceptable) |
| **Sengun** | $65 | ~$47 | **-$18** | Overpay |
| **Sabonis** | $90 | ~$38 | **-$52** | Severe overpay |
| **Jokic** | $190 | ~$83 | **-$107** | Catastrophic overpay |
| **Luka** | $114 | ~$74 | **-$40** | Severe overpay |
| **Wemby** | $111 | ~$65 | **-$46** | Severe overpay |
| **SGA** | $96 | ~$56 | **-$40** | Severe overpay |
| **Giannis** | $94 | ~$56 | **-$38** | Severe overpay |
| **Embiid** | $22 | ~$56 (if healthy, 70.1 FP/G) | **+$34** | Major steal (injury gamble) |
| **Cooper Flagg** | $16 | ~$38 (61.1 FP/G) | **+$22** | Major steal |
| **Dyson Daniels** | $21 | ~$38 (58.3 FP/G) | **+$17** | Solid steal |
| **Derrick White** | $26 | ~$38 (57.8 FP/G) | **+$12** | Good value |

### 6.4. Pattern Recognition

The draft data reveals a structural pattern in your league's auction:

**Overpay zone: Top 6 picks.** The first 6 players drafted (Trae $50, Jokic $190, Luka $114, Wemby $111, SGA $96, Giannis $94) all went for prices significantly above break-even. Combined overpay: **~$280+** across 6 managers.

**Steal zone: Picks 20-50.** The mid-draft is where the best $/FP/G ratios live. Jalen Johnson ($40, 81.8 FP/G), Maxey ($45, 80.1 FP/G), Murray ($40, 75.0 FP/G), Flagg ($16, 61.1 FP/G), Dyson ($21, 58.3 FP/G). These picks funded championship-caliber cores.

**The implication: The winning auction strategy is to be disciplined in the first 20 picks (let others overpay) and aggressive in picks 20-60 (where the value lives).**

---

## 7. The Minimum Bid Floor — "This Guy Is Going Too Cheap"

### 7.1. Absolute Floor Price

Every player who will return positive FP/G has a floor price below which they are an automatic buy. This is the price at which their $/FP/G is so low that **any alternative use of $1 is worse**.

For a $1 player to be worth it, they need:

```
FP/G > Waiver_Wire_Replacement_FP/G
```

In a 16-team league with 160-176 rostered players, the waiver wire floor is roughly 35-40 FP/G. So any player above ~42-45 FP/G at $1 is already a steal. The question is: how high can they go before you should stop?

### 7.2. The "Must Buy" Price for Each FP/G Level

Using the $/FP/G efficient zone ($0.30-0.50):

| FP/G | Must Buy Under | Good Value Under | Fair Under | Overpay Above |
|---:|---:|---:|---:|---:|
| 110+ | $55 | $85 | $100 | $110+ |
| 90-110 | $45 | $70 | $85 | $95+ |
| 80-90 | $40 | $56 | $65 | $75+ |
| 70-80 | $28 | $47 | $55 | $60+ |
| 60-70 | $20 | $38 | $45 | $50+ |
| 50-60 | $12 | $25 | $32 | $38+ |
| 40-50 | $5 | $15 | $20 | $25+ |

### 7.3. Live Draft Heuristic

During the draft, when a player is nominated and bidding is happening:

1. **Look up their projected FP/G** (or your best estimate).
2. **Check the "Must Buy Under" column.** If the current bid is below this, bid immediately — you are getting surplus value that directly improves your TeamPPG.
3. **Check "Overpay Above."** If bidding crosses this threshold, stop. The marginal FP/G you gain doesn't justify the budget damage.
4. **Between "Good Value" and "Fair"**: Bid if the player fills a positional need or you have budget surplus. Don't chase if you're tight on budget.

---

## 8. How MinGames Changes Draft Strategy

### 8.1. The Structural Shift from 25→30

| Dimension | 25-min (10-man) | 30-min (11-man) |
|---|---|---|
| CoreSize | 7-8 | 8-9 |
| Flex/Stream slots | 2-3 | 2-3 |
| Budget per player (even split) | $20 | $18.18 |
| Waiver wire depth | ~290 available | ~250 available |
| FP/G of waiver floor | ~40 | ~37-38 |
| Tolerance for overpaying on stars | Moderate | Low |
| Value of mid-tier depth | High | Very high |
| Streaming skill premium | Moderate | Moderate (more IR absorbs injury pressure) |

### 8.2. Draft Strategy Adjustments at 30-Min

1. **Lower your ceiling on pick 1.** At 25-min, spending $55 on pick 1 is fine (28% of budget). At 30-min with 11 slots, $55 is 27.5% but you have one more mouth to feed. Target $44-50 for pick 1.

2. **Flatten your curve.** Instead of 1 big pick + steep drop-off, aim for 3-4 players in the $25-45 range. You need 9 core contributors, not 7.

3. **The $10-25 range becomes critical.** At 30-min, picks 5-8 can't be throwaway $1-5 bids. You need real 50-58 FP/G players in those slots, which costs $10-25 each.

4. **IR slots change the calculus.** With 3 IR, you can stash injured stars who went cheap (like Embiid at $22 for 70.1 FP/G). The 3-day anti-stash rule prevents abuse, but legitimate injury stashes are more viable and should be budgeted for.

5. **$1 bids still matter.** Even at 30-min, your 10th and 11th picks can be $1-3 players. But you need them to be 42-45 FP/G, not 30-35 FP/G. Scout the end-of-draft tier more carefully.

### 8.3. Budget Template: 30-Min / 11-Man

| Pick # | Target $ | Target FP/G | $/FP/G |
|---:|---:|---:|---:|
| 1 | $44-50 | 75-82 | $0.55-0.65 |
| 2 | $34-42 | 68-76 | $0.50-0.56 |
| 3 | $26-34 | 63-70 | $0.40-0.50 |
| 4 | $20-28 | 58-65 | $0.34-0.44 |
| 5 | $15-22 | 55-60 | $0.28-0.38 |
| 6 | $12-18 | 52-57 | $0.22-0.32 |
| 7 | $8-14 | 48-54 | $0.17-0.26 |
| 8 | $5-10 | 45-50 | $0.11-0.20 |
| 9 | $3-7 | 42-48 | $0.07-0.15 |
| 10-11 | $1-4 each | 40-45 | $0.03-0.10 |
| **Total** | **$200** | **TeamPPG ~57-59** | |

---

## 9. Advanced: The Jokic Problem Formalized

### 9.1. The Question

Jokic is the best fantasy player by a wide margin (112.6 FP/G, 15+ FP/G above #2). In a normal league, he's the obvious #1 pick. In a min-games efficiency league with auction format, is he actually the best $1 you can spend?

### 9.2. Jokic's Team-Level Impact Model

```
Let J = Jokic's FP/G = 112.6
Let P = Price paid for Jokic
Let R = Roster size
Let B = Total budget = $200
Let MinG = Minimum games
Let W = Average FP/G achievable with remaining budget per slot

Remaining budget: B - P = 200 - P
Remaining slots: R - 1
Budget per remaining slot: (200 - P) / (R - 1)

W = f(budget_per_slot)  — this is the FP/G you can buy at a given $ level

TeamPPG(with Jokic) = (J × 3.5 + (R-1) × W × 3.5) / MinG
                     = 3.5/MinG × (J + (R-1) × W)
                     = 3.5/MinG × (J + (R-1) × f((200-P)/(R-1)))
```

The alternative (no Jokic, spread budget):

```
TeamPPG(balanced) = 3.5/MinG × R × f(200/R)
```

Jokic is worth buying when:

```
TeamPPG(with Jokic) > TeamPPG(balanced)
J + (R-1) × f((200-P)/(R-1)) > R × f(200/R)
```

### 9.3. Solving for Jokic's Max Price

Using the empirical $/FP/G curve from Section 3.3, we can approximate `f(budget)`:

For a rough model based on observed data:

```
f(x) ≈ 38 + 2.1 × x^0.55     (for x = dollars per player slot)
```

This gives roughly:
- f($1) ≈ 40 FP/G (waiver floor)
- f($7) ≈ 44 FP/G
- f($10) ≈ 45 FP/G
- f($15) ≈ 47 FP/G
- f($20) ≈ 58 FP/G
- f($30) ≈ 65 FP/G
- f($50) ≈ 75 FP/G

(These are approximations — the real curve is noisy because auction prices depend on market dynamics, not just player quality.)

**At 25-min (R=10):**

```
Balanced: TeamPPG = 3.5/25 × 10 × f($20)
        = 0.14 × 10 × 58 = 81.2? 
```

Wait — this is TeamPPG in the "share" sense. Let me use the direct formula:

```
TeamPPG = Σ(player_i FP/G) / R

Balanced at $20/player avg:
  TeamPPG = f($20) = ~58 FP/G avg across 10 players
  (range: best pick maybe 68, worst maybe 48)
  Effective: 58 × 25 = 1,450

Jokic at $130:
  Remaining: $70/9 = $7.78/player
  f($7.78) ≈ 48.5 FP/G
  TeamPPG = (112.6 + 9 × 48.5) / 10 = (112.6 + 436.5) / 10 = 54.9
  Effective: 54.9 × 25 = 1,373

  Gap: -77 points per week. Jokic at $130 LOSES.

Jokic at $83 (break-even):
  Remaining: $117/9 = $13/player
  f($13) ≈ 52.8 FP/G
  TeamPPG = (112.6 + 9 × 52.8) / 10 = (112.6 + 475.2) / 10 = 58.8
  Effective: 58.8 × 25 = 1,470

  This roughly matches balanced (1,450-1,470). Break-even confirmed at ~$83.
```

**At 30-min (R=11):**

```
Balanced at $18.18/player avg:
  f($18.18) ≈ 56.5 FP/G
  TeamPPG = 56.5
  Effective: 56.5 × 30 = 1,695

Jokic at $130:
  Remaining: $70/10 = $7/player
  f($7) ≈ 47.5 FP/G
  TeamPPG = (112.6 + 10 × 47.5) / 11 = (112.6 + 475) / 11 = 53.4
  Effective: 53.4 × 30 = 1,602

  Gap: -93 points per week. Even worse.

Jokic at $90 (break-even at 30-min):
  Remaining: $110/10 = $11/player
  f($11) ≈ 51.5 FP/G
  TeamPPG = (112.6 + 10 × 51.5) / 11 = (112.6 + 515) / 11 = 57.1
  Effective: 57.1 × 30 = 1,713

  Close to balanced (1,695). Break-even confirmed at ~$90.
```

### 9.4. Summary: Jokic Max Bids

| MinGames | Roster | Jokic Break-Even | Typical Market Price | Overpay |
|---:|---:|---:|---:|---:|
| 25 | 10 | ~$83 | $115-130 | $32-47 |
| 30 | 11 | ~$90 | $115-130 | $25-40 |

**Jokic is only "worth it" if you can get him for roughly $83-90.** At $130, you're sacrificing ~77-93 effective fantasy points per week — the equivalent of replacing a 55 FP/G player with a 45 FP/G player in **two** roster slots.

The 30-min format helps slightly (break-even rises to $90 because you need more total games and Jokic provides reliable ones), but the gap between market price and break-even is still massive.

### 9.5. When Jokic IS Worth Overpaying

The break-even model assumes average conditions. Jokic may be worth a premium above break-even if:

1. **Playoff schedule favors him.** If Denver has a 4-4-4 playoff schedule and your other options have 3-2-3, Jokic's games are worth more in the weeks that matter most.

2. **You're the best in-season manager.** If you can reliably stream 50+ FP/G from your flex slots (instead of the 42-48 the model assumes), your remaining roster doesn't get dragged down as much. The break-even rises by ~$5-10 for elite managers.

3. **The rest of the league overpays worse.** If five other managers spend $90+ on their top pick and gut their rosters, the bar for "competitive TeamPPG" drops, making your Jokic team viable by comparison.

4. **You pair Jokic with one extreme value pick.** If you get Jokic at $130 but also get a Jalen Johnson ($40, 81.8 FP/G) or Flagg ($16, 61.1 FP/G), the surplus from the value pick partially offsets the Jokic overpay.

---

## 10. Connecting to the Tool

This theory should inform the Auction Draft Tool (`160_Feature_Deep_Dive_-_Auction_Draft_Tool.md`) in several ways:

1. **Break-even bid column.** The tool should display a dynamic break-even price for every player based on your current budget, remaining slots, and target TeamPPG.

2. **Surplus value indicator.** Show `Surplus = Break_Even - AdjValue` or `Break_Even - Current_Bid` so you can instantly see steals and overpays.

3. **Budget curve tracking.** Overlay your actual spending curve against the optimal template (Section 5.3 / 8.3) so you can see if you're front-loading too much or leaving value on the table.

4. **"Must Buy" alerts.** When a player's current bid drops below the "Must Buy" threshold for their FP/G tier, flag it in the UI.

5. **Jokic Problem warning.** When your top pick's price exceeds break-even by >$20, show a warning with the projected impact on your remaining roster's average FP/G.

---

## 11. Variance, Floor Games, and the Option Value of Volatile Players

### 11.1. The Same Average, Different Value

Consider two players with identical 50 FP/G averages:

| Game | Player A (Steady) | Player B (Volatile) |
|---:|---:|---:|
| 1 | 50 | 70 |
| 2 | 50 | 30 |
| 3 | 50 | 70 |
| 4 | 50 | 30 |
| **Average** | **50** | **50** |
| **σ (Std Dev)** | **0** | **20** |

In a traditional league where every game counts, these players are interchangeable. In a **min-games efficiency league, they are fundamentally different assets.**

### 11.2. Why Variance Matters Here

The min-games + overage-penalty structure means you are **choosing which games count**. You set lineups daily. This gives you **option value** on volatile players that doesn't exist for steady ones.

With Player B (70/30 pattern), you can:
- **Play the 70 FP games** → they raise your TeamPPG
- **Bench the 30 FP games** → they never drag you down
- Effective FP/G when managed optimally: **closer to 70 than 50**

With Player A (50/50 pattern), every game is the same:
- Play them or don't — you always get 50
- No upside to capture, no downside to avoid
- Effective FP/G: **exactly 50, always**

**The volatile player has higher "managed FP/G"** — the FP/G you actually realize when you have the luxury of choosing which games to start them.

### 11.3. Managed FP/G vs Raw FP/G

Define:

```
Managed FP/G = Average FP of the games you actually START this player
Raw FP/G = Average FP across ALL games played (what the stat sheet shows)
```

For steady players: `Managed FP/G ≈ Raw FP/G`
For volatile players: `Managed FP/G > Raw FP/G` (because you bench the bad games)

The gap depends on:
1. **How volatile the player is** (higher σ = more benchable bad games)
2. **How many games you have to spare** (if you're at exactly MinGames, you can't bench anyone)
3. **How predictable the variance is** (matchup-driven vs random)

### 11.4. The "Need One More Game" Question

When you need exactly 1 more game to hit MinGames, who do you play?

| Situation | Play the Steady (50/50) | Play the Volatile (70/30) |
|---|---|---|
| **You need to hit MinGames to avoid auto-loss** | ✅ Guaranteed 50. Safe floor. | ❌ Could be 30. Risky when survival is at stake. |
| **You're already above MinGames, extra game is "bonus"** | Neutral — 50 is 50 | ✅ If it's a boom game, great. If bust, bench it next time. |
| **Must-win week, need ceiling** | ❌ 50 doesn't move the needle | ✅ The 70 might win you the week |
| **You're way over MinGames (28+ games in a 25-min league)** | ❌ 50 FP dilutes your PPG if team avg is >50 | ❌ Even the 70 might dilute if your team avg is >70 |

**Key insight: The steady player is a floor asset. The volatile player is a ceiling asset with option value.** Which one is more valuable depends entirely on your **games margin** — how far above MinGames you are.

### 11.5. Games Margin and Player Selection

Define:

```
Games Margin = Expected_Team_Games - MinGames
```

| Games Margin | What It Means | Preferred Player Profile |
|---:|---|---|
| **0 or negative** | Scrambling to hit minimum | Steady players. Every game must be playable. You can't afford a 30 FP bust. |
| **1-3** | Tight — slight cushion | Mix. Start your best FP/G players. Steady guys fill the last slots safely. |
| **4-6** | Comfortable | Volatile players become valuable. You can bench their bad games and cherry-pick booms. |
| **7+** | Excess games | High-variance players are ideal. You're benching someone anyway — pick the best games from the volatile guys and sit the rest. |

### 11.6. Practical Impact on Auction Value

This changes how you should price volatile players at auction:

**Steady 50 FP/G player:**
- What you see is what you get
- Worth ~$15-25 at 25-min, ~$16-28 at 30-min (per cheat sheet)
- Reliable floor piece, never exciting

**Volatile 50 FP/G player (σ = 20, booms to 70, busts to 30):**
- If your roster has Games Margin ≥ 4, their **managed FP/G ≈ 58-62**
- That bumps them from the 45-55 tier into the 55-65 tier in effective value
- Worth **$5-10 more** than their raw FP/G suggests
- If your roster is tight on games (margin ≤ 2), they're worth the same or slightly less than the steady player

**Draft heuristic:**
- **If you're building a deep, game-secure roster** (good Games Margin): target volatile players in the mid-to-late rounds. Their auction price reflects raw FP/G, but their managed FP/G is higher. This is free value.
- **If your roster is fragile** (injuries, thin depth, margin ≤ 2): pay the premium for steady players. Floor matters more than ceiling when you can't afford to bench games.

### 11.7. Variance by Archetype — Real Examples

Using the 25/26 season data as reference:

| Archetype | Example | Raw FP/G | Estimated σ | Managed FP/G (margin 4+) | Value Gap |
|---|---|---:|---:|---:|---:|
| **Steady Star** | Scottie Barnes | 69.0 | ~10 | ~71 | Small (+2) |
| **Volatile Star** | Anthony Edwards | 71.9 | ~22 | ~80+ | Large (+8) |
| **Steady Mid** | Bam Adebayo | 66.6 | ~12 | ~69 | Small (+2) |
| **Volatile Mid** | LaMelo Ball | 65.6 | ~25 | ~75+ | Large (+9) |
| **Steady Rotation** | Dyson Daniels | 58.3 | ~12 | ~60 | Small (+2) |
| **Volatile Rotation** | Shaedon Sharpe | 48.0 | ~22 | ~57+ | Large (+9) |
| **Steady Floor** | Mikal Bridges | 50.8 | ~8 | ~51 | Minimal |
| **Volatile Floor** | Embiid (when active) | 70.1 | ~25 | ~82+ | Huge (but availability kills it) |

The "Value Gap" column shows how much extra FP/G you extract by selectively starting their good games. **Volatile players at the mid-to-late tiers are the most underpriced assets in an auction** because the market prices them at raw FP/G while a smart manager realizes managed FP/G.

### 11.8. The Catch: Predictability of Variance

Not all variance is equal:

- **Matchup-driven variance** (good vs bad defensive teams, home/away splits) is **predictable**. You can plan around it. High option value.
- **Random variance** (inconsistent effort, streaky shooting) is **unpredictable**. Harder to cherry-pick the good games. Lower option value.
- **Injury-driven variance** (plays 70 FP/G when active, misses games randomly) is the **worst kind** — you can't predict the zeros, and the missed games often leave you scrambling for MinGames.

When evaluating volatile players at auction, ask: **Can I predict which games will be the booms?** If yes, the option value is real and worth paying for. If no, discount accordingly.

### 11.9. Connection to Roster Construction

This ties back to the budget allocation strategies in Section 5:

- **Stars & Scrubs** builds are naturally high-variance. The scrubs have low FP/G with moderate variance — their bad games are *really* bad. Limited option value because you need every game from them.
- **Balanced** builds give you moderate Games Margin. Some option value on your bottom 2-3 players, but not much room to be selective.
- **Mid-Heavy** builds that target volatile mid-tier players can generate the highest managed TeamPPG because:
  - You have 4-5 players in the 55-75 raw FP/G range
  - Several of them have high σ
  - With 11 players and only 30 games to count, you have margin to cherry-pick
  - Your managed TeamPPG can be 3-5 points higher than your raw TeamPPG

**The ultimate draft strategy combines the Mid-Heavy budget allocation with deliberate targeting of volatile, underpriced mid-tier players.** Their raw FP/G makes them look like $15-25 players. Their managed FP/G makes them play like $30-40 players. That's where championships are built.

---

## 12. Limitations and Future Work

### 12.1. What This Model Doesn't Capture

- **Availability / injury risk.** Embiid at $22 is a "steal" by FP/G but his 38 GP makes that FP/G unreliable. The model needs a GP-weighted adjustment (partially covered in `902` Section 3).
- **Positional scarcity.** The only 90+ FP/G center is Jokic/Wemby. If you need a center, the break-even for those two rises because the alternatives are much worse.
- **Draft dynamics.** Real auctions have nomination strategy, budget pressure, tilt, fatigue, and bidding wars. The model gives pre-draft targets; in-draft adaptation is still a human skill (or bot skill, per `160`).
- **Projection uncertainty.** All FP/G numbers are based on last season. Breakouts, declines, and role changes will shift values.

### 12.2. Future Extensions

- **Monte Carlo draft simulation.** Simulate 10,000 drafts with different allocation strategies to find the optimal budget curve under uncertainty.
- **Dynamic break-even recalculation.** After each pick, recompute all remaining players' break-even based on updated budget and roster.
- **Historical league-specific calibration.** Use 3+ seasons of your league's draft data to build a league-specific $/FP/G curve that accounts for your league's tendencies (e.g., "this league always overpays centers").
- **Combine with playoff schedule projections.** Weight FP/G by expected playoff games for late-season trade and draft valuation.

---

## 13. Quick Reference Cheat Sheet

### At 25-Game Min (10-Man Roster)

| If a player's FP/G is... | Buy under | Fair price | Walk away above |
|---:|---:|---:|---:|
| 110+ | $55 | $83 | $100 |
| 85-110 | $40 | $56-74 | $80 |
| 75-85 | $28 | $47-56 | $60 |
| 65-75 | $20 | $38-47 | $50 |
| 55-65 | $12 | $25-38 | $40 |
| 45-55 | $5 | $15-25 | $30 |
| <45 | $1 | $5-15 | $20 |

### At 30-Game Min (11-Man Roster)

| If a player's FP/G is... | Buy under | Fair price | Walk away above |
|---:|---:|---:|---:|
| 110+ | $60 | $90 | $105 |
| 85-110 | $45 | $60-80 | $85 |
| 75-85 | $30 | $50-60 | $65 |
| 65-75 | $22 | $40-50 | $55 |
| 55-65 | $14 | $28-40 | $45 |
| 45-55 | $6 | $16-28 | $32 |
| <45 | $1 | $6-16 | $22 |

---

_This document should be updated as league parameters change (MinGames, roster size, IR slots) and as more seasons of auction data become available for calibration._
