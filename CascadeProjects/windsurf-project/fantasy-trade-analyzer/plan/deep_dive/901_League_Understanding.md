 # 901 – League Understanding

 _How this league actually plays on the ground, and why FP/G matters more than raw totals._

 This document translates the formal rules from `900_League_Constitution.md` into practical strategy:

 - **What actually wins weekly matchups**
 - **How the 25–game minimum and overage penalties change player value**
 - **How to think about FP/G vs total FPts vs availability**
 - **Implications for drafting, in‑season management, and trades**

---

## 1. League Shape at a Glance

- **16 teams**, **10 roster spots** each.
- **Daily lineups**: 3 G, 3 F, 2 C, 2 Flex (10 active per day).
- **Scoring** is heavily weighted toward real impact stats:
  - Positive: PTS, REB, ST, BLK, 3PTM, AST, OREB, 2D
  - Negative: FG misses, FT misses, TO, TF, EJ
- **Head‑to‑head weekly** with a **25 games per week minimum**, and a specific **penalty for going over**.

The combination of:

- A hard **minimum games requirement**, and
- A **PPG‑based overage penalty**

means that matchups are effectively decided by **how efficiently you can convert ~25 games into fantasy points**, not by who can churn the most total games.

---

## 2. Minimum Games & Overage Penalty – Why FP/G Is the Real Currency

From Article VIII / Amendments:

- **Target**: ~25 games per week (can adjust on odd weeks).
- **If one team is under the minimum and the other meets it** → the under‑min team **auto‑loses**.
- **If both are under** → higher total points wins (no adjustment).
- **If a team goes over**:
  - Penalty = `Games Over × PPG Average` for that week.

### 2.1. Why 25 Games? The 71.4% Baseline

- Before each season, the league minimum is calibrated using a **71.4% expected usage rate**:
  - `Min Games ≈ (Roster Spots × Avg Games per Player per Week) × 0.714`.
- With the current **10‑man roster** and **~3.5 games per player per week**:
  - `10 × 3.5 = 35` theoretical games.
  - `35 × 0.714 ≈ 25` → current minimum.
- In the previous **12‑man (10+2) era**:
  - `12 × 3.5 = 42` theoretical games.
  - `42 × 0.714 ≈ 30` → old minimum.

Intuition:

- The minimum is set so that you are **not forced to perfectly max out every possible game** to compete.
- It bakes in a realistic buffer for injuries, rest, and schedule quirks while still **punishing chronic under‑management**.

Key implication:

- If you end at **exactly 25 games**, total points ≈ `25 × team PPG`.
- If you end at **25 + k games**, your **raw** total points are `PPG × (25 + k)`, but you then lose `PPG × k` as a penalty.
- Net effect: you’re still effectively playing with **25 games worth of scoring**, but now your **PPG includes those extra games**.

This leads to a few critical takeaways:

- **Streaming low‑FP/G players just to add games is neutral at best, harmful at worst.**
  - Added low‑FP/G games lower your weekly PPG.
  - The overage penalty is based on that PPG, so you don’t really “gain” from the extra volume.
- **What really matters is the FP/G quality of the games that actually “count”.**
  - You want ~25 of your best games, not 32 games including a bunch of 18–22 FP grinders.
  - The penalty structure effectively turns this league into a **FP/G efficiency race**, not a raw totals race.

**Bottom line:**

- You still must **hit the minimum games** to avoid auto‑loss.
- But **once you can confidently reach 25 games with your core**, extra available games have sharply diminishing returns.
- That’s why **FP/G and consistency are more important than sheer totals**, especially for your best ~10 lineup spots.

---

## 3. FP/G vs Total FPts vs Availability

It’s not that **total FPts** or **games played** are useless. It’s that, in this league, they are **contextual** to FP/G rather than the other way around.

### 3.1. Core concepts

- **FP/G (Fantasy Points per Game)**
  - Primary measure of how strongly a player moves your matchup when they are in your lineup.
  - In a capped‑games environment, **FP/G per roster spot** is the main driver of weekly ceiling.

- **Total FPts**
  - Still informative over a season (especially for awards / record books / macro view).
  - But for **a given week**, total FPts are heavily constrained by the minimum/penalty structure.
  - Two players with the same total FPts can have very different usefulness if one did it on 30 games and the other on 22.

- **Availability (Games Played / durability)**
  - Matters for **hitting 25 games without relying on junk streamers**.
  - Matters for **playoff reliability** and avoiding weeks where you’re forced to start bad options.
  - But once your roster can reliably reach the minimum with quality players, **additional “volume” is less valuable than extra FP/G.**

### 3.2. High FP/G, Low GP vs Lower FP/G, High GP

This league **does not simply say**:

- "Player with higher FP/G is always better" or
- "Iron‑man volume is automatically king".

Instead, think in terms of **“effective weekly contribution”**:

- Rough mental model:
  - `Effective Weekly FP ≈ FP/G × Expected Games Used` (subject to the 25‑game team target).
  - For a star on a 3‑ or 4‑game week: you’ll almost always use all those games.
  - For a risky / low‑GP player: some weeks you may only get 1–2 games or zero.

Heuristically:

- A **90 FP/G** player who plays **2 games** in a week (~180 FP) can still out‑value a **55 FP/G** player on **4 games** (~220 FP), 
  - but if those extra games push you **far** beyond 25 and force you into low‑impact streams elsewhere, the edge isn’t as large as raw totals suggest.
- A **70 FP/G** but fragile star is worth more if the rest of your roster is durable enough that you **still reach 25 games without streaming junk**.
- If your team is already injury‑prone or shallow, the same fragile star is riskier because missing games could drag you below 25 or force ugly streamers.

So the league’s philosophy is roughly:

- **FP/G is the “currency” of upside and quality.**
- **Availability is the “floor” that ensures you can cash in that currency.**
- **Total FPts is the result, not the target.**

### 3.3. Core FP/G Spots vs Flex/Streamer Slots

- In general, with **~3.5 games per player per week**, your "core" size is roughly:
  - `Core Size ≈ (Min Games Requirement ÷ 3.5)`.
- Examples:
  - `25 ÷ 3.5 ≈ 7.14` → about **7–8 core players** at a 25‑game minimum.
  - `30 ÷ 3.5 ≈ 8.57` → about **8–9 core players** at a 30‑game minimum.
  - `35 ÷ 3.5 = 10` → **10 core players** at a 35‑game minimum.
- Practical takeaway (for current 25‑game setup):
  - Your **top ~7–8 players** are the main drivers of weekly FP/G.
  - The remaining **2–3 roster spots** are primarily **flex/streamer** slots:
    - Used for schedule optimization.
    - Used for **short‑term role spikes** (injury replacements, players moving into a starting role, etc.).

Streaming in this league is **high‑skill, not automatically bad**:

- **Good streams**:
  - News‑driven adds where a player suddenly moves into 30+ minutes or a bigger usage role.
  - Can temporarily behave like a mid‑tier or even high‑tier FP/G asset.
- **Bad streams**:
  - Generic low‑FP/G volume just because "this guy has four games".
  - These are the ones that tend to drag down PPG and trigger overage penalties.

---

## 4. Strategic Archetypes & How the Rules Treat Them

### 4.1. Elite FP/G, Medium Availability (Injury‑prone stars)

- **Upside**: Huge; a few 90+ FP/G games can swing a week.
- **Risk**: Missing games may force you into low‑FP streams or even under 25 if the rest of your roster is thin.
- **Best fit**:
  - Teams with **good depth and healthy cores** that can still hit 25 quality games.
  - Managers willing to manage IL closely and leverage the insurance mechanics in trades.

### 4.2. High FP/G, High Availability (Unicorns)

- **Ideal assets** in this format.
- They help you **hit 25 games with quality**, reduce the need to stream, and keep PPG high.
- These are the players where **consolidation trades** (2‑for‑1, 3‑for‑1) are almost always correct if the FP/G upgrade is significant.

### 4.3. Medium FP/G, Iron‑man Volume Grinders

- Useful for **stabilizing your floor** and ensuring you reach 25 games.
- Less valuable once your roster is already safe on games:
  - Extra low‑ceiling games beyond 25 can **dilute PPG** and generate more negative stats (TO, TF, FG‑, FT‑).
- Good targets when:
  - You’re struggling to hit 25 without streaming multiple sketchy players.
  - You want to "pad" against injury volatility.

### 4.4. Low FP/G, High Volume Streamers

- Here we are specifically talking about **low FP/G, high‑volume schedule plugs**.
- In this league, they are **emergency tools**:
  - Needed if you’re at risk of falling short of the 25‑game minimum.
  - Not something you want to rely on when already at or above 25 games.
- **Smart, news‑driven streaming** (short‑term starters, role spikes) can be very +EV and is closer to temporarily adding a mid‑tier player.
- Over‑streaming the **low‑FP/G** type is exactly how you end up with **mediocre PPG and overage penalties**.

### 4.5. IL Stashes and Risky Upside Bets

- The constitution’s IL rules are strict (locks if you abuse them), so stashing injured players has a **real opportunity cost**.
- Stash logic in this league should be:
  - “Is this player likely to be an **elite FP/G piece in time to matter**, and can I afford the short‑term game loss?”
  - Not “he might be good eventually and I just want to hoard.”

---

## 5. Weekly Matchup Strategy

### 5.1. Hitting the 25‑Game Minimum

- **Non‑negotiable**: If your opponent hits 25 and you don’t, you auto‑lose.
- Plan ahead:
  - Look at **NBA schedule density** early in the week.
  - Track combined projected games for your active players.
  - Use streamers to **patch gaps**, not to chase 35+ games.

### 5.2. Avoiding Over‑Streaming

- Once you’re safely on track to hit ~25–26 games:
  - Be very careful about adding low‑FP/G streamers.
  - Ask: "Does this added game actually raise my final PPG, or just add more turnovers and bricks?"
- Often, **sitting a fringe guy** is correct if:
  - You’re already over 25 projected games.
  - Their FP/G is significantly below your team’s current PPG.

### 5.3. Managing Risk Over the Season

- **Early season**:
  - You can justify **slightly more volume** and experimentation to establish baselines.
  - You may accept more volatility to identify breakouts.

- **Late season / playoffs**:
  - Priority shifts heavily to **FP/G and consistency**.
  - You want a roster that can hit 25 games with **minimal reliance on risky streamers**.

---

## 6. Trade Strategy in This League Context

The trade section of the constitution focuses on fairness, veto logic, and insurance. This section focuses on **optimal strategy given the rules.**

### 6.1. What Makes a “Good Trade” Here

In this league, a trade is strong if it:

- **Improves your top‑end FP/G** across your primary lineup spots.
- **Does not materially weaken your ability to reach 25 quality games**.
- **Improves or maintains your risk profile** (injury + volatility) in a way that fits your team’s situation.

Common patterns that tend to be correct:

- **2‑for‑1 or 3‑for‑1 upgrades** into truly elite FP/G pieces, when you have enough depth.
- Swapping a **volume‑only piece** plus a mid‑tier guy for a **higher FP/G, slightly riskier star**, if your overall durability is good.

Patterns that are often traps:

- Trading away a **high FP/G cornerstone** for two lower FP/G “safe” players just because they play more games.
  - Those extra games may not help if you’re already reaching 25.
- Cutting FP/G too much just to shore up availability on a roster that was already comfortably meeting the games minimum.

### 6.2. When Lower Games but Higher FP/G Is Actually Better

Heuristics for comparing players in trades:

- If Player A has **much higher FP/G** but plays **fewer games**, ask:
  1. **Can I still comfortably reach 25 games without relying heavily on bad streamers if I acquire A?**
  2. **Does A meaningfully raise the FP/G of my best 8–10 spots?**

- If both answers are **yes**, A is often the better piece even with fewer total FPts.
- If acquiring A forces you to:
  - Rely on multiple low‑end streamers just to hit 25, or
  - Regularly dip under the minimum,
  then the total tradeoff starts to favor the more durable player.

### 6.3. Using Insurance and IL Rules in Trade Design

- The **insurance appendix** allows managers to structure trades that hedge missed games risk for specific players over a short window.
- In this context, it’s especially useful when:
  - One side is acquiring a **high FP/G but risky availability** player.
  - The other side wants compensation if the player doesn’t meet a minimum games threshold.
- This makes it easier to trade for upside while respecting how critical availability is to hitting 25 games.

---

## 7. How the Tool Will Align With This Philosophy (High‑Level)

The Trade Suggestions engine and related analytics will be tuned to reflect these principles:

- **FP/G‑centric value**:
  - FP/G is treated as the primary driver of player value per roster spot.
  - Elite FP/G players are rewarded non‑linearly to reflect how they shape weekly matchups.

- **Availability and durability as modifiers, not the main currency**:
  - Games played, injury history, and projected availability adjust value up or down.
  - The goal is to capture “Can this player help me consistently hit 25 games with quality?”

- **Team‑level impact**:
  - Evaluations will focus on how a trade changes your **team’s effective FP/G across key spots**, rather than just summing totals.
  - Volume that only shows up as extra games beyond the penalty threshold will be treated as lower‑impact.

Specific formulas and tuning live in the implementation notes and code, but the guiding philosophy is:

> **Quality of games (FP/G) wins in this league, as long as you can reliably reach the minimum games without drowning in low‑efficiency volume.**

---

## 8. Future Extensions / Notes

Potential areas to explore in future seasons or tool updates:

- **Schedule‑aware recommendations** that highlight weeks where your roster is at risk of missing 25 games.
- **Risk dashboards** that show how many of your key FP/G contributors are injury‑prone vs iron‑men.
- **Playoff‑specific views**, where upcoming schedules and back‑to‑backs are weighted more heavily.

This document should be updated over time as new edge cases, rulings, or league cultural norms develop.

