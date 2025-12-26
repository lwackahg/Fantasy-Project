import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from deap import algorithms, base, creator, tools

from streamlit_compat import dataframe


st.set_page_config(layout="wide", page_title="Best Possible Auction Team")

st.title("Best Possible Auction Team")
st.caption("Uses your Fantrax draft results (auction prices) + Fantrax performance exports (YTD/60/30/14/7) to compute the best $200 roster.")


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DRAFT_RESULTS_PATH = DATA_DIR / "Fantrax-Draft-Results-Mr Squidward s Gay Layup Line.csv"

HORIZON_FILES = {
    "YTD": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(YTD).csv",
    "60": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(60).csv",
    "30": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(30).csv",
    "14": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(14).csv",
    "7": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(7).csv",
}


@st.cache_data(show_spinner=False)
def _load_draft_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Player ID": "ID",
            "Pick": "Pick",
            "Pos": "Pos",
            "Player": "Player",
            "Team": "NBA_Team",
            "Bid": "Bid",
            "Fantasy Team": "FantasyTeam",
        }
    )
    df["ID"] = df["ID"].astype(str)
    df["Player"] = df["Player"].astype(str)
    df["Pos"] = df["Pos"].astype(str)
    df["Bid"] = pd.to_numeric(df["Bid"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def _load_fantrax_export(span: str) -> pd.DataFrame:
    fname = HORIZON_FILES[span]
    path = DATA_DIR / fname
    df = pd.read_csv(path)
    df["ID"] = df["ID"].astype(str)
    df["Player"] = df["Player"].astype(str)
    for col in ["FPts", "FP/G", "GP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _load_all_fantrax_exports() -> dict[str, pd.DataFrame]:
    return {k: _load_fantrax_export(k) for k in HORIZON_FILES.keys()}


def _parse_elig_positions(raw: str) -> set[str]:
    raw = str(raw or "").strip()
    if not raw:
        return set()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # Fantrax can produce values like "G,F" or "F,C".
    # Normalize a bit and only keep G/F/C.
    out: set[str] = set()
    for p in parts:
        p = p.upper()
        if p in {"G", "F", "C"}:
            out.add(p)
    return out


def _roster_feasible(elig_sets: list[set[str]], slots: list[str]) -> bool:
    """Exact feasibility check: can we assign distinct players to each slot?

    Slots are strings in {"G","F","C","Flx"}. "Flx" accepts any of G/F/C.
    Uses a bitmask DP for speed (roster sizes ~10-16 are fine).
    """

    n = len(elig_sets)
    if len(slots) > n:
        return False

    # Build eligibility mask per slot.
    slot_masks: list[int] = []
    for s in slots:
        if s == "Flx":
            allowed = {"G", "F", "C"}
        else:
            allowed = {s}
        mask = 0
        for i, elig in enumerate(elig_sets):
            if elig & allowed:
                mask |= 1 << i
        if mask == 0:
            return False
        slot_masks.append(mask)

    # Sort slots by restrictedness (fewest eligible players first).
    slot_masks = sorted(slot_masks, key=lambda m: int(m.bit_count()))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _can_fill(slot_idx: int, used_mask: int) -> bool:
        if slot_idx >= len(slot_masks):
            return True
        available = slot_masks[slot_idx] & (~used_mask)
        while available:
            lsb = available & -available
            available -= lsb
            if _can_fill(slot_idx + 1, used_mask | lsb):
                return True
        return False

    return _can_fill(0, 0)


def _build_player_pool(draft_df: pd.DataFrame, perf_by_span: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Build a single perf frame with per-span FP/G.
    base = None
    for span, df in perf_by_span.items():
        cols = [c for c in ["ID", "Player", "Team", "Position", "FPts", "FP/G", "GP"] if c in df.columns]
        tmp = df[cols].copy()
        if "FP/G" in tmp.columns:
            tmp = tmp.rename(columns={"FP/G": f"FP/G_{span}"})
        if "FPts" in tmp.columns:
            tmp = tmp.rename(columns={"FPts": f"FPts_{span}"})
        if "GP" in tmp.columns:
            tmp = tmp.rename(columns={"GP": f"GP_{span}"})

        if base is None:
            base = tmp
        else:
            # Keep Player/Position from the first span encountered; merge the metrics.
            metric_cols = [c for c in tmp.columns if c not in {"Player", "Team", "Position"}]
            base = base.merge(tmp[["ID"] + metric_cols], on="ID", how="outer")

    if base is None:
        base = pd.DataFrame(columns=["ID"])

    merged = draft_df.merge(base, on="ID", how="left")

    if "Position" in merged.columns:
        merged["FantraxPosition"] = merged["Position"]
    else:
        merged["FantraxPosition"] = ""

    # Prefer Fantrax player name if we have it.
    if "Player_y" in merged.columns and "Player_x" in merged.columns:
        merged["Player"] = merged["Player_y"].fillna(merged["Player_x"])
    elif "Player" not in merged.columns and "Player_x" in merged.columns:
        merged["Player"] = merged["Player_x"]

    # Ensure metric columns exist
    for span in HORIZON_FILES.keys():
        for col in [f"FP/G_{span}", f"FPts_{span}", f"GP_{span}"]:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    pool = (
        merged.groupby(["ID", "Player"], as_index=False)
        .agg(
            Bid=("Bid", "min"),
            Pos=("Pos", "first"),
            FantraxPosition=("FantraxPosition", "first"),
            **{f"FP/G_{span}": (f"FP/G_{span}", "max") for span in HORIZON_FILES.keys()},
            **{f"FPts_{span}": (f"FPts_{span}", "max") for span in HORIZON_FILES.keys()},
            **{f"GP_{span}": (f"GP_{span}", "max") for span in HORIZON_FILES.keys()},
        )
    )

    pool["Bid"] = pd.to_numeric(pool["Bid"], errors="coerce").fillna(0).astype(int)
    pool["Pos"] = pool["Pos"].astype(str)
    pool["FantraxPosition"] = pool["FantraxPosition"].astype(str)

    # Convenience: current-span columns for display (YTD)
    pool["FP/G"] = pool.get("FP/G_YTD", 0.0)
    pool["FPts"] = pool.get("FPts_YTD", 0.0)
    pool["GP"] = pool.get("GP_YTD", 0.0)

    pool = pool.sort_values(["FP/G_YTD", "FPts_YTD"], ascending=False).reset_index(drop=True)
    return pool


def _ga_optimize_roster(
    pool: pd.DataFrame,
    budget: int,
    roster: dict,
    objective: str,
    window_weights: dict[str, float],
    time_limit_s: float,
    population_size: int,
    generations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    required_g = int(roster.get("G", 0))
    required_f = int(roster.get("F", 0))
    required_c = int(roster.get("C", 0))
    required_flx = int(roster.get("Flx", 0))
    required_bench = int(roster.get("Bench", 0))
    total_spots = int(sum(int(v) for v in roster.values()))

    if total_spots <= 0:
        return pd.DataFrame()

    # Build a blended per-player score from FP/G across windows.
    wsum = float(sum(float(v) for v in window_weights.values()))
    if wsum <= 0:
        window_weights = {k: 1.0 for k in HORIZON_FILES.keys()}
        wsum = float(len(window_weights))

    # Normalize weights
    norm_w = {k: float(v) / wsum for k, v in window_weights.items()}

    fpg_cols = [f"FP/G_{k}" for k in HORIZON_FILES.keys()]
    fpg_mat = np.vstack([pd.to_numeric(pool.get(c, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in fpg_cols]).T
    weights_vec = np.array([norm_w[k] for k in HORIZON_FILES.keys()], dtype=float)
    blended_fpg = (fpg_mat * weights_vec).sum(axis=1)

    bids = pool["Bid"].to_numpy(dtype=int)
    denom = (bids.astype(float) + 1.0)
    value_per_dollar = blended_fpg / denom

    if objective == "Blended FP/G":
        scores = blended_fpg
    elif objective == "Blended FP/G per $":
        scores = value_per_dollar
    elif objective == "Blend: FP/G + Value":
        # Mixed objective: reward both raw output and value.
        scores = blended_fpg + (value_per_dollar * 25.0)
    else:
        scores = blended_fpg

    candidate_idx = list(pool.index)
    if len(candidate_idx) < total_spots:
        return pd.DataFrame()

    if not hasattr(creator, "AuctionFitnessMax"):
        creator.create("AuctionFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "AuctionIndividual"):
        creator.create("AuctionIndividual", list, fitness=creator.AuctionFitnessMax)

    toolbox = base.Toolbox()

    elig_raw = pool.get("FantraxPosition", pool.get("Pos", "")).astype(str).to_numpy()
    elig_sets = [_parse_elig_positions(x) for x in elig_raw]

    weights = scores.copy()
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    def _init_individual():
        picks = rng.choice(candidate_idx, size=total_spots, replace=False, p=weights).tolist()
        return creator.AuctionIndividual(picks)

    toolbox.register("individual", tools.initIterate, creator.AuctionIndividual, _init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def _evaluate(individual):
        if len(set(individual)) != len(individual):
            return (-1e9,)

        total_cost = int(bids[individual].sum())
        if total_cost > budget:
            return (-1e9 - float(total_cost - budget) * 1000.0,)

        # Exact slot feasibility using Fantrax multi-position eligibility.
        slots = (["G"] * required_g) + (["F"] * required_f) + (["C"] * required_c) + (["Flx"] * required_flx)
        sel_elig = [elig_sets[i] for i in individual]
        if not _roster_feasible(sel_elig, slots):
            return (-1e9,)

        return (float(scores[individual].sum()),)

    def _mutate(individual, indpb=0.2):
        for i in range(len(individual)):
            if rng.random() < indpb:
                existing = set(individual)
                for _ in range(20):
                    cand = int(rng.choice(candidate_idx, p=weights))
                    if cand not in existing:
                        existing.remove(individual[i])
                        individual[i] = cand
                        existing.add(cand)
                        break
        return (individual,)

    toolbox.register("evaluate", _evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    start = time.time()
    gen = 0
    while (time.time() - start) < time_limit_s and gen < generations:
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.55, mutpb=0.35)
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))
        hof.update(pop)
        gen += 1

    if not hof:
        return pd.DataFrame()

    best = list(hof[0])
    out = pool.loc[best].copy()
    # Attach blended columns for display
    out["BlendedFP/G"] = blended_fpg[best]
    out["ValuePer$"] = value_per_dollar[best]
    out["Score"] = scores[best]
    out = out.sort_values(["Score", "Bid"], ascending=[False, True]).reset_index(drop=True)
    return out


left, right = st.columns([0.42, 0.58])

with left:
    st.subheader("Scoring")
    objective = st.selectbox(
        "Objective",
        options=["Blended FP/G", "Blended FP/G per $", "Blend: FP/G + Value"],
        index=0,
        key="best_team_obj",
    )

    st.markdown("---")
    st.subheader("Window weights (FP/G)")
    wcols = st.columns(5)
    w_ytd = wcols[0].number_input("YTD", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="best_team_w_ytd")
    w_60 = wcols[1].number_input("60d", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="best_team_w_60")
    w_30 = wcols[2].number_input("30d", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="best_team_w_30")
    w_14 = wcols[3].number_input("14d", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="best_team_w_14")
    w_7 = wcols[4].number_input("7d", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="best_team_w_7")

    budget = st.number_input("Budget cap ($)", min_value=1, max_value=1000, value=200, step=5, key="best_team_budget")

    st.markdown("---")
    st.subheader("Roster slots")
    rcols = st.columns(2)
    g = rcols[0].number_input("G", min_value=0, max_value=20, value=4, step=1, key="best_team_g")
    f = rcols[1].number_input("F", min_value=0, max_value=20, value=4, step=1, key="best_team_f")
    c = rcols[0].number_input("C", min_value=0, max_value=20, value=2, step=1, key="best_team_c")
    flx = rcols[1].number_input("Flx", min_value=0, max_value=20, value=2, step=1, key="best_team_flx")
    bench = rcols[0].number_input("Bench", min_value=0, max_value=40, value=3, step=1, key="best_team_bench")

    st.markdown("---")
    st.subheader("Optimizer settings")
    time_limit_s = st.slider("Time limit (seconds)", min_value=1.0, max_value=20.0, value=4.0, step=0.5, key="best_team_time")
    population_size = st.slider("Population", min_value=50, max_value=600, value=200, step=25, key="best_team_pop")
    generations = st.slider("Generations", min_value=10, max_value=400, value=120, step=10, key="best_team_gen")
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=7, step=1, key="best_team_seed")

    run = st.button("Compute best possible team", type="primary", use_container_width=True)

with right:
    st.subheader("Actual drafted teams (context only)")
    try:
        draft_df = _load_draft_results(DRAFT_RESULTS_PATH)
        perf_by_span = _load_all_fantrax_exports()
        pool = _build_player_pool(draft_df, perf_by_span)

        # Compute blended FP/G for the draft pool using current weights
        w_map = {"YTD": float(w_ytd), "60": float(w_60), "30": float(w_30), "14": float(w_14), "7": float(w_7)}
        wsum = sum(w_map.values())
        if wsum <= 0:
            wsum = 1.0
        w_norm = {k: v / wsum for k, v in w_map.items()}
        pool_by_id = pool.set_index("ID")
        blended = (
            pool_by_id["FP/G_YTD"] * w_norm["YTD"]
            + pool_by_id["FP/G_60"] * w_norm["60"]
            + pool_by_id["FP/G_30"] * w_norm["30"]
            + pool_by_id["FP/G_14"] * w_norm["14"]
            + pool_by_id["FP/G_7"] * w_norm["7"]
        )

        draft_perf = draft_df.merge(blended.rename("BlendedFP/G"), left_on="ID", right_index=True, how="left")
        draft_perf["BlendedFP/G"] = pd.to_numeric(draft_perf.get("BlendedFP/G", 0), errors="coerce").fillna(0.0)
        team_agg = (
            draft_perf.groupby("FantasyTeam", as_index=False)
            .agg(
                Spent=("Bid", "sum"),
                Players=("Player", "count"),
                AvgBlendedFPG=("BlendedFP/G", "mean"),
            )
            .sort_values(["AvgBlendedFPG", "Spent"], ascending=[False, True])
        )
        dataframe(team_agg, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Could not load draft/performance data: {exc}")

st.markdown("---")

if run:
    try:
        with st.spinner("Optimizing..."):
            draft_df = _load_draft_results(DRAFT_RESULTS_PATH)
            perf_by_span = _load_all_fantrax_exports()
            pool = _build_player_pool(draft_df, perf_by_span)

            roster = {"G": int(g), "F": int(f), "C": int(c), "Flx": int(flx), "Bench": int(bench)}
            window_weights = {"YTD": float(w_ytd), "60": float(w_60), "30": float(w_30), "14": float(w_14), "7": float(w_7)}
            best_df = _ga_optimize_roster(
                pool=pool,
                budget=int(budget),
                roster=roster,
                objective=objective,
                window_weights=window_weights,
                time_limit_s=float(time_limit_s),
                population_size=int(population_size),
                generations=int(generations),
                seed=int(seed),
            )

        if best_df.empty:
            st.warning("No roster found with the current constraints/settings.")
        else:
            total_cost = int(pd.to_numeric(best_df["Bid"], errors="coerce").fillna(0).sum())
            total_score = float(pd.to_numeric(best_df.get("Score", 0), errors="coerce").fillna(0.0).sum())
            total_blended = float(pd.to_numeric(best_df.get("BlendedFP/G", 0), errors="coerce").fillna(0.0).sum())

            top = st.columns([0.25, 0.25, 0.25, 0.25])
            top[0].metric("Total cost", f"${total_cost}")
            top[1].metric("Total Score", f"{total_score:.1f}")
            top[2].metric("Players", f"{len(best_df)}")
            top[3].metric("Total Blended FP/G", f"{total_blended:.1f}")

            show_cols = [c for c in ["Player", "FantraxPosition", "Bid", "BlendedFP/G", "ValuePer$", "Score", "FP/G_YTD", "FP/G_60", "FP/G_30", "FP/G_14", "FP/G_7"] if c in best_df.columns]
            dataframe(best_df[show_cols], width="stretch", hide_index=True)

            try:
                csv_bytes = best_df[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download best roster (CSV)",
                    data=csv_bytes,
                    file_name="best_auction_roster.csv",
                    mime="text/csv",
                )
            except Exception:
                pass

    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
