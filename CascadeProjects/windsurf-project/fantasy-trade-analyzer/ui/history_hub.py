import streamlit as st
import pandas as pd
import plotly.express as px

from streamlit_compat import dataframe, plotly_chart

from modules.draft_history import load_draft_history
from modules.manager_ids import load_manager_ids, get_manager_list

from pathlib import Path
import time
import numpy as np
from deap import algorithms, base, creator, tools


def render_best_team_optimizer() -> None:
    st.subheader("🧠 Best Possible $200 Team (10-man roster)")

    st.markdown(
        """
**What does “Blended” mean?**

“Blended FP/G” is a **weighted average** of FP/G from multiple time windows (YTD, 60d, 30d, 14d, 7d).
Example: if you set YTD=1 and 60d=1 (others 0), then:

`Blended FP/G = 0.5 * FP/G_YTD + 0.5 * FP/G_60`

**Objectives**

- **Blended FP/G**: maximize raw blended FP/G.
- **Blended FP/G per $**: maximize blended FP/G divided by (Bid + 1). Strongly favors value/cheap players.
- **Blend: FP/G + Value**: a hybrid that tries to balance stars and value.
"""
    )

    DATA_DIR = Path(__file__).resolve().parents[1] / "data"

    draft_csvs = sorted(list(DATA_DIR.rglob("Fantrax-Draft-Results-*.csv")))
    if not draft_csvs:
        st.warning("No Fantrax draft results CSVs found in the data folder.")
        return

    horizon_files = {
        "YTD": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(YTD).csv",
        "60": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(60).csv",
        "30": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(30).csv",
        "14": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(14).csv",
        "7": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(7).csv",
    }

    hist_ytd_dir = DATA_DIR / "historical_ytd"
    hist_ytd_files = []
    if hist_ytd_dir.exists():
        hist_ytd_files = sorted(list(hist_ytd_dir.glob("Fantrax-Players-*-YTD-*.csv")))

    @st.cache_data(show_spinner=False)
    def _load_draft_prices(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.rename(columns={"Player ID": "ID", "Bid": "Bid"})
        df["ID"] = df["ID"].astype(str)
        df["Bid"] = pd.to_numeric(df.get("Bid", 0), errors="coerce")
        df = df.dropna(subset=["ID"]).copy()
        df["Bid"] = df["Bid"].fillna(0).astype(int)
        df = df[["ID", "Bid"]].copy()
        df = df.groupby("ID", as_index=False).agg(Bid=("Bid", "max"))
        return df

    @st.cache_data(show_spinner=False)
    def _load_export_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "ID" in df.columns:
            df["ID"] = df["ID"].astype(str)
        for col in ["FPts", "FP/G", "GP"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @st.cache_data(show_spinner=False)
    def _load_all_exports(ytd_path: Path) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        out["YTD"] = _load_export_csv(ytd_path)
        for k, fname in horizon_files.items():
            if k == "YTD":
                continue
            out[k] = _load_export_csv(DATA_DIR / fname)
        return out

    def _parse_elig_positions(raw: str) -> set[str]:
        raw = str(raw or "").strip()
        if not raw:
            return set()
        parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
        return {p for p in parts if p in {"G", "F", "C"}}

    def _roster_feasible(elig_sets: list[set[str]], slots: list[str]) -> bool:
        n = len(elig_sets)
        if len(slots) > n:
            return False
        slot_masks: list[int] = []
        for s in slots:
            allowed = {"G", "F", "C"} if s == "Flx" else {s}
            mask = 0
            for i, elig in enumerate(elig_sets):
                if elig & allowed:
                    mask |= 1 << i
            if mask == 0:
                return False
            slot_masks.append(mask)

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

    def _build_pool(perf_by_span: dict[str, pd.DataFrame], bid_df: pd.DataFrame) -> pd.DataFrame:
        base = None
        for span, df_span in perf_by_span.items():
            cols = [c for c in ["ID", "Player", "Team", "Position", "FP/G", "FPts", "GP"] if c in df_span.columns]
            tmp = df_span[cols].copy()
            if "FP/G" in tmp.columns:
                tmp = tmp.rename(columns={"FP/G": f"FP/G_{span}"})
            if "FPts" in tmp.columns:
                tmp = tmp.rename(columns={"FPts": f"FPts_{span}"})
            if "GP" in tmp.columns:
                tmp = tmp.rename(columns={"GP": f"GP_{span}"})

            if base is None:
                base = tmp
            else:
                metric_cols = [c for c in tmp.columns if c not in {"ID", "Player", "Team", "Position"}]
                base = base.merge(tmp[["ID"] + metric_cols], on="ID", how="outer")

        if base is None:
            base = pd.DataFrame(columns=["ID", "Player", "Position"])

        if "ID" in base.columns:
            base = base.dropna(subset=["ID"]).copy()
            base["ID"] = base["ID"].astype(str)
            base = base.drop_duplicates(subset=["ID"], keep="first")

        bid_df = bid_df.copy()
        bid_df["ID"] = bid_df["ID"].astype(str)

        pool = base.merge(bid_df, on="ID", how="left")
        pool["HasDraftBid"] = pool["Bid"].notna()
        pool["Bid"] = pd.to_numeric(pool.get("Bid", 0), errors="coerce").fillna(0).astype(int)
        pool["Waiver"] = ~pool["HasDraftBid"]

        if "Position" not in pool.columns:
            pool["Position"] = ""
        pool["Position"] = pool["Position"].astype(str)

        for span in horizon_files.keys():
            for c in [f"FP/G_{span}", f"FPts_{span}", f"GP_{span}"]:
                if c not in pool.columns:
                    pool[c] = 0.0
                pool[c] = pd.to_numeric(pool[c], errors="coerce").fillna(0.0)

        pool["Player"] = pool.get("Player", "").astype(str)
        pool = pool.loc[:, ~pool.columns.duplicated()].copy()
        return pool

    def _compute_scores(pool: pd.DataFrame, w: dict[str, float], objective: str):
        wsum = float(sum(float(v) for v in w.values()))
        if wsum <= 0:
            w = {k: 1.0 for k in horizon_files.keys()}
            wsum = float(len(w))
        norm_w = {k: float(v) / wsum for k, v in w.items()}

        fpg_cols = [f"FP/G_{k}" for k in horizon_files.keys()]
        fpg_mat = np.vstack([pd.to_numeric(pool.get(col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) for col in fpg_cols]).T
        weights_vec = np.array([norm_w[k] for k in horizon_files.keys()], dtype=float)
        blended_fpg = (fpg_mat * weights_vec).sum(axis=1)

        gp_cols = [f"GP_{k}" for k in horizon_files.keys()]
        gp_mat = np.vstack([pd.to_numeric(pool.get(col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) for col in gp_cols]).T
        blended_gp = (gp_mat * weights_vec).sum(axis=1)

        bids = pd.to_numeric(pool.get("Bid", 0), errors="coerce").fillna(0).to_numpy(dtype=int)
        denom = bids.astype(float) + 1.0
        value_per_dollar = blended_fpg / denom

        if objective == "Blended FP/G":
            scores = blended_fpg
        elif objective == "Blended FP/G per $":
            scores = value_per_dollar
        else:
            scores = blended_fpg + (value_per_dollar * 25.0)
        return blended_fpg, blended_gp, value_per_dollar, scores

    def _hill_climb_best(
        best_ind: list[int],
        candidate_idx: list[int],
        scores_adj: np.ndarray,
        bids: np.ndarray,
        waiver_flags: np.ndarray,
        budget: int,
        max_waivers: int,
        elig_sets: list[set[str]],
        slots: list[str],
        try_k: int,
        max_passes: int,
    ) -> list[int]:
        curr = list(best_ind)
        curr_set = set(curr)

        def _is_valid(sel: list[int]) -> bool:
            if len(set(sel)) != 10:
                return False
            if int(bids[sel].sum()) > int(budget):
                return False
            if int(waiver_flags[sel].sum()) > int(max_waivers):
                return False
            sel_elig = [elig_sets[i] for i in sel]
            return _roster_feasible(sel_elig, slots)

        def _score(sel: list[int]) -> float:
            return float(scores_adj[sel].sum())

        if not _is_valid(curr):
            return curr

        base_score = _score(curr)

        # Candidates to try (best by scores_adj).
        order = np.argsort(-scores_adj)
        top_cands = [int(i) for i in order[: max(int(try_k), 10)] if int(i) in candidate_idx]

        for _ in range(int(max_passes)):
            improved = False
            best_swap = None
            best_swap_score = base_score
            for out_pos in range(10):
                out_idx = curr[out_pos]
                for in_idx in top_cands:
                    if in_idx in curr_set:
                        continue
                    trial = curr.copy()
                    trial[out_pos] = in_idx
                    if not _is_valid(trial):
                        continue
                    s = _score(trial)
                    if s > best_swap_score + 1e-9:
                        best_swap_score = s
                        best_swap = (out_pos, in_idx)

            if best_swap is None:
                break

            out_pos, in_idx = best_swap
            curr_set.remove(curr[out_pos])
            curr[out_pos] = in_idx
            curr_set.add(in_idx)
            base_score = best_swap_score
            improved = True
            if not improved:
                break

        return curr

    settings_col, status_col = st.columns([0.55, 0.45])

    with settings_col:
        draft_labels = [str(p.relative_to(DATA_DIR)).replace("\\", "/") for p in draft_csvs]
        draft_choice = st.selectbox("Draft results CSV", options=draft_labels, index=0, key="best_draft_csv")
        st.caption("Select which auction draft results to use for player costs (Bid).")

        ytd_mode = st.selectbox(
            "YTD data",
            options=["Current", "Historical"],
            index=0,
            key="best_draft_ytd_mode",
        )
        ytd_path = DATA_DIR / horizon_files["YTD"]
        if ytd_mode == "Historical":
            if not hist_ytd_files:
                st.warning("No historical YTD files found in data/historical_ytd.")
            else:
                ytd_labels = [p.name for p in hist_ytd_files]
                ytd_pick = st.selectbox("Historical YTD file", options=ytd_labels, index=len(ytd_labels) - 1, key="best_draft_ytd_file")
                ytd_path = hist_ytd_dir / ytd_pick
        st.caption("60/30/14/7 windows use the current exports. Historical is only wired for YTD right now.")

        budget_cap = st.number_input("Budget cap ($)", min_value=1, max_value=1000, value=200, step=5, key="best_draft_budget")
        st.caption("Total auction money available for your 10 drafted players.")

        objective = st.selectbox(
            "Objective",
            options=["Blended FP/G", "Blended FP/G per $", "Blend: FP/G + Value"],
            index=0,
            key="best_draft_objective",
        )
        st.caption("What the optimizer tries to maximize. Value-per-$ favors cheaper players.")

        roster_cols = st.columns(4)
        g_req = roster_cols[0].number_input("G", min_value=0, max_value=10, value=3, step=1, key="best_draft_g")
        f_req = roster_cols[1].number_input("F", min_value=0, max_value=10, value=3, step=1, key="best_draft_f")
        c_req = roster_cols[2].number_input("C", min_value=0, max_value=10, value=2, step=1, key="best_draft_c")
        flx_req = roster_cols[3].number_input("Flx", min_value=0, max_value=10, value=2, step=1, key="best_draft_flx")
        st.caption("Roster must total exactly 10. Flx accepts G/F/C.")
        if int(g_req + f_req + c_req + flx_req) != 10:
            st.warning("Roster must total exactly 10 players: G + F + C + Flx = 10")

        st.markdown("**Window weights (FP/G)**")
        st.caption("Weighted blend across time windows. Higher = more influence.")
        wcols = st.columns(5)
        w_ytd = wcols[0].number_input("YTD", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="best_draft_w_ytd")
        w_60 = wcols[1].number_input("60d", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="best_draft_w_60")
        w_30 = wcols[2].number_input("30d", min_value=0.0, max_value=10.0, value=0.25, step=0.25, key="best_draft_w_30")
        w_14 = wcols[3].number_input("14d", min_value=0.0, max_value=10.0, value=0.0, step=0.25, key="best_draft_w_14")
        w_7 = wcols[4].number_input("7d", min_value=0.0, max_value=10.0, value=0.0, step=0.25, key="best_draft_w_7")

        st.markdown("**Games Played (reliability)**")
        use_gp = st.toggle("Adjust scores for low GP", value=True, key="best_draft_use_gp")
        st.caption("Penalizes small-sample FP/G so 2-game heaters don't dominate.")
        gp_cols2 = st.columns(2)
        min_gp = gp_cols2[0].number_input("Min blended GP", min_value=0.0, max_value=82.0, value=0.0, step=1.0, key="best_draft_min_gp")
        gp_smooth = gp_cols2[1].number_input("GP smoothing (k)", min_value=0.0, max_value=50.0, value=10.0, step=1.0, key="best_draft_gp_smooth")
        st.caption("**Min blended GP**: exclude players below this blended GP (0 = allow anyone).")
        st.caption("**GP smoothing (k)**: reliability = GP / (GP + k). Example: GP=5, k=10 => 5/15=0.33 multiplier.")

        st.markdown("**Waiver wire**")
        max_waivers = st.slider("Max waiver pickups", min_value=0, max_value=10, value=10, step=1, key="best_draft_max_waivers")
        st.caption("Waiver pickup = player not found in the selected draft results CSV (treated as $0).")

        st.markdown("**Optimizer settings**")
        time_limit_s = st.slider("Time limit (seconds)", min_value=5.0, max_value=300.0, value=30.0, step=5.0, key="best_draft_time")
        st.caption("Stops automatically after this many seconds and returns the best found so far.")
        population_size = st.slider("Population", min_value=50, max_value=1200, value=300, step=25, key="best_draft_pop")
        st.caption("How many candidate rosters are kept each generation. More = slower but often better.")
        generations = st.slider("Generations", min_value=10, max_value=2000, value=500, step=10, key="best_draft_gen")
        st.caption("Max evolutionary steps. Can stop early due to time limit or Stop button.")
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=7, step=1, key="best_draft_seed")
        st.caption("Same seed + same settings produces the same result.")
        print_every = st.slider("Print every N generations", min_value=1, max_value=200, value=25, step=1, key="best_draft_print_every")
        chunk_gens = st.slider("Generations per UI tick", min_value=1, max_value=50, value=5, step=1, key="best_draft_chunk")

        st.markdown("**Stagnation handling**")
        stagnate_gens = st.slider("Stagnation patience (gens)", min_value=10, max_value=500, value=100, step=10, key="best_draft_stagnate")
        diversify_frac = st.slider("Refresh fraction on stagnation", min_value=0.0, max_value=0.9, value=0.4, step=0.1, key="best_draft_refresh_frac")
        hillclimb_k = st.slider("Hill-climb candidate pool", min_value=25, max_value=300, value=120, step=25, key="best_draft_hill_k")
        hillclimb_passes = st.slider("Hill-climb passes", min_value=1, max_value=10, value=3, step=1, key="best_draft_hill_passes")
        st.caption("If the best score hasn't improved for N generations, we try a deterministic local-improvement swap (hill-climb) and refresh part of the population.")

        btn_cols = st.columns(2)
        start_btn = btn_cols[0].button("Start / Resume", type="primary", use_container_width=True, key="best_draft_start")
        stop_btn = btn_cols[1].button("Stop", use_container_width=True, key="best_draft_stop")

    state_key = "best_draft_state"
    if state_key not in st.session_state:
        st.session_state[state_key] = {}
    state = st.session_state[state_key]

    if stop_btn:
        state["stop_requested"] = True

    def _state_reset():
        st.session_state[state_key] = {}

    with status_col:
        st.caption("Waiver pickups: players not in the selected draft results are treated as $0.")

        prog = st.progress(0.0)
        timer_ph = st.empty()
        best_ph = st.empty()
        log_ph = st.empty()

        if start_btn and int(g_req + f_req + c_req + flx_req) == 10:
            if not state.get("initialized"):
                try:
                    perf_by_span = _load_all_exports(ytd_path)
                    bids_df = _load_draft_prices(DATA_DIR / draft_choice)
                    pool = _build_pool(perf_by_span, bids_df)

                    weights = {"YTD": float(w_ytd), "60": float(w_60), "30": float(w_30), "14": float(w_14), "7": float(w_7)}
                    blended_fpg, blended_gp, value_per_dollar, scores = _compute_scores(pool, weights, str(objective))

                    scores_adj = scores
                    if bool(use_gp):
                        gp = np.maximum(blended_gp, 0.0)
                        rel = gp / (gp + float(gp_smooth) + 1e-9)
                        scores_adj = scores * rel
                    if float(min_gp) > 0:
                        mask = blended_gp >= float(min_gp)
                        scores_adj = np.where(mask, scores_adj, -1e9)

                    elig_sets = [_parse_elig_positions(x) for x in pool.get("Position", "").astype(str).tolist()]
                    bids = pd.to_numeric(pool.get("Bid", 0), errors="coerce").fillna(0).to_numpy(dtype=int)
                    waiver_flags = pd.to_numeric(pool.get("Waiver", False), errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)

                    if not hasattr(creator, "BestDraftFitnessMax"):
                        creator.create("BestDraftFitnessMax", base.Fitness, weights=(1.0,))
                    if not hasattr(creator, "BestDraftIndividual"):
                        creator.create("BestDraftIndividual", list, fitness=creator.BestDraftFitnessMax)

                    rng = np.random.default_rng(int(seed))
                    candidate_idx = list(range(len(pool)))
                    wts = np.maximum(scores_adj, 0.0)
                    if wts.sum() <= 0:
                        wts = np.ones_like(wts)
                    wts = wts / wts.sum()

                    def _init_individual():
                        picks = rng.choice(candidate_idx, size=10, replace=False, p=wts).tolist()
                        return creator.BestDraftIndividual(picks)

                    slots = (["G"] * int(g_req)) + (["F"] * int(f_req)) + (["C"] * int(c_req)) + (["Flx"] * int(flx_req))

                    def _evaluate(individual):
                        if len(set(individual)) != 10:
                            return (-1e9,)
                        cost = int(bids[individual].sum())
                        if cost > int(budget_cap):
                            return (-1e9 - float(cost - int(budget_cap)) * 1000.0,)
                        if int(waiver_flags[individual].sum()) > int(max_waivers):
                            return (-1e9 - float(int(waiver_flags[individual].sum()) - int(max_waivers)) * 10000.0,)
                        sel_elig = [elig_sets[i] for i in individual]
                        if not _roster_feasible(sel_elig, slots):
                            return (-1e9,)
                        return (float(scores_adj[individual].sum()),)

                    def _mutate(individual, indpb=0.2):
                        for i in range(len(individual)):
                            if rng.random() < indpb:
                                existing = set(individual)
                                for _ in range(20):
                                    cand = int(rng.choice(candidate_idx, p=wts))
                                    if cand not in existing:
                                        existing.remove(individual[i])
                                        individual[i] = cand
                                        existing.add(cand)
                                        break
                        return (individual,)

                    toolbox = base.Toolbox()
                    toolbox.register("individual", tools.initIterate, creator.BestDraftIndividual, _init_individual)
                    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                    toolbox.register("evaluate", _evaluate)
                    toolbox.register("mate", tools.cxTwoPoint)
                    toolbox.register("mutate", _mutate)
                    toolbox.register("select", tools.selTournament, tournsize=3)

                    pop = toolbox.population(n=int(population_size))
                    for ind in pop:
                        ind.fitness.values = toolbox.evaluate(ind)

                    state.update(
                        {
                            "initialized": True,
                            "start_time": time.time(),
                            "gen": 0,
                            "pool": pool,
                            "bids": bids,
                            "waiver_flags": waiver_flags,
                            "blended_fpg": blended_fpg,
                            "blended_gp": blended_gp,
                            "value_per_dollar": value_per_dollar,
                            "scores": scores,
                            "scores_adj": scores_adj,
                            "toolbox": toolbox,
                            "pop": pop,
                            "hof": tools.HallOfFame(1),
                            "log_lines": [],
                            "stop_requested": False,
                            "running": True,
                            "budget": int(budget_cap),
                            "time_limit": float(time_limit_s),
                            "generations": int(generations),
                            "print_every": int(print_every),
                            "max_waivers": int(max_waivers),
                            "best_fit": None,
                            "no_improve": 0,
                        }
                    )
                    state["hof"].update(pop)
                except Exception as exc:
                    _state_reset()
                    st.error(f"Optimization failed: {exc}")

            else:
                state["stop_requested"] = False
                state["running"] = True

        if state.get("initialized"):
            elapsed = float(time.time() - float(state.get("start_time", time.time())))
            timer_ph.metric("Elapsed", f"{elapsed:.1f}s")

            done = False
            if bool(state.get("stop_requested")):
                done = True
            if elapsed >= float(state.get("time_limit", 0.0)):
                done = True
            if int(state.get("gen", 0)) >= int(state.get("generations", 0)):
                done = True

            if not done and bool(state.get("running", False)):
                steps = int(chunk_gens)
                did_step = False
                for _ in range(steps):
                    elapsed = float(time.time() - float(state.get("start_time", time.time())))
                    if bool(state.get("stop_requested")) or elapsed >= float(state.get("time_limit", 0.0)) or int(state.get("gen", 0)) >= int(state.get("generations", 0)):
                        break

                    toolbox = state["toolbox"]
                    pop = state["pop"]
                    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.55, mutpb=0.35)
                    fits = map(toolbox.evaluate, offspring)
                    for fit, ind in zip(fits, offspring):
                        ind.fitness.values = fit
                    pop = toolbox.select(offspring, k=len(pop))
                    state["pop"] = pop
                    state["hof"].update(pop)
                    state["gen"] = int(state.get("gen", 0)) + 1
                    did_step = True

                    # Stagnation tracking.
                    if len(state.get("hof", [])):
                        curr_best = float(state["hof"][0].fitness.values[0])
                        prev_best = state.get("best_fit")
                        if prev_best is None or curr_best > float(prev_best) + 1e-9:
                            state["best_fit"] = curr_best
                            state["no_improve"] = 0
                        else:
                            state["no_improve"] = int(state.get("no_improve", 0)) + 1

                    # Escape local optimum if stuck.
                    if int(state.get("no_improve", 0)) >= int(stagnate_gens):
                        try:
                            pool = state["pool"]
                            toolbox = state["toolbox"]
                            pop = state["pop"]
                            bids = state["bids"]
                            waiver_flags = state.get("waiver_flags")
                            scores_adj = state.get("scores_adj")
                            elig_sets = [_parse_elig_positions(x) for x in pool.get("Position", "").astype(str).tolist()]
                            candidate_idx = list(range(len(pool)))
                            slots = (["G"] * int(g_req)) + (["F"] * int(f_req)) + (["C"] * int(c_req)) + (["Flx"] * int(flx_req))

                            if waiver_flags is None:
                                waiver_flags = pd.to_numeric(pool.get("Waiver", False), errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)

                            best = list(state["hof"][0]) if len(state.get("hof", [])) else None
                            if best is not None and scores_adj is not None:
                                improved = _hill_climb_best(
                                    best_ind=best,
                                    candidate_idx=candidate_idx,
                                    scores_adj=scores_adj,
                                    bids=bids,
                                    waiver_flags=waiver_flags,
                                    budget=int(state.get("budget", budget_cap)),
                                    max_waivers=int(state.get("max_waivers", max_waivers)),
                                    elig_sets=elig_sets,
                                    slots=slots,
                                    try_k=int(hillclimb_k),
                                    max_passes=int(hillclimb_passes),
                                )
                                if improved is not None:
                                    improved_fit = toolbox.evaluate(improved)
                                    try:
                                        improved_ind = creator.BestDraftIndividual(improved)
                                        improved_ind.fitness.values = improved_fit
                                        pop[0] = improved_ind
                                    except Exception:
                                        pass

                            # Refresh some of the population to reintroduce randomness.
                            refresh_n = int(max(0, min(len(pop), int(len(pop) * float(diversify_frac)))))
                            if refresh_n > 0:
                                for i in range(1, refresh_n + 1):
                                    ind = toolbox.individual()
                                    ind.fitness.values = toolbox.evaluate(ind)
                                    pop[-i] = ind
                                state["pop"] = pop
                                state["hof"].update(pop)
                                state["log_lines"].append(f"Stagnation escape at gen {int(state.get('gen', 0))}: refreshed {refresh_n} individuals")

                            state["no_improve"] = 0
                            state["best_fit"] = float(state["hof"][0].fitness.values[0]) if len(state.get("hof", [])) else state.get("best_fit")
                        except Exception:
                            state["no_improve"] = 0

                    gen_now = int(state.get("gen", 0))
                    if gen_now % int(state.get("print_every", 25)) == 0:
                        best_fit = float(state["hof"][0].fitness.values[0]) if len(state["hof"]) else float("nan")
                        state["log_lines"].append(f"Gen {gen_now}: best={best_fit:.2f}")

                state["_did_step"] = bool(did_step)

            gen_now = int(state.get("gen", 0))
            gen_total = int(state.get("generations", 1))
            prog.progress(min(1.0, float(gen_now) / float(gen_total)))

            if state.get("log_lines"):
                log_ph.text("\n".join(state["log_lines"][-15:]))

            if len(state.get("hof", [])):
                pool = state["pool"]
                best = list(state["hof"][0])
                out = pool.iloc[best].copy()
                out["BlendedFP/G"] = state["blended_fpg"][best]
                out["BlendedGP"] = state.get("blended_gp", np.zeros(len(pool)))[best]
                out["ValuePer$"] = state["value_per_dollar"][best]
                out["Score"] = state.get("scores_adj", state.get("scores"))[best]
                out["Bid"] = pd.to_numeric(out["Bid"], errors="coerce").fillna(0).astype(int)
                cost = int(out["Bid"].sum())
                score = float(pd.to_numeric(out.get("Score", 0), errors="coerce").fillna(0.0).sum())
                waiver_ct = int(out.get("Waiver", False).sum()) if "Waiver" in out.columns else 0

                m = best_ph.columns(3)
                m[0].metric("Best cost", f"${cost}")
                m[1].metric("Best score", f"{score:.1f}")
                m[2].metric("Waiver pickups", f"{waiver_ct}")

                show_cols = [c for c in ["Player", "Position", "Bid", "Waiver", "BlendedFP/G", "BlendedGP", "ValuePer$", "Score", "FP/G_YTD", "GP_YTD", "FP/G_60", "GP_60", "FP/G_30", "GP_30", "FP/G_14", "GP_14", "FP/G_7", "GP_7"] if c in out.columns]
                dataframe(out[show_cols].sort_values(["Score", "Bid"], ascending=[False, True]).reset_index(drop=True), width="stretch", hide_index=True)

                try:
                    csv_bytes = out[show_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download best team (CSV)",
                        data=csv_bytes,
                        file_name="best_possible_10_man_team.csv",
                        mime="text/csv",
                        key="best_draft_download",
                    )
                except Exception:
                    pass

            if not done and bool(state.get("running", False)) and bool(state.get("_did_step", False)):
                time.sleep(0.25)
                st.rerun()

        if state.get("initialized") and st.button("Reset run", use_container_width=True, key="best_draft_reset"):
            _state_reset()
            st.rerun()


def render_draft_history() -> None:
    st.subheader("📜 Draft History - S1 (2021-22) to S5 (2025-26)")

    st.markdown(
        """
Use this view to explore how draft prices and habits have evolved across seasons.

- Inspect **per-season spend** and **max bids**.
- Track **player price trajectories** year-over-year.
- See how much of a **200-budget** a single player consumed (e.g., 190/200 on Jokic).
- Compare **teams' total spend vs theoretical pots** as league size changes.
"""
    )

    df = load_draft_history()

    if df is None or df.empty:
        st.warning(
            "No draft history found. Make sure S1–S4 draft CSVs and at least one Fantrax draft results CSV exist in the data folder."
        )
        return

    df = df.copy()
    if "Bid" in df.columns:
        df["Bid"] = pd.to_numeric(df["Bid"], errors="coerce")

    df["TeamLabel"] = df["FantasyTeamCanonical"].fillna(df["FantasyTeamRaw"])
    df["TeamLabel"] = df["TeamLabel"].astype(str).str.strip()
    df.loc[df["TeamLabel"].isin(["", "nan", "NaN", "None", "N/A", "(N/A)"]), "TeamLabel"] = pd.NA

    def _norm_text(val) -> str:
        try:
            s = "" if val is None else str(val)
        except Exception:
            s = ""
        s = s.strip().lower()
        s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
        s = s.replace("â", "'").replace("â", "'")
        s = s.replace("â€™", "'").replace("â€˜", "'")
        s = " ".join(s.split())
        return s

    df["TeamLabelNorm"] = df["TeamLabel"].map(_norm_text)
    if "FantasyTeamRaw" in df.columns:
        df["FantasyTeamRawNorm"] = df["FantasyTeamRaw"].map(_norm_text)
    else:
        df["FantasyTeamRawNorm"] = ""
    if "FantasyTeamCanonical" in df.columns:
        df["FantasyTeamCanonicalNorm"] = df["FantasyTeamCanonical"].map(_norm_text)
    else:
        df["FantasyTeamCanonicalNorm"] = ""

    if "Player" in df.columns:
        df["Player"] = df["Player"].astype(str).str.strip()
        df.loc[df["Player"].isin(["", "nan", "NaN"]), "Player"] = pd.NA

    drop_mask = df["Player"].isna() & df["TeamLabel"].isna()
    if "Bid" in df.columns:
        drop_mask &= df["Bid"].isna()
    df = df[~drop_mask].copy()

    season_order = {f"S{i}": i for i in range(1, 10)}
    df["SeasonNum"] = df["SeasonKey"].map(season_order)

    mid_df = load_manager_ids()
    mgr_list = pd.DataFrame()
    if mid_df is not None and not mid_df.empty:
        mgr_list = get_manager_list(mid_df)
    use_manager_ids = (
        mid_df is not None
        and not mid_df.empty
        and mgr_list is not None
        and not mgr_list.empty
    )

    selected_manager_id = None
    manager_seasons_df = pd.DataFrame()
    selected_teams = []

    st.markdown("## Filters & Settings")

    filters_col1, filters_col2, filters_col3 = st.columns([1.2, 1.2, 0.8])

    with filters_col1:
        seasons = sorted(df["SeasonKey"].dropna().unique(), key=lambda k: season_order.get(k, 999))
        selected_seasons = st.multiselect(
            "Seasons",
            options=seasons,
            default=seasons,
            key="draft_history_seasons",
        )

    with filters_col2:
        if use_manager_ids:
            manager_options = mgr_list["label"].tolist()
            if manager_options:
                selected_manager_label = st.selectbox(
                    "Manager",
                    options=manager_options,
                    index=0,
                    key="draft_history_manager",
                )
                selected_row = mgr_list[mgr_list["label"] == selected_manager_label].iloc[0]
                selected_manager_id = selected_row["managerid"]
        else:
            all_teams = sorted(df["TeamLabel"].dropna().unique())
            selected_teams = st.multiselect(
                "Fantasy Teams (canonical)",
                options=all_teams,
                default=all_teams,
                key="draft_history_teams",
            )

    with filters_col3:
        player_query = st.text_input("Player name contains", "", key="draft_history_player_query")
        budget_per_team = st.number_input(
            "Assumed budget per team ($)",
            min_value=1,
            max_value=10000,
            value=200,
            step=10,
            key="draft_history_budget",
        )

    f = df.copy()
    if selected_seasons:
        f = f[f["SeasonKey"].isin(selected_seasons)]

    if use_manager_ids and selected_manager_id is not None:
        manager_seasons_df = mid_df[mid_df["managerid"] == selected_manager_id].copy()
        if not manager_seasons_df.empty:
            team_names = manager_seasons_df["team_name"].map(_norm_text)
            team_abbrs = manager_seasons_df["team_abbreviation"].map(_norm_text)
            team_name_set = set(team_names[team_names.ne("")])
            team_abbr_set = set(team_abbrs[team_abbrs.ne("")])
            if team_name_set or team_abbr_set:
                mask = pd.Series(False, index=f.index)
                if team_name_set:
                    mask = mask | f["FantasyTeamRawNorm"].isin(team_name_set) | f["FantasyTeamCanonicalNorm"].isin(team_name_set) | f["TeamLabelNorm"].isin(team_name_set)
                if team_abbr_set:
                    mask = mask | f["FantasyTeamRawNorm"].isin(team_abbr_set) | f["FantasyTeamCanonicalNorm"].isin(team_abbr_set) | f["TeamLabelNorm"].isin(team_abbr_set)
                f = f[mask]
    else:
        if selected_teams:
            f = f[f["TeamLabel"].isin(selected_teams)]

    if player_query.strip():
        q = player_query.strip().lower()
        f = f[f["Player"].astype(str).str.lower().str.contains(q)]

    if f.empty:
        st.info("No rows match the current filters.")
        return

    if "Bid" in f.columns:
        f["BidPctOfTeamBudget"] = (f["Bid"] / float(budget_per_team)) * 100.0
    else:
        f["BidPctOfTeamBudget"] = pd.NA

    st.markdown("## Season Overview")

    season_summary = (
        f.groupby("SeasonKey")
        .agg(
            TotalSpend=("Bid", "sum"),
            MaxBid=("Bid", "max"),
            AvgBid=("Bid", "mean"),
            PlayersDrafted=("Player", "count"),
            UniqueTeams=("TeamLabel", "nunique"),
        )
        .reset_index()
    )

    season_summary["TheoreticalPot"] = season_summary["UniqueTeams"] * float(budget_per_team)
    season_summary["UtilizationPct"] = (
        season_summary["TotalSpend"] / season_summary["TheoreticalPot"] * 100.0
    ).where(season_summary["TheoreticalPot"] > 0)

    season_summary = season_summary.sort_values("SeasonKey", key=lambda s: s.map(season_order))

    dataframe(
        season_summary.round({"TotalSpend": 1, "MaxBid": 1, "AvgBid": 2, "UtilizationPct": 1}),
        width="stretch",
    )

    if "Bid" in f.columns and f["Bid"].notna().any():
        fig_season = px.bar(
            season_summary,
            x="SeasonKey",
            y="TotalSpend",
            hover_data=["MaxBid", "AvgBid", "PlayersDrafted", "UtilizationPct"],
            title="Total Draft Spend by Season",
            labels={"TotalSpend": "Total Spend ($)", "SeasonKey": "Season"},
        )
        plotly_chart(fig_season, width="stretch")

        clean_bids = f.dropna(subset=["Bid"]).copy()
        if not clean_bids.empty:
            fig_bid_dist = px.box(
                clean_bids,
                x="SeasonKey",
                y="Bid",
                points="outliers",
                title="Bid Distribution by Season",
                labels={"Bid": "Bid ($)", "SeasonKey": "Season"},
            )
            plotly_chart(fig_bid_dist, width="stretch")

    st.markdown("## Player Price Trajectories")

    players_for_select = sorted(f["Player"].dropna().unique())
    selected_player = st.selectbox(
        "Select a player to view YoY draft prices",
        options=[""] + players_for_select,
        index=0,
        key="draft_history_selected_player",
    )

    if selected_player:
        p_df = f[f["Player"] == selected_player].copy()
        p_df = p_df.sort_values("SeasonNum")

        col1, col2 = st.columns([2, 1])

        with col1:
            if "Bid" in p_df.columns and p_df["Bid"].notna().any():
                fig_player = px.line(
                    p_df,
                    x="SeasonKey",
                    y="Bid",
                    markers=True,
                    title=f"{selected_player} - Draft Price by Season",
                    labels={"Bid": "Bid ($)", "SeasonKey": "Season"},
                )
                plotly_chart(fig_player, width="stretch")
            else:
                st.info("No bid data available for this player across the selected seasons.")

        with col2:
            if "BidPctOfTeamBudget" in p_df.columns and p_df["BidPctOfTeamBudget"].notna().any():
                p_df["BidPctOfTeamBudget"] = p_df["BidPctOfTeamBudget"].astype(float)
                st.markdown("**Bid as % of team budget**")
                dataframe(
                    p_df[["SeasonKey", "Bid", "BidPctOfTeamBudget", "FantasyTeamCanonical", "FantasyTeamRaw"]]
                    .rename(columns={"BidPctOfTeamBudget": "% of 200-budget"})
                    .round({"Bid": 1, "% of 200-budget": 1}),
                    width="stretch",
                )
            else:
                st.info("No bid data to compute % of budget.")

    st.markdown("## Biggest Single-Player Splurges")

    if "BidPctOfTeamBudget" in f.columns and f["BidPctOfTeamBudget"].notna().any():
        overpays = (
            f.dropna(subset=["Bid", "BidPctOfTeamBudget"])
            .sort_values("BidPctOfTeamBudget", ascending=False)
            .head(25)
            .copy()
        )
        overpays["BidPctOfTeamBudget"] = overpays["BidPctOfTeamBudget"].astype(float)
        dataframe(
            overpays[["SeasonKey", "Player", "Bid", "BidPctOfTeamBudget", "FantasyTeamCanonical", "FantasyTeamRaw"]]
            .rename(columns={"BidPctOfTeamBudget": "% of 200-budget"})
            .round({"Bid": 1, "% of 200-budget": 1}),
            width="stretch",
        )
    else:
        st.info("No bid data found to compute overpays.")

    st.markdown("## Rosters by Season")

    if use_manager_ids and selected_manager_id is not None:
        roster_df = f.copy()
        if not roster_df.empty:
            roster_df = roster_df.sort_values(["SeasonNum", "Player"])
            seasons_for_roster = roster_df["SeasonKey"].dropna().unique().tolist()
            seasons_for_roster = sorted(seasons_for_roster, key=lambda k: season_order.get(k, 999))
            if seasons_for_roster:
                tabs = st.tabs([f"{sk} Roster" for sk in seasons_for_roster])
                for sk, tab in zip(seasons_for_roster, tabs):
                    with tab:
                        season_slice = roster_df[roster_df["SeasonKey"] == sk]
                        if season_slice.empty:
                            continue
                        cols = ["Player", "Bid", "Pos"]
                        existing_cols = [c for c in cols if c in season_slice.columns]
                        display_roster = season_slice[existing_cols].copy()
                        if "Bid" in display_roster.columns:
                            display_roster["Bid"] = display_roster["Bid"].round(1)
                        dataframe(display_roster, hide_index=True, width="stretch")
    else:
        st.caption("Rosters by season are available when ManagerIDs are configured.")

    st.markdown("## Raw Draft Records (filtered)")

    display_cols = ["SeasonKey", "Player", "TeamLabel"]
    for c in ("Bid", "BidPctOfTeamBudget", "Pick", "Pos", "Time (EST)"):
        if c in f.columns:
            display_cols.append(c)

    display_df = f[display_cols].copy()
    if "BidPctOfTeamBudget" in display_df.columns:
        display_df.rename(columns={"BidPctOfTeamBudget": "% of 200-budget"}, inplace=True)
    if "TeamLabel" in display_df.columns:
        display_df.rename(columns={"TeamLabel": "Team"}, inplace=True)

    dataframe(display_df.round(1), width="stretch")
