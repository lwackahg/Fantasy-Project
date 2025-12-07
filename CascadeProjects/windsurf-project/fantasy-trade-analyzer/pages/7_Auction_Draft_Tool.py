import streamlit as st
import pandas as pd
import numpy as np
import random
from pathlib import Path
from logic.auction_tool import (calculate_initial_values, recalculate_dynamic_values, BASE_VALUE_MODELS, SCARCITY_MODELS, calculate_realistic_price)
from modules.data_preparation import generate_pps_projections
from logic.smart_auction_bot import SmartAuctionBot
from logic.ui_components import (
    render_setup_page, render_sidebar_in_draft, render_value_calculation_expander,
    render_draft_board, render_team_rosters, render_drafting_form, 
    render_player_analysis_metrics, render_draft_summary
)
from modules.team_mappings import TEAM_MAPPINGS

st.set_page_config(layout="wide", page_title="Auction Draft Tool")

# --- Centralized Session State Initialization ---
def initialize_session_state():
    # Draft status and settings
    if 'draft_started' not in st.session_state:
        st.session_state.draft_started = False
    if 'draft_history' not in st.session_state:
        st.session_state.draft_history = []
    if 'teams' not in st.session_state:
        st.session_state.teams = {}
    if 'main_team' not in st.session_state:
        st.session_state.main_team = None
    if 'my_team_name' not in st.session_state:
        st.session_state.my_team_name = None

    # Draft flow control
    if 'current_nominating_team_index' not in st.session_state:
        st.session_state.current_nominating_team_index = 0
    if 'player_on_the_block' not in st.session_state:
        st.session_state.player_on_the_block = None
    if '_reset_player_on_block' not in st.session_state:
        st.session_state._reset_player_on_block = False

    # Other session state variables
    if 'projections_generated' not in st.session_state:
        st.session_state.projections_generated = False
    if 'num_teams' not in st.session_state:
        st.session_state.num_teams = len(TEAM_MAPPINGS)
    if 'budget_per_team' not in st.session_state:
        st.session_state.budget_per_team = 200
    if 'roster_spots_per_team' not in st.session_state:
        st.session_state.roster_spots_per_team = 10
    if 'games_in_season' not in st.session_state:
        st.session_state.games_in_season = 75
    if 'team_names' not in st.session_state:
        st.session_state.team_names = list(TEAM_MAPPINGS.values())[:st.session_state.num_teams]
    if 'available_players' not in st.session_state:
        st.session_state.available_players = pd.DataFrame()
    if 'initial_pos_counts' not in st.session_state:
        st.session_state.initial_pos_counts = pd.Series(dtype='int64')
    if 'initial_tier_counts' not in st.session_state:
        st.session_state.initial_tier_counts = pd.Series(dtype='int64')
    if 'drafted_players' not in st.session_state:
        st.session_state.drafted_players = []
    if 'total_money_spent' not in st.session_state:
        st.session_state.total_money_spent = 0
    if 'base_value_models' not in st.session_state:
        st.session_state.base_value_models = [BASE_VALUE_MODELS[0]]
    if 'scarcity_models' not in st.session_state:
        st.session_state.scarcity_models = [SCARCITY_MODELS[0]]
    if 'tier_cutoffs' not in st.session_state:
        st.session_state.tier_cutoffs = {'Tier 1': 1.00, 'Tier 2': 0.88, 'Tier 3': 0.70, 'Tier 4': 0.45}
    if 'draft_order' not in st.session_state:
        st.session_state.draft_order = []
    if 'trend_weights' not in st.session_state:
        trend_weights = {'S1': 0.00, 'S2': 0.10, 'S3': 0.25, 'S4': 0.65}
        total_weight = sum(trend_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in trend_weights.items()}
        else:
            normalized_weights = trend_weights
        st.session_state.trend_weights = normalized_weights
    if 'roster_composition' not in st.session_state:
        # Use session weights to avoid referencing a local variable that may not exist
        st.session_state.roster_composition = {'G': 3, 'F': 3, 'C': 2, 'Flx': 2, 'Bench': 0}
    # Realistic pricing controls
    if 'realism_enabled' not in st.session_state:
        st.session_state.realism_enabled = True
    if 'realism_aggression' not in st.session_state:
        st.session_state.realism_aggression = 1.0
    # Nomination strategy defaults
    if 'nomination_strategy' not in st.session_state:
        st.session_state.nomination_strategy = 'Blended (recommended)'
    if 'nomination_weights' not in st.session_state:
        st.session_state.nomination_weights = {'budget_pressure': 0.5, 'positional_scarcity': 0.3, 'value_inflation': 0.2}

    if not st.session_state.teams:
        st.session_state.teams = {name: {'budget': st.session_state.budget_per_team, 'players': []} for name in st.session_state.team_names}

    # Load injured players from JSON file, ensuring it runs only once
    if 'injured_players_text' not in st.session_state:
        try:
            from pathlib import Path
            import json
            injured_players_path = Path(__file__).resolve().parent.parent / "data" / "injured_players.json"
            if injured_players_path.exists():
                with open(injured_players_path, 'r') as f:
                    injured_players_dict = json.load(f)
                    injured_text = "\n".join([f"{player} ({status})" for player, status in injured_players_dict.items()])
                    st.session_state.injured_players_text = injured_text
            else:
                st.session_state.injured_players_text = ""
        except Exception:
            st.session_state.injured_players_text = "" # Default to empty if any error occurs

initialize_session_state()

def _parse_injured_players_text(raw_text: str) -> dict:
    """Convert the injured players textarea text into a mapping of name -> status."""
    injuries = {}
    if not raw_text:
        return injuries
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or '(' not in line or ')' not in line:
            continue
        name_part, status_part = line.rsplit('(', 1)
        player_name = name_part.strip()
        status = status_part.replace(')', '').strip()
        if player_name and status:
            injuries[player_name] = status
    return injuries

def _annotate_injury_flags(df: pd.DataFrame, injury_map: dict) -> pd.DataFrame:
    """Attach InjuryStatus/IsInjured/DisplayName columns to a player DataFrame."""
    if df is None or df.empty:
        return df
    out = df.copy()
    # Determine the base player name series
    if 'PlayerName' in out.columns:
        base_names = out['PlayerName'].astype(str)
    else:
        base_names = pd.Series(out.index.astype(str), index=out.index)
    out['InjuryStatus'] = base_names.map(injury_map or {}).fillna('')
    out['IsInjured'] = out['InjuryStatus'].ne('')
    out['DisplayName'] = np.where(out['IsInjured'], base_names + ' (INJ)', base_names)
    return out

def clear_player_on_block():
    """Callback function to reset the player on the block."""
    # Set a flag to reset before widget creation to avoid Streamlit mutation error
    st.session_state._reset_player_on_block = True

def drafting_callback(selected_player_name, team_selection, draft_price, assigned_position, player_values):
    """Handles the logic for drafting a player and then clearing the selection."""
    # Server-side safety: enforce max bid = current budget - $1 per remaining roster spot after this pick
    team_data = st.session_state.teams[team_selection]
    team_budget = team_data['budget']
    rostered = len(team_data.get('players', []))
    total_spots = st.session_state.get('roster_spots_per_team', 0)
    spots_remaining_now = max(total_spots - rostered, 0)
    spots_after_this_pick = max(spots_remaining_now - 1, 0)
    reserve_for_min_bids = spots_after_this_pick
    max_allowed_bid = max(0, team_budget - reserve_for_min_bids)

    if draft_price > max_allowed_bid:
        st.error(
            f"Bid exceeds max allowed ${max_allowed_bid}. You must keep $1 for each of the {spots_after_this_pick} remaining spots after this pick."
        )
        return

    if draft_price > st.session_state.teams[team_selection]['budget']:
        st.error(f"{team_selection} cannot afford {selected_player_name} at this price.")
        return
    # Roster capacity enforcement (including Flex/Bench)
    req = {k: int(v) for k, v in st.session_state.roster_composition.items()}
    current_players = st.session_state.teams[team_selection].get('players', [])
    filled = pd.Series([p.get('Position', '') for p in current_players]).value_counts().to_dict() if current_players else {}
    def remaining(pos):
        return max(0, int(req.get(pos, 0)) - int(filled.get(pos, 0)))

    # If the assigned core slot is full, auto-fallback to Flex if available and player can play Flex
    fallback_used = False
    if assigned_position not in ('Flx', 'Bench') and remaining(assigned_position) <= 0:
        if req.get('Flx', 0) > 0 and remaining('Flx') > 0:
            assigned_position = 'Flx'
            fallback_used = True
        else:
            st.error(f"No available {assigned_position} or Flex slots left for {team_selection}.")
            return

    if assigned_position == 'Flx' and remaining('Flx') <= 0:
        st.error(f"No available Flex slots left for {team_selection}.")
        return
    if assigned_position == 'Bench' and (req.get('Bench', 0) <= 0 or remaining('Bench') <= 0):
        st.error(f"No available Bench slots left for {team_selection}.")
        return

    # Deduct from budget and add to team roster
    st.session_state.teams[team_selection]['budget'] -= draft_price
    # Ensure we expand a mapping, not a raw Series, into the roster record
    extra_attrs = {}
    try:
        if hasattr(player_values, 'to_dict'):
            extra_attrs = player_values.to_dict()
        elif hasattr(player_values, 'keys'):
            extra_attrs = dict(player_values)
    except Exception:
        extra_attrs = {}

    player_data = {
        'Player': selected_player_name,
        'Price': draft_price,
        **extra_attrs
    }
    # Ensure assigned position overrides any 'Position' coming from extra_attrs
    player_data['Position'] = assigned_position
    st.session_state.teams[team_selection]['players'].append(player_data)

    # Add to draft history
    draft_entry = {
        'Player': selected_player_name,
        'Team': team_selection,
        'Price': draft_price,
        'Position': assigned_position
    }
    st.session_state.draft_history.append(draft_entry)

    # Notify if auto-assigned to Flex
    if fallback_used:
        st.info("Selected slot was full; assigned to Flex instead.")

    # Update total money spent
    if 'total_money_spent' not in st.session_state:
        st.session_state.total_money_spent = 0
    st.session_state.total_money_spent += draft_price

    # Maintain drafted_players list used by summaries and recalculation logic
    base_val = player_values.get('BaseValue', 0) if hasattr(player_values, 'get') else 0
    adj_val = player_values.get('AdjValue', 0) if hasattr(player_values, 'get') else 0
    drafted_entry = {
        'PlayerName': selected_player_name,
        'Team': team_selection,
        'DraftPrice': draft_price,
        'Position': assigned_position,
        'BaseValue': base_val,
        'AdjValue': adj_val
    }
    st.session_state.drafted_players.append(drafted_entry)

    # Cache the removed player's row for potential Undo action
    try:
        if 'removed_player_rows' not in st.session_state:
            st.session_state.removed_player_rows = {}
        row_dict = dict(player_values) if hasattr(player_values, 'keys') else {}
        if 'PlayerName' not in row_dict:
            row_dict['PlayerName'] = selected_player_name
        st.session_state.removed_player_rows[selected_player_name] = row_dict
    except Exception:
        pass

    # Remove drafted player from available players so they disappear from lists
    if (not st.session_state.available_players.empty) and ('PlayerName' in st.session_state.available_players.columns):
        st.session_state.available_players = st.session_state.available_players[
            st.session_state.available_players['PlayerName'] != selected_player_name
        ]

    # Advance to the next team in the draft order (guard against empty order)
    if st.session_state.draft_order:
        st.session_state.current_nominating_team_index = (
            st.session_state.current_nominating_team_index + 1
        ) % len(st.session_state.draft_order)

    # Clear the player on the block after drafting
    clear_player_on_block()
    # Trigger a rerun so the reset happens before the selectbox is created
    st.rerun()

# --- Main App Flow ---
if not st.session_state.draft_started:
    render_setup_page()

    st.markdown("---")
    st.header("Set Draft Order")
    # Reset support to avoid preloaded selections
    if 'draft_order_version' not in st.session_state:
        st.session_state.draft_order_version = 0
    reset_cols = st.columns([2,1,1])
    with reset_cols[1]:
        if st.button("Reset Order"):
            st.session_state.draft_order = []
            st.session_state.draft_order_version += 1
            st.rerun()
    with reset_cols[2]:
        if st.button("Random Order"):
            n = len(st.session_state.team_names)
            st.session_state.draft_order = random.sample(st.session_state.team_names, n)
            st.session_state.draft_order_version += 1
            st.rerun()

    cols = st.columns([1, 3])
    with cols[0]:
        st.write("Slot")
    with cols[1]:
        st.write("Team")
    order = st.session_state.get('draft_order', [])
    n = len(st.session_state.team_names)
    if not order or len(order) != n:
        order = [None] * n
    chosen = set([x for x in order if x])
    for i in range(n):
        remaining = [t for t in st.session_state.team_names if (t not in chosen) or (order[i] == t)]
        c1, c2 = st.columns([1, 3])
        with c1:
            st.write(f"{i+1}")
        with c2:
            opts = ["-- Select --"] + remaining
            current = order[i] if (order[i] in remaining) else "-- Select --"
            idx = opts.index(current) if current in opts else 0
            sel = st.selectbox(
                label=f"Select Team for Slot {i+1}",
                options=opts,
                index=idx,
                key=f"draft_order_slot_{i}_{st.session_state.draft_order_version}",
                label_visibility="collapsed"
            ) if opts else "-- Select --"
            order[i] = (sel if sel != "-- Select --" else None)
        chosen = set([x for x in order if x])
    st.session_state.draft_order = [x for x in order if x]

    st.markdown("---")
    st.header("Actions")
    col1, col2 = st.columns(2)

    with col1:
        if not st.session_state.projections_generated:
            if st.button("Generate Projections", use_container_width=True):
                with st.spinner("Generating PPS projections..."):
                    success = generate_pps_projections(trend_weights=st.session_state.trend_weights, games_in_season=st.session_state.games_in_season)
                    if success:
                        st.session_state.projections_generated = True
                        st.success("Projections generated!")
                        st.rerun()
                    else:
                        st.error("Failed to generate projections.")
        else:
            st.success("Projections have been generated.")
            sources = st.session_state.get("auction_data_sources", {})
            if sources:
                error_msg = sources.get("error")
                if error_msg:
                    st.error(f"Auction data sources: {error_msg}")
                else:
                    fp_src = sources.get("pps_fp_gp_source")
                    mv_src = sources.get("pps_market_value_source")
                    parts = []
                    if fp_src == "historical_ytd":
                        parts.append("FP/G & GP from historical YTD (DB-backed)")
                    elif fp_src == "csv":
                        parts.append("FP/G & GP from legacy CSVs")
                    if mv_src == "draft_history":
                        parts.append("Market values from draft history (bids)")
                    elif mv_src == "bids_csv":
                        parts.append("Market values from legacy bid CSV")
                    if parts:
                        text = "; ".join(parts)
                        if fp_src == "csv" or mv_src == "bids_csv":
                            st.warning(f"Auction inputs: {text}")
                        else:
                            st.caption(f"Auction inputs: {text}")

    with col2:
        if st.session_state.projections_generated:
            if st.button("Start Draft", type="primary", use_container_width=True):
                st.session_state.draft_started = True
                injured_players = {}
                text = st.session_state.get('injured_players_text', '').strip()
                if text:
                    for line in text.split('\n'):
                        line = line.strip()
                        if '(' in line and ')' in line:
                            parts = line.split('(')
                            player_name = parts[0].strip()
                            status = parts[1].replace(')', '').strip()
                            injured_players[player_name] = status
                # Save for later display usage
                st.session_state.injury_map = injured_players
                
                with st.spinner("Calculating initial player values..."):
                    data_dir = Path(__file__).resolve().parent.parent / "data"
                    auction_dir = data_dir / "auction"
                    proj_path = auction_dir / 'player_projections.csv'
                    if not proj_path.exists():
                        proj_path = data_dir / 'player_projections.csv'
                    pps_df = pd.read_csv(proj_path)
                    st.session_state.available_players, st.session_state.initial_pos_counts, st.session_state.initial_tier_counts = calculate_initial_values(
                        pps_df=pps_df,
                        num_teams=st.session_state.num_teams,
                        budget_per_team=st.session_state.budget_per_team,
                        roster_composition=st.session_state.roster_composition,
                        base_value_models=st.session_state.base_value_models,
                        tier_cutoffs=st.session_state.tier_cutoffs,
                        injured_players=injured_players,
                        ec_settings=(st.session_state.get('ec_settings', None) if st.session_state.get('ec_enabled', True) else None),
                        gp_rel_settings=st.session_state.get('gp_rel_settings', None)
                    )
                    # Annotate for UI so injury is obvious
                    st.session_state.available_players = _annotate_injury_flags(
                        st.session_state.available_players, st.session_state.get('injury_map', {})
                    )
                st.rerun()
        else:
            st.info("Generate projections before you can start the draft.")

else: # Draft is in progress
    # Process any pending reset BEFORE any widgets that use 'player_on_the_block' are created
    if st.session_state.get('_reset_player_on_block', False):
        st.session_state.player_on_the_block = None
        st.session_state._reset_player_on_block = False

    render_sidebar_in_draft()

    # Realistic pricing controls (inline for now)
    with st.expander("Realistic Pricing Settings", expanded=False):
        st.session_state.realism_enabled = st.checkbox("Show Realistic Price (with range)", value=st.session_state.get('realism_enabled', True))
        st.session_state.realism_aggression = st.slider("Aggression (higher = wilder bidding wars)", min_value=0.5, max_value=1.5, value=float(st.session_state.get('realism_aggression', 1.0)), step=0.05)

    # --- Value Recalculation and Bot Logic ---
    recalculated_df = pd.DataFrame()
    if not st.session_state.available_players.empty:
        total_league_money = st.session_state.num_teams * st.session_state.budget_per_team
        total_money_spent = sum(p['DraftPrice'] for p in st.session_state.drafted_players)
        remaining_money_pool = total_league_money - total_money_spent

        recalculated_df = recalculate_dynamic_values(
            available_players_df=st.session_state.available_players.copy(),
            remaining_money_pool=remaining_money_pool,
            total_league_money=total_league_money,
            base_value_models=st.session_state.base_value_models,
            scarcity_models=st.session_state.scarcity_models,
            initial_tier_counts=st.session_state.initial_tier_counts,
            initial_pos_counts=st.session_state.initial_pos_counts,
            teams_data=st.session_state.teams,
            tier_cutoffs=st.session_state.tier_cutoffs,
            roster_composition=st.session_state.roster_composition,
            num_teams=st.session_state.num_teams
        )
        # Carry injury flags forward for UI
        recalculated_df = _annotate_injury_flags(recalculated_df, st.session_state.get('injury_map', {}))

        # Compute Realistic Price + ranges if enabled
        try:
            if st.session_state.get('realism_enabled', True):
                rp = calculate_realistic_price(
                    recalculated_df,
                    num_teams=st.session_state.num_teams,
                    budget_per_team=st.session_state.budget_per_team,
                    teams_data=st.session_state.teams,
                    roster_composition=st.session_state.roster_composition,
                    aggression=float(st.session_state.get('realism_aggression', 1.0)),
                    rng_seed=None,
                )
                recalculated_df['RealisticPrice'] = rp
                # Bands are attached by function as RealisticLow/RealisticHigh if possible
                if 'RealisticLow' not in recalculated_df.columns:
                    recalculated_df['RealisticLow'] = np.nan
                if 'RealisticHigh' not in recalculated_df.columns:
                    recalculated_df['RealisticHigh'] = np.nan
        except Exception:
            pass

    # --- Main 3-Column Layout ---
    left_col, mid_col, right_col = st.columns([1.2, 2, 1])

    with left_col:
        nominating_team_name = st.session_state.draft_order[st.session_state.current_nominating_team_index] if st.session_state.draft_order else 'N/A'
        # Compute draft phase from budget depletion ratio
        try:
            total_initial_budget = sum(d['budget'] + sum(p.get('Price', 0) for p in d.get('players', [])) for d in st.session_state.teams.values())
            current_budget = sum(d['budget'] for d in st.session_state.teams.values())
            depletion = 1 - (current_budget / total_initial_budget) if total_initial_budget > 0 else 0.0
        except Exception:
            depletion = 0.0
        phase = 'Early'
        if depletion >= 0.33 and depletion < 0.66:
            phase = 'Mid'
        elif depletion >= 0.66:
            phase = 'Late'
        st.subheader(f"On the Clock: {nominating_team_name}  ")
        st.caption(f"Phase: {phase} (budget used: {depletion:.0%})")

        st.subheader("Analyze or Draft")
        player_options = []
        if not recalculated_df.empty:
            if recalculated_df.index.name != 'PlayerName':
                recalculated_df.set_index('PlayerName', inplace=True)
            player_options = recalculated_df.index.tolist()

        if 'player_on_the_block' not in st.session_state:
            st.session_state.player_on_the_block = None

        # Use the widget key to manage selection; avoid recomputing index to reduce lag
        def _player_label(name: str) -> str:
            try:
                if name in recalculated_df.index and 'DisplayName' in recalculated_df.columns:
                    return str(recalculated_df.loc[name, 'DisplayName'])
            except Exception:
                pass
            return name

        st.selectbox(
            "Select a player to analyze or draft:",
            options=player_options,
            key='player_on_the_block',
            placeholder="Choose a player...",
            format_func=_player_label
        )

        if st.session_state.get('player_on_the_block') and not recalculated_df.empty:
            player_series = recalculated_df.loc[st.session_state.player_on_the_block]
            render_drafting_form(player_series, drafting_callback)

    with mid_col:
        render_team_rosters()

    # --- Bot Recommendations (calculated once per run) ---
    drafted_player_names = [p['Player'] for p in st.session_state.draft_history]
    undrafted_df = recalculated_df[~recalculated_df.index.isin(drafted_player_names)]

    weights = st.session_state.get('nomination_weights', {'budget_pressure': 0.5, 'positional_scarcity': 0.3, 'value_inflation': 0.2})
    bot = SmartAuctionBot(
        my_team_name=st.session_state.main_team,
        teams=st.session_state.teams,
        available_players=undrafted_df,
        draft_history=st.session_state.draft_history,
        budget=st.session_state.budget_per_team,
        roster_composition=st.session_state.roster_composition
    )
    top5_recommendations = bot.get_nomination_recommendation(weights=weights)

    # --- Right Column: Nomination Targets and Draft History ---
    with right_col:
        st.subheader("Top 5 Nomination Targets")
        if top5_recommendations and top5_recommendations[0].get('player') is not None:
            # Filter out players with Full Season injuries; augment labels for others
            filtered = []
            for rec in top5_recommendations:
                name = rec.get('player')
                try:
                    inj = str(recalculated_df.loc[name, 'InjuryStatus']) if (name in recalculated_df.index and 'InjuryStatus' in recalculated_df.columns) else ''
                except Exception:
                    inj = ''
                if inj.strip().lower() == 'full season':
                    continue
                rec = dict(rec)
                rec['injury_status'] = inj
                filtered.append(rec)

            for i, rec in enumerate(filtered):
                worth_whole = int(round(rec.get('adj_value', 0)))
                target_whole = int(rec.get('target_nom_price', 0))
                label = rec['player']
                try:
                    if rec['player'] in recalculated_df.index and 'DisplayName' in recalculated_df.columns:
                        label = str(recalculated_df.loc[rec['player'], 'DisplayName'])
                except Exception:
                    pass
                if rec.get('injury_status'):
                    label = f"{label} (Inj: {rec['injury_status']})"
                st.markdown(f"**{i+1}. {label}** – Worth: ${worth_whole:,} | Target: ${target_whole:,}")
                # Plain-English one-liner
                det = rec.get('details', {})
                bp = det.get('budget_pressure', {})
                sc = det.get('positional_scarcity', {})
                inf = det.get('value_inflation', {})
                needy = bp.get('needy_count', 0)
                avg_bud = bp.get('avg_budget', 0)
                pos = sc.get('position', None) or 'this position'
                st_tier = sc.get('tier', None)
                tier_label = f"Tier {int(st_tier)}" if isinstance(st_tier, (int, float)) else "this tier"
                same_tier_left = sc.get('same_tier_count', 0)
                infl_pct = inf.get('inflation_ratio', 0)
                st.caption(f"Why: {needy} team(s) still need {pos}; {tier_label} left: {same_tier_left}; likely overpay ~{infl_pct:.0%}; avg needy budget ~${avg_bud:,.0f}.")
                # Secondary explanation for why this target amount
                explain_text = rec.get('explain')
                if explain_text:
                    with st.expander("Why this target?", expanded=False):
                        st.markdown(explain_text)
        else:
            st.write("No recommendations available at this time.")

    st.markdown("---_**Player Analysis**_---")
    if st.session_state.get('player_on_the_block') and not recalculated_df.empty:
        render_player_analysis_metrics(st.session_state.player_on_the_block, recalculated_df)
    else:
        st.info("Select a player from the dropdown above to see detailed analysis.")

    st.markdown("---_**Draft Summary**_---")
    render_draft_summary()

    # --- Model Accuracy (Live Residuals) ---
    top_help_cols = st.columns([1, 0.06])
    with top_help_cols[0]:
        st.markdown("**Model Accuracy (Live)**")
    with top_help_cols[1]:
        st.button("ℹ️", key="model_accuracy_help", help=(
            "Shows how close the current model is to actual paid prices during this draft.\n\n"
            "ME = Mean Error (Paid - Model). Positive means the room is paying more than the model.\n"
            "MAE = Mean Absolute Error. Lower is better.\n"
            "Calibration suggestions are preview-only multipliers (capped ±10%) to align with observed prices."
        ))

    with st.expander("Details", expanded=False):
        # Outlier handling controls
        st.markdown("**Outlier handling**")
        method = st.selectbox(
            "Exclude outliers by",
            options=["None", "IQR (1.5x)", "IQR (3.0x)", "Absolute Error > $X"],
            index=0,
            key="resid_outlier_method"
        )
        abs_x = 50
        if method == "Absolute Error > $X":
            abs_x = st.number_input("X ($)", min_value=1, max_value=1000, value=100, step=5, key="resid_abs_x")
        try:
            drafted = pd.DataFrame(st.session_state.get('drafted_players', []))
            if not drafted.empty:
                # Attach cached features (Tier/Position/etc.) from when the player was drafted
                removed_cache = st.session_state.get('removed_player_rows', {})
                cache_df = pd.DataFrame(list(removed_cache.values())) if removed_cache else pd.DataFrame()
                if not cache_df.empty:
                    if 'PlayerName' not in cache_df.columns and 'Player' in cache_df.columns:
                        cache_df.rename(columns={'Player': 'PlayerName'}, inplace=True)
                    # Only keep relevant columns to avoid collisions
                    cache_keep = [c for c in cache_df.columns if c in ['PlayerName', 'Tier', 'Position']]
                    cache_df = cache_df[cache_keep].copy()
                    drafted = drafted.merge(cache_df, on='PlayerName', how='left')

                # Compute residuals vs AdjValue and BaseValue at time of draft
                drafted['Error_Adj'] = pd.to_numeric(drafted.get('DraftPrice', 0), errors='coerce') - pd.to_numeric(drafted.get('AdjValue', 0), errors='coerce')
                drafted['AbsError_Adj'] = drafted['Error_Adj'].abs()
                drafted['Error_Base'] = pd.to_numeric(drafted.get('DraftPrice', 0), errors='coerce') - pd.to_numeric(drafted.get('BaseValue', 0), errors='coerce')
                drafted['AbsError_Base'] = drafted['Error_Base'].abs()

                # Outlier filtering mask based on selection
                mask = pd.Series(True, index=drafted.index)
                if method.startswith("IQR"):
                    k = 1.5 if "1.5" in method else 3.0
                    q1 = drafted['Error_Adj'].quantile(0.25)
                    q3 = drafted['Error_Adj'].quantile(0.75)
                    iqr = q3 - q1
                    low, high = q1 - k * iqr, q3 + k * iqr
                    mask = (drafted['Error_Adj'] >= low) & (drafted['Error_Adj'] <= high)
                elif method == "Absolute Error > $X":
                    mask = drafted['AbsError_Adj'] <= abs_x
                drafted_f = drafted[mask].copy()

                # Overall metrics
                overall = {
                    'Count': len(drafted_f),
                    'Adj ME ($)': round(drafted_f['Error_Adj'].mean(), 2),
                    'Adj MAE ($)': round(drafted_f['AbsError_Adj'].mean(), 2),
                    'Base ME ($)': round(drafted_f['Error_Base'].mean(), 2),
                    'Base MAE ($)': round(drafted_f['AbsError_Base'].mean(), 2),
                }
                st.markdown("**Overall**")
                st.dataframe(pd.DataFrame([overall]), hide_index=True, width="stretch")

                # By Tier
                if 'Tier' in drafted_f.columns:
                    by_tier = drafted_f.groupby('Tier').agg(
                        Count=('PlayerName', 'count'),
                        Adj_ME=('Error_Adj', 'mean'),
                        Adj_MAE=('AbsError_Adj', 'mean'),
                        Base_ME=('Error_Base', 'mean'),
                        Base_MAE=('AbsError_Base', 'mean'),
                    ).reset_index()
                    by_tier[['Adj_ME','Adj_MAE','Base_ME','Base_MAE']] = by_tier[['Adj_ME','Adj_MAE','Base_ME','Base_MAE']].round(2)
                    st.markdown("**By Tier**")
                    st.dataframe(by_tier, hide_index=True, width="stretch")

                    # Suggest non-destructive calibration multipliers by Tier (not applied)
                    try:
                        tier_avg_adj = drafted_f.groupby('Tier')['AdjValue'].mean().rename('AdjMean')
                        tier_me = drafted_f.groupby('Tier')['Error_Adj'].mean().rename('AdjME')
                        tier_sugg = pd.concat([tier_avg_adj, tier_me], axis=1).dropna()
                        # multiplier ≈ 1 - ME/AdjMean, clamped to [0.90, 1.10]
                        tier_sugg['SuggestedMultiplier'] = (1.0 - (tier_sugg['AdjME'] / tier_sugg['AdjMean']).replace([np.inf, -np.inf], 0)).clip(0.90, 1.10)
                        tier_sugg['SuggestedMultiplierPct'] = ((tier_sugg['SuggestedMultiplier'] - 1.0) * 100).round(1)
                        tier_sugg = tier_sugg.reset_index()[['Tier','AdjMean','AdjME','SuggestedMultiplier','SuggestedMultiplierPct']]
                        st.markdown("**Calibration suggestions (by Tier)** – informational only, not applied")
                        st.dataframe(tier_sugg, hide_index=True, width="stretch")
                    except Exception:
                        pass

                # By Position
                if 'Position' in drafted_f.columns:
                    by_pos = drafted_f.groupby('Position').agg(
                        Count=('PlayerName', 'count'),
                        Adj_ME=('Error_Adj', 'mean'),
                        Adj_MAE=('AbsError_Adj', 'mean'),
                        Base_ME=('Error_Base', 'mean'),
                        Base_MAE=('AbsError_Base', 'mean'),
                    ).reset_index()
                    by_pos[['Adj_ME','Adj_MAE','Base_ME','Base_MAE']] = by_pos[['Adj_ME','Adj_MAE','Base_ME','Base_MAE']].round(2)
                    st.markdown("**By Position**")
                    st.dataframe(by_pos, hide_index=True, width="stretch")

                    # Suggest non-destructive calibration multipliers by Position (not applied)
                    try:
                        pos_avg_adj = drafted_f.groupby('Position')['AdjValue'].mean().rename('AdjMean')
                        pos_me = drafted_f.groupby('Position')['Error_Adj'].mean().rename('AdjME')
                        pos_sugg = pd.concat([pos_avg_adj, pos_me], axis=1).dropna()
                        pos_sugg['SuggestedMultiplier'] = (1.0 - (pos_sugg['AdjME'] / pos_sugg['AdjMean']).replace([np.inf, -np.inf], 0)).clip(0.90, 1.10)
                        pos_sugg['SuggestedMultiplierPct'] = ((pos_sugg['SuggestedMultiplier'] - 1.0) * 100).round(1)
                        pos_sugg = pos_sugg.reset_index()[['Position','AdjMean','AdjME','SuggestedMultiplier','SuggestedMultiplierPct']]
                        st.markdown("**Calibration suggestions (by Position)** – informational only, not applied")
                        st.dataframe(pos_sugg, hide_index=True, width="stretch")
                    except Exception:
                        pass

                # Visualizations
                try:
                    # Ensure pick numbers
                    if 'PickNumber' not in drafted.columns:
                        drafted['PickNumber'] = range(1, len(drafted) + 1)
                    # Scatter of residuals and cumulative MAE
                    vis_cols = st.columns(2)
                    with vis_cols[0]:
                        st.markdown("Residuals by pick (Adj)")
                        st.line_chart(drafted.set_index('PickNumber')['Error_Adj'])
                    with vis_cols[1]:
                        st.markdown("Cumulative MAE (Adj)")
                        cum_mae = drafted['AbsError_Adj'].expanding().mean()
                        st.line_chart(pd.DataFrame({ 'Cumulative_MAE': cum_mae }).reset_index(drop=True))
                except Exception:
                    pass

                # Download residuals
                try:
                    csv_bytes = drafted.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download residuals (CSV)",
                        data=csv_bytes,
                        file_name="auction_model_residuals.csv",
                        mime="text/csv"
                    )
                    # Draft log bundle (JSON)
                    try:
                        bundle = {
                            'settings': {
                                'num_teams': st.session_state.num_teams,
                                'budget_per_team': st.session_state.budget_per_team,
                                'roster_composition': st.session_state.roster_composition,
                                'base_value_models': st.session_state.base_value_models,
                                'scarcity_models': st.session_state.scarcity_models,
                            },
                            'drafted_players': st.session_state.get('drafted_players', []),
                            'residuals': drafted.to_dict(orient='records')
                        }
                        import json as _json
                        st.download_button(
                            label="Download draft log (JSON)",
                            data=_json.dumps(bundle, indent=2),
                            file_name="auction_draft_log.json",
                            mime="application/json"
                        )
                    except Exception:
                        pass
                    # Optional: download calibration suggestions bundle
                    try:
                        calib_bundle = {}
                        if 'tier_sugg' in locals():
                            calib_bundle['tier'] = tier_sugg.to_dict(orient='records')
                        if 'pos_sugg' in locals():
                            calib_bundle['position'] = pos_sugg.to_dict(orient='records')
                        if calib_bundle:
                            import json as _json
                            st.download_button(
                                label="Download calibration suggestions (JSON)",
                                data=_json.dumps(calib_bundle, indent=2),
                                file_name="auction_calibration_suggestions.json",
                                mime="application/json"
                            )
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                st.info("No drafted players yet. Residuals will appear after picks are made.")
        except Exception as e:
            st.warning(f"Could not compute residuals: {e}")

    # --- All Player Values Table (based on selected models) ---
    st.markdown("---_**All Player Values**_---")
    if not recalculated_df.empty:
        display_df = recalculated_df.copy()
        if display_df.index.name == 'PlayerName':
            display_df = display_df.reset_index()
        # Ensure injury flags present for table; show DisplayName if available
        display_df = _annotate_injury_flags(display_df, st.session_state.get('injury_map', {}))
        if 'DisplayName' in display_df.columns:
            display_df['PlayerName'] = display_df['DisplayName']

        # Compute average scarcity premium (from SP_* columns) for currently selected scarcity models only
        try:
            def _col_name(model_name, prefix):
                clean = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
                return f"{prefix}{clean}"
            selected_scarcity = [m for m in st.session_state.get('scarcity_models', []) if m != 'No Scarcity Adjustment']
            selected_sp_cols = [_col_name(m, 'SP_') for m in selected_scarcity]
            sp_cols_present = [c for c in selected_sp_cols if c in display_df.columns]
            if sp_cols_present:
                display_df['AvgScarcityPremiumPct'] = (display_df[sp_cols_present].mean(axis=1) * 100).round(0)
        except Exception:
            pass

        # Curated, readable columns only (Adj first for emphasis). Include Injury column for clarity.
        wanted_cols = [
            'PlayerName', 'Position', 'Tier', 'PPS', 'AdjValue', 'BaseValue',
            'RealisticPrice', 'RealisticLow', 'RealisticHigh',
            'MarketValue', 'Confidence', 'AvgScarcityPremiumPct', 'InjuryStatus',
            'S4_FP/G', 'S4_GP'
        ]
        cols_present = [c for c in wanted_cols if c in display_df.columns]
        display_df = display_df[cols_present].copy()

        # Friendly labels and light formatting (keep numeric for sorting)
        rename_map = {
            'PlayerName': 'Player',
            'Position': 'Pos',
            'Tier': 'Tier',
            'PPS': 'PPS',
            'BaseValue': 'Base $',
            'AdjValue': 'Adj $',
            'RealisticPrice': 'Realistic $',
            'RealisticLow': 'R Low',
            'RealisticHigh': 'R High',
            'MarketValue': 'Market $',
            'Confidence': 'Conf %',
            'AvgScarcityPremiumPct': 'Scarcity %',
            'InjuryStatus': 'Injury',
            'S4_FP/G': 'S4 FP/G',
            'S4_GP': 'S4 GP'
        }

        # Round key numeric columns if present
        for c in ['BaseValue', 'AdjValue', 'MarketValue', 'RealisticPrice', 'RealisticLow', 'RealisticHigh']:
            if c in display_df.columns:
                display_df[c] = pd.to_numeric(display_df[c], errors='coerce').fillna(0).round(0).astype(int)
        if 'PPS' in display_df.columns:
            display_df['PPS'] = pd.to_numeric(display_df['PPS'], errors='coerce').fillna(0).round(2)
        if 'Confidence' in display_df.columns:
            display_df['Confidence'] = pd.to_numeric(display_df['Confidence'], errors='coerce').fillna(0).round(1)
        # Last season stat columns
        if 'S4_FP/G' in display_df.columns:
            display_df['S4_FP/G'] = pd.to_numeric(display_df['S4_FP/G'], errors='coerce').round(1)
        if 'S4_GP' in display_df.columns:
            display_df['S4_GP'] = pd.to_numeric(display_df['S4_GP'], errors='coerce').fillna(0).astype(int)

        # Sort by Adj $ desc if available
        sort_col = 'AdjValue' if 'AdjValue' in display_df.columns else (cols_present[0] if cols_present else None)
        if sort_col:
            display_df = display_df.sort_values(sort_col, ascending=False)

        display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
        # Style for readability while keeping underlying dtypes numeric for CSV/sort
        try:
            style_formats = {}
            if 'Adj $' in display_df.columns:
                style_formats['Adj $'] = '${:,.0f}'
            if 'Base $' in display_df.columns:
                style_formats['Base $'] = '${:,.0f}'
            if 'Market $' in display_df.columns:
                style_formats['Market $'] = '${:,.0f}'
            if 'Realistic $' in display_df.columns:
                style_formats['Realistic $'] = '${:,.0f}'
            if 'R Low' in display_df.columns:
                style_formats['R Low'] = '${:,.0f}'
            if 'R High' in display_df.columns:
                style_formats['R High'] = '${:,.0f}'
            if 'Conf %' in display_df.columns:
                style_formats['Conf %'] = '{:.1f}%'
            if 'Scarcity %' in display_df.columns:
                style_formats['Scarcity %'] = '{:.0f}%'

            def _injury_row_style(row):
                try:
                    if str(row.get('Injury', '')).strip() != '':
                        return ['background-color: rgba(255, 0, 0, 0.10)'] * len(row)
                except Exception:
                    pass
                return [''] * len(row)

            styled = display_df.style.format(style_formats).apply(_injury_row_style, axis=1)
            st.dataframe(styled, width="stretch", hide_index=True)
        except Exception:
            st.dataframe(display_df, width="stretch", hide_index=True)

        # Download current curated values
        try:
            csv_bytes = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download current player values (CSV)",
                data=csv_bytes,
                file_name="player_values_current.csv",
                mime="text/csv"
            )
        except Exception:
            pass

        # Optionally show drafted players' last-known values from cache (curated to same columns)
        removed_cache = st.session_state.get('removed_player_rows', {})
        if removed_cache:
            with st.expander("Show drafted players' last-known values", expanded=False):
                drafted_vals = pd.DataFrame(list(removed_cache.values()))
                if not drafted_vals.empty:
                    if drafted_vals.index.name == 'PlayerName':
                        drafted_vals = drafted_vals.reset_index()
                    if 'PlayerName' not in drafted_vals.columns and 'Player' in drafted_vals.columns:
                        drafted_vals.rename(columns={'Player': 'PlayerName'}, inplace=True)

                    # Compute average scarcity premium for drafted cache too (only selected models)
                    try:
                        def _col_name_d(model_name, prefix):
                            clean = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
                            return f"{prefix}{clean}"
                        selected_scarcity_d = [m for m in st.session_state.get('scarcity_models', []) if m != 'No Scarcity Adjustment']
                        selected_sp_cols_d = [_col_name_d(m, 'SP_') for m in selected_scarcity_d]
                        sp_cols_present_d = [c for c in selected_sp_cols_d if c in drafted_vals.columns]
                        if sp_cols_present_d:
                            drafted_vals['AvgScarcityPremiumPct'] = (drafted_vals[sp_cols_present_d].mean(axis=1) * 100).round(0)
                    except Exception:
                        pass

                    cur_cols = [c for c in wanted_cols if c in drafted_vals.columns]
                    drafted_vals = drafted_vals[cur_cols].copy()

                    # Apply same rounding and renaming
                    for c in ['BaseValue', 'AdjValue', 'MarketValue']:
                        if c in drafted_vals.columns:
                            drafted_vals[c] = pd.to_numeric(drafted_vals[c], errors='coerce').fillna(0).round(0).astype(int)
                    if 'PPS' in drafted_vals.columns:
                        drafted_vals['PPS'] = pd.to_numeric(drafted_vals['PPS'], errors='coerce').fillna(0).round(2)
                    if 'Confidence' in drafted_vals.columns:
                        drafted_vals['Confidence'] = pd.to_numeric(drafted_vals['Confidence'], errors='coerce').fillna(0).round(1)
                    if 'S4_FP/G' in drafted_vals.columns:
                        drafted_vals['S4_FP/G'] = pd.to_numeric(drafted_vals['S4_FP/G'], errors='coerce').round(1)
                    if 'S4_GP' in drafted_vals.columns:
                        drafted_vals['S4_GP'] = pd.to_numeric(drafted_vals['S4_GP'], errors='coerce').fillna(0).astype(int)
                    drafted_vals = drafted_vals.rename(columns={k: v for k, v in rename_map.items() if k in drafted_vals.columns})
                    try:
                        style_formats_d = {}
                        if 'Adj $' in drafted_vals.columns:
                            style_formats_d['Adj $'] = '${:,.0f}'
                        if 'Base $' in drafted_vals.columns:
                            style_formats_d['Base $'] = '${:,.0f}'
                        if 'Market $' in drafted_vals.columns:
                            style_formats_d['Market $'] = '${:,.0f}'
                        if 'Conf %' in drafted_vals.columns:
                            style_formats_d['Conf %'] = '{:.1f}%'
                        if 'Scarcity %' in drafted_vals.columns:
                            style_formats_d['Scarcity %'] = '{:.0f}%'
                        st.dataframe(drafted_vals.style.format(style_formats_d), width="stretch", hide_index=True)
                    except Exception:
                        st.dataframe(drafted_vals, width="stretch", hide_index=True)
                else:
                    st.write("No cached values for drafted players.")
        # Quick legend
        with st.expander("Column legend", expanded=False):
            st.markdown("- **Adj $**: Current value after scarcity models\n- **Base $**: Average of base models\n- **Realistic $ / R Low / R High**: Realistic target with P10–P90 range (room dynamics + bounded noise)\n- **Market $**: Raw market estimate (if available)\n- **Conf %**: Agreement across models (higher = more consistent)\n- **Scarcity %**: Avg scarcity premium applied across selected scarcity models")
    else:
        st.info("Values will appear here once players are loaded.")

    if recalculated_df.empty and st.session_state.draft_started:
        st.balloons()
        st.success("Draft Complete!")
