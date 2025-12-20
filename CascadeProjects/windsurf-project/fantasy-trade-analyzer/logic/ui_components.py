import streamlit as st
import pandas as pd
import re
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import json
from streamlit_compat import dataframe
from logic.auction_tool import BASE_VALUE_MODELS, SCARCITY_MODELS
from logic.team_optimizer import run_team_optimizer

def _get_model_col_name(model_name, prefix):
    """Helper to create a consistent column name from a model name."""
    clean_name = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
    return f"{prefix}{clean_name}"

def render_setup_page():
    """Renders the main setup interface on the page before the draft starts."""
    st.title("Auction Draft Tool Setup")
    st.markdown("""
    Welcome to the Auction Draft Tool. Configure your league settings below. 
    You can import/export settings to save your configuration for later use.
    """)

    # --- Settings Import/Export ---
    st.sidebar.subheader("Manage Settings")
    settings_data = {
        'num_teams': st.session_state.num_teams,
        'budget_per_team': st.session_state.budget_per_team,
        'roster_spots_per_team': st.session_state.roster_spots_per_team,
        'games_in_season': st.session_state.games_in_season,
        'roster_composition': st.session_state.roster_composition,
        'base_value_models': st.session_state.base_value_models,
        'scarcity_models': st.session_state.scarcity_models,
        'tier_cutoffs': st.session_state.tier_cutoffs,
        'injured_players_text': st.session_state.injured_players_text,
        'trend_weights': st.session_state.trend_weights
    }
    st.sidebar.download_button(
        label="Export Settings",
        data=json.dumps(settings_data, indent=4),
        file_name="draft_settings.json",
        mime="application/json",
        width="stretch"
    )
    
    uploaded_file = st.sidebar.file_uploader("Import Settings", type=['json'])
    if uploaded_file is not None:
        try:
            new_settings = json.load(uploaded_file)
            for key, value in new_settings.items():
                st.session_state[key] = value
            st.sidebar.success("Settings imported successfully!")
            st.rerun()
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON file.")
        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")

    st.markdown("---")

    with st.expander("League & Roster Setup", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("League Setup")
            st.session_state.num_teams = st.number_input("Number of Teams", min_value=8, max_value=20, value=st.session_state.num_teams)
            st.session_state.budget_per_team = st.number_input("Budget per Team ($)", min_value=100, max_value=500, value=st.session_state.budget_per_team, step=10)
            st.session_state.roster_spots_per_team = st.number_input("Roster Spots per Team", min_value=5, max_value=20, value=st.session_state.roster_spots_per_team)
            st.session_state.games_in_season = st.number_input("Games in a Season", min_value=1, max_value=82, value=st.session_state.games_in_season)
        with col2:
            st.subheader("Roster Composition")
            st.session_state.roster_composition['G'] = st.number_input("Guard (G)", min_value=0, value=st.session_state.roster_composition.get('G', 3))
            st.session_state.roster_composition['F'] = st.number_input("Forward (F)", min_value=0, value=st.session_state.roster_composition.get('F', 3))
            st.session_state.roster_composition['C'] = st.number_input("Center (C)", min_value=0, value=st.session_state.roster_composition.get('C', 2))
            st.session_state.roster_composition['Flx'] = st.number_input("Flex (Flx)", min_value=0, value=st.session_state.roster_composition.get('Flx', 2))
            st.session_state.roster_composition['Bench'] = st.number_input("Bench", min_value=0, value=st.session_state.roster_composition.get('Bench', 5))
            total_roster_spots = sum(st.session_state.roster_composition.values())
            if total_roster_spots != st.session_state.roster_spots_per_team:
                st.warning(f"Roster size ({total_roster_spots}) doesn't match setting ({st.session_state.roster_spots_per_team}).")
            else:
                st.info(f"Total Roster Size: {total_roster_spots} players")

    with st.expander("Valuation & Scarcity Models", expanded=True):
        st.write("**Select the models to use for player valuation.**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Base Value Calculation Model(s)**")
            selected_base_models = []
            for model in BASE_VALUE_MODELS:
                if st.checkbox(model, key=f"base_{model}", value=(model in st.session_state.base_value_models)):
                    selected_base_models.append(model)
            st.session_state.base_value_models = selected_base_models
        with col2:
            st.write("**In-Draft Scarcity Model(s)**")
            selected_scarcity_models = []
            for model in SCARCITY_MODELS:
                if st.checkbox(model, key=f"scarcity_{model}", value=(model in st.session_state.scarcity_models)):
                    selected_scarcity_models.append(model)
            st.session_state.scarcity_models = selected_scarcity_models

    with st.expander("Projections & Adjustments", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Injured Players")
            st.info("Enter a player name per line. Use the buttons to append a status for the last player entered.")

            # Define callback function to append status
            def add_status_to_text(status_to_add):
                current_text = st.session_state.get("injured_players_text", "").strip()
                # Append status and a newline to encourage one player per line
                st.session_state.injured_players_text = f"{current_text} {status_to_add}\n"

            st.text_area(
                "Injured Players List",
                height=150,
                key="injured_players_text",
                label_visibility="collapsed",
                help="Enter one player per line. Click a status button to add it to the last player in the list."
            )

            btn_cols = st.columns(4)
            statuses = ["(1/4 Season)", "(Half Season)", "(3/4 Season)", "(Full Season)"]
            
            for i, status in enumerate(statuses):
                btn_cols[i].button(
                    status.strip('()'), 
                    key=f"status_btn_{i}", 
                    on_click=add_status_to_text,
                    args=(status,), # Pass the status string to the callback
                    width="stretch"
                )
        with col2:
            st.subheader("Tier Percentiles")
            st.info("Define the PPS percentile cutoffs for each tier.")
            st.session_state.tier_cutoffs['Tier 1'] = st.slider("Tier 1 Cutoff (Top %)", 0.90, 1.0, st.session_state.tier_cutoffs.get('Tier 1', 0.98), 0.01)
            st.session_state.tier_cutoffs['Tier 2'] = st.slider("Tier 2 Cutoff", 0.70, st.session_state.tier_cutoffs['Tier 1'], st.session_state.tier_cutoffs.get('Tier 2', 0.90), 0.01)
            st.session_state.tier_cutoffs['Tier 3'] = st.slider("Tier 3 Cutoff", 0.50, st.session_state.tier_cutoffs['Tier 2'], st.session_state.tier_cutoffs.get('Tier 3', 0.70), 0.01)
            st.session_state.tier_cutoffs['Tier 4'] = st.slider("Tier 4 Cutoff", 0.30, st.session_state.tier_cutoffs['Tier 3'], st.session_state.tier_cutoffs.get('Tier 4', 0.45), 0.01)

        # Early-Career Model Settings
        st.markdown("---")
        st.subheader("Early-Career Model Settings")
        st.session_state.ec_enabled = st.checkbox("Enable Early-Career Model", value=bool(st.session_state.get('ec_enabled', True)))
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            elite_thr = st.number_input("Elite FP/G threshold", min_value=50.0, max_value=140.0, value=float(st.session_state.get('ec_settings', {}).get('elite_fpg_threshold', 85.0)), step=0.5, help="Tier 1 requires Top-10 pick and FP/G >= this value")
        with ec2:
            trend_delta = st.number_input("Trend delta (FP/G)", min_value=0.0, max_value=30.0, value=float(st.session_state.get('ec_settings', {}).get('trend_delta_min', 5.0)), step=0.5, help="Minimum S4−S3 increase to qualify Tier 2")
        with ec3:
            st.write("")
        ecm1, ecm2, ecm3 = st.columns(3)
        with ecm1:
            t1 = st.number_input("Tier 1 boost (%)", min_value=0.0, max_value=50.0, value=float(st.session_state.get('ec_settings', {}).get('tier1_pct', 15.0)), step=0.5)
        with ecm2:
            t2 = st.number_input("Tier 2 boost (%)", min_value=0.0, max_value=40.0, value=float(st.session_state.get('ec_settings', {}).get('tier2_pct', 10.0)), step=0.5)
        with ecm3:
            t3 = st.number_input("Tier 3 boost (%)", min_value=0.0, max_value=30.0, value=float(st.session_state.get('ec_settings', {}).get('tier3_pct', 5.0)), step=0.5)
        st.session_state.ec_settings = {
            'elite_fpg_threshold': elite_thr,
            'trend_delta_min': trend_delta,
            'tier1_pct': t1,
            'tier2_pct': t2,
            'tier3_pct': t3,
        }

        # GP Reliability Adjustment (uses user's Trend Weights as lookback weights)
        st.markdown("---")
        st.subheader("GP Reliability Adjustment")
        st.session_state.gp_rel_enabled = st.checkbox("Enable GP Reliability Adjustment", value=bool(st.session_state.get('gp_rel_enabled', True)))
        gr1, gr2, gr3 = st.columns(3)
        with gr1:
            gp_target = st.number_input("GP target for full trust", min_value=60.0, max_value=82.0, value=float(st.session_state.get('gp_rel_settings', {}).get('gp_target', 72.0)), step=1.0)
        with gr2:
            strictness = st.selectbox("Strictness curve", options=["Mild", "Standard", "Strict"], index={"Mild":0,"Standard":1,"Strict":2}.get(str(st.session_state.get('gp_rel_settings', {}).get('strictness','Standard')).title(),1))
        with gr3:
            st.write("")
        gf1, gf2 = st.columns(2)
        with gf1:
            veteran_floor = st.number_input("Veteran floor", min_value=0.60, max_value=0.95, value=float(st.session_state.get('gp_rel_settings', {}).get('veteran_floor', 0.70)), step=0.01, help="Minimum PPS scale for veterans with low GP")
        with gf2:
            ec_floor = st.number_input("Early-career floor", min_value=0.70, max_value=0.99, value=float(st.session_state.get('gp_rel_settings', {}).get('ec_floor', 0.85)), step=0.01, help="Minimum PPS scale for Seasons ≤ 3")
        s1, s2 = st.columns(2)
        with s1:
            gp_severe_threshold = st.number_input("Severe GP threshold (ratio)", min_value=0.10, max_value=0.90, value=float(st.session_state.get('gp_rel_settings', {}).get('gp_severe_threshold', 0.50)), step=0.05, help="If weighted GP / target < this, enforce severe minimum factor")
        with s2:
            severe_min_factor = st.number_input("Severe min factor", min_value=0.05, max_value=0.50, value=float(st.session_state.get('gp_rel_settings', {}).get('severe_min_factor', 0.10)), step=0.01, help="Harsher floor multiplier for severe low-GP cases")
        # Persist settings and include current trend weights for lookback
        st.session_state.gp_rel_settings = {
            'enabled': st.session_state.gp_rel_enabled,
            'gp_target': gp_target,
            'strictness': strictness.lower(),
            'veteran_floor': veteran_floor,
            'ec_floor': ec_floor,
            'gp_severe_threshold': gp_severe_threshold,
            'severe_min_factor': severe_min_factor,
            'trend_weights': st.session_state.get('trend_weights', {'S4':0.65,'S3':0.25,'S2':0.10,'S1':0.00}),
        }

    with st.expander("Customize Trend Weights", expanded=False):
        st.info("These weights determine the importance of each of the last four seasons when calculating a player's Power Score (PPS). The weights must sum to 1.0.")
        weights = {}
        default_weights = {'S4': 0.65, 'S3': 0.25, 'S2': 0.10, 'S1': 0.00}
        for i, (season_label, default_weight) in enumerate(default_weights.items()):
            weight_label = f"Most Recent Season ({season_label}) Weight" if i == 0 else f"{season_label} Weight"
            weights[season_label] = st.slider(
                weight_label, 0.0, 1.0, 
                st.session_state.trend_weights.get(season_label, default_weight), 
                0.01
            )
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            normalized_weights = weights

        st.session_state.trend_weights = normalized_weights
        st.info(f"Weights have been normalized to sum to 1.0")

def render_draft_board(df):
    """Renders the main AgGrid component for the draft board."""
    st.markdown("---_**Available Players**_---")
    if df.empty:
        st.warning("No available players to display.")
        return

    # Define the core columns that should always be present.
    core_cols = ['PlayerName', 'Position', 'Tier', 'PPS', 'Confidence', 'BaseValue', 'AdjValue']
    
    # Add individual model columns only if more than one model is selected.
    model_cols = []
    if len(st.session_state.base_value_models) > 1:
        for model in st.session_state.base_value_models:
            model_cols.append(_get_model_col_name(model, 'BV_'))
    if len(st.session_state.scarcity_models) > 1:
        for model in st.session_state.scarcity_models:
            model_cols.append(_get_model_col_name(model, 'AV_'))

    # Combine and filter columns to ensure they exist in the dataframe.
    display_cols = core_cols + model_cols
    final_display_df = df[[col for col in display_cols if col in df.columns]]

    gb = GridOptionsBuilder.from_dataframe(final_display_df)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
    # --- Configure Core Columns ---
    gb.configure_column("PlayerName", headerName="Player", pinned='left', width=180)
    gb.configure_column("Position", headerName="Pos", width=80)
    gb.configure_column("BaseValue", headerName="Avg. Base Value", width=120, type=["numericColumn", "numberColumnFilter"], valueFormatter="'$' + Math.round(value)")
    gb.configure_column("AdjValue", headerName="Avg. Adj. Value", width=120, type=["numericColumn", "numberColumnFilter"], valueFormatter="'$' + Math.round(value)")
    gb.configure_column("PPS", headerName="PPS", width=80, type=["numericColumn"], valueFormatter="value.toFixed(2)")

    # Tier styling
    js_tier_cell_style = JsCode("""function(params) { 
        if (params.value == 1) { return {'color': 'white', 'backgroundColor': '#6a0dad'}; } // Purple
        if (params.value == 2) { return {'color': 'white', 'backgroundColor': '#007bff'}; } // Blue
        if (params.value == 3) { return {'color': 'black', 'backgroundColor': '#17a2b8'}; } // Cyan
        return {}; 
    }""")
    gb.configure_column("Tier", width=80, cellStyle=js_tier_cell_style)

    # Confidence styling
    js_confidence_cell_style = JsCode("""function(params) {
        if (params.value >= 90) { return {'color': 'white', 'backgroundColor': 'darkgreen'}; }
        if (params.value >= 75) { return {'color': 'black', 'backgroundColor': 'gold'}; }
        if (params.value < 75) { return {'color': 'white', 'backgroundColor': 'darkred'}; }
        return {};
    }""")
    gb.configure_column("Confidence", width=110, cellStyle=js_confidence_cell_style, valueFormatter="value.toFixed(1) + '%'", type=["numericColumn", "numberColumnFilter"])

    # --- Configure Optional Model Columns (if more than one model is selected) ---
    if len(st.session_state.base_value_models) > 1:
        for model in st.session_state.base_value_models:
            col_name = _get_model_col_name(model, 'BV_')
            if col_name in final_display_df.columns:
                gb.configure_column(col_name, f"BV: {model}", width=100, type=["numericColumn"], valueFormatter="'$' + Math.round(value)")
    
    if len(st.session_state.scarcity_models) > 1:
        for model in st.session_state.scarcity_models:
            col_name = _get_model_col_name(model, 'AV_')
            if col_name in final_display_df.columns:
                gb.configure_column(col_name, f"AV: {model}", width=100, type=["numericColumn"], valueFormatter="'$' + Math.round(value)")

    gb.configure_side_bar(filters_panel=True, columns_panel=True, defaultToolPanel="columns")
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    AgGrid(final_display_df, gridOptions=gridOptions, enable_enterprise_modules=False, allow_unsafe_jscode=True, update_mode=GridUpdateMode.MODEL_CHANGED, height=600)

def render_team_rosters():
    """Renders the rosters and budgets for each team in a 5-column grid."""
    st.subheader("Team Rosters & Budgets")
    team_names = st.session_state.team_names
    num_teams = len(team_names)
    
    # Calculate number of rows needed for a 5-column grid
    num_rows = (num_teams + 4) // 5

    for i in range(num_rows):
        cols = st.columns(5)
        for j in range(5):
            team_index = i * 5 + j
            if team_index < num_teams:
                team_name = team_names[team_index]
                team_data = st.session_state.teams[team_name]
                with cols[j]:
                    with st.expander(f"{team_name} (${team_data['budget']})", expanded=False):
                        if team_data['players']:
                            roster_df = pd.DataFrame(team_data['players'])
                            dataframe(roster_df[['Player', 'Position', 'Price']], hide_index=True, width="stretch")
                        else:
                            st.write("No players drafted.")

def render_drafting_form(player_series, drafting_callback):
    """Renders the form for drafting a player."""
    st.subheader("Draft Player")

    # Prepare position options BEFORE the form to prevent state lag
    # Split by comma or slash to handle formats like 'G/F' or 'C,Flx'
    position_options = [pos.strip() for pos in str(player_series.get('Position', '')).replace(',', '/').split('/') if pos.strip()]
    # Include Flex if league supports it
    if st.session_state.roster_composition.get('Flx', 0) > 0 and 'Flx' not in position_options:
        position_options.append('Flx')
    # Only include Bench if league has bench slots
    if st.session_state.roster_composition.get('Bench', 0) > 0 and 'Bench' not in position_options:
        position_options.append('Bench')

    with st.form(key=f"draft_{player_series.name}"):
        team_selection = st.selectbox("Select Team", options=st.session_state.team_names, key=f"team_select_{player_series.name}")

        # Compute available slots for the selected team
        req = {k: int(v) for k, v in st.session_state.roster_composition.items()}
        team_players = st.session_state.teams.get(team_selection, {}).get('players', [])
        filled_counts = pd.Series([p.get('Position', '') for p in team_players]).value_counts().to_dict() if team_players else {}
        def remaining(pos):
            return max(0, int(req.get(pos, 0)) - int(filled_counts.get(pos, 0)))

        # Rebuild options by filtering to positions with remaining capacity
        eligible_core = [p for p in position_options if p not in ('Flx','Bench') and remaining(p) > 0]
        options_final = []
        options_final.extend(eligible_core)
        if req.get('Flx', 0) > 0 and remaining('Flx') > 0 and 'Flx' in position_options:
            options_final.append('Flx')
        if req.get('Bench', 0) > 0 and remaining('Bench') > 0 and 'Bench' in position_options:
            options_final.append('Bench')

        # Fallback: if no core slots left but Flex available, ensure Flex shows; else Bench if available
        if not options_final:
            if req.get('Flx', 0) > 0 and remaining('Flx') > 0:
                options_final = ['Flx']
            elif req.get('Bench', 0) > 0 and remaining('Bench') > 0:
                options_final = ['Bench']

        # Ensure the default value is numeric and at least 1 to prevent crashes
        raw_default = player_series.get('AdjValue', 1)
        try:
            default_price = int(raw_default) if pd.notna(raw_default) else 1
        except Exception:
            default_price = 1
        default_price = max(1, default_price)

        # Set the max bid to: team_budget - $1 for each remaining roster spot AFTER this pick
        try:
            team_data = st.session_state.teams.get(team_selection, {})
            team_budget = int(team_data.get('budget', 0))
            rostered = len(team_data.get('players', []))
        except Exception:
            team_budget, rostered = 0, 0

        total_spots = int(st.session_state.get('roster_spots_per_team', 0))
        spots_remaining_now = max(total_spots - rostered, 0)
        spots_after_this_pick = max(spots_remaining_now - 1, 0)
        reserve_for_min_bids = spots_after_this_pick  # $1 per remaining spot
        max_allowed = max(0, team_budget - reserve_for_min_bids)

        # Determine min based on max_allowed: if you can't bid at least $1, allow 0
        min_val = 1 if max_allowed >= 1 else 0

        # Clamp default into allowed range [min_val, max_allowed]
        default_price = max(min_val, min(default_price, max_allowed))
        draft_price = st.number_input(
            "Draft Price",
            min_value=min_val,
            max_value=max_allowed,
            value=default_price,
            key=f"price_input_{player_series.name}"
        )
        st.caption(
            f"Max bid for {team_selection}: ${max_allowed} (Budget ${team_budget} − $1 × {spots_after_this_pick} remaining spots after this pick)"
        )
        
        assigned_position = st.radio("Assign Position", options=position_options, horizontal=True, key=f"pos_radio_{player_series.name}")

        submitted = st.form_submit_button("Draft Player", width="stretch", type="primary")
        if submitted:
            # Recompute max at submit time to guard against any stale widget state
            try:
                team_data_submit = st.session_state.teams.get(team_selection, {})
                team_budget_submit = int(team_data_submit.get('budget', 0))
                rostered_submit = len(team_data_submit.get('players', []))
            except Exception:
                team_budget_submit, rostered_submit = 0, 0

            total_spots_submit = int(st.session_state.get('roster_spots_per_team', 0))
            spots_remaining_now_submit = max(total_spots_submit - rostered_submit, 0)
            spots_after_this_pick_submit = max(spots_remaining_now_submit - 1, 0)
            reserve_for_min_bids_submit = spots_after_this_pick_submit
            max_allowed_submit = max(0, team_budget_submit - reserve_for_min_bids_submit)

            if draft_price > max_allowed_submit:
                st.error(
                    f"Bid exceeds max allowed ${max_allowed_submit}. You must keep $1 for each of the {spots_after_this_pick_submit} remaining spots after this pick."
                )
            elif draft_price > team_budget_submit:
                st.error(f"{team_selection} cannot afford this bid. Budget: ${team_budget_submit}")
            else:
                drafting_callback(player_series.name, team_selection, draft_price, assigned_position, player_series)

def render_player_analysis_metrics(selected_player_name, recalculated_df):
    """Renders the metrics and analysis for a selected player, handling missing keys gracefully.

    If injury metadata is present (InjuryStatus/IsInjured/DisplayName), it is surfaced prominently.
    """
    player_series = recalculated_df.loc[selected_player_name]

    # Prefer DisplayName when available (so '(INJ)' shows up), fall back to selected_player_name
    name_to_show = player_series.get('DisplayName', selected_player_name)
    st.subheader(f"Analysis for: {name_to_show}")

    # Show an obvious injury banner with duration if available
    try:
        is_injured = bool(player_series.get('IsInjured', False))
        injury_status = str(player_series.get('InjuryStatus', '')).strip()
        if is_injured:
            if injury_status:
                st.error(f"Injury: {injury_status}")
            else:
                st.error("Injury: Status not specified")
    except Exception:
        pass

    # --- Main two columns for player metrics ---
    main_col1, main_col2 = st.columns(2)

    with main_col1:
        st.markdown("##### Average Values & Confidence")
        confidence = player_series.get('Confidence', 0.0)
        st.progress(confidence / 100, text=f"Confidence: {confidence:.1f}%")
        
        val_col1, val_col2, val_col3 = st.columns(3)
        val_col1.metric("Adjusted Value", f"${player_series.get('AdjValue', 0.0):.2f}")
        val_col2.metric("Base Value", f"${player_series.get('BaseValue', 0.0):.2f}")
        val_col3.metric("Market Value", f"${player_series.get('MarketValue', 0.0):.2f}")

    with main_col2:
        st.markdown("##### Historical Performance (Last 4 Seasons)")
        hist_data = {}
        for col in player_series.index:
            match = re.match(r'S(\d)_(GP|FP/G)', col)
            if match:
                season_num, metric = match.groups()
                # Only process seasons with a weight > 0
                if st.session_state.trend_weights.get(f'S{season_num}', 0) > 0:
                    season_label = f"Last Season" if int(season_num) == 4 else f"Season -{4 - int(season_num)}"
                    if season_label not in hist_data:
                        hist_data[season_label] = {}
                    value = player_series.get(col, 'N/A')
                    hist_data[season_label][metric] = f"{value:.1f}" if metric == 'FP/G' and isinstance(value, (int, float)) else value

        # Sort seasons from most recent to oldest
        sorted_seasons = sorted(hist_data.keys(), key=lambda x: (4 if 'Last' in x else int(x.split('-')[1])), reverse=True)
        if sorted_seasons:
            perf_data = [[season, hist_data[season].get('GP', 'N/A'), hist_data[season].get('FP/G', 'N/A')] for season in sorted_seasons]
            df_perf = pd.DataFrame(perf_data, columns=["Season", "GP", "FP/G"])
            dataframe(df_perf, hide_index=True, width="stretch")
        else:
            st.info("No historical performance data available.")

    st.markdown("---_**Valuation Models Used**_---")
    base_models_used = st.session_state.get('base_value_models', ['N/A'])
    scarcity_models_used = st.session_state.get('scarcity_models', ['N/A'])
    st.write(f"**Base Model(s):** {', '.join(base_models_used)}")
    st.write(f"**Scarcity Model(s):** {', '.join(scarcity_models_used)}")

    with st.expander("Detailed Model Values", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Base Model Values**")
            base_models = base_models_used
            for model_name in base_models:
                if model_name != "No Adjustment":
                    col_name = _get_model_col_name(model_name, 'BV_')
                    value = player_series.get(col_name, 0.0)
                    st.metric(label=model_name, value=f"${value:.2f}")

        with col2:
            st.markdown("**Scarcity Premiums & Values**")
            scarcity_models = st.session_state.get('scarcity_models', [])
            for model_name in scarcity_models:
                if model_name != "No Scarcity Adjustment":
                    premium_col = _get_model_col_name(model_name, 'SP_')
                    adj_val_col = _get_model_col_name(model_name, 'AV_')
                    premium = player_series.get(premium_col, 0.0)
                    adj_value = player_series.get(adj_val_col, 0.0)
                    st.metric(label=f"{model_name} Adj. Value", value=f"${adj_value:.2f}",
                              help=f"Scarcity Premium: {premium:.2%}")

    # Removed: Optimal roster CTA per user request

def render_draft_summary():
    """Renders the draft summary table."""
    if not st.session_state.drafted_players:
        return

    st.subheader("Draft Summary")
    
    # Pick-by-pick list
    pick_df = pd.DataFrame(st.session_state.drafted_players)
    pick_df['PickNumber'] = range(1, len(pick_df) + 1)
    pick_df = pick_df.sort_values('PickNumber', ascending=True)
    pick_df['Pick'] = pick_df['PickNumber'].astype(str)
    pick_df['Player'] = pick_df['PlayerName']
    pick_df['Team'] = pick_df['Team']
    pick_df['Price'] = pick_df['DraftPrice'].apply(lambda x: f"${x:,.0f}")
    pick_df['Position'] = pick_df['Position']
    
    st.markdown("**Pick-by-Pick Results**")
    pick_df['Value'] = pick_df['AdjValue'] - pick_df['DraftPrice']
    pick_df['Value'] = pick_df['Value'].apply(lambda v: f"${v:,.0f}")
    
    # Team summary columns
    team_summary = pick_df.groupby('Team').agg(
        Players_Drafted=('PlayerName', 'count'),
        Budget_Spent=('DraftPrice', 'sum'),
        Total_BaseValue=('BaseValue', 'sum'),
        Total_AdjValue=('AdjValue', 'sum')
    ).reset_index()
    
    # Merge team summary into pick_df
    pick_df = pick_df.merge(team_summary, on='Team', how='left')
    pick_df['Players_Drafted'] = pick_df['Players_Drafted'].astype(int)
    pick_df['Budget_Spent'] = pick_df['Budget_Spent'].apply(lambda x: f"${x:,.0f}")
    pick_df['Total_BaseValue'] = pick_df['Total_BaseValue'].apply(lambda x: f"${x:,.0f}")
    pick_df['Total_AdjValue'] = pick_df['Total_AdjValue'].apply(lambda x: f"${x:,.0f}")
    
    dataframe(pick_df[['Pick', 'Player', 'Position', 'Team', 'Price', 'Total_BaseValue', 'Total_AdjValue']], width="stretch")

def render_sidebar_in_draft():
    """Renders the sidebar controls for an active draft."""
    st.sidebar.success("Draft in Progress!")
    st.sidebar.write(f"**{st.session_state.num_teams}** Teams")
    st.sidebar.write(f"**${st.session_state.budget_per_team}** Budget")
    st.sidebar.selectbox(
        "Select Your Main Team (for Advisor)",
        options=st.session_state.team_names,
        key='main_team',
        index=0 if st.session_state.main_team is None else st.session_state.team_names.index(st.session_state.main_team)
    )

    # --- Nomination Strategy Controls ---
    st.sidebar.subheader("Nomination Strategy")
    strategy_options = [
        'Blended (recommended)',
        'Budget Pressure',
        'Positional Scarcity',
        'Value Inflation',
        'Custom'
    ]
    strategy = st.sidebar.selectbox("Strategy", options=strategy_options, key='nomination_strategy')

    def _set_weights(bp, ps, vi):
        total = bp + ps + vi
        if total <= 0:
            st.session_state.nomination_weights = {'budget_pressure': 0.5, 'positional_scarcity': 0.3, 'value_inflation': 0.2}
        else:
            st.session_state.nomination_weights = {
                'budget_pressure': bp / total,
                'positional_scarcity': ps / total,
                'value_inflation': vi / total,
            }

    if strategy == 'Blended (recommended)':
        _set_weights(0.5, 0.3, 0.2)
        st.sidebar.caption("Balanced: drain budgets, respect scarcity, nudge inflation")
    elif strategy == 'Budget Pressure':
        _set_weights(1.0, 0.0, 0.0)
        st.sidebar.caption("Focus on forcing opponents to overspend")
    elif strategy == 'Positional Scarcity':
        _set_weights(0.0, 1.0, 0.0)
        st.sidebar.caption("Target scarce positions among top tiers")
    elif strategy == 'Value Inflation':
        _set_weights(0.0, 0.0, 1.0)
        st.sidebar.caption("Nominate likely overpays to create money pits")
    else:
        bp = st.sidebar.slider("Budget Pressure", 0.0, 1.0, st.session_state.nomination_weights.get('budget_pressure', 0.5), 0.05)
        ps = st.sidebar.slider("Positional Scarcity", 0.0, 1.0, st.session_state.nomination_weights.get('positional_scarcity', 0.3), 0.05)
        vi = st.sidebar.slider("Value Inflation", 0.0, 1.0, st.session_state.nomination_weights.get('value_inflation', 0.2), 0.05)
        _set_weights(bp, ps, vi)

    if st.sidebar.button("Reset Draft"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.session_state.draft_history:
        if st.sidebar.button("Undo Last Pick"):
            last_draft = st.session_state.draft_history.pop()
            player_name = last_draft['Player']
            team_name = last_draft['Team']
            draft_price = last_draft['Price']

            # Restore budget and remove player from team
            st.session_state.teams[team_name]['budget'] += draft_price
            st.session_state.teams[team_name]['players'] = [p for p in st.session_state.teams[team_name]['players'] if p.get('Player') != player_name]

            st.session_state.total_money_spent -= draft_price

            # Remove from drafted_players (last matching entry)
            try:
                for i in range(len(st.session_state.drafted_players) - 1, -1, -1):
                    dp = st.session_state.drafted_players[i]
                    if dp.get('PlayerName') == player_name and dp.get('Team') == team_name and dp.get('DraftPrice') == draft_price:
                        del st.session_state.drafted_players[i]
                        break
            except Exception:
                pass

            # Restore to available_players using cached row if present
            try:
                row_cache = st.session_state.get('removed_player_rows', {})
                row = row_cache.pop(player_name, None)
                if row is not None:
                    import pandas as pd
                    df = st.session_state.available_players
                    row_df = pd.DataFrame([row])
                    # Ensure PlayerName column present in both
                    if 'PlayerName' not in df.columns:
                        df['PlayerName'] = pd.Series(dtype='string')
                    if 'PlayerName' not in row_df.columns:
                        row_df['PlayerName'] = player_name
                    # Align columns
                    for c in set(row_df.columns) - set(df.columns):
                        df[c] = pd.NA
                    for c in set(df.columns) - set(row_df.columns):
                        row_df[c] = pd.NA
                    st.session_state.available_players = pd.concat([df, row_df[df.columns]], ignore_index=True)
            except Exception:
                pass

            st.success(f"Reversed the pick of {player_name} by {team_name}.")
            # Move the nominating team back to the previous team
            if st.session_state.draft_order:
                current_index = st.session_state.get('current_nominating_team_index', 0)
                prev_index = (current_index - 1) % len(st.session_state.draft_order)
                st.session_state.current_nominating_team_index = prev_index
            st.rerun()

def render_value_calculation_expander():
    """Renders the expander that explains how values are calculated."""
    with st.expander("How Values Are Calculated"):
        st.markdown("""
        ### Core Concepts
        - **PPS (Player Power Score):** A weighted average of a player's fantasy points per game over the last four seasons, adjusted for games played. The trend weights are configurable in the sidebar.
        - **Tier:** Players are grouped into five tiers based on their PPS percentile rank against the *remaining* player pool. Tiers are recalculated after every pick.
        - **VORP (Value Over Replacement Player):** Measures a player's value relative to the best player likely available on the waiver wire.
        
        ### Base Value Models (Pre-Draft Estimates)
        - **Blended (VORP + Market):** A hybrid model that averages the `Pure VORP` and `Pure Market Value` models.
        - **Pure VORP:** Value is determined by a player's VORP score as a percentage of the total VORP available.
        - **Pure Market Value:** Uses historical auction data to determine a player's price.
        - **Risk-Adjusted VORP:** Discounts players based on their injury history and games played.
        - **Expert Consensus Value (ECV):** Blends VORP with expert rankings.

        ### Scarcity & Adjustment Models (In-Draft Dynamics)
        - **Tier Scarcity:** Remaining players in a tier become more valuable as others are drafted.
        - **Position Scarcity:** Premium based on the scarcity of a player's primary position.
        - **Roster Slot Demand:** Dynamically shifts between Position and Tier Scarcity.
        - **Contrarian Fade:** Applies a discount to players in tiers/positions being drafted heavily.
        - **None:** Applies no in-draft adjustment.
        """)
