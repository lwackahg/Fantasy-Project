import streamlit as st
import pandas as pd
import re
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import json
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
        use_container_width=True
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
                    use_container_width=True
                )
        with col2:
            st.subheader("Tier Percentiles")
            st.info("Define the PPS percentile cutoffs for each tier.")
            st.session_state.tier_cutoffs['Tier 1'] = st.slider("Tier 1 Cutoff (Top %)", 0.90, 1.0, st.session_state.tier_cutoffs.get('Tier 1', 0.98), 0.01)
            st.session_state.tier_cutoffs['Tier 2'] = st.slider("Tier 2 Cutoff", 0.70, st.session_state.tier_cutoffs['Tier 1'], st.session_state.tier_cutoffs.get('Tier 2', 0.90), 0.01)
            st.session_state.tier_cutoffs['Tier 3'] = st.slider("Tier 3 Cutoff", 0.50, st.session_state.tier_cutoffs['Tier 2'], st.session_state.tier_cutoffs.get('Tier 3', 0.70), 0.01)
            st.session_state.tier_cutoffs['Tier 4'] = st.slider("Tier 4 Cutoff", 0.30, st.session_state.tier_cutoffs['Tier 3'], st.session_state.tier_cutoffs.get('Tier 4', 0.45), 0.01)

    with st.expander("Customize Trend Weights", expanded=False):
        st.info("These weights determine the importance of each of the last four seasons when calculating a player's Power Score (PPS). The weights must sum to 1.0.")
        weights = {}
        default_weights = {'S4': 0.57, 'S3': 0.29, 'S2': 0.14, 'S1': 0.00}
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
                            st.dataframe(roster_df[['Player', 'Position', 'Price']])
                        else:
                            st.write("No players drafted.")

def render_drafting_form(player_series, drafting_callback):
    """Renders the form for drafting a player."""
    st.subheader("Draft Player")

    # Prepare position options BEFORE the form to prevent state lag
    # Split by comma or slash to handle formats like 'G/F' or 'C,Flx'
    position_options = [pos.strip() for pos in player_series['Position'].replace(',', '/').split('/')]
    if 'Bench' not in position_options:
        position_options.append('Bench')

    with st.form(key=f"draft_{player_series.name}"):
        team_selection = st.selectbox("Select Team", options=st.session_state.team_names, key=f"team_select_{player_series.name}")
        
        # Ensure the default value is at least 1 to prevent crashes
        default_price = max(1, int(player_series.get('AdjValue', 1)))
        draft_price = st.number_input("Draft Price", min_value=1, value=default_price, key=f"price_input_{player_series.name}")
        
        assigned_position = st.radio("Assign Position", options=position_options, horizontal=True, key=f"pos_radio_{player_series.name}")

        submitted = st.form_submit_button("Draft Player", use_container_width=True, type="primary")
        if submitted:
            drafting_callback(player_series.name, team_selection, draft_price, assigned_position, player_series)

def render_player_analysis_metrics(selected_player_name, recalculated_df):
    """Renders the metrics and analysis for a selected player, handling missing keys gracefully."""
    st.subheader(f"Analysis for: {selected_player_name}")
    player_series = recalculated_df.loc[selected_player_name]

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
            st.dataframe(df_perf, hide_index=True, use_container_width=True)
        else:
            st.info("No historical performance data available.")

    st.markdown("---_**Valuation Models Used**_---")
    base_models_used = st.session_state.get('base_value_models', ['N/A'])
    scarcity_models_used = st.session_state.get('scarcity_models', ['N/A'])
    st.write(f"**Base Model(s):** {', '.join(base_models_used)}")
    st.write(f"**Scarcity Model(s):** {', '.join(scarcity_models_used)}")

    with st.expander("Detailed Model Values"):
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

    # --- Single column for optimal roster ---
    st.markdown("---")
    if st.button("Find Optimal Roster for this Player", use_container_width=True):
        st.info("Optimal roster feature coming soon!")
    st.markdown("---")

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
    
    st.dataframe(pick_df[['Pick', 'Player', 'Position', 'Team', 'Price', 'Total_BaseValue', 'Total_AdjValue']], use_container_width=True)



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
