import streamlit as st
import pandas as pd
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
        
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Weights should sum to 1.0 (current sum: {total_weight:.2f})")
        else:
            st.success(f"Weights Normalized (Sum: {total_weight:.2f})")
        st.session_state.trend_weights = weights

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
            if model != "No Scarcity Adjustment":
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
    """Renders the compact team rosters and budgets in columns."""
    st.markdown("---_**Team Rosters & Budgets**_---")
    team_cols = st.columns(st.session_state.num_teams)
    for i, team_name in enumerate(st.session_state.team_names):
        with team_cols[i]:
            st.markdown(f"**{team_name}**")
            st.markdown(f"*Budget: ${st.session_state.teams[team_name]['budget']}*", help="Remaining Budget")
            team_roster_df = pd.DataFrame(st.session_state.teams[team_name]['players'])
            if not team_roster_df.empty:
                st.dataframe(team_roster_df[['PlayerName', 'Position', 'DraftPrice']], hide_index=True, use_container_width=True)
            else:
                st.text("No players drafted.")

def render_player_analysis(selected_player_name, recalculated_df):
    """Renders the analysis section for a selected player, including the draft form."""
    if not selected_player_name:
        return

    # Ensure the index is what we expect, if not, set it.
    if 'PlayerName' in recalculated_df.columns:
        recalculated_df = recalculated_df.set_index('PlayerName')

    if selected_player_name not in recalculated_df.index:
        st.warning(f"Could not find player: {selected_player_name}")
        return

    player_values = recalculated_df.loc[selected_player_name]

    with st.form(key='draft_form'):
        st.markdown(f"#### Draft **{selected_player_name}**")
        draft_col1, draft_col2, draft_col3 = st.columns(3)
        team_selection = draft_col1.selectbox("Select Team", options=st.session_state.team_names)
        draft_price = draft_col2.number_input("Draft Price", min_value=1, value=max(1, int(player_values.get('AdjValue', 1))))
        col1, col2 = st.columns([1, 1])
        with col1:
            draft_button = st.form_submit_button(f"Draft Player", use_container_width=True)
        with col2:
            on_the_block_button = st.form_submit_button("Set On The Block", use_container_width=True, type="secondary")

    if on_the_block_button:
        st.session_state.player_on_the_block = selected_player_name
        st.toast(f"{selected_player_name} is now on the block!")

    if draft_button:
            if draft_price > st.session_state.teams[team_selection]['budget']:
                st.error(f"{team_selection} cannot afford {selected_player_name} at this price.")
            else:
                    # Get the full player row as a dictionary
                draft_entry = st.session_state.available_players[st.session_state.available_players['PlayerName'] == selected_player_name].to_dict('records')[0]
                draft_entry.update({'DraftPrice': draft_price, 'Team': team_selection})

                # Add to drafted list and team's roster
                st.session_state.drafted_players.append(draft_entry)
                st.session_state.teams[team_selection]['players'].append(draft_entry)
                st.session_state.teams[team_selection]['budget'] -= draft_price
                st.session_state.total_money_spent += draft_price

                # Remove from available players
                st.session_state.available_players = st.session_state.available_players[st.session_state.available_players['PlayerName'] != selected_player_name]

                # Advance to the next nominating team
                if st.session_state.draft_order:
                    current_index = st.session_state.get('current_nominating_team_index', 0)
                    next_index = (current_index + 1) % len(st.session_state.draft_order)
                    st.session_state.current_nominating_team_index = next_index

                st.toast(f"{selected_player_name} drafted by {team_selection} for ${draft_price}!", icon="🎉")
                st.session_state.player_on_the_block = None # Clear player on block
                st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)
    col1.metric("Valuation Confidence", f"{player_values.get('Confidence', 100.0):.1f}%")
    col2.metric("Average Valuation", f"${player_values.get('ValueMean', 0):.0f}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Base Valuations**")
        st.markdown(f"**_Average Base Value_**: **${player_values.get('BaseValue', 0):,.0f}**")
        st.markdown("&nbsp;")
        for model in st.session_state.base_value_models:
            st.markdown(f"_{model}_: **${player_values.get(_get_model_col_name(model, 'BV_'), 0):,.0f}**")
    with col2:
        st.markdown("**Adjusted Valuations (In-Draft)**")
        st.markdown(f"**_Average Adj. Value_**: **${player_values.get('AdjValue', 0):,.0f}**")
        st.markdown("&nbsp;")
        for model in st.session_state.scarcity_models:
            st.markdown(f"_{model}_: **${player_values.get(_get_model_col_name(model, 'AV_'), 0):,.0f}**")
    st.markdown("---")

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("**Key Player Metrics**")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Tier", str(player_values.get('Tier', 'N/A')))
        m_col2.metric("VORP", f"{player_values.get('VORP', 0):.2f}")
        m_col3.metric("Market Value", f"${player_values.get('MarketValue', 0):.0f}")
        stats_df = pd.DataFrame({
            'Season': ['S4', 'S3', 'S2', 'S1'],
            'GP': [player_values.get(f'S{i}_GP', 'N/A') for i in range(4, 0, -1)],
            'FP/G': [f"{player_values.get(f'S{i}_FP/G', 0):.2f}" for i in range(4, 0, -1)]
        })
        st.markdown("**Historical Performance**")
        st.dataframe(stats_df, use_container_width=True)
    with col2:
        st.markdown("**Team Optimizer**")
        if not st.session_state.main_team:
            st.warning("Select your main team from the sidebar to use the optimizer.")
        else:
            if st.button("Find Optimal Roster", key="run_optimizer_btn", use_container_width=True):
                with st.spinner("Running genetic algorithm to find the best team..."):
                    # Determine remaining budget and roster spots
                    main_team_data = st.session_state.teams[st.session_state.main_team]
                    remaining_budget = main_team_data['budget']
                    current_roster_df = pd.DataFrame(main_team_data['players'])

                    # Calculate remaining roster composition
                    remaining_composition = st.session_state.roster_composition.copy()
                    if not current_roster_df.empty:
                        current_pos_counts = current_roster_df['Position'].value_counts().to_dict()
                        for pos, count in current_pos_counts.items():
                            if pos in remaining_composition:
                                remaining_composition[pos] -= count
                    
                    # Filter out positions that are already full
                    remaining_composition = {pos: count for pos, count in remaining_composition.items() if count > 0}

                    if not remaining_composition:
                        st.session_state.optimal_team = pd.DataFrame()
                        st.warning("Your roster is already full!")
                    else:
                        # Run the optimizer
                        optimal_team_df = run_team_optimizer(
                            available_players=st.session_state.available_players,
                            budget=remaining_budget,
                            roster_composition=remaining_composition
                        )
                        st.session_state.optimal_team = optimal_team_df

            if 'optimal_team' in st.session_state and not st.session_state.optimal_team.empty:
                st.success("Optimal Roster Found!")
                st.info("This is the optimal set of players to draft for your remaining spots and budget.")

                # Combine with current team for a full view
                current_team_df = pd.DataFrame(st.session_state.teams[st.session_state.main_team]['players'])
                # Ensure columns align for concatenation
                current_team_df_display = pd.DataFrame({
                    'PlayerName': current_team_df['PlayerName'],
                    'Position': current_team_df['Position'],
                    'AdjValue': current_team_df['DraftPrice'],
                    'VORP': current_team_df['VORP']
                })
                final_roster_df = pd.concat([current_team_df_display, st.session_state.optimal_team[['PlayerName', 'Position', 'AdjValue', 'VORP']]], ignore_index=True)

                st.subheader("Projected Final Roster")
                st.dataframe(final_roster_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Projected VORP", f"{final_roster_df['VORP'].sum():.2f}")
                with col2:
                    st.metric("Total Team Cost", f"${final_roster_df['AdjValue'].sum():.0f}")

def render_draft_summary():
    """Renders the draft summary table."""
    if not st.session_state.drafted_players:
        return

    st.subheader("Draft Summary")
    drafted_df = pd.DataFrame(st.session_state.drafted_players)
    
    for col in ['BaseValue', 'AdjValue', 'DraftPrice']:
        if col not in drafted_df.columns:
            drafted_df[col] = 0

    summary_df = drafted_df.groupby('Team').agg(
        Players_Drafted=('PlayerName', 'count'),
        Budget_Spent=('DraftPrice', 'sum'),
        Total_BaseValue=('BaseValue', 'sum'),
        Total_AdjValue=('AdjValue', 'sum')
    ).reset_index()

    summary_df['Budget_Remaining'] = summary_df['Team'].apply(lambda team: st.session_state.teams[team]['budget'])
    summary_df['Value'] = summary_df['Total_AdjValue'] - summary_df['Budget_Spent']

    # Formatting for display
    for col in ['Budget_Spent', 'Total_BaseValue', 'Total_AdjValue', 'Budget_Remaining']:
        summary_df[col] = summary_df[col].apply(lambda x: f"${x:,.0f}")
    
    def color_value(val):
        if val > 0:
            return f'<span style="color: green;">${val:,.0f} (Bargain)</span>'
        elif val < 0:
            return f'<span style="color: red;">${val:,.0f} (Reach)</span>'
        else:
            return f"${val:,.0f}"
    summary_df['Value'] = summary_df['Value'].apply(color_value)

    st.markdown(summary_df.to_html(escape=False), unsafe_allow_html=True)

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

    if st.sidebar.button("Reset Draft"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.session_state.drafted_players:
        if st.sidebar.button("Undo Last Pick"):
            last_draft = st.session_state.drafted_players.pop()
            player_to_restore = last_draft.copy()
            player_name = player_to_restore.pop('PlayerName')
            draft_price = player_to_restore.pop('DraftPrice')
            team_name = player_to_restore.pop('Team')

            st.session_state.teams[team_name]['budget'] += draft_price
            st.session_state.teams[team_name]['players'] = [p for p in st.session_state.teams[team_name]['players'] if p['PlayerName'] != player_name]

            player_to_restore_df = pd.DataFrame([player_to_restore], index=[player_name])
            player_to_restore_df.index.name = 'PlayerName'
            player_to_restore_df.reset_index(inplace=True)
            st.session_state.available_players = pd.concat([st.session_state.available_players, player_to_restore_df], ignore_index=True).sort_values(by='PPS', ascending=False)

            st.session_state.total_money_spent -= draft_price
            st.success(f"Reversed the pick of {player_name} by {team_name}.")
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
