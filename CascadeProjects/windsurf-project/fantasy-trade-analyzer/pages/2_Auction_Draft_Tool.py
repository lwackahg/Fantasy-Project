import streamlit as st
import pandas as pd
from logic.auction_tool import (calculate_initial_values, recalculate_dynamic_values, generate_draft_advice, BASE_VALUE_MODELS, SCARCITY_MODELS)
from modules.data_preparation import generate_pps_projections
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, DataReturnMode, GridUpdateMode

# --- Constants ---
BASE_VALUE_MODELS = [
    "Blended (VORP + Market)", 
    "Pure VORP", 
    "Pure Market Value",
    "Risk-Adjusted VORP",
    "Expert Consensus Value (ECV)"
]

SCARCITY_MODELS = [
    "Tier Scarcity", 
    "Position Scarcity", 
    "Roster Slot Demand",
    "Contrarian Fade",
    "Opponent Budget Targeting",
    "None"
]

st.set_page_config(layout="wide")

st.title("Live Auction Draft Tool")

# --- State Initialization ---
def initialize_session_state():
    """Initializes all the necessary variables in Streamlit's session state."""
    if 'draft_started' not in st.session_state:
        st.session_state.draft_started = False
        st.session_state.projections_generated = False
        st.session_state.num_teams = 16
        st.session_state.budget_per_team = 200
        st.session_state.roster_spots_per_team = 10
        st.session_state.games_in_season = 82
        st.session_state.team_names = [f"Team {i+1}" for i in range(st.session_state.num_teams)]
        st.session_state.available_players = pd.DataFrame()
        if 'initial_pos_counts' not in st.session_state:
            st.session_state.initial_pos_counts = pd.Series()
        if 'initial_tier_counts' not in st.session_state:
            st.session_state.initial_tier_counts = pd.Series()
        st.session_state.drafted_players = []
        st.session_state.total_money_spent = 0
        # Initialize team-specific states
        st.session_state.teams = {name: {'budget': st.session_state.budget_per_team, 'players': []} for name in st.session_state.team_names}
        st.session_state.base_value_models = [BASE_VALUE_MODELS[0]]
        st.session_state.scarcity_models = [SCARCITY_MODELS[0]]
        if 'tier_cutoffs' not in st.session_state:
            st.session_state.tier_cutoffs = {
                'Tier 1': 0.98, 'Tier 2': 0.90, 'Tier 3': 0.75, 'Tier 4': 0.50
            }
        if 'main_team' not in st.session_state:
            st.session_state.main_team = None
        if 'injured_players_text' not in st.session_state:
            st.session_state.injured_players_text = """Tyrese Haliburton (Full Season)
Damian Lillard (Full Season)
Kyrie Irving (Half Season)
Jayson Tatum (Full Season)
Dejounte Murray (Half Season)"""

initialize_session_state()

# --- Sidebar for Setup ---
st.sidebar.header("League Setup")

if not st.session_state.draft_started:
    st.session_state.num_teams = st.sidebar.number_input("Number of Teams", min_value=8, max_value=20, value=st.session_state.num_teams)
    st.session_state.budget_per_team = st.sidebar.number_input("Budget per Team ($)", min_value=100, max_value=500, value=st.session_state.budget_per_team, step=10)
    st.session_state.roster_spots_per_team = st.sidebar.number_input("Roster Spots per Team", min_value=5, max_value=20, value=st.session_state.roster_spots_per_team)

    st.sidebar.header("Projection Settings")
    st.session_state.games_in_season = st.sidebar.number_input("Games in a Season", min_value=1, max_value=82, value=st.session_state.games_in_season, disabled=st.session_state.draft_started)

    with st.sidebar.expander("Valuation Model Settings", expanded=True):
        st.write("**Base Value Calculation Model(s)**")
        selected_base_models = []
        for model in BASE_VALUE_MODELS:
            if st.checkbox(model, key=f"base_{model}", value=(model in st.session_state.base_value_models)):
                selected_base_models.append(model)
        st.session_state.base_value_models = selected_base_models

        st.write("**In-Draft Scarcity Model(s)**")
        selected_scarcity_models = []
        for model in SCARCITY_MODELS:
            if st.checkbox(model, key=f"scarcity_{model}", value=(model in st.session_state.scarcity_models)):
                selected_scarcity_models.append(model)
        st.session_state.scarcity_models = selected_scarcity_models

        # For backward compatibility and single-model logic, we still need a primary model.
        if st.session_state.base_value_models:
            st.session_state.base_value_model = st.session_state.base_value_models[0]
        else:
            st.session_state.base_value_model = BASE_VALUE_MODELS[0]

        if st.session_state.scarcity_models:
            st.session_state.scarcity_model = st.session_state.scarcity_models[0]
        else:
            st.session_state.scarcity_model = SCARCITY_MODELS[0]

    # Roster composition is now always available
    with st.sidebar.expander("Roster Composition", expanded=True):
        st.write("Define your league's starting roster spots.")
        if 'roster_composition' not in st.session_state:
            st.session_state.roster_composition = {
                'G': 3, 'F': 3, 'C': 2, 'Flx': 2, 'Bench': 0
            }
        
        st.session_state.roster_composition['G'] = st.number_input("Guard (G) Spots", min_value=0, value=st.session_state.roster_composition.get('G', 3), key='g_spots')
        st.session_state.roster_composition['F'] = st.number_input("Forward (F) Spots", min_value=0, value=st.session_state.roster_composition.get('F', 3), key='f_spots')
        st.session_state.roster_composition['C'] = st.number_input("Center (C) Spots", min_value=0, value=st.session_state.roster_composition.get('C', 2), key='c_spots')
        st.session_state.roster_composition['Flx'] = st.number_input("Flex (Flx) Spots", min_value=0, value=st.session_state.roster_composition.get('Flx', 2), key='flx_spots')
        st.session_state.roster_composition['Bench'] = st.number_input("Bench Spots", min_value=0, value=st.session_state.roster_composition.get('Bench', 5), key='bench_spots')

        total_roster_spots = sum(st.session_state.roster_composition.values())
        st.info(f"Total Roster Size: {total_roster_spots} players")
        
        if total_roster_spots != st.session_state.roster_spots_per_team:
            st.warning(f"Roster size ({total_roster_spots}) doesn't match 'Roster Spots per Team' ({st.session_state.roster_spots_per_team}).")

    with st.sidebar.expander("Customize Tier Percentiles"):
        st.info("Define the PPS percentile cutoffs for each tier.")
        st.session_state.tier_cutoffs['Tier 1'] = st.slider(
            "Tier 1 Cutoff (Top %)", 0.90, 1.0, st.session_state.tier_cutoffs.get('Tier 1', 0.98), 0.01,
            help="Players with PPS in this percentile or higher are Tier 1."
        )
        st.session_state.tier_cutoffs['Tier 2'] = st.slider(
            "Tier 2 Cutoff", 0.70, st.session_state.tier_cutoffs['Tier 1'], st.session_state.tier_cutoffs.get('Tier 2', 0.90), 0.01,
            help="Players between this and Tier 1 cutoff are Tier 2."
        )
        st.session_state.tier_cutoffs['Tier 3'] = st.slider(
            "Tier 3 Cutoff", 0.50, st.session_state.tier_cutoffs['Tier 2'], st.session_state.tier_cutoffs.get('Tier 3', 0.75), 0.01,
            help="Players between this and Tier 2 cutoff are Tier 3."
        )
        st.session_state.tier_cutoffs['Tier 4'] = st.slider(
            "Tier 4 Cutoff", 0.30, st.session_state.tier_cutoffs['Tier 3'], st.session_state.tier_cutoffs.get('Tier 4', 0.50), 0.01,
            help="Players between this and Tier 3 cutoff are Tier 4. All others are Tier 5."
        )

    with st.sidebar.expander("Injury Adjustments"):
        st.info("Enter one player per line, followed by `(Full Season)` or `(Half Season)`.")
        st.session_state.injured_players_text = st.text_area(
            "Injured Players List", 
            st.session_state.injured_players_text, 
            height=150,
            disabled=st.session_state.draft_started
        )

    with st.sidebar.expander("Customize Trend Weights"):
        st.session_state.s4_weight = st.slider("Most Recent Season (S4) Weight", 0.0, 1.0, 0.95, 0.05, disabled=st.session_state.draft_started)
        st.session_state.s3_weight = st.slider("S3 Weight", 0.0, 1.0, 0.10, 0.05, disabled=st.session_state.draft_started)
        st.session_state.s2_weight = st.slider("S2 Weight", 0.0, 1.0, 0.05, 0.05, disabled=st.session_state.draft_started)
        st.session_state.s1_weight = st.slider("Oldest Season (S1) Weight", 0.0, 1.0, 0.0, 0.05, disabled=st.session_state.draft_started)

        # Normalize weights
        total_weight = st.session_state.s1_weight + st.session_state.s2_weight + st.session_state.s3_weight + st.session_state.s4_weight
        if total_weight > 0:
            st.session_state.trend_weights = {
                'S1': st.session_state.s1_weight / total_weight,
                'S2': st.session_state.s2_weight / total_weight,
                'S3': st.session_state.s3_weight / total_weight,
                'S4': st.session_state.s4_weight / total_weight
            }
            st.success(f"Weights Normalized (Sum: {sum(st.session_state.trend_weights.values()):.2f})")
        else:
            st.session_state.trend_weights = {'S1': 0.15, 'S2': 0.20, 'S3': 0.30, 'S4': 0.35} # Default
            st.warning("All weights are zero. Using default weights.")
    
    if not st.session_state.projections_generated:
        if st.sidebar.button("Generate Projections"):
            with st.spinner("Generating PPS projections..."):
                success = generate_pps_projections(trend_weights=st.session_state.trend_weights, games_in_season=st.session_state.games_in_season)
                if success:
                    st.session_state.projections_generated = True
                    st.sidebar.success("Projections generated!")
                else:
                    st.sidebar.error("Failed to generate projections.")

    if st.session_state.projections_generated:
        if st.sidebar.button("Start Draft"):
            st.session_state.draft_started = True

            # Parse injured players from text area
            injured_players = {}
            for line in st.session_state.injured_players_text.strip().split('\n'):
                if '(Full Season)' in line:
                    player = line.split('(Full Season)')[0].strip()
                    injured_players[player] = 'Full Season'
                elif '(Half Season)' in line:
                    player = line.split('(Half Season)')[0].strip()
                    injured_players[player] = 'Half Season'

            # Load projections and calculate initial values
            try:
                # Load the projections and merge historical stats
                pps_df = pd.read_csv('data/player_projections.csv')
                try:
                    gp_df = pd.read_csv('data/PlayerGPOverYears.csv').rename(columns={'Player': 'PlayerName'})
                    fpg_df = pd.read_csv('data/PlayerFPperGameOverYears.csv').rename(columns={'Player': 'PlayerName'})
                    
                    # Define the columns to be used from the historical data files
                    gp_cols_to_use = ['PlayerName'] + [f'S{i} GP' for i in range(1, 5)]
                    fpg_cols_to_use = ['PlayerName'] + [f'S{i} FP/G' for i in range(1, 5)]

                    # Select only the necessary columns
                    gp_df = gp_df[gp_cols_to_use]
                    fpg_df = fpg_df[fpg_cols_to_use]

                    # Create renaming dictionaries
                    gp_rename_mapping = {f'S{i} GP': f'GP (S{i})' for i in range(1, 5)}
                    fpg_rename_mapping = {f'S{i} FP/G': f'FP/G (S{i})' for i in range(1, 5)}

                    # Rename the columns in the historical dataframes
                    gp_df.rename(columns=gp_rename_mapping, inplace=True)
                    fpg_df.rename(columns=fpg_rename_mapping, inplace=True)

                    # Merge the renamed dataframes
                    pps_df = pd.merge(pps_df, gp_df, on='PlayerName', how='left')
                    pps_df = pd.merge(pps_df, fpg_df, on='PlayerName', how='left')

                except FileNotFoundError:
                    st.warning("Could not find historical player data (GP, FP/G). Stats will be incomplete.")

                st.session_state.available_players, st.session_state.initial_tier_counts, st.session_state.initial_pos_counts = calculate_initial_values(
                    pps_df=pps_df,
                    num_teams=st.session_state.num_teams,
                    roster_spots_per_team=st.session_state.roster_spots_per_team,
                    budget_per_team=st.session_state.budget_per_team,
                    base_value_models=st.session_state.base_value_models,
                    tier_cutoffs=st.session_state.tier_cutoffs,
                    injured_players=injured_players
                )
                st.rerun()

            except FileNotFoundError:
                st.error("The 'player_projections.csv' file was not found. Please ensure it was generated correctly.")
                st.session_state.draft_started = False
    else:
        st.sidebar.warning("You must generate projections before starting the draft.")

else:
    st.sidebar.success("Draft in Progress!")
    st.sidebar.write(f"**{st.session_state.num_teams}** Teams")
    st.sidebar.write(f"**${st.session_state.budget_per_team}** Budget")
    st.sidebar.selectbox(
        "Select Your Main Team (for Advisor)",
        options=st.session_state.team_names,
        key='main_team',
        index=0 if st.session_state.main_team is None and st.session_state.team_names else st.session_state.team_names.index(st.session_state.main_team) if st.session_state.main_team in st.session_state.team_names else 0
    )

    if st.sidebar.button("Reset Draft"):
        # Clear session state and rerun to go back to the setup screen
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Add the 'Undo Last Pick' button if there are drafted players
    if st.session_state.drafted_players:
        if st.sidebar.button("Undo Last Pick"):
            # Get the last drafted player entry
            last_draft = st.session_state.drafted_players.pop()
            player_to_restore = last_draft.copy()
            player_name = player_to_restore.pop('PlayerName')
            draft_price = player_to_restore.pop('DraftPrice')
            team_name = player_to_restore.pop('Team')

            # --- Revert State ---
            # 1. Restore team budget and remove player from team's list
            st.session_state.teams[team_name]['budget'] += draft_price
            st.session_state.teams[team_name]['players'] = [p for p in st.session_state.teams[team_name]['players'] if p['PlayerName'] != player_name]

            # 2. Add player back to the available players dataframe
            player_to_restore_df = pd.DataFrame([player_to_restore], index=[player_name])
            player_to_restore_df.index.name = 'PlayerName'
            player_to_restore_df.reset_index(inplace=True)
            st.session_state.available_players = pd.concat([st.session_state.available_players, player_to_restore_df], ignore_index=True).sort_values(by='PPS', ascending=False)

            # 3. Update total money spent
            st.session_state.total_money_spent -= draft_price
            
            st.success(f"Reversed the pick of {player_name} by {team_name}.")
            st.rerun()

# --- Main Draft Interface (will be built out next) ---
def _get_model_col_name(model_name, prefix):
    """Helper to create a consistent column name from a model name."""
    clean_name = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
    return f"{prefix}{clean_name}"

def _get_model_col_name(model_name, prefix):
    """Helper to create a consistent column name from a model name."""
    clean_name = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
    return f"{prefix}{clean_name}"

if st.session_state.draft_started:
    def _get_model_col_name(model_name, prefix):
        """Helper to create a consistent column name from a model name."""
        clean_name = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
        return f"{prefix}{clean_name}"
    st.header("Draft Board")

    with st.expander("How Values Are Calculated"):
        st.markdown("""
        ### Core Concepts
        - **PPS (Player Power Score):** A weighted average of a player's fantasy points per game over the last four seasons, adjusted for games played. The trend weights are configurable in the sidebar.
        
        - **Tier:** Players are grouped into five tiers based on their PPS percentile rank against the *remaining* player pool. Tiers are recalculated after every pick.
            - **Tier 1:** Top 2% of players
            - **Tier 2:** Next 8% (90th-98th percentile)
            - **Tier 3:** Next 15% (75th-90th percentile)
            - **Tier 4:** Next 25% (50th-75th percentile)
            - **Tier 5:** Bottom 50%

        - **VORP (Value Over Replacement Player):** Measures a player's value relative to the best player likely available on the waiver wire. It's calculated as `Player's PPS - Replacement Level PPS`.
        - **Tier:** Players are grouped into five tiers based on their PPS percentile rank against the *remaining* player pool. Tiers are recalculated after every pick.

        --- 

        ### Base Value Models (Pre-Draft Estimates)
        These models provide a starting valuation for each player before the draft begins. You can select multiple models to see a range of values.

        - **Blended (VORP + Market):** A hybrid model that averages the `Pure VORP` and `Pure Market Value` models. This is often the most balanced and reliable starting point.
        - **Pure VORP:** A purely statistical approach. Value is determined by a player's VORP score as a percentage of the total VORP available for all draftable players, multiplied by the total auction budget for the league.
        - **Pure Market Value:** Ignores this year's stats and instead uses historical auction data to determine a player's price. It's useful for gauging public perception and hype.
        - **Risk-Adjusted VORP:** A variation of `Pure VORP` that discounts players based on their injury history and games played. Players with a clean bill of health get a premium.
        - **Expert Consensus Value (ECV):** A model that blends VORP with rankings and values from fantasy industry experts to create a more robust, consensus-driven price.

        --- 

        ### Scarcity & Adjustment Models (In-Draft Dynamics)
        These models adjust player values *during* the draft based on which players have already been selected. They apply a premium or discount to the `Base Value`.

        - **Tier Scarcity:** The classic model. As players from a specific tier (e.g., Tier 1) are drafted, the remaining players in that tier become more valuable due to scarcity. The premium is capped at +25%.
        - **Position Scarcity:** Similar to Tier Scarcity, but applies a premium based on the scarcity of a player's primary position (G, F, C). 
        - **Roster Slot Demand:** A more advanced model that dynamically shifts between `Position Scarcity` and `Tier Scarcity` based on which is more constrained relative to your league's specific roster settings.
        - **Contrarian Fade:** The opposite of a scarcity model. It applies a *discount* (up to -25%) to players in tiers or positions that are being drafted heavily, assuming those pools are overvalued. This helps identify players who might be relative bargains.
        - **Opponent Budget Targeting:** (Future Model) A placeholder model that currently mirrors `Tier Scarcity`. It will be developed to increase the value of top players when your opponents have large remaining budgets.
        - **None:** Applies no in-draft adjustment. The player's value will only change based on the amount of money remaining in the auction.
        """)

    total_league_money = st.session_state.num_teams * st.session_state.budget_per_team
    remaining_money_pool = total_league_money - st.session_state.total_money_spent

    # Recalculate dynamic values before displaying anything
    if not st.session_state.available_players.empty:
        recalculated_df = recalculate_dynamic_values(
            available_players_df=st.session_state.available_players.copy(),
            remaining_money_pool=remaining_money_pool,
            total_league_money=total_league_money,
            base_value_models=st.session_state.base_value_models, # Pass base models for confidence score
            scarcity_models=st.session_state.scarcity_models, # Pass the list of models
            initial_tier_counts=st.session_state.initial_tier_counts,
            initial_pos_counts=st.session_state.initial_pos_counts,
            tier_cutoffs=st.session_state.tier_cutoffs,
            roster_composition=st.session_state.get('roster_composition'),
            num_teams=st.session_state.get('num_teams')
        )
    else:
        recalculated_df = pd.DataFrame()

    # --- [1] DRAFT PLAYER SECTION ---
    st.subheader("Draft a Player")
    available_player_names = recalculated_df['PlayerName'].tolist() if not recalculated_df.empty else []
    selected_player_name = st.selectbox("Select Player to analyze or draft", options=available_player_names, label_visibility="collapsed")

    # This container holds the draft form and all the analysis for the selected player
    draft_analysis_container = st.container()

    # --- [5] AVAILABLE PLAYERS GRID ---
    st.markdown("---_**Available Players**_---")
    if not recalculated_df.empty:
        gb = GridOptionsBuilder.from_dataframe(recalculated_df)
        
        # Define and configure all columns for the grid
        display_cols = ['PlayerName', 'Position', 'Tier', 'PPS', 'Confidence', 'BaseValue', 'AdjValue']
        if len(st.session_state.base_value_models) > 1:
            for model in st.session_state.base_value_models:
                display_cols.append(_get_model_col_name(model, 'BV_'))
        if len(st.session_state.scarcity_models) > 1:
            for model in st.session_state.scarcity_models:
                if model != "None": display_cols.append(_get_model_col_name(model, 'AV_'))
        for col in ['VORP', 'MarketValue', 'Points']:
            if col in recalculated_df.columns and col not in display_cols: display_cols.append(col)

        final_display_df = recalculated_df[[col for col in display_cols if col in recalculated_df.columns]]

        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
        gb.configure_column("PlayerName", headerName="Player", pinned='left', width=180)
        gb.configure_column("Position", headerName="Pos", width=60)
        gb.configure_column("BaseValue", "Base Val", width=80, type=["numericColumn", "numberColumnFilter", "customNumericFormat"], valueFormatter="'$' + value.toFixed(0)")
        gb.configure_column("AdjValue", "Adj. Val", width=80, type=["numericColumn", "numberColumnFilter", "customNumericFormat"], valueFormatter="'$' + value.toFixed(0)")
        gb.configure_column("PPS", "Weighted FP/G", width=120, type=["numericColumn", "numberColumnFilter"], valueFormatter="value.toFixed(2)")

        js_tier_cell_style = JsCode("""function(params) { if (params.value == 1) { return {'color': 'white', 'backgroundColor': 'purple'}; } else if (params.value == 2) { return {'color': 'white', 'backgroundColor': 'blue'}; } else if (params.value == 3) { return {'color': 'black', 'backgroundColor': 'lightblue'}; } return {}; }""")
        gb.configure_column("Tier", width=60, cellStyle=js_tier_cell_style)

        js_confidence_cell_style = JsCode("""function(params) { if (params.value >= 90) { return {'color': 'white', 'backgroundColor': 'darkgreen'}; } else if (params.value >= 75) { return {'color': 'black', 'backgroundColor': 'gold'}; } else if (params.value < 75) { return {'color': 'white', 'backgroundColor': 'darkred'}; } return {}; }""")
        gb.configure_column("Confidence", width=100, cellStyle=js_confidence_cell_style, valueFormatter="value.toFixed(1) + '%'", type=["numericColumn", "numberColumnFilter"])

        for model in st.session_state.base_value_models: gb.configure_column(_get_model_col_name(model, 'BV_'), f"BV: {model}", width=100, type=["numericColumn", "numberColumnFilter", "customNumericFormat"], valueFormatter="'$' + value.toFixed(0)")
        for model in st.session_state.scarcity_models: gb.configure_column(_get_model_col_name(model, 'AV_'), f"AV: {model}", width=100, type=["numericColumn", "numberColumnFilter", "customNumericFormat"], valueFormatter="'$' + value.toFixed(0)")

        gb.configure_side_bar(filters_panel=True, columns_panel=True, defaultToolPanel="columns")
        gb.configure_grid_options(domLayout='normal')
        gridOptions = gb.build()
        AgGrid(final_display_df, gridOptions=gridOptions, enable_enterprise_modules=False, allow_unsafe_jscode=True, update_mode=GridUpdateMode.MODEL_CHANGED, height=600)

    # --- [6] TEAM ROSTERS & BUDGETS ---
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

    # --- DRAFT ANALYSIS CONTAINER (Populated if a player is selected) ---
    if selected_player_name:
        with draft_analysis_container:
            player_values = recalculated_df[recalculated_df['PlayerName'] == selected_player_name].iloc[0]

            # --- DRAFT FORM ---
            with st.form(key='draft_form'):
                st.markdown(f"#### Draft **{selected_player_name}**")
                draft_col1, draft_col2, draft_col3 = st.columns(3)
                team_selection = draft_col1.selectbox("Select Team", options=st.session_state.team_names)
                draft_price = draft_col2.number_input("Draft Price", min_value=1, value=int(player_values.get('AdjValue', 1)))
                submit_button = draft_col3.form_submit_button(label=f"Draft Player")

                if submit_button:
                    if draft_price > st.session_state.teams[team_selection]['budget']:
                        st.error(f"{team_selection} cannot afford {selected_player_name} at this price.")
                    else:
                        st.session_state.teams[team_selection]['budget'] -= draft_price
                        player_data = {
                            'PlayerName': selected_player_name,
                            'Position': player_values['Position'],
                            'DraftPrice': draft_price,
                            'Tier': player_values.get('Tier', 'N/A'),
                            'BaseValue': player_values.get('Value', 0), # Using 'Value' as BaseValue
                            'Confidence': f"{player_values.get('Confidence', 100.0):.1f}%"
                        }
                        st.session_state.teams[team_selection]['players'].append(player_data)
                        st.session_state.total_money_spent += draft_price
                        draft_entry = player_values.to_dict()
                        draft_entry.update({'DraftPrice': draft_price, 'Team': team_selection})
                        st.session_state.drafted_players.append(draft_entry)
                        st.session_state.available_players = st.session_state.available_players[st.session_state.available_players['PlayerName'] != selected_player_name]
                        st.success(f"{selected_player_name} drafted by {team_selection} for ${draft_price}!")
                        st.rerun()
            
            st.markdown("---")

            # --- [2] VALUATION CONFIDENCE & AVERAGE ---
            col1, col2 = st.columns(2)
            col1.metric("Valuation Confidence", f"{player_values.get('Confidence', 100.0):.1f}%")
            col2.metric("Average Valuation", f"${player_values.get('ValueMean', 0):.0f}")
            st.markdown("---")

            # --- [3] BASE & ADJUSTED VALUATIONS ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Base Valuations**")
                for model in st.session_state.base_value_models:
                    st.markdown(f"_{model}_: **${player_values.get(_get_model_col_name(model, 'BV_'), 0):,.0f}**")
            with col2:
                st.markdown("**Adjusted Valuations (In-Draft)**")
                for model in st.session_state.scarcity_models:
                    st.markdown(f"_{model}_: **${player_values.get(_get_model_col_name(model, 'AV_'), 0):,.0f}**")
            st.markdown("---")

            # --- [4] KEY PLAYER METRICS & ADVISOR ---
            col1, col2 = st.columns([2,1])
            with col1:
                st.markdown("**Key Player Metrics**")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Tier", str(player_values.get('Tier', 'N/A')))
                m_col2.metric("VORP", f"{player_values.get('VORP', 0):.2f}")
                m_col3.metric("Market Value", f"${player_values.get('MarketValue', 0):.0f}")

                stats_df = pd.DataFrame({
                    'Season': ['S1', 'S2', 'S3', 'S4'],
                    'GP': [player_values.get(f'GP (S{i})', 'N/A') for i in range(1, 5)],
                    'FP/G': [f"{player_values.get(f'FP/G (S{i})', 0):.2f}" for i in range(1, 5)]
                })
                st.markdown("**Historical Performance**")
                st.dataframe(stats_df, use_container_width=True)
            with col2:
                st.markdown("**Draft Advisor**")
                if st.session_state.main_team:
                    advice_list = generate_draft_advice(
                        selected_player=player_values,
                        team_data=st.session_state.teams[st.session_state.main_team],
                        roster_composition=st.session_state.roster_composition,
                        roster_spots_per_team=st.session_state.roster_spots_per_team
                    )
                    for advice in advice_list:
                        st.markdown(f"- {advice}")
                else:
                    st.warning("Select a 'Main Team' in the sidebar to get personalized advice.")

                # Display trend weights used for projections
                if st.session_state.trend_weights:
                    weights = st.session_state.trend_weights
                    weights_str = ", ".join([f"{season}: {weight:.0%}" for season, weight in sorted(weights.items(), reverse=True)])
                    st.info(f"**Projections based on trend weights:** {weights_str}")
    else:
        st.warning("All players have been drafted!")

    st.subheader("Team Rosters & Budgets")
    for team_name, team_data in st.session_state.teams.items():
        with st.expander(f"{team_name} - Budget: ${team_data['budget']}"):
            if team_data['players']:
                roster_df = pd.DataFrame(team_data['players'])
                st.dataframe(roster_df[['PlayerName', 'Position', 'Tier', 'DraftPrice', 'BaseValue', 'Confidence']], use_container_width=True)
            else:
                st.write("No players drafted yet.")

    # --- Draft Summary ---
    if st.session_state.drafted_players:
        st.subheader("Draft Summary")
        drafted_df = pd.DataFrame(st.session_state.drafted_players)
        
        # Ensure required columns exist, fill with 0 if not (for robustness)
        for col in ['BaseValue', 'AdjValue', 'DraftPrice']:
            if col not in drafted_df.columns:
                drafted_df[col] = 0

        drafted_df['Value'] = drafted_df['AdjValue'] - drafted_df['DraftPrice']
        
        # Format for display
        summary_df = drafted_df[['PlayerName', 'Team', 'Position', 'DraftPrice', 'AdjValue', 'BaseValue', 'Value']].copy()
        summary_df['DraftPrice'] = summary_df['DraftPrice'].apply(lambda x: f"${x:,.0f}")
        summary_df['AdjValue'] = summary_df['AdjValue'].apply(lambda x: f"${x:,.0f}")
        summary_df['BaseValue'] = summary_df['BaseValue'].apply(lambda x: f"${x:,.0f}")
        
        # Color the 'Value' column based on whether it's a bargain or a reach
        def color_value(val):
            if val > 0:
                return f'<span style="color: green;">${val:,.0f} (Bargain)</span>'
            elif val < 0:
                return f'<span style="color: red;">${val:,.0f} (Reach)</span>'
            else:
                return f"${val:,.0f}"
        summary_df['Value'] = summary_df['Value'].apply(color_value)

        st.markdown(summary_df.to_html(escape=False), unsafe_allow_html=True)

else:
    st.info("Please configure your league settings in the sidebar and click 'Start Draft' to begin.")
