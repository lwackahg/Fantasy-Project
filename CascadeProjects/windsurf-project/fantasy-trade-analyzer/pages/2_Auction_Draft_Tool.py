import streamlit as st
import pandas as pd
import json
from pathlib import Path
from logic.auction_tool import calculate_initial_values, recalculate_dynamic_values
from modules.data_preparation import generate_pps_projections

# --- Persistence Functions ---
INJURY_FILE = Path('data/injured_players.json')

def load_injured_players():
    """Loads the dictionary of injured players from a JSON file."""
    if INJURY_FILE.exists():
        try:
            with open(INJURY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_injured_players(injured_dict):
    """Saves the dictionary of injured players to a JSON file."""
    INJURY_FILE.parent.mkdir(exist_ok=True)
    with open(INJURY_FILE, 'w') as f:
        json.dump(injured_dict, f, indent=4)

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
        st.session_state.injured_players = load_injured_players()
        st.session_state.draft_started = False
        st.session_state.projections_generated = False
        st.session_state.num_teams = 16
        st.session_state.budget_per_team = 200
        st.session_state.roster_spots_per_team = 10
        st.session_state.games_in_season = 82
        st.session_state.team_names = [f"Team {i+1}" for i in range(st.session_state.num_teams)]
        st.session_state.available_players = pd.DataFrame()
        st.session_state.initial_tier_counts = {}
        st.session_state.drafted_players = []
        st.session_state.total_money_spent = 0
        # Initialize team-specific states
        st.session_state.teams = {name: {'budget': st.session_state.budget_per_team, 'players': []} for name in st.session_state.team_names}
        st.session_state.base_value_models = [BASE_VALUE_MODELS[0]]
        st.session_state.scarcity_models = [SCARCITY_MODELS[0]]

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

    # --- Injury Settings ---
    with st.sidebar.expander("Injury Settings", expanded=True):
        # Load player names for the dropdown
        try:
            all_players_df = pd.read_csv('data/player_projections.csv')
            all_player_names = sorted(list(all_players_df['PlayerName'].dropna().astype(str).unique()))
        except FileNotFoundError:
            all_player_names = []
            st.warning("Player projections file not found. Cannot populate injury list.")

        # UI for adding a new injured player
        col1, col2 = st.columns([3, 2])
        with col1:
            player_to_injure = st.selectbox("Select Player to Injure", options=all_player_names, index=None, placeholder="Choose a player...", key="player_injure_select")
        with col2:
            injury_type = st.selectbox("Injury Duration", options=["Full Season", "Half Season"], key="injury_type_select")

        if st.button("Add Injured Player", use_container_width=True, disabled=st.session_state.draft_started):
            if player_to_injure and player_to_injure not in st.session_state.injured_players:
                st.session_state.injured_players[player_to_injure] = injury_type
                save_injured_players(st.session_state.injured_players)
                st.success(f"Added {player_to_injure} to injured list ({injury_type}).")
                st.rerun()
            elif player_to_injure:
                st.warning(f"{player_to_injure} is already on the injured list.")

        # Display and manage the list of currently injured players
        if st.session_state.injured_players:
            st.markdown("**Current Injured List**")
            for player, status in list(st.session_state.injured_players.items()):
                col1, col2 = st.columns([3, 1])
                col1.text(f"{player} ({status})")
                if col2.button(f"Remove {player}", key=f"remove_{player}", use_container_width=True, disabled=st.session_state.draft_started):
                    del st.session_state.injured_players[player]
                    save_injured_players(st.session_state.injured_players)
                    st.rerun()

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
    
    if st.sidebar.button("Generate Projections"):
        with st.spinner("Generating PPS projections..."):
            success = generate_pps_projections(trend_weights=st.session_state.trend_weights, injured_players=st.session_state.injured_players, games_in_season=st.session_state.games_in_season)
            if success:
                st.session_state.projections_generated = True
                st.sidebar.success("Projections generated!")
            else:
                st.sidebar.error("Failed to generate projections.")

    if st.session_state.projections_generated:
        if st.sidebar.button("Start Draft"):
            st.session_state.draft_started = True
            # Load projections and calculate initial values
            try:
                # Load the projections and merge historical stats
                pps_df = pd.read_csv('data/player_projections.csv')
                try:
                    gp_df = pd.read_csv('data/PlayerGPOverYears.csv').rename(columns={'Player': 'PlayerName'})
                    fpg_df = pd.read_csv('data/PlayerFPperGameOverYears.csv').rename(columns={'Player': 'PlayerName'})
                    
                    # Merge historical data into the main projections dataframe
                    pps_df = pd.merge(pps_df, gp_df, on='PlayerName', how='left')
                    pps_df = pd.merge(pps_df, fpg_df, on='PlayerName', how='left')

                except FileNotFoundError:
                    st.warning("Could not find historical player data (GP, FP/G). Stats will be incomplete.")



                st.session_state.available_players, st.session_state.initial_tier_counts, st.session_state.initial_pos_counts = calculate_initial_values(
                    pps_df=pps_df,
                    num_teams=st.session_state.num_teams,
                    roster_spots_per_team=st.session_state.roster_spots_per_team,
                    budget_per_team=st.session_state.budget_per_team,
                    base_value_models=st.session_state.base_value_models # Pass the list of models
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
    st.session_state.my_team = st.sidebar.selectbox("Select Your Team", options=st.session_state.team_names)

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
if st.session_state.draft_started:
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
            scarcity_models=st.session_state.scarcity_models, # Pass the list of models
            initial_tier_counts=st.session_state.initial_tier_counts,
            initial_pos_counts=st.session_state.initial_pos_counts,
            roster_composition=st.session_state.get('roster_composition'),
            num_teams=st.session_state.get('num_teams')
        )
    else:
        recalculated_df = pd.DataFrame()

    # Display Key Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Money in Pool", f"${total_league_money:,}")
    col2.metric("Money Spent", f"${st.session_state.total_money_spent:,}")
    col3.metric("Remaining Money", f"${remaining_money_pool:,}", delta_color="inverse")

    # --- Draft Player Form ---
    st.subheader("Draft a Player")
    available_player_names = recalculated_df['PlayerName'].tolist() if not recalculated_df.empty else []
    selected_player_name = st.selectbox("Select Player", options=available_player_names)

    # Display selected player's values from all selected models
    if selected_player_name:
        player_values = recalculated_df[recalculated_df['PlayerName'] == selected_player_name].iloc[0]

        st.markdown("---")
        col1, col2 = st.columns(2)

        def get_model_col_name(model_name, prefix):
            clean_name = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
            return f"{prefix}{clean_name}"

        with col1:
            st.markdown("**Base Valuations**")
            for model in st.session_state.base_value_models:
                col_name = get_model_col_name(model, 'BV_')
                value = player_values.get(col_name, 0)
                st.markdown(f"_{model}_: **${value:,.0f}**")
        
        with col2:
            st.markdown("**Adjusted Valuations (In-Draft)**")
            for model in st.session_state.scarcity_models:
                col_name = get_model_col_name(model, 'AV_')
                value = player_values.get(col_name, 0)
                st.markdown(f"_{model}_: **${value:,.0f}**")
        st.markdown("---")

        # --- Player Stats Display ---
        st.markdown("**Key Player Metrics**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tier", str(player_values.get('Tier', 'N/A')))
        col2.metric("VORP", f"{player_values.get('VORP', 0):.2f}")
        col3.metric("PPS", f"{player_values.get('PPS', 0):.2f}")
        col4.metric("Market Value", f"${player_values.get('MarketValue', 0):.0f}")

        # Display historical stats in a more structured way
        stats_df = pd.DataFrame({
            'Season': ['S1', 'S2', 'S3', 'S4'],
            'GP': [
                player_values.get('S1_GP', 'N/A'), 
                player_values.get('S2_GP', 'N/A'), 
                player_values.get('S3_GP', 'N/A'), 
                player_values.get('S4_GP', 'N/A')
            ],
            'FP/G': [
                f"{player_values.get('S1_FP/G', 0):.2f}",
                f"{player_values.get('S2_FP/G', 0):.2f}",
                f"{player_values.get('S3_FP/G', 0):.2f}",
                f"{player_values.get('S4_FP/G', 0):.2f}"
            ]
        }).set_index('Season')

        st.dataframe(stats_df.T, use_container_width=True)
        st.markdown("---</br>", unsafe_allow_html=True)

    with st.form("draft_form"):
        col1, col2 = st.columns(2)
        with col1:
            drafting_team = st.selectbox("Select Team", options=st.session_state.team_names)
        with col2:
            draft_price = st.number_input("Draft Price ($)", min_value=1, max_value=st.session_state.budget_per_team, value=10)
        
        submitted = st.form_submit_button("Draft Player")
        if submitted:
            team_budget = st.session_state.teams[drafting_team]['budget']
            roster_spots_to_fill = st.session_state.roster_spots_per_team - len(st.session_state.teams[drafting_team]['players'])
            max_bid = team_budget - (roster_spots_to_fill - 1)

            if draft_price > max_bid:
                st.error(f"{drafting_team} cannot bid more than ${max_bid} to save room for their remaining {roster_spots_to_fill} roster spots.")
            else:
                # --- Update State Logic ---
                player_to_draft_series = st.session_state.available_players[st.session_state.available_players['PlayerName'] == selected_player_name].iloc[0].copy()
                
                # Add draft-specific info to the player's data
                player_to_draft_series['DraftPrice'] = draft_price
                player_to_draft_series['Team'] = drafting_team
                
                # Store the complete player data (as a Series) in the drafted list
                st.session_state.drafted_players.append(player_to_draft_series)

                # Update team-specific state
                st.session_state.teams[drafting_team]['players'].append(player_to_draft_series)
                st.session_state.teams[drafting_team]['budget'] -= draft_price
                
                # Update global state
                st.session_state.total_money_spent += draft_price
                st.session_state.available_players = st.session_state.available_players[st.session_state.available_players['PlayerName'] != selected_player_name]
                
                st.success(f"Drafted {selected_player_name} to {drafting_team} for ${draft_price}!")
                st.rerun()

    # --- Display Tables ---
    st.subheader("Available Players")
    if not recalculated_df.empty:
        # --- Dynamic Column Display Logic ---
        def get_model_col_name(model_name, prefix):
            clean_name = ''.join(word.title() for word in model_name.replace('(', '').replace(')', '').replace('-', '').split())
            return f"{prefix}{clean_name}"

        # Start with core columns
        display_cols = ['PlayerName', 'Position', 'Tier', 'BaseValue', 'AdjValue']

        # Add columns for selected Base Value models
        if len(st.session_state.base_value_models) > 1:
            for model in st.session_state.base_value_models:
                col_name = get_model_col_name(model, 'BV_')
                if col_name in recalculated_df.columns and col_name not in display_cols:
                    display_cols.append(col_name)

        # Add columns for selected Scarcity models
        if len(st.session_state.scarcity_models) > 1:
            for model in st.session_state.scarcity_models:
                col_name = get_model_col_name(model, 'AV_')
                if col_name in recalculated_df.columns and col_name not in display_cols:
                    display_cols.append(col_name)
        
        # Add other important metrics
        for col in ['VORP', 'MarketValue', 'Points', 'PPS']:
            if col in recalculated_df.columns and col not in display_cols:
                display_cols.append(col)

        # Ensure all display_cols exist in the dataframe before showing
        final_display_cols = [col for col in display_cols if col in recalculated_df.columns]

        st.dataframe(recalculated_df[final_display_cols], use_container_width=True, height=500)

        # Display the trend weights used for the calculation
        if 'trend_weights' in st.session_state and st.session_state.trend_weights:
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
                st.dataframe(roster_df[['PlayerName', 'Position', 'Tier', 'DraftPrice', 'BaseValue']], use_container_width=True)
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
