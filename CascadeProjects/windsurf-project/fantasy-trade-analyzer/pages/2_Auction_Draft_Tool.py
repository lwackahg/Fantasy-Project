import streamlit as st
import pandas as pd
from logic.auction_tool import (calculate_initial_values, recalculate_dynamic_values, BASE_VALUE_MODELS, SCARCITY_MODELS)
from modules.data_preparation import generate_pps_projections
from logic.smart_auction_bot import SmartAuctionBot

from logic.ui_components import (
    render_setup_page, render_sidebar_in_draft, render_value_calculation_expander,
    render_draft_board, render_team_rosters, render_player_analysis, render_draft_summary
)

def clear_player_on_block():
    """Callback function to reset the player on the block."""
    st.session_state.player_on_the_block = None

def drafting_callback(selected_player_name, team_selection, draft_price, assigned_position, player_values):
    """Handles the logic for drafting a player and then clearing the selection."""
    if draft_price > st.session_state.teams[team_selection]['budget']:
        st.error(f"{team_selection} cannot afford {selected_player_name} at this price.")
        return

    if not assigned_position or assigned_position == '':
        st.error("You must assign a position to this player before drafting.")
        return

    # Proceed with draft
    draft_entry = player_values.to_dict()
    draft_entry['PlayerName'] = player_values.name
    draft_entry.update({'DraftPrice': draft_price, 'Team': team_selection, 'Position': assigned_position})

    st.session_state.drafted_players.append(draft_entry)
    st.session_state.teams[team_selection]['players'].append(draft_entry)
    st.session_state.teams[team_selection]['budget'] -= draft_price
    st.session_state.total_money_spent += draft_price
    st.session_state.available_players = st.session_state.available_players[st.session_state.available_players['PlayerName'] != selected_player_name]

    if st.session_state.draft_order:
        current_index = st.session_state.get('current_nominating_team_index', 0)
        next_index = (current_index + 1) % len(st.session_state.draft_order)
        st.session_state.current_nominating_team_index = next_index

    st.toast(f"{selected_player_name} drafted by {team_selection} for ${draft_price}!", icon="🎉")
    st.session_state.player_on_the_block = None

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
        st.session_state.games_in_season = 75
        st.session_state.team_names = [f"Team {i+1}" for i in range(st.session_state.num_teams)]
        st.session_state.available_players = pd.DataFrame()
        st.session_state.initial_pos_counts = pd.Series(dtype='int64')
        st.session_state.initial_tier_counts = pd.Series(dtype='int64')
        st.session_state.drafted_players = []
        st.session_state.total_money_spent = 0
        st.session_state.teams = {name: {'budget': st.session_state.budget_per_team, 'players': []} for name in st.session_state.team_names}
        st.session_state.base_value_models = [BASE_VALUE_MODELS[0]]
        st.session_state.scarcity_models = [SCARCITY_MODELS[0]]
        st.session_state.tier_cutoffs = {'Tier 1': 1.00, 'Tier 2': 0.88, 'Tier 3': 0.70, 'Tier 4': 0.45}
        st.session_state.main_team = None
        st.session_state.trend_weights = {'S1': 0.0, 'S2': 0.05, 'S3': 0.10, 'S4': 1.00}
        st.session_state.roster_composition = {'G': 3, 'F': 3, 'C': 2, 'Flx': 2, 'Bench': 0}
        st.session_state.draft_order = []
        st.session_state.current_nominating_team_index = 0


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


# --- Main App Flow ---
if not st.session_state.draft_started:
    render_setup_page()

    st.markdown("---")
    st.header("Set Draft Order")
    st.info("Drag and drop the teams to set the nomination order for the draft.")
    st.session_state.draft_order = st.multiselect(
        'Draft Order',
        options=st.session_state.team_names,
        default=st.session_state.team_names,
        label_visibility="collapsed"
    )

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
                
                with st.spinner("Calculating initial player values..."):
                    pps_df = pd.read_csv('data/player_projections.csv')
                    st.session_state.available_players, st.session_state.initial_pos_counts, st.session_state.initial_tier_counts = calculate_initial_values(
                        pps_df=pps_df,
                        num_teams=st.session_state.num_teams,
                        budget_per_team=st.session_state.budget_per_team,
                        roster_composition=st.session_state.roster_composition,
                        base_value_models=st.session_state.base_value_models,
                        tier_cutoffs=st.session_state.tier_cutoffs,
                        injured_players=injured_players
                    )
                st.rerun()
        else:
            st.info("Generate projections before you can start the draft.")

else: # Draft is in progress
    render_sidebar_in_draft()

    st.header("Draft Board")
    with st.expander("How are these values calculated?"):
        st.markdown("""
        ### Core Player Valuations
        - **Base Value**: Calculated from player projections using a z-score methodology against a replacement-level player. This is a stable, pre-draft valuation.
        - **Market Value**: Reflects the expected auction price based on data from similar leagues.
        - **Adjusted Value (In-Draft)**: This is the most important number. It adjusts the Base Value in real-time based on:
            - **Positional Scarcity**: As players at a certain position are drafted, the value of remaining players at that position increases.
            - **Inflation/Deflation**: The model tracks the total money spent versus the total value of players drafted. If teams overpay, inflation occurs, and the value of remaining players goes up. If players are acquired for less than their value, deflation occurs.
        - **VORP (Value Over Replacement Player)**: A measure of how much a player is expected to contribute beyond a readily available replacement player.
        """)


    # --- Nomination and Bidding --- 
    nominating_team_name = st.session_state.draft_order[st.session_state.current_nominating_team_index] if st.session_state.draft_order else 'N/A'
    st.subheader(f"On the Clock: {nominating_team_name}")

    # --- Value Recalculation and Bot Logic ---
    if not st.session_state.available_players.empty:
        # Define money pool variables before they are used
        total_league_money = st.session_state.num_teams * st.session_state.budget_per_team
        total_money_spent = sum(player['DraftPrice'] for player in st.session_state.drafted_players)
        remaining_money_pool = total_league_money - total_money_spent

        # 1. Recalculate dynamic values for the current available players.
        # The result is stored in a temporary variable for this script run and not written back to session_state
        # to prevent data corruption. The session_state.available_players is the source of truth.
        recalculated_df = recalculate_dynamic_values(
            available_players_df=st.session_state.available_players.copy(), # Use a copy to be safe
            remaining_money_pool=remaining_money_pool,
            total_league_money=total_league_money,
            base_value_models=st.session_state.base_value_models,
            scarcity_models=st.session_state.scarcity_models,
            initial_tier_counts=st.session_state.initial_tier_counts,
            initial_pos_counts=st.session_state.initial_pos_counts,
            tier_cutoffs=st.session_state.tier_cutoffs,
            roster_composition=st.session_state.get('roster_composition'),
            num_teams=st.session_state.get('num_teams')
        )

        # 2. Now, initialize and use the Smart Auction Bot with the recalculated data
        my_team_name = st.session_state.get('main_team', st.session_state.team_names[0])
        bot = SmartAuctionBot(
            my_team_name=my_team_name,
            teams=st.session_state.teams,
            available_players=recalculated_df, # Pass the recalculated df
            draft_history=st.session_state.drafted_players,
            budget=st.session_state.budget_per_team,
            roster_composition=st.session_state.roster_composition
        )

        if nominating_team_name == my_team_name:
            with st.expander("🤖 **Nomination Advice**", expanded=True):
                advice = bot.get_nomination_recommendation()
                if advice and advice.get('player'):
                    st.metric("Recommended Nomination", advice['player'])
                    st.info(f"**Reasoning:** {advice['reason']}")
                else:
                    st.warning("No nomination recommendation available at this time.")

    else:
        recalculated_df = pd.DataFrame()
        st.balloons()
        st.success("Draft Complete!")

    st.subheader("Analyze or Draft a Player")
    player_options = []
    if not recalculated_df.empty:
        # First, get the list of player names for the dropdown
        if 'PlayerName' in recalculated_df.columns:
            player_options = recalculated_df['PlayerName'].tolist()
        
        # Then, set the index for fast lookups later
        if 'PlayerName' in recalculated_df.columns:
            recalculated_df.set_index('PlayerName', inplace=True)

    # The selectbox now directly controls the 'player_on_the_block' state
    # Initialize the key in session state if it doesn't exist
    if 'player_on_the_block' not in st.session_state:
        st.session_state.player_on_the_block = None

    # Determine the index of the currently selected player to ensure the selectbox is synchronized
    try:
        current_selection_index = player_options.index(st.session_state.player_on_the_block)
    except (ValueError, TypeError):
        current_selection_index = None # Handles case where player is not in list or is None

    # The selectbox's state is now fully managed
    st.selectbox(
        "Select a player to put on the block:",
        options=player_options,
        index=current_selection_index,
        key='player_on_the_block',
        placeholder="Choose a player..."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    render_team_rosters()
    render_draft_summary()

    draft_analysis_container = st.container()
    with draft_analysis_container:
        # Check for a truthy value to ensure a player is actually selected
        if st.session_state.get('player_on_the_block'):
            render_player_analysis(st.session_state.player_on_the_block, recalculated_df, drafting_callback)
