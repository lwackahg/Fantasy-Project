import streamlit as st
import pandas as pd
from logic.auction_tool import (calculate_initial_values, recalculate_dynamic_values, BASE_VALUE_MODELS, SCARCITY_MODELS)
from modules.data_preparation import generate_pps_projections
from logic.smart_auction_bot import SmartAuctionBot
from logic.ui_components import (
    render_setup_page, render_sidebar_in_draft, render_value_calculation_expander,
    render_draft_board, render_team_rosters, render_player_analysis, render_draft_summary
)

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

        ### Smart Auction Bot Logic
        The bot provides two types of recommendations:
        - **Nomination Advice**: This is a strategic recommendation for who you should nominate next. It is based on the overall state of the draft and does not change when you select different players in the dropdown below. The bot uses several tactics:
            - **Target High Value**: Recommends a player who provides the highest *marginal value* specifically for your team's build.
            - **Pressure Opponents**: Suggests nominating a top player you don't need to force other teams to spend their budget.
            - **Find a Bargain**: Identifies players with high potential (VORP) relative to their expected cost.
        - **Bidding Advice**: This advice appears when a player is 'On The Block'. The bot calculates a **Max Bid** based on:
            - The player's **Adjusted Value**.
            - The player's **Marginal Value**: The bot runs a rapid simulation to measure how much that specific player improves your team's optimal lineup. A high marginal value means the player is a perfect fit for your roster, justifying a higher bid.
        """)


    # --- Nomination and Bidding --- 
    nominating_team_name = st.session_state.draft_order[st.session_state.current_nominating_team_index] if st.session_state.draft_order else 'N/A'
    st.subheader(f"On the Clock: {nominating_team_name}")

    if 'available_players' in st.session_state and not st.session_state.available_players.empty:
        # Find the user's team name, default to the first team if not set
        my_team_name = st.session_state.get('main_team')
        if not my_team_name:
            my_team_name = st.session_state.team_names[0]
        
        bot = SmartAuctionBot(
            available_players=st.session_state.available_players,
            my_team=st.session_state.teams[my_team_name].get('players', pd.DataFrame()),
            budget=st.session_state.budget_per_team,
            roster_composition=st.session_state.roster_composition,
            team_strategy_name=st.session_state.get('team_strategy', 'balanced')
        )

        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            my_team_name = st.session_state.get('main_team')
            if nominating_team_name == my_team_name:
                with st.expander("🤖 **Nomination Advice**", expanded=True):
                    # Display simple advice by default or if no advanced advice has been generated yet
                    if 'nomination_reason' not in st.session_state or st.session_state.nomination_reason is None:
                        simple_advice = bot.get_simple_nomination_recommendation()
                        st.session_state.nomination_reason = simple_advice.get('reason')
                        st.session_state.recommended_nominee = simple_advice.get('player')

                    # Display the current recommendation
                    if st.session_state.get('recommended_nominee'):
                        st.metric("Recommended Nomination", st.session_state.recommended_nominee)
                        st.info(f"Reasoning: {st.session_state.nomination_reason}")
                    else:
                        st.warning("No nomination recommendations at this time.")

                    # Button to get advanced advice
                    if st.button("Get Advanced Advice"):
                        with st.spinner("Running advanced analysis..."):
                            advanced_advice = bot.get_nomination_recommendation()
                            st.session_state.nomination_reason = advanced_advice['reason']
                            st.session_state.recommended_nominee = advanced_advice['player']
                            st.rerun()
            else:
                st.empty()
        
        with rec_col2:
            with st.expander("💰 **Bidding Advice**", expanded=True):
                player_on_block = st.session_state.get('player_on_the_block')
                if player_on_block:
                    st.markdown(f"#### Advice for **{player_on_block}**")
                    advice = bot.get_bidding_advice(player_on_block)
                    st.metric(label="Recommended Max Bid", value=f"${advice['max_bid']}")
                    st.markdown(f"**Reasoning:** {advice['reason']}")
                else:
                    st.info("Select a player from the dropdown below, then click 'Set On The Block' to get bidding advice.")

    total_league_money = st.session_state.num_teams * st.session_state.budget_per_team
    remaining_money_pool = total_league_money - st.session_state.total_money_spent

    if not st.session_state.available_players.empty:
        recalculated_df = recalculate_dynamic_values(
            available_players_df=st.session_state.available_players.copy(),
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
    else:
        recalculated_df = pd.DataFrame()
        st.balloons()
        st.success("Draft Complete!")

    st.subheader("Analyze or Draft a Player")
    if not recalculated_df.empty:
        if 'PlayerName' in recalculated_df.columns:
            recalculated_df.set_index('PlayerName', inplace=True)
    
    player_options = st.session_state.available_players['PlayerName'].tolist()
    selected_player_for_block = st.selectbox(
        "Select a player to put on the block:", 
        options=player_options, 
        index=None, 
        key="player_select_for_block"
    )

    # Button to set the player on the block
    if st.button("Set On The Block", key="set_on_block_button"):
        if selected_player_for_block:
            st.session_state.player_on_the_block = selected_player_for_block
            st.rerun()
        else:
            st.warning("Please select a player first.")

    st.markdown("---")

    col1, col2 = st.columns(2)
    render_team_rosters()
    render_draft_summary()

    draft_analysis_container = st.container()
    with draft_analysis_container:
        if 'player_on_the_block' in st.session_state:
            render_player_analysis(st.session_state.player_on_the_block, recalculated_df)
