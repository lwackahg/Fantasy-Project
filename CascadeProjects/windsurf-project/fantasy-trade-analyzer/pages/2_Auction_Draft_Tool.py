import streamlit as st
import pandas as pd
from logic.auction_tool import (calculate_initial_values, recalculate_dynamic_values, BASE_VALUE_MODELS, SCARCITY_MODELS)
from modules.data_preparation import generate_pps_projections
from logic.smart_auction_bot import SmartAuctionBot
from logic.ui_components import (
    render_setup_page, render_sidebar_in_draft, render_value_calculation_expander,
    render_draft_board, render_team_rosters, render_drafting_form, 
    render_player_analysis_metrics, render_draft_summary
)

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
        st.session_state.num_teams = 16
    if 'budget_per_team' not in st.session_state:
        st.session_state.budget_per_team = 200
    if 'roster_spots_per_team' not in st.session_state:
        st.session_state.roster_spots_per_team = 10
    if 'games_in_season' not in st.session_state:
        st.session_state.games_in_season = 75
    if 'team_names' not in st.session_state:
        st.session_state.team_names = [f"Team {i+1}" for i in range(st.session_state.num_teams)]
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
        trend_weights = {'S1': 0.0, 'S2': 0.05, 'S3': 0.10, 'S4': 1.00}
        total_weight = sum(trend_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in trend_weights.items()}
        else:
            normalized_weights = trend_weights
        st.session_state.trend_weights = normalized_weights
    if 'roster_composition' not in st.session_state:
        st.info(f"Weights Normalized. Total: {sum(normalized_weights.values()):.2f}")
        st.session_state.roster_composition = {'G': 3, 'F': 3, 'C': 2, 'Flx': 2, 'Bench': 0}
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

def clear_player_on_block():
    """Callback function to reset the player on the block."""
    # Set a flag to reset before widget creation to avoid Streamlit mutation error
    st.session_state._reset_player_on_block = True

def drafting_callback(selected_player_name, team_selection, draft_price, assigned_position, player_values):
    """Handles the logic for drafting a player and then clearing the selection."""
    if draft_price > st.session_state.teams[team_selection]['budget']:
        st.error(f"{team_selection} cannot afford {selected_player_name} at this price.")
        return
    # Deduct from budget and add to team roster
    st.session_state.teams[team_selection]['budget'] -= draft_price
    player_data = {
        'Player': selected_player_name,
        'Position': assigned_position,
        'Price': draft_price,
        **player_values
    }
    st.session_state.teams[team_selection]['players'].append(player_data)

    # Add to draft history
    draft_entry = {
        'Player': selected_player_name,
        'Team': team_selection,
        'Price': draft_price,
        'Position': assigned_position
    }
    st.session_state.draft_history.append(draft_entry)

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

    # Remove drafted player from available players so they disappear from lists
    if (not st.session_state.available_players.empty) and ('PlayerName' in st.session_state.available_players.columns):
        st.session_state.available_players = st.session_state.available_players[
            st.session_state.available_players['PlayerName'] != selected_player_name
        ]

    # Advance to the next team in the draft order
    st.session_state.current_nominating_team_index = (st.session_state.current_nominating_team_index + 1) % len(st.session_state.draft_order)

    # Clear the player on the block after drafting
    clear_player_on_block()
    # Trigger a rerun so the reset happens before the selectbox is created
    st.rerun()

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
    # Process any pending reset BEFORE any widgets that use 'player_on_the_block' are created
    if st.session_state.get('_reset_player_on_block', False):
        st.session_state.player_on_the_block = None
        st.session_state._reset_player_on_block = False

    render_sidebar_in_draft()

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

    # --- Main 3-Column Layout ---
    left_col, mid_col, right_col = st.columns([1.2, 2, 1])

    with left_col:
        nominating_team_name = st.session_state.draft_order[st.session_state.current_nominating_team_index] if st.session_state.draft_order else 'N/A'
        st.subheader(f"On the Clock: {nominating_team_name}")

        st.subheader("Analyze or Draft")
        player_options = []
        if not recalculated_df.empty:
            if recalculated_df.index.name != 'PlayerName':
                recalculated_df.set_index('PlayerName', inplace=True)
            player_options = recalculated_df.index.tolist()

        if 'player_on_the_block' not in st.session_state:
            st.session_state.player_on_the_block = None

        # Use the widget key to manage selection; avoid recomputing index to reduce lag
        st.selectbox(
            "Select a player to analyze or draft:",
            options=player_options,
            key='player_on_the_block',
            placeholder="Choose a player..."
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
            for i, rec in enumerate(top5_recommendations):
                st.markdown(f"**{i+1}. {rec['player']}** – Worth: ${rec['adj_value']:.2f} | Target: ${rec['target_nom_price']:.2f}")
                st.caption(rec['reason'])
        else:
            st.write("No recommendations available at this time.")
        
        st.subheader("Draft History")
        if st.session_state.draft_history:
            history_df = pd.DataFrame(st.session_state.draft_history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.write("No players drafted yet.")

    st.markdown("---_**Player Analysis**_---")
    if st.session_state.get('player_on_the_block') and not recalculated_df.empty:
        render_player_analysis_metrics(st.session_state.player_on_the_block, recalculated_df)
    else:
        st.info("Select a player from the dropdown above to see detailed analysis.")

    st.markdown("---_**Draft Summary**_---")
    render_draft_summary()

    if recalculated_df.empty and st.session_state.draft_started:
        st.balloons()
        st.success("Draft Complete!")
