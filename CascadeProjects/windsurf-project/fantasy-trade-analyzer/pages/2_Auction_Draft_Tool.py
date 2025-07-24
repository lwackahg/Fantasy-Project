import streamlit as st
import pandas as pd
from logic.auction_tool import (calculate_initial_values, recalculate_dynamic_values, BASE_VALUE_MODELS, SCARCITY_MODELS)
from modules.data_preparation import generate_pps_projections
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
    render_value_calculation_expander()

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

    st.subheader("Draft a Player")
    available_player_names = recalculated_df['PlayerName'].tolist() if not recalculated_df.empty else []
    selected_player_name = st.selectbox("Select Player to analyze or draft", options=available_player_names, label_visibility="collapsed")

    draft_analysis_container = st.container()
    with draft_analysis_container:
        if selected_player_name:
            render_player_analysis(selected_player_name, recalculated_df)

    render_draft_board(recalculated_df)
    render_team_rosters()
    render_draft_summary()
