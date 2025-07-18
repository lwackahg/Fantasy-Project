import streamlit as st
import pandas as pd
from logic.auction_tool import calculate_initial_values, recalculate_dynamic_values
from modules.data_preparation import generate_pps_projections

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
        st.session_state.drafted_players = []
        st.session_state.total_money_spent = 0
        # Initialize team-specific states
        st.session_state.teams = {name: {'budget': st.session_state.budget_per_team, 'players': []} for name in st.session_state.team_names}

initialize_session_state()

# --- Sidebar for Setup ---
st.sidebar.header("League Setup")

if not st.session_state.draft_started:
    st.session_state.num_teams = st.sidebar.number_input("Number of Teams", min_value=8, max_value=20, value=st.session_state.num_teams)
    st.session_state.budget_per_team = st.sidebar.number_input("Budget per Team ($)", min_value=100, max_value=500, value=st.session_state.budget_per_team, step=10)
    st.session_state.roster_spots_per_team = st.sidebar.number_input("Roster Spots per Team", min_value=5, max_value=20, value=st.session_state.roster_spots_per_team)

    st.sidebar.header("Projection Settings")
    st.session_state.games_in_season = st.sidebar.number_input("Games in a Season", min_value=1, max_value=82, value=st.session_state.games_in_season)
    
    if st.sidebar.button("Generate Projections"):
        with st.spinner("Generating PPS projections..."):
            if generate_pps_projections(st.session_state.games_in_season):
                st.session_state.projections_generated = True
                st.sidebar.success("Projections generated!")
            else:
                st.sidebar.error("Failed to generate projections.")

    if st.session_state.projections_generated:
        if st.sidebar.button("Start Draft"):
            st.session_state.draft_started = True
            # Load projections and calculate initial values
            try:
                pps_df = pd.read_csv('data/player_projections.csv')
                
                initial_df = calculate_initial_values(
                    pps_df, 
                    st.session_state.num_teams,
                    st.session_state.roster_spots_per_team,
                    st.session_state.budget_per_team
                )
                st.session_state.available_players = initial_df
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
    if st.sidebar.button("Reset Draft"):
        # Clear session state and rerun to go back to the setup screen
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main Draft Interface (will be built out next) ---
if st.session_state.draft_started:
    st.header("Draft Board")
    total_league_money = st.session_state.num_teams * st.session_state.budget_per_team
    remaining_money_pool = total_league_money - st.session_state.total_money_spent

    # Recalculate dynamic values before displaying anything
    if not st.session_state.available_players.empty:
        recalculated_df = recalculate_dynamic_values(st.session_state.available_players.copy(), remaining_money_pool, total_league_money)
    else:
        recalculated_df = pd.DataFrame()

    # Display Key Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Money in Pool", f"${total_league_money:,}")
    col2.metric("Money Spent", f"${st.session_state.total_money_spent:,}")
    col3.metric("Remaining Money", f"${remaining_money_pool:,}", delta_color="inverse")

    # --- Draft Player Form ---
    st.subheader("Draft a Player")
    with st.form("draft_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            available_player_names = st.session_state.available_players['PlayerName'].tolist()
            selected_player = st.selectbox("Select Player", options=available_player_names)
        with col2:
            drafting_team = st.selectbox("Select Team", options=st.session_state.team_names)
        with col3:
            draft_price = st.number_input("Draft Price ($)", min_value=1, max_value=st.session_state.budget_per_team, value=10)
        
        submitted = st.form_submit_button("Draft Player")
        if submitted:
            team_budget = st.session_state.teams[drafting_team]['budget']
            if draft_price > team_budget:
                st.error(f"{drafting_team} only has ${team_budget} remaining!")
            else:
                # --- Update State Logic ---
                player_to_draft = st.session_state.available_players[st.session_state.available_players['PlayerName'] == selected_player].iloc[0]
                
                draft_entry = player_to_draft.to_dict()
                draft_entry['DraftPrice'] = draft_price
                draft_entry['Team'] = drafting_team
                st.session_state.drafted_players.append(draft_entry)

                # Update team-specific state
                st.session_state.teams[drafting_team]['players'].append(draft_entry)
                st.session_state.teams[drafting_team]['budget'] -= draft_price
                
                # Update global state
                st.session_state.total_money_spent += draft_price
                st.session_state.available_players = st.session_state.available_players[st.session_state.available_players['PlayerName'] != selected_player]
                
                st.success(f"Drafted {selected_player} to {drafting_team} for ${draft_price}!")
                st.rerun()

    # --- Display Tables ---
    st.subheader("Available Players")
    if not recalculated_df.empty:
        display_cols = ['PlayerName', 'Position', 'Tier', 'PPS', 'VORP', 'BaseValue', 'AdjValue']
        st.dataframe(recalculated_df[display_cols], use_container_width=True, height=500)
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

else:
    st.info("Please configure your league settings in the sidebar and click 'Start Draft' to begin.")
