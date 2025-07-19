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
        st.session_state.injured_players = {}
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

initialize_session_state()

# --- Sidebar for Setup ---
st.sidebar.header("League Setup")

if not st.session_state.draft_started:
    st.session_state.num_teams = st.sidebar.number_input("Number of Teams", min_value=8, max_value=20, value=st.session_state.num_teams)
    st.session_state.budget_per_team = st.sidebar.number_input("Budget per Team ($)", min_value=100, max_value=500, value=st.session_state.budget_per_team, step=10)
    st.session_state.roster_spots_per_team = st.sidebar.number_input("Roster Spots per Team", min_value=5, max_value=20, value=st.session_state.roster_spots_per_team)

    st.sidebar.header("Projection Settings")
    st.session_state.games_in_season = st.sidebar.number_input("Games in a Season", min_value=1, max_value=82, value=st.session_state.games_in_season, disabled=st.session_state.draft_started)

    with st.sidebar.expander("Valuation Model Settings"):
        st.session_state.base_value_model = st.selectbox(
            "Base Value Calculation",
            options=["Blended (VORP + Market)", "Pure VORP", "Pure Market Value"],
            index=0,
            help="Choose how the initial Base Value is calculated. This cannot be changed after the draft starts.",
            disabled=st.session_state.draft_started
        )
        st.session_state.scarcity_model = st.selectbox(
            "In-Draft Scarcity Model",
            options=["Tier Scarcity", "Position Scarcity", "None"],
            index=0,
            help="Choose the model for applying a scarcity premium to the Adjusted Value. This cannot be changed after the draft starts.",
            disabled=st.session_state.draft_started
        )

    with st.sidebar.expander("Injury Settings"):
        # Load all possible player names for the multiselect
        try:
            # Use the same source as the projection engine for consistency
            all_players_df = pd.read_csv('data/PlayerFPperGameOverYears.csv')
            # Clean the player list by dropping missing values and ensuring string type
            valid_players = all_players_df['Player'].dropna().astype(str).unique()
            all_player_names = sorted(list(valid_players))
        except FileNotFoundError:
            all_player_names = []
            st.warning("Player list not found. Cannot set injuries.")

        injured_player = st.selectbox(
            "Select Player to Add to Injury List",
            options=all_player_names,
            index=None,
            placeholder="Search for a player...",
            disabled=st.session_state.projections_generated
        )
        injury_status = st.radio(
            "Injury Duration",
            options=["Half Season", "Out for Season"],
            horizontal=True,
            disabled=st.session_state.projections_generated
        )

        if st.button("Add Injury", disabled=st.session_state.projections_generated):
            if injured_player:
                st.session_state.injured_players[injured_player] = injury_status
                st.success(f"Added {injured_player} to injury list as '{injury_status}'.")
            else:
                st.error("Please select a player first.")

        if st.session_state.injured_players:
            st.write("**Current Injury List:**")
            for player, status in st.session_state.injured_players.items():
                st.write(f"- {player}: {status}")
            if st.button("Clear Injury List", disabled=st.session_state.projections_generated):
                st.session_state.injured_players = {}
                st.rerun()

    with st.sidebar.expander("Customize Trend Weights"):
        st.session_state.s4_weight = st.slider("Most Recent Season (S4) Weight", 0.0, 1.0, 0.95, 0.05, disabled=st.session_state.projections_generated)
        st.session_state.s3_weight = st.slider("S3 Weight", 0.0, 1.0, 0.10, 0.05, disabled=st.session_state.projections_generated)
        st.session_state.s2_weight = st.slider("S2 Weight", 0.0, 1.0, 0.05, 0.05, disabled=st.session_state.projections_generated)
        st.session_state.s1_weight = st.slider("Oldest Season (S1) Weight", 0.0, 1.0, 0.0, 0.05, disabled=st.session_state.projections_generated)

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
                # Load the projections from the CSV file before using them
                pps_df = pd.read_csv('data/player_projections.csv')

                st.session_state.available_players, st.session_state.initial_tier_counts, st.session_state.initial_pos_counts = calculate_initial_values(
                    pps_df=pps_df,
                    num_teams=st.session_state.num_teams,
                    roster_spots_per_team=st.session_state.roster_spots_per_team,
                    budget_per_team=st.session_state.budget_per_team,
                    base_value_model=st.session_state.base_value_model
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
        if st.sidebar.button("↩️ Undo Last Pick"):
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
        - **PPS (Player Power Score):** A weighted average of a player's fantasy points per game over the last four seasons, adjusted for games played. The trend weights are configurable in the sidebar.
        
        - **Tier:** Players are grouped into five tiers based on their PPS percentile rank against the *remaining* player pool. Tiers are recalculated after every pick.
            - **Tier 1:** Top 2% of players
            - **Tier 2:** Next 8% (90th-98th percentile)
            - **Tier 3:** Next 15% (75th-90th percentile)
            - **Tier 4:** Next 25% (50th-75th percentile)
            - **Tier 5:** Bottom 50%

        - **VORP (Value Over Replacement Player):** Measures a player's value relative to the best player likely available on the waiver wire. It's calculated as `Player's PPS - Replacement Level PPS`.

        - **Base Value:** The initial estimated auction value. It's a 50/50 blend of the player's VORP-based value and their historical **Market Value** (average auction price from past seasons).

        - **Adjusted Value:** The dynamic, in-draft value, recalculated after every pick. The calculation is a multi-step process:
            1.  **Re-Tiering:** The entire remaining player pool is re-tiered based on their PPS percentiles. As top players are drafted, others move up into higher tiers.
            2.  **Preliminary Value:** A new preliminary value is calculated based on the player's share of the remaining talent pool (VORP) and the remaining money in the league.
            3.  **Scarcity Premium:** A premium is applied based on how many players in a given tier have been drafted. As a tier becomes more scarce, the value of the remaining players in that tier increases (capped at +25%).
        """)

    total_league_money = st.session_state.num_teams * st.session_state.budget_per_team
    remaining_money_pool = total_league_money - st.session_state.total_money_spent

    # Recalculate dynamic values before displaying anything
    if not st.session_state.available_players.empty:
        recalculated_df = recalculate_dynamic_values(
            available_players_df=st.session_state.available_players.copy(), 
            remaining_money_pool=remaining_money_pool, 
            total_league_money=total_league_money, 
            scarcity_model=st.session_state.scarcity_model,
            initial_tier_counts=st.session_state.initial_tier_counts,
            initial_pos_counts=st.session_state.initial_pos_counts
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

    # Display selected player's values
    if selected_player_name:
        player_values = recalculated_df[recalculated_df['PlayerName'] == selected_player_name].iloc[0]
        base_value = player_values['BaseValue']
        adj_value = player_values['AdjValue']
        st.markdown(f"Base Value: ${base_value:,.2f}")
        st.markdown(f"Adjusted Value: ${adj_value:,.2f}")

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
        display_cols = ['PlayerName', 'Position', 'Tier', 'PPS', 'VORP', 'BaseValue', 'AdjValue']
        st.dataframe(recalculated_df[display_cols], use_container_width=True, height=500)

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
