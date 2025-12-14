import streamlit as st
from debug import debug_manager

def init_session_state(extra_defaults=None):
    """Initialize session state variables."""
    default_states = {
        'data_ranges': {},
        'combined_data': None,
        'current_range': None,
        'debug_manager': debug_manager,
        'trade_analyzer': None,
        'trade_analysis': None,
        'csv_timestamp': "CSV timestamp not available",
        'schedule_view_type': "List View",
        'schedule_selected_period': "All Periods",
        'schedule_selected_team': "All Teams",
        'schedule_selected_periods': [],
        'schedule_selected_teams': [],
        'schedule_swap_team1': None,
        'schedule_swap_team2': None,
        'schedule_swap_performed': False,
    }

    if extra_defaults:
        try:
            default_states.update(extra_defaults)
        except Exception:
            pass

    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def init_auction_draft_session_state():
    from pathlib import Path
    import json
    import pandas as pd

    from logic.auction_tool import BASE_VALUE_MODELS, SCARCITY_MODELS
    from modules.team_mappings import TEAM_MAPPINGS

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

    if 'current_nominating_team_index' not in st.session_state:
        st.session_state.current_nominating_team_index = 0
    if 'player_on_the_block' not in st.session_state:
        st.session_state.player_on_the_block = None
    if '_reset_player_on_block' not in st.session_state:
        st.session_state._reset_player_on_block = False

    if 'projections_generated' not in st.session_state:
        st.session_state.projections_generated = False
    if 'num_teams' not in st.session_state:
        st.session_state.num_teams = len(TEAM_MAPPINGS)
    if 'budget_per_team' not in st.session_state:
        st.session_state.budget_per_team = 200
    if 'roster_spots_per_team' not in st.session_state:
        st.session_state.roster_spots_per_team = 10
    if 'games_in_season' not in st.session_state:
        st.session_state.games_in_season = 75
    if 'team_names' not in st.session_state:
        st.session_state.team_names = list(TEAM_MAPPINGS.values())[:st.session_state.num_teams]
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
        trend_weights = {'S1': 0.00, 'S2': 0.10, 'S3': 0.25, 'S4': 0.65}
        total_weight = sum(trend_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in trend_weights.items()}
        else:
            normalized_weights = trend_weights
        st.session_state.trend_weights = normalized_weights
    if 'roster_composition' not in st.session_state:
        st.session_state.roster_composition = {'G': 3, 'F': 3, 'C': 2, 'Flx': 2, 'Bench': 0}
    if 'realism_enabled' not in st.session_state:
        st.session_state.realism_enabled = True
    if 'realism_aggression' not in st.session_state:
        st.session_state.realism_aggression = 1.0
    if 'nomination_strategy' not in st.session_state:
        st.session_state.nomination_strategy = 'Blended (recommended)'
    if 'nomination_weights' not in st.session_state:
        st.session_state.nomination_weights = {'budget_pressure': 0.5, 'positional_scarcity': 0.3, 'value_inflation': 0.2}

    if not st.session_state.teams:
        st.session_state.teams = {name: {'budget': st.session_state.budget_per_team, 'players': []} for name in st.session_state.team_names}

    if 'injured_players_text' not in st.session_state:
        try:
            injured_players_path = Path(__file__).resolve().parent / "data" / "injured_players.json"
            if injured_players_path.exists():
                with open(injured_players_path, 'r') as f:
                    injured_players_dict = json.load(f)
                    injured_text = "\n".join([f"{player} ({status})" for player, status in injured_players_dict.items()])
                    st.session_state.injured_players_text = injured_text
            else:
                st.session_state.injured_players_text = ""
        except Exception:
            st.session_state.injured_players_text = ""