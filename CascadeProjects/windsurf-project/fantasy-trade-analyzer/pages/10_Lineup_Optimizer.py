"""Tonight's Lineup Decision Helper page.

Thin wrapper that exposes the Phase 1 lineup optimizer UI:
compare a handful of players for a specific date using cached game logs.
"""

import streamlit as st

try:
	from league_config import FANTRAX_DEFAULT_LEAGUE_ID
except ImportError:
	FANTRAX_DEFAULT_LEAGUE_ID = ""

from modules.lineup_optimizer.ui import show_tonight_decision_helper, show_weekly_planner, show_stat_line_calculator
from modules.sidebar.ui import display_global_sidebar


st.set_page_config(page_title="Lineup Optimizer", page_icon="🧮", layout="wide")

display_global_sidebar()

tab_tonight, tab_weekly, tab_stat_calc = st.tabs([
	"🕒 Tonight's Helper",
	"📅 Weekly Planner (Grid)",
	"📊 Stat Line Calculator",
])

with tab_tonight:
	show_tonight_decision_helper(default_league_id=FANTRAX_DEFAULT_LEAGUE_ID)

with tab_weekly:
	show_weekly_planner(default_league_id=FANTRAX_DEFAULT_LEAGUE_ID)

with tab_stat_calc:
	show_stat_line_calculator()
