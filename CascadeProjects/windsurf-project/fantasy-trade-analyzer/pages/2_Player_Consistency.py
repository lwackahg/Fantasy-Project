"""Public-facing player consistency viewer - no scraping, just viewing cached data."""
import streamlit as st
from modules.player_game_log_scraper.ui_viewer import show_player_consistency_viewer

st.set_page_config(page_title="Player Consistency", page_icon="📊", layout="wide")

show_player_consistency_viewer()
