import streamlit as st
from modules.auth.ui import check_password
from modules.fantrax_downloader.ui import display_downloader_ui
from modules.player_game_log_scraper.ui import show_player_game_log_scraper
from modules.standings_adjuster.ui import show_standings_adjuster
from modules.weekly_standings_analyzer.ui import show_weekly_standings_analyzer

st.set_page_config(page_title="Admin Tools", page_icon="🔐", layout="wide")

# Password protection
if not check_password():
	st.stop()

st.title("🔐 Admin Tools")
st.write("Commissioner-only tools for league management and data scraping.")

# Create tabs for different admin tools
tab1, tab2, tab3, tab4 = st.tabs([
	"📥 Downloader",
	"📊 Player Game Logs", 
	"📈 Weekly Standings",
	"⚙️ Standings Adjuster"
])

with tab1:
	display_downloader_ui()

with tab2:
	show_player_game_log_scraper()

with tab3:
	show_weekly_standings_analyzer()

with tab4:
	show_standings_adjuster()
