import streamlit as st
from modules.auth.ui import check_password
from modules.fantrax_downloader.ui import display_downloader_ui
from modules.player_game_log_scraper.ui import show_player_game_log_scraper
from modules.standings_adjuster.ui import show_standings_adjuster
from modules.weekly_standings_analyzer.ui import show_weekly_standings_analyzer
from modules.historical_ytd_downloader.ui import display_historical_ytd_ui

st.set_page_config(page_title="Admin Tools", page_icon="🔐", layout="wide")

# Password protection
if not check_password():
	st.stop()

st.title("🔐 Admin Tools")
st.write("Commissioner-only tools for league management and data scraping.")

# Create tabs for different admin tools
tab1, tab2, tab3, tab4, tab5 = st.tabs([
	"📥 Downloader for Current Season (Trading Files)",
	"📅 Historical YTD (When Significant Data Available)",
	"📊 Player Game Logs (when Signifcant Data Available)", 
	"📈 Weekly Standings",
	"⚙️ Weekly Standings Adjuster"
])

with tab1:
	display_downloader_ui()

with tab2:
	display_historical_ytd_ui()

with tab3:
	show_player_game_log_scraper()

with tab4:
	show_weekly_standings_analyzer()

with tab5:
	show_standings_adjuster()
