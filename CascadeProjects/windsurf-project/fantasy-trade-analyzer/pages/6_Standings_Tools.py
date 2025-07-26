import streamlit as st
from modules.auth.ui import check_password
from modules.weekly_standings_analyzer.ui import show_weekly_standings_analyzer
from modules.standings_adjuster.ui import show_standings_adjuster

def main():
    st.set_page_config(layout="wide")
    st.title("Standings Tools")

    if check_password():
        tab1, tab2 = st.tabs(["Weekly Standings Analyzer", "Standings Adjuster"])

        with tab1:
            show_weekly_standings_analyzer()
        
        with tab2:
            show_standings_adjuster()

if __name__ == "__main__":
    main()
