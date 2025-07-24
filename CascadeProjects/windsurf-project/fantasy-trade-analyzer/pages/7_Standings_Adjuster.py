import streamlit as st
from modules.standings_adjuster.ui import show_standings_adjuster

def main():
    st.set_page_config(layout="wide")
    show_standings_adjuster()

if __name__ == "__main__":
    main()
