import streamlit as st
from modules.weekly_standings_analyzer.ui import show_weekly_standings_analyzer

def main():
    """
    This page displays the Weekly Standings Analyzer tool.
    """
    show_weekly_standings_analyzer()

if __name__ == "__main__":
    main()
