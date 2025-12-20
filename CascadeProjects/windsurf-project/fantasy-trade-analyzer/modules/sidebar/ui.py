"""
This module contains the global sidebar for the Fantasy Basketball Trade Analyzer.
"""

import streamlit as st
from pathlib import Path
from data_loader import csv_time

def display_global_sidebar():
    """Displays the global sidebar with navigation and high-level data status.

    League loading and management now live under the Admin Tools page; the
    sidebar only shows a read-only summary.
    """
    st.markdown(
        """
        <style>
          [data-testid="stSidebarNav"] { display: none; }
          [data-testid="stSidebarNavItems"] { display: none; }
          [data-testid^="stSidebarNav"] { display: none; }
          [data-testid="stPageNav"] { display: none; }
          section[data-testid="stSidebar"] nav { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title(f":rainbow[Fantasy Trade Analyzer]")
        st.caption(f"Last Data Update: {csv_time()}")

        st.markdown("---")
        st.subheader("Navigation")
        st.page_link("Home.py", label="Home", icon="🏠")
        st.page_link("pages/1_Trade_Analysis.py", label="Trade Analysis", icon="💱")
        st.page_link("pages/5_Trade_Targets.py", label="Trade Targets", icon="🎯")
        st.page_link("pages/4_Player_Full_Data.py", label="Player Full Data", icon="📄")
        st.page_link("pages/3_Schedule_Analysis.py", label="Schedule Analysis", icon="📅")
        st.page_link("pages/10_Lineup_Optimizer.py", label="Lineup Optimizer", icon="🧮")
        st.page_link("pages/7_Auction_Draft_Tool.py", label="Auction Draft Tool", icon="🧾")
        st.page_link("pages/9_Player_Value_Analyzer.py", label="Player Value & Consistency", icon="🏆")
        st.page_link("pages/12_Manager_History.py", label="History Hub", icon="📚")
        st.page_link("pages/6_Admin_Tools.py", label="Admin Tools", icon="🔐")

        st.markdown("---")
        st.subheader("League Data")
        if st.session_state.get('data_loaded', False):
            st.success(f"Loaded: **{st.session_state.get('loaded_league_name', 'Default')}**")
        else:
            st.warning("Default dataset not found or failed to load.")
        st.caption("Manage and change leagues from the 🔐 Admin Tools page.")
        st.markdown("--- ")
        st.caption("CSV/DB data is updated regularly. Ping the commish if it looks stale.")
