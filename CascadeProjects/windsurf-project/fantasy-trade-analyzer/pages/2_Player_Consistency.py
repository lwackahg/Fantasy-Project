"""Thin wrapper page for backwards compatibility.

Player Consistency has been moved into the Player Value & Consistency hub
under the "📊 Player Consistency Browser" tab. This page now just redirects
there without loading any heavy modules.
"""

import streamlit as st

from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Player Consistency (Moved)", page_icon="📊", layout="wide")

display_global_sidebar()

st.title("📊 Player Consistency")
st.info(
	"This view has been moved into the 🏆 Player Value & Consistency Hub.\n\n"
	"Use the **📊 Player Consistency Browser** tab on the Player Value & Consistency page."
)

st.page_link("pages/9_Player_Value_Analyzer.py", label="Go to Player Value & Consistency Hub", icon="🏆")
st.stop()
