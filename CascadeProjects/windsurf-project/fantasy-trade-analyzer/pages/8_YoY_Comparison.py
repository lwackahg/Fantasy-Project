"""Thin wrapper page for backwards compatibility.

YoY comparison has been moved into the Player Value & Consistency hub
under the "📊 YoY Trends" tab. This page now just redirects there
without loading any heavy modules.
"""

import streamlit as st

st.set_page_config(page_title="YoY Comparison (Moved)", page_icon="📊", layout="wide")

st.title("📊 Year-over-Year Comparison")
st.info(
	"This view has been moved into the 🏆 Player Value & Consistency Hub.\n\n"
	"Use the **📊 YoY Trends** tab on the Player Value & Consistency page."
)

st.page_link("pages/9_Player_Value_Analyzer.py", label="Go to Player Value & Consistency Hub", icon="🏆")
st.stop()
