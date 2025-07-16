import streamlit as st
import os
from modules.fantrax_downloader.logic import download_all_ranges

def _get_league_choices():
    """Gets league choices from environment variables."""
    ids = os.environ.get("FANTRAX_LEAGUE_IDS", "").split(",")
    names = os.environ.get("FANTRAX_LEAGUE_NAMES", "").split(",")
    if not ids or not names or len(ids) != len(names) or not ids[0]:
        return {}
    return {n.strip(): i.strip() for i, n in zip(ids, names) if i.strip() and n.strip()}

def display_downloader_ui():
    """Displays the UI for downloading data from Fantrax."""
    st.header("Download Fantrax Data")
    st.markdown("""
        Use this section to download the latest player stats from Fantrax.
        This will download data for all standard time ranges (YTD, 60 days, 30 days, 14 days, 7 days).
    """)

    league_choices = _get_league_choices()
    if not league_choices:
        st.warning("No Fantrax leagues configured. Please set `FANTRAX_LEAGUE_IDS` and `FANTRAX_LEAGUE_NAMES` in your `fantrax.env` file.")
        return

    selected_league_name = st.selectbox(
        "Select League to Download Data For",
        options=list(league_choices.keys())
    )

    if st.button("Download All Standard Ranges", type="primary"):
        if selected_league_name:
            league_id = league_choices[selected_league_name]
            
            progress_bar = st.progress(0, text="Starting download...")
            
            with st.spinner("Downloading in progress... please wait."):
                results = download_all_ranges(league_id, lambda p, m: progress_bar.progress(p, text=m))
            
            st.success("Download process complete!")
            
            st.subheader("Download Log")
            with st.expander("Click to see full log", expanded=False):
                for msg in results:
                    if "Success" in msg:
                        st.success(msg)
                    elif "Error" in msg:
                        st.error(msg)
                    else:
                        st.info(msg)
