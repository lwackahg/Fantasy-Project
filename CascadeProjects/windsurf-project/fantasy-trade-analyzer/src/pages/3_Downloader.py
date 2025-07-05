import streamlit as st
from modules.fantrax_downloader.ui import display_downloader_ui as display_downloader_page
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(page_title="Downloader", layout="wide")

display_global_sidebar()

st.title(":orange[Downloader]")
display_downloader_page()
