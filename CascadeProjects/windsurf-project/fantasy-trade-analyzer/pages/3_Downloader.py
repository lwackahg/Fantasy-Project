"""
This page provides an interface for downloading the latest Fantrax data.
"""

import streamlit as st
from modules.fantrax_downloader.ui import display_downloader_ui
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, MENUITEMS

st.set_page_config(page_title=f"Downloader - {PAGE_TITLE}", page_icon=PAGE_ICON, layout=LAYOUT, menu_items=MENUITEMS)

st.title(":rainbow[Download Latest Fantrax Data]")

st.write("Use the button below to download the latest player data from Fantrax. This may take a moment.")

display_downloader_ui()
