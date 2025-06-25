import streamlit as st
from datetime import datetime, timedelta
import os
from fantrax_downloader import download_players_csv, get_chrome_driver, FANTRAX_USERNAME, FANTRAX_PASSWORD

# Helper to get league mapping from env

def get_league_choices():
    ids = os.environ.get("FANTRAX_LEAGUE_IDS", "").split(",")
    names = os.environ.get("FANTRAX_LEAGUE_NAMES", "").split(",")
    return [(i.strip(), n.strip()) for i, n in zip(ids, names) if i.strip() and n.strip()]

# Helper for default ranges
DEFAULT_RANGES = [
    ("YTD", None, None),
    ("60 Days", 60, None),
    ("30 Days", 30, None),
    ("14 Days", 14, None),
    ("7 Days", 7, None)
]

def get_date_range(days, end_date):
    if days is None:
        return None, None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=days-1)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def downloader_sidebar():
    st.sidebar.subheader(":blue[Download Player Stats]")
    league_choices = get_league_choices()
    league_ids = [i for i, n in league_choices]
    league_names = [n for i, n in league_choices]
    league_idx = 0
    if not league_choices:
        st.sidebar.warning("No leagues configured in environment.")
        return
    league_idx = 0
    league_name_to_id = {n: i for i, n in league_choices}
    selected_league_name = st.sidebar.selectbox("Select League", league_names, index=league_idx)
    selected_league_id = league_name_to_id[selected_league_name]

    # Default or custom range
    st.sidebar.markdown("---")
    st.sidebar.write("Select stat ranges to download:")
    default_labels = [r[0] for r in DEFAULT_RANGES]
    selected_defaults = st.sidebar.multiselect("Default Ranges", default_labels, default=default_labels)
    custom_range = st.sidebar.checkbox("Custom Range")
    custom_start = custom_end = None
    if custom_range:
        custom_start = st.sidebar.date_input("Custom Start Date", datetime.now() - timedelta(days=7))
        custom_end = st.sidebar.date_input("Custom End Date", datetime.now())
        if custom_start > custom_end:
            st.sidebar.error("Start date must be before end date.")
            return
    download_btn = st.sidebar.button("Download Selected Ranges")
    if download_btn:
        with st.spinner("Logging in and downloading..."):
            driver = get_chrome_driver(os.getenv("FANTRAX_DOWNLOAD_DIR", os.getcwd()))
            results = []
            try:
                from fantrax_downloader import login_to_fantrax
                login_to_fantrax(driver, FANTRAX_USERNAME, FANTRAX_PASSWORD)
                today = datetime.now().strftime("%Y-%m-%d")
                from fantrax_downloader import DEFAULT_START, DEFAULT_END
                # Download selected default ranges
                for label, days, _ in DEFAULT_RANGES:
                    if label in selected_defaults:
                        st.sidebar.info(f"Downloading: {label}")
                        try:
                            if label == "YTD":
                                download_players_csv(driver, DEFAULT_START, DEFAULT_END, selected_league_id)
                                results.append((label, True, None))
                            else:
                                start, end = get_date_range(days, today)
                                download_players_csv(driver, start, end, selected_league_id)
                                results.append((label, True, None))
                        except Exception as e:
                            st.sidebar.error(f"Failed: {label} ({e})")
                            results.append((label, False, str(e)))
                # Download custom range if selected
                if custom_range and custom_start and custom_end:
                    label = f"Custom: {custom_start.strftime('%Y-%m-%d')} to {custom_end.strftime('%Y-%m-%d')}"
                    st.sidebar.info(f"Downloading: {label}")
                    try:
                        download_players_csv(driver, custom_start.strftime("%Y-%m-%d"), custom_end.strftime("%Y-%m-%d"), selected_league_id)
                        results.append((label, True, None))
                    except Exception as e:
                        st.sidebar.error(f"Failed: {label} ({e})")
                        results.append((label, False, str(e)))
                # Summary
                success_count = sum(1 for _, ok, _ in results if ok)
                fail_count = sum(1 for _, ok, _ in results if not ok)
                if success_count:
                    st.sidebar.success(f"{success_count} download(s) succeeded.")
                if fail_count:
                    st.sidebar.error(f"{fail_count} download(s) failed.")
            except Exception as e:
                st.sidebar.error(f"Download failed: {e}")
            finally:
                driver.quit()
    st.sidebar.markdown("---")
