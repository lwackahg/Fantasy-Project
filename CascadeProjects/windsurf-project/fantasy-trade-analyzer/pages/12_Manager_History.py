import streamlit as st
import pandas as pd

from modules.manager_ids import load_manager_ids, get_manager_list
from modules.sidebar.ui import display_global_sidebar
from streamlit_compat import dataframe
from ui.history_hub import render_best_team_optimizer, render_draft_history


st.set_page_config(page_title="Manager History", page_icon="👤", layout="wide")

display_global_sidebar()

st.title("👤 Manager History")
st.caption("Manager timeline + Draft history in one place.")

tab_mgr, tab_draft, tab_best = st.tabs(["👤 Manager History", "📜 Draft History", "🧠 Best $200 Team"])

with tab_mgr:
    # Load manager identity data
    mid_df = load_manager_ids()
    if mid_df.empty:
        st.warning("ManagerIDs.csv not found or contains no rows.")
        st.stop()

    mgr_list = get_manager_list(mid_df)
    if mgr_list.empty:
        st.warning("No manager records available.")
        st.stop()

    # Sidebar-like selector in the main body
    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.subheader("Select Manager")
        options = mgr_list["label"].tolist()
        default_index = 0
        selected_label = st.selectbox("Manager", options=options, index=default_index)
        # Map back to ManagerID
        selected_row = mgr_list[mgr_list["label"] == selected_label].iloc[0]
        selected_id = selected_row["managerid"]

    with right_col:
        st.subheader("Manager Overview")
        st.markdown(f"**Manager ID:** `{selected_id}`")

        # Filter seasons for this manager
        seasons_df = mid_df[mid_df["managerid"] == selected_id].copy()
        if seasons_df.empty:
            st.info("No season records found for this manager.")
        else:
            # Sort by season string; relies on consistent formatting like '2021-22'
            seasons_df = seasons_df.sort_values("season")

            st.markdown("### Seasons & Team Identities")
            display_cols = [
                "season",
                "team_name",
                "team_abbreviation",
            ]
            pretty = seasons_df[display_cols].rename(
                columns={
                    "season": "Season",
                    "team_name": "Team Name",
                    "team_abbreviation": "Team Abbrev",
                }
            )
            dataframe(pretty, hide_index=True, width="stretch")

            # Simple timeline-style text summary
            st.markdown("### Timeline")
            timeline_lines = []
            for _, row in seasons_df.iterrows():
                season = row.get("season", "?")
                tname = row.get("team_name", "?")
                tabbr = row.get("team_abbreviation", "")
                if tabbr:
                    timeline_lines.append(f"- **{season}** – {tname} (`{tabbr}`)")
                else:
                    timeline_lines.append(f"- **{season}** – {tname}")
            if timeline_lines:
                st.markdown("\n".join(timeline_lines))
            else:
                st.caption("No timeline entries available.")

    st.markdown("---")
    st.caption(
        "Rosters and advanced performance history by manager can be layered onto this view "
        "once historical rosters are available from the DB/log scrapers."
    )

with tab_draft:
    render_draft_history()

with tab_best:
    render_best_team_optimizer()
