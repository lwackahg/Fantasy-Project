import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from .logic import (
    get_cached_standings_and_min_games,
    calculate_adjusted_scores,
    submit_adjustments_to_fantrax,
    get_cached_periods,
    FANTRAX_USERNAME,
    FANTRAX_PASSWORD
)
from .audit_log import append_adjustment_to_log, get_audit_log_info, get_audit_log_path, reset_audit_log

from ..weekly_standings_analyzer.logic import get_league_name_map, FANTRAX_DEFAULT_LEAGUE_ID

def show_standings_adjuster():
    st.title("Standings Adjuster")
    st.write("This tool submits the calculated adjustments from the Weekly Standings Analyzer to Fantrax.")

    league_name_map = get_league_name_map()
    league_names = list(league_name_map.keys())
    default_league_name = next((name for name, id in league_name_map.items() if id == FANTRAX_DEFAULT_LEAGUE_ID), None)

    selected_league_name = st.selectbox("Select League", options=league_names, index=league_names.index(default_league_name) if default_league_name in league_names else 0)
    league_id = league_name_map[selected_league_name]

    cached_periods_data = get_cached_periods(league_id)
    cached_periods_options = sorted(cached_periods_data.keys())

    if not cached_periods_options:
        st.warning("No cached standings found for this league ID. Please run the Weekly Standings Analyzer first.")
        return
	
    # Audit Log Section
    st.markdown("---")
    st.subheader("📋 Audit Log")
    
    audit_info = get_audit_log_info(league_id, selected_league_name)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if audit_info:
            st.success(f"✅ Audit log exists: **{audit_info['periods_logged']}** period(s) logged")
            st.caption(f"File: `{audit_info['filepath'].name}` ({audit_info['file_size']:,} bytes)")
        else:
            st.info("📝 No audit log yet. It will be created when you log your first adjustment.")
    
    with col2:
        if audit_info:
            filepath = get_audit_log_path(league_id, selected_league_name)
            # Read file data once and close it immediately
            with open(filepath, 'rb') as f:
                file_data = f.read()
            
            st.download_button(
                label="⬇️ Download Audit Log",
                data=file_data,
                file_name=filepath.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                key="download_audit_log"
            )
    
    # Reset Audit Log Section (for testing)
    if audit_info:
        with st.expander("⚠️ Danger Zone - Reset Audit Log"):
            st.error("**WARNING:** This will permanently delete all audit history for this league!")
            st.write(f"Current audit log contains **{audit_info['periods_logged']} period(s)** of data.")
            
            # Triple confirmation checkboxes
            confirm1 = st.checkbox(
                "I understand this will delete all audit history",
                key="reset_confirm1"
            )
            
            confirm2 = st.checkbox(
                "I have downloaded a backup of the audit log if needed",
                key="reset_confirm2",
                disabled=not confirm1
            )
            
            confirm3 = st.checkbox(
                "I am absolutely sure I want to permanently delete this audit log",
                key="reset_confirm3",
                disabled=not confirm2
            )
            
            # Reset button only enabled after all confirmations
            if confirm1 and confirm2 and confirm3:
                if st.button("🗑️ PERMANENTLY DELETE AUDIT LOG", type="secondary", width="stretch"):
                    success, message = reset_audit_log(league_id, selected_league_name)
                    if success:
                        st.success(message)
                        st.info("The audit log has been deleted. A new one will be created on the next log entry.")
                        st.rerun()
                    else:
                        st.error(message)
    
    st.markdown("---")

    selected_period = st.selectbox("Select Period to Adjust", options=cached_periods_options)

    if selected_period:
        raw_df, min_games = get_cached_standings_and_min_games(league_id, selected_period)
        
        if raw_df is not None:
            st.info(f"Using minimum games value of **{min_games}** from cache for Period {selected_period}.")
            adjustments_df = calculate_adjusted_scores(raw_df.copy(), min_games)

            st.subheader(f"Review Adjustments for Period {selected_period}")
            
            # Calculate the final adjustment value for submission (negative, rounded whole number)
            adjustments_df['Final Adjustment'] = -(adjustments_df['Adjustment'].round(0).astype(int))
            
            st.dataframe(adjustments_df[['team_name', 'Adjustment', 'Final Adjustment']])
			
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📝 Log to Audit File", width="stretch", type="primary"):
                    with st.spinner(f"Logging Period {selected_period} to audit file..."):
                        success, message, filepath = append_adjustment_to_log(
                            league_id, 
                            selected_period, 
                            selected_league_name
                        )
                        if success:
                            st.success(message)
                            st.info(f"📁 Updated: `{filepath.name}`")
                            st.rerun()
                        else:
                            st.error(message)
            
            with col2:
                if st.button("🚀 Submit to Fantrax", width="stretch"):
                    with st.spinner("Submitting adjustments..."):
                        success, message = submit_adjustments_to_fantrax(
                            league_id,
                            selected_period,
                            FANTRAX_USERNAME,
                            FANTRAX_PASSWORD,
                            adjustments_df
                        )
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        else:
            st.error(f"Could not load cached data for period {selected_period}.")
