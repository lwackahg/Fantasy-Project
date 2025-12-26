import pandas as pd
import streamlit as st
from pathlib import Path
from streamlit_compat import dataframe
from modules.sidebar.ui import display_global_sidebar

st.set_page_config(layout="wide", page_title="FP/G + GP Lookup (Fantrax)")

display_global_sidebar()

st.title("FP/G + GP Lookup (Fantrax)")
st.caption("Paste player names (one per line). Data source: Fantrax exports (YTD, 60, 30, 14, 7).")

HORIZON_FILES = {
    "YTD": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(YTD).csv",
    "60": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(60).csv",
    "30": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(30).csv",
    "14": "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(14).csv",
    "7":  "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(7).csv",
}

def load_span(span: str) -> pd.DataFrame:
    fname = HORIZON_FILES.get(span, HORIZON_FILES["YTD"])
    path = Path(__file__).resolve().parents[1] / "data" / fname
    return pd.read_csv(path)

names_raw = st.text_area(
    "Players (one per line)",
    value="",
    height=200,
    key="fp_gp_lookup_names"
)

span = st.selectbox("Window", options=list(HORIZON_FILES.keys()), index=0, key="fp_gp_span")

if st.button("Lookup FP/G & GP", type="primary"):
    names = [n.strip() for n in names_raw.splitlines() if n.strip()]
    if not names:
        st.warning("Paste at least one player name.")
    else:
        try:
            df = load_span(span)
            subset = df[df['Player'].isin(names)]
            if subset.empty:
                st.warning(f"No matches for the provided names in Fantrax {span}.")
            else:
                cols = [c for c in ['Player', 'FP/G', 'GP', 'Team', 'Position'] if c in subset.columns]
                subset = subset[cols].sort_values('Player')
                dataframe(subset, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Lookup failed: {exc}")

st.markdown("---")
st.caption("Sources: data/Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(YTD|60|30|14|7).csv")
