import streamlit as st
import pandas as pd
import plotly.express as px

from modules.draft_history import load_draft_history
from modules.manager_ids import load_manager_ids, get_manager_list


st.set_page_config(page_title="Draft History", page_icon="📜", layout="wide")

st.title("📜 Draft History - S1 (2021-22) to S5 (2025-26)")

st.markdown(
    """
Use this view to explore how draft prices and habits have evolved across seasons.

- Inspect **per-season spend** and **max bids**.
- Track **player price trajectories** year-over-year.
- See how much of a **200-budget** a single player consumed (e.g., 190/200 on Jokic).
- Compare **teams' total spend vs theoretical pots** as league size changes.
"""
)

# --- Load unified draft history ---

df = load_draft_history()

if df is None or df.empty:
    st.warning("No draft history found. Make sure S1–S4 draft CSVs and at least one Fantrax draft results CSV exist in the data folder.")
    st.stop()

# Ensure Bid is numeric for downstream calcs
df = df.copy()
if "Bid" in df.columns:
    df["Bid"] = pd.to_numeric(df["Bid"], errors="coerce")

# Clean up fantasy team labels and obvious junk rows
df["TeamLabel"] = df["FantasyTeamCanonical"].fillna(df["FantasyTeamRaw"])
df["TeamLabel"] = df["TeamLabel"].astype(str).str.strip()
df.loc[df["TeamLabel"].isin(["", "nan", "NaN", "None", "N/A", "(N/A)"]), "TeamLabel"] = pd.NA
df["TeamLabelNorm"] = df["TeamLabel"].astype(str).str.strip().str.lower()
if "FantasyTeamRaw" in df.columns:
	df["FantasyTeamRawNorm"] = df["FantasyTeamRaw"].astype(str).str.strip().str.lower()
else:
	df["FantasyTeamRawNorm"] = ""
if "FantasyTeamCanonical" in df.columns:
	df["FantasyTeamCanonicalNorm"] = df["FantasyTeamCanonical"].astype(str).str.strip().str.lower()
else:
	df["FantasyTeamCanonicalNorm"] = ""

# Normalize player names and treat placeholder nan strings as missing
if "Player" in df.columns:
    df["Player"] = df["Player"].astype(str).str.strip()
    df.loc[df["Player"].isin(["", "nan", "NaN"]), "Player"] = pd.NA

# Drop rows that have no real player, no team, and no bid (junk footer/header rows)
drop_mask = df["Player"].isna() & df["TeamLabel"].isna()
if "Bid" in df.columns:
    drop_mask &= df["Bid"].isna()
df = df[~drop_mask].copy()

# Derive a stable season sort order (S1..S5)
season_order = {f"S{i}": i for i in range(1, 10)}
df["SeasonNum"] = df["SeasonKey"].map(season_order)

mid_df = load_manager_ids()
mgr_list = pd.DataFrame()
if mid_df is not None and not mid_df.empty:
    mgr_list = get_manager_list(mid_df)
use_manager_ids = (
    mid_df is not None
    and not mid_df.empty
    and mgr_list is not None
    and not mgr_list.empty
)
selected_manager_id = None
selected_manager_label = None
manager_seasons_df = pd.DataFrame()
selected_teams = []

# --- Global controls ---

st.markdown("## Filters & Settings")

filters_col1, filters_col2, filters_col3 = st.columns([1.2, 1.2, 0.8])

with filters_col1:
    seasons = sorted(df["SeasonKey"].dropna().unique(), key=lambda k: season_order.get(k, 999))
    selected_seasons = st.multiselect(
        "Seasons",
        options=seasons,
        default=seasons,
    )

with filters_col2:
    if use_manager_ids:
        manager_options = mgr_list["label"].tolist()
        if manager_options:
            default_index = 0
            selected_manager_label = st.selectbox(
                "Manager",
                options=manager_options,
                index=default_index,
            )
            selected_row = mgr_list[mgr_list["label"] == selected_manager_label].iloc[0]
            selected_manager_id = selected_row["managerid"]
    else:
        all_teams = sorted(df["TeamLabel"].dropna().unique())
        selected_teams = st.multiselect(
            "Fantasy Teams (canonical)",
            options=all_teams,
            default=all_teams,
        )

with filters_col3:
    player_query = st.text_input("Player name contains", "")
    budget_per_team = st.number_input(
        "Assumed budget per team ($)",
        min_value=1,
        max_value=10000,
        value=200,
        step=10,
    )

# Apply filters

f = df.copy()
if selected_seasons:
    f = f[f["SeasonKey"].isin(selected_seasons)]

if use_manager_ids and selected_manager_id is not None:
    manager_seasons_df = mid_df[mid_df["managerid"] == selected_manager_id].copy()
    if not manager_seasons_df.empty:
        team_names = manager_seasons_df["team_name"].astype(str).str.strip().str.lower()
        team_abbrs = manager_seasons_df["team_abbreviation"].astype(str).str.strip().str.lower()
        team_name_set = set(team_names[team_names.ne("")])
        team_abbr_set = set(team_abbrs[team_abbrs.ne("")])
        if team_name_set or team_abbr_set:
            mask = pd.Series(False, index=f.index)
            if team_name_set:
                mask = mask | f["FantasyTeamRawNorm"].isin(team_name_set) | f["FantasyTeamCanonicalNorm"].isin(team_name_set) | f["TeamLabelNorm"].isin(team_name_set)
            if team_abbr_set:
                mask = mask | f["FantasyTeamRawNorm"].isin(team_abbr_set) | f["FantasyTeamCanonicalNorm"].isin(team_abbr_set) | f["TeamLabelNorm"].isin(team_abbr_set)
            f = f[mask]
else:
    if selected_teams:
        f = f[f["TeamLabel"].isin(selected_teams)]

if player_query.strip():
    q = player_query.strip().lower()
    f = f[f["Player"].astype(str).str.lower().str.contains(q)]

if f.empty:
    st.info("No rows match the current filters.")
    st.stop()

# Derived metrics for budget context

if "Bid" in f.columns:
    f["BidPctOfTeamBudget"] = (f["Bid"] / float(budget_per_team)) * 100.0
else:
    f["BidPctOfTeamBudget"] = pd.NA

# --- Season overview ---

if use_manager_ids and selected_manager_id is not None:
    st.markdown("## Manager History")
    seasons_df = manager_seasons_df.copy()
    if seasons_df.empty:
        st.info("No season records found for this manager.")
    else:
        seasons_df = seasons_df.sort_values("season")
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
        st.dataframe(pretty, hide_index=True, use_container_width=True)
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

st.markdown("## Season Overview")

# Compute per-season aggregates

season_summary = (
    f.groupby("SeasonKey")
    .agg(
        TotalSpend=("Bid", "sum"),
        MaxBid=("Bid", "max"),
        AvgBid=("Bid", "mean"),
        PlayersDrafted=("Player", "count"),
        UniqueTeams=("TeamLabel", "nunique"),
    )
    .reset_index()
)

# Theoretical league pot and utilization
season_summary["TheoreticalPot"] = season_summary["UniqueTeams"] * float(budget_per_team)
season_summary["UtilizationPct"] = (
    season_summary["TotalSpend"] / season_summary["TheoreticalPot"] * 100.0
).where(season_summary["TheoreticalPot"] > 0)

season_summary = season_summary.sort_values("SeasonKey", key=lambda s: s.map(season_order))

st.dataframe(
    season_summary.round({"TotalSpend": 1, "MaxBid": 1, "AvgBid": 2, "UtilizationPct": 1}),
    use_container_width=True,
)

if "Bid" in f.columns and f["Bid"].notna().any():
    fig_season = px.bar(
        season_summary,
        x="SeasonKey",
        y="TotalSpend",
        hover_data=["MaxBid", "AvgBid", "PlayersDrafted", "UtilizationPct"],
        title="Total Draft Spend by Season",
        labels={"TotalSpend": "Total Spend ($)", "SeasonKey": "Season"},
    )
    st.plotly_chart(fig_season, use_container_width=True)

    clean_bids = f.dropna(subset=["Bid"]).copy()
    if not clean_bids.empty:
        fig_bid_dist = px.box(
            clean_bids,
            x="SeasonKey",
            y="Bid",
            points="outliers",
            title="Bid Distribution by Season",
            labels={"Bid": "Bid ($)", "SeasonKey": "Season"},
        )
        st.plotly_chart(fig_bid_dist, use_container_width=True)

# --- Player-level YoY trends ---

st.markdown("## Player Price Trajectories")

players_for_select = sorted(f["Player"].unique())
selected_player = st.selectbox(
    "Select a player to view YoY draft prices",
    options=[""] + players_for_select,
    index=0,
)

if selected_player:
    p_df = f[f["Player"] == selected_player].copy()
    p_df = p_df.sort_values("SeasonNum")

    col1, col2 = st.columns([2, 1])

    with col1:
        if "Bid" in p_df.columns and p_df["Bid"].notna().any():
            fig_player = px.line(
                p_df,
                x="SeasonKey",
                y="Bid",
                markers=True,
                title=f"{selected_player} - Draft Price by Season",
                labels={"Bid": "Bid ($)", "SeasonKey": "Season"},
            )
            st.plotly_chart(fig_player, use_container_width=True)
        else:
            st.info("No bid data available for this player across the selected seasons.")

    with col2:
        if "BidPctOfTeamBudget" in p_df.columns and p_df["BidPctOfTeamBudget"].notna().any():
            p_df["BidPctOfTeamBudget"] = p_df["BidPctOfTeamBudget"].astype(float)
            st.markdown("**Bid as % of team budget**")
            st.dataframe(
                p_df[["SeasonKey", "Bid", "BidPctOfTeamBudget", "FantasyTeamCanonical", "FantasyTeamRaw"]]
                .rename(columns={"BidPctOfTeamBudget": "% of 200-budget"})
                .round({"Bid": 1, "% of 200-budget": 1}),
                use_container_width=True,
            )
        else:
            st.info("No bid data to compute % of budget.")

# --- Big overpays / outliers ---

st.markdown("## Biggest Single-Player Splurges")

if "BidPctOfTeamBudget" in f.columns and f["BidPctOfTeamBudget"].notna().any():
    overpays = (
        f.dropna(subset=["Bid", "BidPctOfTeamBudget"])
        .sort_values("BidPctOfTeamBudget", ascending=False)
        .head(25)
        .copy()
    )
    overpays["BidPctOfTeamBudget"] = overpays["BidPctOfTeamBudget"].astype(float)
    st.dataframe(
        overpays[
            [
                "SeasonKey",
                "Player",
                "Bid",
                "BidPctOfTeamBudget",
                "FantasyTeamCanonical",
                "FantasyTeamRaw",
            ]
        ]
        .rename(columns={"BidPctOfTeamBudget": "% of 200-budget"})
        .round({"Bid": 1, "% of 200-budget": 1}),
        use_container_width=True,
    )
else:
    st.info("No bid data found to compute overpays.")

st.markdown("## Rosters by Season")

if use_manager_ids and selected_manager_id is not None:
    roster_df = f.copy()
    if not roster_df.empty:
        roster_df = roster_df.sort_values(["SeasonNum", "Player"])
        seasons_for_roster = roster_df["SeasonKey"].dropna().unique().tolist()
        seasons_for_roster = sorted(seasons_for_roster, key=lambda k: season_order.get(k, 999))
        if seasons_for_roster:
            tabs = st.tabs([f"{sk} Roster" for sk in seasons_for_roster])
            for sk, tab in zip(seasons_for_roster, tabs):
                with tab:
                    season_slice = roster_df[roster_df["SeasonKey"] == sk]
                    if season_slice.empty:
                        continue
                    cols = [
                        "Player",
                        "Bid",
                        "Pos",
                    ]
                    existing_cols = [c for c in cols if c in season_slice.columns]
                    display_roster = season_slice[existing_cols].copy()
                    if "Bid" in display_roster.columns:
                        display_roster["Bid"] = display_roster["Bid"].round(1)
                    st.dataframe(display_roster, hide_index=True, use_container_width=True)
else:
    st.caption("Rosters by season are available when ManagerIDs are configured.")

# --- Raw table ---

st.markdown("## Raw Draft Records (filtered)")

display_cols = [
    "SeasonKey",
    "Player",
    "TeamLabel",
]
for c in ("Bid", "BidPctOfTeamBudget", "Pick", "Pos", "Time (EST)"):
    if c in f.columns:
        display_cols.append(c)

display_df = f[display_cols].copy()
if "BidPctOfTeamBudget" in display_df.columns:
    display_df.rename(columns={"BidPctOfTeamBudget": "% of 200-budget"}, inplace=True)
if "TeamLabel" in display_df.columns:
    display_df.rename(columns={"TeamLabel": "Team"}, inplace=True)

st.dataframe(display_df.round(1), use_container_width=True)
