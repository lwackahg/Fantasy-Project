import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANAGER_IDS_FILE = DATA_DIR / "ManagerIDs.csv"


def load_manager_ids() -> pd.DataFrame:
    """Load ManagerIDs.csv with basic cleaning.

    Expects the file to have an initial 'Columns:' line, followed by a
    header row:
        ManagerID,Season,Team Name,Team Abbreviation
    """
    if not MANAGER_IDS_FILE.exists():
        return pd.DataFrame()

    # Skip the first descriptive line so the next line becomes the header
    df = pd.read_csv(MANAGER_IDS_FILE, skiprows=1)
    if df.empty:
        return df

    # Normalize column names for internal use
    col_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        col_map[col] = key
    df = df.rename(columns=col_map)

    # Ensure expected columns exist
    for c in ("managerid", "season", "team_name", "team_abbreviation"):
        if c not in df.columns:
            df[c] = ""

    df["managerid"] = df["managerid"].astype(str).str.strip()
    df["season"] = df["season"].astype(str).str.strip()
    df["team_name"] = df["team_name"].astype(str).str.strip()
    df["team_abbreviation"] = df["team_abbreviation"].astype(str).str.strip()

    # Drop any rows that have no manager id or season
    df = df[df["managerid"].ne("") & df["season"].ne("")]
    return df


def get_manager_list(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per manager with a simple display label.

    The label uses the **most recent** season's team name as a shorthand so
    the dropdown reflects how managers are currently known in the league.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["managerid", "label"])

    # Sort by season so we can pick the most recent entry per manager.
    tmp = df.copy()
    tmp["season_sort"] = tmp["season"]

    grouped = []
    for mid, sub in tmp.groupby("managerid"):
        sub = sub.sort_values("season_sort")
        # Use the *latest* season's team name for the label
        example_team = sub["team_name"].iloc[-1] if not sub.empty else ""
        label = f"{mid} – {example_team}" if example_team else str(mid)
        grouped.append({"managerid": mid, "label": label})

    return pd.DataFrame(grouped)


def get_team_name_for_season(df: pd.DataFrame, manager_id: str, season: str) -> str:
    """Get the team name for a specific manager in a specific season.
    
    Args:
        df: ManagerIDs DataFrame
        manager_id: Manager ID (e.g., 'M001')
        season: Season string (e.g., '2025-26')
    
    Returns:
        Team name string, or empty string if not found
    """
    if df is None or df.empty:
        return ""
    
    match = df[(df["managerid"] == manager_id) & (df["season"] == season)]
    if match.empty:
        return ""
    
    return str(match.iloc[0]["team_name"]).strip()


def get_current_team_name(df: pd.DataFrame, manager_id: str) -> str:
    """Get the most recent team name for a manager.
    
    Args:
        df: ManagerIDs DataFrame
        manager_id: Manager ID (e.g., 'M001')
    
    Returns:
        Most recent team name, or empty string if not found
    """
    if df is None or df.empty:
        return ""
    
    mgr_data = df[df["managerid"] == manager_id].copy()
    if mgr_data.empty:
        return ""
    
    mgr_data = mgr_data.sort_values("season")
    return str(mgr_data.iloc[-1]["team_name"]).strip()


def get_manager_id_from_team(df: pd.DataFrame, team_name: str, season: str = None) -> str:
    """Get manager ID from team name, optionally filtered by season.
    
    Args:
        df: ManagerIDs DataFrame
        team_name: Team name to search for
        season: Optional season to filter by (e.g., '2025-26')
    
    Returns:
        Manager ID string, or empty string if not found
    """
    if df is None or df.empty:
        return ""
    
    # Normalize for matching
    def normalize(s: str) -> str:
        return "".join(ch.lower() for ch in str(s) if ch.isalnum())
    
    target = normalize(team_name)
    search_df = df.copy()
    
    if season:
        search_df = search_df[search_df["season"] == season]
    
    search_df["_norm"] = search_df["team_name"].apply(normalize)
    match = search_df[search_df["_norm"] == target]
    
    if match.empty:
        return ""
    
    return str(match.iloc[0]["managerid"]).strip()


def get_all_seasons_for_manager(df: pd.DataFrame, manager_id: str) -> pd.DataFrame:
    """Get all season records for a specific manager.
    
    Args:
        df: ManagerIDs DataFrame
        manager_id: Manager ID (e.g., 'M001')
    
    Returns:
        DataFrame with all seasons for this manager, sorted by season
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    mgr_data = df[df["managerid"] == manager_id].copy()
    return mgr_data.sort_values("season")
