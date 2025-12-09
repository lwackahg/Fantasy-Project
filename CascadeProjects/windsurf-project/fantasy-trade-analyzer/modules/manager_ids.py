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
