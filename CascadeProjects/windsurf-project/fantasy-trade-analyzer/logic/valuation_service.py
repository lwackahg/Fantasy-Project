from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from logic.auction_tool import (
    BASE_VALUE_MODELS,
    SCARCITY_MODELS,
    calculate_initial_values,
    recalculate_dynamic_values,
)
from modules.historical_trade_analyzer.logic import build_historical_combined_data
from modules.historical_ytd_downloader.logic import (
    get_available_seasons,
    load_and_compare_seasons,
)


FRIEND_DOLLAR_BUCKETS: List[Tuple[float, float]] = [
    (125.0, 22.0),
    (120.0, 20.0),
    (115.0, 18.0),
    (110.0, 16.0),
    (105.0, 14.0),
    (100.0, 12.0),
    (95.0, 10.0),
    (90.0, 8.0),
    (85.0, 6.0),
    (80.0, 4.0),
    (70.0, 3.0),
    (60.0, 2.0),
    (50.0, 1.0),
    (0.0, 0.0),
]


def _friend_dollar_value(fpg: float) -> float:
    try:
        value = float(fpg)
    except (TypeError, ValueError):
        value = 0.0
    for threshold, dollars in FRIEND_DOLLAR_BUCKETS:
        if value >= threshold:
            return dollars
    return 0.0


@dataclass
class LeagueContext:
    num_teams: int = 12
    budget_per_team: int = 200
    roster_composition: Dict[str, int] = field(default_factory=lambda: {"G": 3, "F": 3, "C": 2, "Flx": 2, "Bench": 0})
    tier_cutoffs: Optional[Dict[str, float]] = None
    base_value_models: Optional[List[str]] = None


class ValuationService:
    """
    Shared valuation service that centralizes projection loading, initial valuation calculations,
    dynamic recalculations, and historical backtests so multiple features can consume the same outputs.
    """

    def __init__(self, data_root: Optional[Path] = None):
        default_root = Path(__file__).resolve().parent.parent / "data"
        self.data_root = data_root or default_root
        self._projections_cache: Optional[pd.DataFrame] = None
        self._draft_history_cache: Optional[pd.DataFrame] = None
        self._history_by_source_cache: Dict[str, pd.DataFrame] = {}
        self._yoy_cache: Dict[str, pd.DataFrame] = {}
        self._league_context: LeagueContext = LeagueContext()
        self._replacement_level_pps: Optional[float] = None
        self._backtest_summary: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update_league_context(
        self,
        num_teams: int,
        budget_per_team: int,
        roster_composition: Dict[str, int],
        tier_cutoffs: Optional[Dict[str, float]] = None,
        base_value_models: Optional[List[str]] = None,
    ) -> None:
        self._league_context = LeagueContext(
            num_teams=num_teams,
            budget_per_team=budget_per_team,
            roster_composition=roster_composition,
            tier_cutoffs=tier_cutoffs,
            base_value_models=base_value_models,
        )

    def load_projections(self) -> pd.DataFrame:
        if self._projections_cache is not None:
            return self._projections_cache.copy()

        auction_dir = self.data_root / "auction"
        proj_path = auction_dir / "player_projections.csv"
        if not proj_path.exists():
            proj_path = self.data_root / "player_projections.csv"
        if not proj_path.exists():
            raise FileNotFoundError("player_projections.csv not found in data directory.")

        self._projections_cache = pd.read_csv(proj_path)
        return self._projections_cache.copy()

    def build_initial_valuations(
        self,
        *,
        injured_players: Optional[Dict[str, str]] = None,
        tier_cutoffs: Optional[Dict[str, float]] = None,
        base_value_models: Optional[List[str]] = None,
        ec_settings: Optional[Dict] = None,
        gp_rel_settings: Optional[Dict] = None,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        ctx = self._derive_context(tier_cutoffs, base_value_models)
        pps_df = self.load_projections()
        valuations, pos_counts, tier_counts = calculate_initial_values(
            pps_df=pps_df,
            num_teams=ctx.num_teams,
            budget_per_team=ctx.budget_per_team,
            roster_composition=ctx.roster_composition,
            base_value_models=ctx.base_value_models or BASE_VALUE_MODELS[:1],
            tier_cutoffs=ctx.tier_cutoffs,
            injured_players=injured_players,
            ec_settings=ec_settings,
            gp_rel_settings=gp_rel_settings,
        )
        self._replacement_level_pps = self._estimate_replacement_level(valuations)
        valuations = self._attach_friend_dollars(valuations)
        valuations = self._attach_value_breakdown(valuations)
        self._backtest_summary = None  # Reset so it can be recomputed with new data
        return valuations, pos_counts, tier_counts

    def attach_value_context(self, df: pd.DataFrame) -> pd.DataFrame:
        annotated = self._attach_friend_dollars(df.copy())
        annotated = self._attach_value_breakdown(annotated)
        return annotated

    def recalc_dynamic_values(
        self,
        available_players_df: pd.DataFrame,
        *,
        remaining_money_pool: float,
        total_league_money: float,
        base_value_models: Optional[List[str]],
        scarcity_models: Optional[List[str]],
        initial_tier_counts: Dict,
        initial_pos_counts: Dict,
        teams_data: Dict,
    ) -> pd.DataFrame:
        ctx = self._league_context
        recalculated = recalculate_dynamic_values(
            available_players_df=available_players_df,
            remaining_money_pool=remaining_money_pool,
            total_league_money=total_league_money,
            base_value_models=base_value_models or ctx.base_value_models or BASE_VALUE_MODELS[:1],
            scarcity_models=scarcity_models or [SCARCITY_MODELS[0]],
            initial_tier_counts=initial_tier_counts,
            initial_pos_counts=initial_pos_counts,
            teams_data=teams_data,
            tier_cutoffs=ctx.tier_cutoffs,
            roster_composition=ctx.roster_composition,
            num_teams=ctx.num_teams,
        )
        recalculated = self.attach_value_context(recalculated)
        return recalculated

    def get_backtest_summary(self, valuations: pd.DataFrame) -> Optional[Dict[str, float]]:
        if valuations is None or valuations.empty:
            return None
        if self._backtest_summary is not None:
            return self._backtest_summary
        history = self._load_draft_history()
        if history is None or history.empty:
            return None
        needed_cols = ["PlayerName", "BaseValue", "AdjValue", "FriendDollarValue", "Tier"]
        present_cols = [c for c in needed_cols if c in valuations.columns]
        if "BaseValue" not in present_cols:
            return None  # nothing to compare against
        merged = history.merge(
            valuations[present_cols],
            on="PlayerName",
            how="left",
        ).dropna(subset=["BaseValue"])
        if merged.empty:
            return None
        merged["BaseError"] = merged["Bid"] - merged["BaseValue"]
        merged["AdjError"] = merged["Bid"] - merged.get("AdjValue", merged["BaseValue"])
        merged["AbsBaseError"] = merged["BaseError"].abs()
        merged["AbsAdjError"] = merged["AdjError"].abs()
        summary = {
            "samples": int(len(merged)),
            "avg_bid": float(merged["Bid"].mean()),
            "avg_base_error": float(merged["BaseError"].mean()),
            "avg_adj_error": float(merged["AdjError"].mean()),
            "mae_base": float(merged["AbsBaseError"].mean()),
            "mae_adj": float(merged["AbsAdjError"].mean()),
        }
        tier_mae = merged.groupby("Tier")["AbsBaseError"].mean().to_dict()
        summary["tier_mae"] = {int(k): float(v) for k, v in tier_mae.items()}
        self._backtest_summary = summary
        return summary

    def get_available_yoy_seasons(self) -> List[str]:
        return get_available_seasons()

    def get_yoy_comparison(self, league_name: str, seasons: List[str]) -> Optional[pd.DataFrame]:
        if not league_name or not seasons:
            return None
        sanitized = self._sanitize_league_name(league_name)
        key = f"{sanitized}__{'|'.join(sorted(seasons))}"
        if key in self._yoy_cache:
            return self._yoy_cache[key].copy()
        df = load_and_compare_seasons(sanitized, seasons)
        if df is None or df.empty:
            return None
        self._yoy_cache[key] = df
        return df.copy()

    def build_historical_snapshot(
        self,
        *,
        league_id: str,
        season: str,
        trade_date: str,
        rosters_by_team: Dict[str, List[str]],
    ) -> Optional[pd.DataFrame]:
        if not league_id or not season or not trade_date or not rosters_by_team:
            return None
        try:
            parsed_date = datetime.fromisoformat(trade_date).date()
        except ValueError:
            return None
        try:
            snapshot = build_historical_combined_data(parsed_date, league_id, season, rosters_by_team)
        except Exception:
            return None
        return snapshot

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _derive_context(
        self,
        tier_cutoffs: Optional[Dict[str, float]],
        base_value_models: Optional[List[str]],
    ) -> LeagueContext:
        ctx = self._league_context
        if tier_cutoffs or base_value_models:
            ctx = LeagueContext(
                num_teams=ctx.num_teams,
                budget_per_team=ctx.budget_per_team,
                roster_composition=ctx.roster_composition,
                tier_cutoffs=tier_cutoffs or ctx.tier_cutoffs,
                base_value_models=base_value_models or ctx.base_value_models,
            )
        if ctx.base_value_models is None:
            ctx.base_value_models = BASE_VALUE_MODELS[:1]
        return ctx

    def _attach_friend_dollars(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "FriendDollarValue" in df.columns:
            return df
        pps_col = "PPS" if "PPS" in df.columns else "FP/G"
        df["FriendDollarValue"] = df[pps_col].apply(_friend_dollar_value)
        if self._replacement_level_pps:
            df["SlotEquivalent"] = df[pps_col].apply(
                lambda val: round(float(val) / self._replacement_level_pps, 2) if self._replacement_level_pps else 0.0
            )
        else:
            df["SlotEquivalent"] = 0.0
        return df

    def _attach_value_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        def _summary(row):
            parts = []
            if pd.notna(row.get("PPS")):
                parts.append(f"PPS {row['PPS']:.1f}")
            if pd.notna(row.get("Tier")):
                parts.append(f"Tier {int(row['Tier'])}")
            if "VORPValue" in row and pd.notna(row["VORPValue"]):
                parts.append(f"VORP ${row['VORPValue']:.0f}")
            if "BaseValue" in row and pd.notna(row["BaseValue"]):
                parts.append(f"Base ${row['BaseValue']:.0f}")
            if "AdjValue" in row and pd.notna(row["AdjValue"]):
                parts.append(f"Adj ${row['AdjValue']:.0f}")
            if "FriendDollarValue" in row and pd.notna(row["FriendDollarValue"]):
                parts.append(f"FD ${row['FriendDollarValue']:.0f}")
            return " | ".join(parts)
        df["ValueBreakdown"] = df.apply(_summary, axis=1)
        return df

    def _load_draft_history(self) -> Optional[pd.DataFrame]:
        if self._draft_history_cache is not None:
            return self._draft_history_cache.copy()

        csv_files = list(self.data_root.glob("Fantrax-Draft-Results-*.csv"))
        if not csv_files:
            return None
        frames = []
        for path in csv_files:
            try:
                df = pd.read_csv(path)
                if "Player" not in df.columns or "Bid" not in df.columns:
                    continue
                df = df.rename(columns={"Player": "PlayerName"})
                df["Source"] = path.name
                frames.append(df[["PlayerName", "Bid", "Source"]].copy())
            except Exception:
                continue
        if not frames:
            return None
        self._draft_history_cache = pd.concat(frames, ignore_index=True)
        return self._draft_history_cache.copy()

    def list_draft_history_sources(self) -> List[str]:
        """Return available draft history CSV filenames in the data directory."""
        csv_files = list(self.data_root.glob("Fantrax-Draft-Results-*.csv"))
        return [f.name for f in csv_files]

    def get_draft_history_by_source(self, source: str) -> Optional[pd.DataFrame]:
        """Load a specific draft history CSV by filename."""
        if not source:
            return None
        if source in self._history_by_source_cache:
            return self._history_by_source_cache[source].copy()
        path = self.data_root / source
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            self._history_by_source_cache[source] = df
            return df.copy()
        except Exception:
            return None

    def _estimate_replacement_level(self, df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty or "PPS" not in df.columns:
            return None
        try:
            pps_sorted = df["PPS"].sort_values(ascending=False)
            idx = int(len(pps_sorted) * 0.85)
            idx = min(max(idx, 0), len(pps_sorted) - 1)
            return float(pps_sorted.iloc[idx])
        except Exception:
            return None

    def _sanitize_league_name(self, name: str) -> str:
        import re

        return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().replace(" ", "_")) or "League"
