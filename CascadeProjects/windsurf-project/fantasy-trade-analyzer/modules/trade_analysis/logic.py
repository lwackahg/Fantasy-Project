"""
Business logic for the trade analysis feature.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time
import uuid

from data_loader import TEAM_MAPPINGS
from debug import debug_manager
from modules.trade_analysis.consistency_integration import (
	load_player_consistency,
	load_all_player_consistency,
	enrich_roster_with_consistency,
	build_league_consistency_index,
	CONSISTENCY_VERY_MAX_CV,
	CONSISTENCY_MODERATE_MAX_CV,
)
from modules.player_value.logic import build_player_value_profiles

# Import league config
try:
	from league_config import FANTRAX_DEFAULT_LEAGUE_ID
except ImportError:
	FANTRAX_DEFAULT_LEAGUE_ID = ""


def _get_trade_history_key() -> str:
	"""Build a key for trade history based on league id or loaded league name."""
	league_id = st.session_state.get("league_id") or FANTRAX_DEFAULT_LEAGUE_ID
	if league_id:
		return f"league_{league_id}"
	league_name = st.session_state.get("loaded_league_name")
	if league_name:
		return f"league_{str(league_name).replace(' ', '_')}"
	return "league_default"


def _get_trade_history_path() -> Path:
	"""Return the filesystem path for the current league's trade history cache."""
	project_root = Path(__file__).resolve().parents[2]
	history_dir = project_root / "data" / "trade_history"
	history_dir.mkdir(parents=True, exist_ok=True)
	return history_dir / f"{_get_trade_history_key()}_trades.json"


def _load_trade_history() -> List[Dict[str, Any]]:
	"""Load trade history for the current league from disk, if available."""
	try:
		path = _get_trade_history_path()
		if not path.exists():
			return []
		with path.open("r", encoding="utf-8") as f:
			data = json.load(f)
		if isinstance(data, list):
			return data
	except Exception as exc:  # pragma: no cover - best-effort logging
		debug_manager.log(f"Failed to load trade history: {exc}", level="warning")
	return []


def _save_trade_history(history: List[Dict[str, Any]]) -> None:
	"""Persist trade history for the current league to disk."""
	try:
		path = _get_trade_history_path()
		with path.open("w", encoding="utf-8") as f:
			json.dump(history, f, ensure_ascii=False, indent=2)
	except Exception as exc:  # pragma: no cover - best-effort logging
		debug_manager.log(f"Failed to save trade history: {exc}", level="warning")


class TradeAnalyzer:
    """Analyzes fantasy basketball trades."""

    def __init__(self, data: pd.DataFrame):
        """Initialize the trade analyzer with player data."""
        self.data = data
        self.trade_history = []
        debug_manager.log("TradeAnalyzer initialized", level='info')

    def update_data(self, data: pd.DataFrame):
        """Update the player data used for analysis."""
        if not isinstance(data, pd.DataFrame):
            debug_manager.log("Invalid data type provided to update_data", level='error')
            return
        self.data = data
        debug_manager.log("Data updated successfully", level='info')

    def get_team_players(self, team: str) -> pd.DataFrame:
        """Get all players for a given team."""
        if self.data is None:
            debug_manager.log("No data available", level='error')
            return pd.DataFrame()

        team_data = self.data[self.data['Status'] == team].copy()
        debug_manager.log(f"Retrieved {len(team_data)} players for team {team}", level='debug')
        return team_data

    def evaluate_trade_fairness(
        self,
        trade_teams: Dict[str, Dict[str, str]],
        num_top_players: int = 10,
        include_advanced_metrics: bool = True,
        assumed_fpg_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate the fairness of a trade between teams.

        If include_advanced_metrics is False, skips consistency and value
        profile computations to speed up analysis.
        """
        analysis_results = {}

        for team, players in trade_teams.items():
            team_data = self.data[self.data['Status'] == team].copy()
            time_ranges = ['YTD', '60 Days', '30 Days', '14 Days', '7 Days']
            pre_trade_rosters = {}
            post_trade_rosters = {}
            outgoing_players = list(players.keys())
            incoming_players = []
            for other_team, other_players in trade_teams.items():
                for player, dest in other_players.items():
                    if dest == team:
                        incoming_players.append(player)

            for time_range in time_ranges:
                range_data = team_data[team_data['Timestamp'] == time_range].reset_index()
                if not range_data.empty:
                    # Apply assumed FP/G overrides to this team's data for this time range, if provided
                    if assumed_fpg_overrides:
                        override_players = [
                            p for p in range_data['Player'].unique()
                            if p in assumed_fpg_overrides
                        ]
                        if override_players:
                            range_data = range_data.copy()
                            for player_name in override_players:
                                try:
                                    new_fpg = float(assumed_fpg_overrides.get(player_name))
                                except (TypeError, ValueError):
                                    continue
                                mask = range_data['Player'] == player_name
                                range_data.loc[mask, 'FP/G'] = new_fpg
                                if 'GP' in range_data.columns and 'FPts' in range_data.columns:
                                    gp_vals = pd.to_numeric(range_data.loc[mask, 'GP'], errors='coerce').fillna(0.0)
                                    range_data.loc[mask, 'FPts'] = new_fpg * gp_vals

                    pre_trade_rosters[time_range] = range_data.nlargest(num_top_players, 'FP/G')[['Player', 'Team', 'FPts', 'FP/G', 'GP']].to_dict('records')
                    post_trade_data = range_data[~range_data['Player'].isin(outgoing_players)].copy()
                    incoming_data = []
                    for player in incoming_players:
                        player_data = self.data[
                            (self.data.index == player) &
                            (self.data['Timestamp'] == time_range)
                        ].reset_index()
                        if not player_data.empty:
                            # Apply assumed FP/G overrides for incoming players, if provided
                            if assumed_fpg_overrides and player in assumed_fpg_overrides:
                                try:
                                    new_fpg = float(assumed_fpg_overrides.get(player))
                                except (TypeError, ValueError):
                                    new_fpg = None
                                if new_fpg is not None:
                                    player_data = player_data.copy()
                                    player_data['FP/G'] = new_fpg
                                    if 'GP' in player_data.columns and 'FPts' in player_data.columns:
                                        gp_vals = pd.to_numeric(player_data['GP'], errors='coerce').fillna(0.0)
                                        player_data['FPts'] = new_fpg * gp_vals
                            player_data['Status'] = team
                            incoming_data.append(player_data)

                    if incoming_data:
                        incoming_df = pd.concat(incoming_data, ignore_index=True)
                        post_trade_data = pd.concat([post_trade_data, incoming_df], ignore_index=True)

                    post_trade_rosters[time_range] = post_trade_data.nlargest(num_top_players, 'FP/G')[['Player', 'Team', 'FPts', 'FP/G', 'GP']].to_dict('records')

            pre_trade_metrics = {}
            post_trade_metrics = {}
            pre_trade_consistency = {}
            post_trade_consistency = {}
            pre_trade_value_scores = {}
            post_trade_value_scores = {}

            # Get league ID from session state or use default
            league_id = st.session_state.get('league_id', FANTRAX_DEFAULT_LEAGUE_ID)
            value_profiles_df = None
            consistency_index = None
            if include_advanced_metrics and league_id:
                # Cache player value profiles per league
                vp_cache_key = "player_value_profiles_cache"
                vp_cache = st.session_state.get(vp_cache_key) or {}
                if isinstance(vp_cache, dict) and league_id in vp_cache:
                    value_profiles_df = vp_cache.get(league_id)
                else:
                    try:
                        value_profiles_df = build_player_value_profiles(league_id)
                    except Exception:
                        value_profiles_df = None
                    if value_profiles_df is not None:
                        vp_cache[league_id] = value_profiles_df
                        st.session_state[vp_cache_key] = vp_cache

                # Cache league-wide consistency index per league
                ci_cache_key = "consistency_index_cache"
                ci_cache = st.session_state.get(ci_cache_key) or {}
                if isinstance(ci_cache, dict) and league_id in ci_cache:
                    consistency_index = ci_cache.get(league_id)
                else:
                    try:
                        consistency_index = build_league_consistency_index(league_id)
                    except Exception:
                        consistency_index = None
                    if consistency_index is not None:
                        ci_cache[league_id] = consistency_index
                        st.session_state[ci_cache_key] = ci_cache

            for time_range in time_ranges:
                if time_range in pre_trade_rosters and pre_trade_rosters.get(time_range):
                    pre_roster_df = pd.DataFrame(pre_trade_rosters[time_range])
                    pre_trade_metrics[time_range] = {
                        'mean_fpg': pre_roster_df['FP/G'].mean(),
                        'median_fpg': pre_roster_df['FP/G'].median(),
                        'std_dev': pre_roster_df['FP/G'].std(),
                        'total_fpts': pre_roster_df['FPts'].sum(),
                        'avg_gp': pre_roster_df['GP'].mean()
                    }

                    # Add consistency metrics for this time range using cached index
                    if include_advanced_metrics and league_id:
                        enriched_pre = enrich_roster_with_consistency(
                            pre_roster_df.copy(),
                            league_id,
                            consistency_index=consistency_index,
                        )
                        if 'CV%' in enriched_pre.columns:
                            cv_values = enriched_pre['CV%'].dropna()
                            if len(cv_values) > 0:
                                pre_trade_consistency[time_range] = {
                                    'avg_cv': cv_values.mean(),
                                    'players_with_data': len(cv_values),
                                    'very_consistent': len(cv_values[cv_values < CONSISTENCY_VERY_MAX_CV]),
                                    'moderate': len(cv_values[(cv_values >= CONSISTENCY_VERY_MAX_CV) & (cv_values <= CONSISTENCY_MODERATE_MAX_CV)]),
                                    'volatile': len(cv_values[cv_values > CONSISTENCY_MODERATE_MAX_CV])
                                }

                if time_range in post_trade_rosters and post_trade_rosters.get(time_range):
                    post_roster_df = pd.DataFrame(post_trade_rosters[time_range])
                    post_trade_metrics[time_range] = {
                        'mean_fpg': post_roster_df['FP/G'].mean(),
                        'median_fpg': post_roster_df['FP/G'].median(),
                        'std_dev': post_roster_df['FP/G'].std(),
                        'total_fpts': post_roster_df['FPts'].sum(),
                        'avg_gp': post_roster_df['GP'].mean()
                    }

                    # Add consistency metrics for this time range using cached index
                    if include_advanced_metrics and league_id:
                        enriched_post = enrich_roster_with_consistency(
                            post_roster_df.copy(),
                            league_id,
                            consistency_index=consistency_index,
                        )
                        if 'CV%' in enriched_post.columns:
                            cv_values = enriched_post['CV%'].dropna()
                            if len(cv_values) > 0:
                                post_trade_consistency[time_range] = {
                                    'avg_cv': cv_values.mean(),
                                    'players_with_data': len(cv_values),
                                    'very_consistent': len(cv_values[cv_values < 20]),
                                    'moderate': len(cv_values[(cv_values >= 20) & (cv_values <= 30)]),
                                    'volatile': len(cv_values[cv_values > 30])
                                }

                # Add value score aggregates if profiles are available and requested
                if include_advanced_metrics and value_profiles_df is not None and not value_profiles_df.empty:
                    if time_range in pre_trade_rosters and pre_trade_rosters.get(time_range):
                        merged_pre = pd.DataFrame(pre_trade_rosters[time_range]).merge(
                            value_profiles_df[['Player', 'ValueScore']],
                            on='Player',
                            how='left'
                        )
                        if not merged_pre.empty:
                            pre_trade_value_scores[time_range] = {
                                'total_value_score': float(merged_pre['ValueScore'].fillna(0).sum()),
                                'avg_value_score': float(merged_pre['ValueScore'].fillna(0).mean())
                            }

                    if time_range in post_trade_rosters and post_trade_rosters.get(time_range):
                        merged_post = pd.DataFrame(post_trade_rosters[time_range]).merge(
                            value_profiles_df[['Player', 'ValueScore']],
                            on='Player',
                            how='left'
                        )
                        if not merged_post.empty:
                            post_trade_value_scores[time_range] = {
                                'total_value_score': float(merged_post['ValueScore'].fillna(0).sum()),
                                'avg_value_score': float(merged_post['ValueScore'].fillna(0).mean())
                            }

            # Aggregate value changes per time range
            value_changes = {}
            for time_range in time_ranges:
                if time_range in pre_trade_metrics and time_range in post_trade_metrics:
                    change_entry = {
                        'mean_fpg_change': post_trade_metrics[time_range]['mean_fpg'] - pre_trade_metrics[time_range]['mean_fpg'],
                        'total_fpts_change': post_trade_metrics[time_range]['total_fpts'] - pre_trade_metrics[time_range]['total_fpts'],
                        'avg_gp_change': post_trade_metrics[time_range]['avg_gp'] - pre_trade_metrics[time_range]['avg_gp']
                    }
                    if time_range in pre_trade_value_scores and time_range in post_trade_value_scores:
                        change_entry['value_score_change'] = (
                            post_trade_value_scores[time_range]['total_value_score']
                            - pre_trade_value_scores[time_range]['total_value_score']
                        )
                    value_changes[time_range] = change_entry

            analysis_results[team] = {
                'outgoing_players': outgoing_players,
                'incoming_players': incoming_players,
                'pre_trade_metrics': pre_trade_metrics,
                'post_trade_metrics': post_trade_metrics,
                'pre_trade_consistency': pre_trade_consistency,
                'post_trade_consistency': post_trade_consistency,
                'pre_trade_value_scores': pre_trade_value_scores,
                'post_trade_value_scores': post_trade_value_scores,
                'value_changes': value_changes,
                'pre_trade_rosters': pre_trade_rosters,
                'post_trade_rosters': post_trade_rosters,
                'league_id': league_id,
                'include_advanced_metrics': include_advanced_metrics,
            }
        return analysis_results

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get the history of analyzed trades as structured entries."""
        return self.trade_history

    def _generate_trade_summary(self, analysis_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate a text summary of the trade analysis."""
        summary_parts = []
        for team_name, results in analysis_results.items():
            outgoing = ', '.join(results['outgoing_players']) if results['outgoing_players'] else 'none'
            incoming = ', '.join(results['incoming_players']) if results['incoming_players'] else 'none'
            for range_name, value_changes in results['value_changes'].items():
                fpg_change = value_changes['mean_fpg_change']
                total_change = value_changes['total_fpts_change']
                summary = f"{team_name} ({range_name}):\n"
                summary += f"  Giving: {outgoing}\n"
                summary += f"  Getting: {incoming}\n"
                summary += f"  Impact: {fpg_change:+.1f} FP/G, {total_change:+.0f} Total FPts\n"
                summary_parts.append(summary)
        return "\n".join(summary_parts)

def get_team_name(team_id: str) -> str:
    """Get full team name from team ID."""
    return TEAM_MAPPINGS.get(team_id, team_id)

def get_all_teams() -> List[str]:
    """Get a list of all teams from the data."""
    # This function relies on TEAM_MAPPINGS, which is a complete list of all teams.
    # It does not need to check session state.
    return sorted(TEAM_MAPPINGS.keys())

def run_trade_analysis(
    trade_teams: Dict[str, Dict[str, str]],
    num_players: int,
    trade_label: str = "",
    trade_date: Optional[str] = None,
    include_advanced_metrics: bool = True,
    assumed_fpg_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run the trade analysis, record it in history, and return the results."""
    if st.session_state.trade_analyzer:
        start_time = time.perf_counter()
        st.session_state.trade_analyzer.update_data(st.session_state.combined_data)
        try:
            results = st.session_state.trade_analyzer.evaluate_trade_fairness(
                trade_teams,
                num_players,
                include_advanced_metrics=include_advanced_metrics,
                assumed_fpg_overrides=assumed_fpg_overrides,
            )
        except TypeError:
            # Backwards compatibility for existing TradeAnalyzer instances: reinitialize
            st.session_state.trade_analyzer = TradeAnalyzer(st.session_state.combined_data)
            results = st.session_state.trade_analyzer.evaluate_trade_fairness(
                trade_teams,
                num_players,
                include_advanced_metrics=include_advanced_metrics,
                assumed_fpg_overrides=assumed_fpg_overrides,
            )
        duration = time.perf_counter() - start_time
        try:
            st.session_state["trade_analysis_last_duration_sec"] = float(duration)
        except Exception:
            pass
        if results:
            try:
                summary = st.session_state.trade_analyzer._generate_trade_summary(results)
            except Exception:
                summary = ""
            history_entry = {
                "id": uuid.uuid4().hex,
                "trade_teams": trade_teams,
                "summary": summary,
                "label": trade_label or "",
                "date": trade_date or "",
                "num_players": num_players,
            }
            st.session_state.trade_analyzer.trade_history.append(history_entry)
        return results
    return {}
