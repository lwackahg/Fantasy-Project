"""
Business logic for the trade analysis feature.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple

from data_loader import TEAM_MAPPINGS
from debug import debug_manager
from modules.trade_analysis.consistency_integration import (
	load_player_consistency,
	load_all_player_consistency,
	enrich_roster_with_consistency
)
from modules.player_value.logic import build_player_value_profiles

# Import league config
try:
	from league_config import FANTRAX_DEFAULT_LEAGUE_ID
except ImportError:
	FANTRAX_DEFAULT_LEAGUE_ID = ""


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

    def evaluate_trade_fairness(self, trade_teams: Dict[str, Dict[str, str]], num_top_players: int = 10) -> Dict[str, Dict[str, Any]]:
        """Evaluate the fairness of a trade between teams."""
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
                    pre_trade_rosters[time_range] = range_data.nlargest(num_top_players, 'FP/G')[['Player', 'Team', 'FPts', 'FP/G', 'GP']].to_dict('records')
                    post_trade_data = range_data[~range_data['Player'].isin(outgoing_players)].copy()
                    incoming_data = []
                    for player in incoming_players:
                        player_data = self.data[
                            (self.data.index == player) &
                            (self.data['Timestamp'] == time_range)
                        ].reset_index()
                        if not player_data.empty:
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
            if league_id:
                try:
                    value_profiles_df = build_player_value_profiles(league_id)
                except Exception:
                    value_profiles_df = None
            
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
                    
                    # Add consistency metrics if available
                    if league_id:
                        enriched_pre = enrich_roster_with_consistency(pre_roster_df.copy(), league_id)
                        if 'CV%' in enriched_pre.columns:
                            cv_values = enriched_pre['CV%'].dropna()
                            if len(cv_values) > 0:
                                pre_trade_consistency[time_range] = {
                                    'avg_cv': cv_values.mean(),
                                    'players_with_data': len(cv_values),
                                    'very_consistent': len(cv_values[cv_values < 20]),
                                    'moderate': len(cv_values[(cv_values >= 20) & (cv_values <= 30)]),
                                    'volatile': len(cv_values[cv_values > 30])
                                }
                    # Add value score aggregates if profiles are available
                    if value_profiles_df is not None and not value_profiles_df.empty:
                        merged_pre = pre_roster_df.merge(
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
                    post_roster_df = pd.DataFrame(post_trade_rosters[time_range])
                    post_trade_metrics[time_range] = {
                        'mean_fpg': post_roster_df['FP/G'].mean(),
                        'median_fpg': post_roster_df['FP/G'].median(),
                        'std_dev': post_roster_df['FP/G'].std(),
                        'total_fpts': post_roster_df['FPts'].sum(),
                        'avg_gp': post_roster_df['GP'].mean()
                    }
                    
                    # Add consistency metrics if available
                    if league_id:
                        enriched_post = enrich_roster_with_consistency(post_roster_df.copy(), league_id)
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
                    # Add value score aggregates if profiles are available
                    if value_profiles_df is not None and not value_profiles_df.empty:
                        merged_post = post_roster_df.merge(
                            value_profiles_df[['Player', 'ValueScore']],
                            on='Player',
                            how='left'
                        )
                        if not merged_post.empty:
                            post_trade_value_scores[time_range] = {
                                'total_value_score': float(merged_post['ValueScore'].fillna(0).sum()),
                                'avg_value_score': float(merged_post['ValueScore'].fillna(0).mean())
                            }

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
                'league_id': league_id
            }
        return analysis_results

    def get_trade_history(self) -> List[Tuple[Dict[str, Dict[str, str]], str]]:
        """Get the history of analyzed trades."""
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

def run_trade_analysis(trade_teams: Dict[str, Dict[str, str]], num_players: int) -> Dict[str, Any]:
    """Run the trade analysis and return the results."""
    if st.session_state.trade_analyzer:
        st.session_state.trade_analyzer.update_data(st.session_state.combined_data)
        results = st.session_state.trade_analyzer.evaluate_trade_fairness(trade_teams, num_players)
        return results
    return {}
