"""Trade analysis page component."""
import streamlit as st
from typing import Dict, List
from ..analyze_data import analyze_team_performance
from ..visualization import DataVisualizer

def display_trade_analysis_page():
    """Display the trade analysis interface."""
    st.header("Trade Analysis")
    
    if not st.session_state.data_ranges:
        st.warning("Please upload player data files first.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Team 1")
        team1 = st.selectbox("Select Team 1", get_all_teams(), key="team1")
        
    with col2:
        st.subheader("Team 2")
        team2 = st.selectbox("Select Team 2", get_all_teams(), key="team2")
        
    if team1 and team2:
        setup_trade(team1, team2)

def setup_trade(team1: str, team2: str):
    """Setup the trade interface for two teams."""
    current_data = st.session_state.data_ranges[st.session_state.current_range]
    
    team1_players = current_data[current_data['Status'] == team1]['Player'].tolist()
    team2_players = current_data[current_data['Status'] == team2]['Player'].tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        team1_selected = st.multiselect(
            f"Select players from {team1}",
            team1_players,
            key="team1_players"
        )
        
    with col2:
        team2_selected = st.multiselect(
            f"Select players from {team2}",
            team2_players,
            key="team2_players"
        )
        
    if team1_selected and team2_selected:
        analyze_trade(team1, team2, team1_selected, team2_selected)

def analyze_trade(team1: str, team2: str, team1_players: List[str], team2_players: List[str]):
    """Analyze the proposed trade."""
    if st.button("Analyze Trade"):
        trade_analysis = st.session_state.trade_analyzer.analyze_trade(
            {team1: team1_players, team2: team2_players}
        )
        display_trade_results(trade_analysis)

def display_trade_results(analysis: Dict):
    """Display the results of the trade analysis."""
    st.subheader("Trade Analysis Results")
    
    for team, results in analysis.items():
        st.write(f"### {team}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Value Change", f"{results['value_change']:.1f}")
        with col2:
            st.metric("Fairness Score", f"{results['fairness_score']:.2%}")
        with col3:
            st.metric("Risk Score", f"{results['risk_score']:.2f}")

def get_all_teams() -> List[str]:
    """Get list of all teams from current data."""
    if not st.session_state.data_ranges or not st.session_state.current_range:
        return []
    
    current_data = st.session_state.data_ranges[st.session_state.current_range]
    return sorted(current_data['Status'].unique())
