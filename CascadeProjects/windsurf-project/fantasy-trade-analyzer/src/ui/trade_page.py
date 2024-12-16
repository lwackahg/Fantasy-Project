"""Trade analysis page component."""
import streamlit as st
import pandas as pd
from shared_utils import calculate_z_scores, get_player_stats
from visualization import create_radar_chart, plot_player_comparison, DataVisualizer
from trade_analysis import analyze_trade_impact
from analyze_data import analyze_team_performance

def display_trade_setup():
    """Display the trade setup interface for selecting teams and players."""
    # Get unique team names
    teams = sorted(st.session_state.data['Team'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Team 1")
        team1 = st.selectbox("Select Team 1", teams, key='team1')
        team1_players = st.session_state.data[st.session_state.data['Team'] == team1]['Player'].unique()
        selected_players1 = st.multiselect("Select Players from Team 1", team1_players, key='players1')
        
    with col2:
        st.write("### Team 2")
        team2 = st.selectbox("Select Team 2", teams, key='team2')
        team2_players = st.session_state.data[st.session_state.data['Team'] == team2]['Player'].unique()
        selected_players2 = st.multiselect("Select Players from Team 2", team2_players, key='players2')
    
    return team1, team2, selected_players1, selected_players2

def display_trade_analysis(team1, team2, players1, players2):
    """Display trade analysis results."""
    if not (players1 and players2):
        st.warning("Please select players from both teams to analyze the trade.")
        return
        
    # Get player stats
    team1_stats = get_player_stats(st.session_state.data, players1)
    team2_stats = get_player_stats(st.session_state.data, players2)
    
    # Analyze trade impact
    trade_analysis = analyze_trade_impact(team1_stats, team2_stats)
    
    # Display results
    st.write("## Trade Analysis Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"### {team1} Receives:")
        for player in players2:
            st.write(f"- {player}")
            
    with col2:
        st.write(f"### {team2} Receives:")
        for player in players1:
            st.write(f"- {player}")
            
    # Display trade impact visualization
    create_radar_chart(trade_analysis, f"Trade Impact: {team1} vs {team2}")
    
    # Show player comparison
    if len(players1) == 1 and len(players2) == 1:
        plot_player_comparison(
            st.session_state.data,
            players1[0],
            players2[0],
            st.session_state.categories
        )

def render_trade_page():
    """Main function to render the trade analysis page."""
    st.write("## Trade Analysis")
    
    # Ensure data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload player data first.")
        return
        
    # Display trade setup interface
    team1, team2, players1, players2 = display_trade_setup()
    
    # Display trade analysis if players are selected
    if st.button("Analyze Trade"):
        display_trade_analysis(team1, team2, players1, players2)

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