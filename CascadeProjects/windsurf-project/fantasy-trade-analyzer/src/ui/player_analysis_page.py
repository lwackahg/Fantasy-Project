"""Player analysis page component."""
import streamlit as st
import pandas as pd
from shared_utils import calculate_z_scores, get_player_stats
from visualization import plot_player_stats, plot_player_trends
from analyze_data import analyze_player_performance

def display_player_comparison():
    """Display player comparison interface."""
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload player data first.")
        return
        
    st.write("## Player Comparison")
    
    # Get all players
    all_players = sorted(st.session_state.data['Player'].unique())
    
    # Player selection
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Select Player 1", all_players, key='player1')
    with col2:
        player2 = st.selectbox("Select Player 2", all_players, key='player2')
        
    if player1 and player2:
        # Get player stats
        stats1 = get_player_stats(st.session_state.data, [player1])
        stats2 = get_player_stats(st.session_state.data, [player2])
        
        # Plot comparison
        plot_player_stats(
            stats1,
            stats2,
            player1,
            player2,
            st.session_state.categories
        )

def display_player_trends():
    """Display player performance trends over time."""
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload player data first.")
        return
        
    st.write("## Player Performance Trends")
    
    # Get all players
    all_players = sorted(st.session_state.data['Player'].unique())
    
    # Player selection
    selected_player = st.selectbox("Select Player", all_players)
    
    # Category selection
    categories = st.multiselect(
        "Select Categories to Analyze",
        st.session_state.categories,
        default=st.session_state.categories[:3]
    )
    
    if selected_player and categories:
        plot_player_trends(
            st.session_state.data,
            selected_player,
            categories
        )

def display_player_analysis():
    """Display detailed player analysis."""
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload player data first.")
        return
        
    st.write("## Player Analysis")
    
    # Get all players
    all_players = sorted(st.session_state.data['Player'].unique())
    
    # Player selection
    selected_player = st.selectbox("Select Player for Analysis", all_players)
    
    if selected_player:
        # Get player analysis
        analysis = analyze_player_performance(
            st.session_state.data,
            selected_player
        )
        
        # Display analysis results
        st.write("### Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Value Score", f"{analysis['value_score']:.2f}")
        with col2:
            st.metric("Consistency Score", f"{analysis['consistency']:.2f}")
        with col3:
            st.metric("Trend Score", f"{analysis['trend']:.2f}")
            
        # Display detailed stats
        st.write("### Detailed Statistics")
        st.dataframe(analysis['detailed_stats'])

def render_player_analysis_page():
    """Main function to render the player analysis page."""
    st.write("# Player Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "Player Comparison",
        "Player Trends",
        "Detailed Analysis"
    ])
    
    with tab1:
        display_player_comparison()
        
    with tab2:
        display_player_trends()
        
    with tab3:
        display_player_analysis()

def display_player_stats(player: str):
    """Display comprehensive statistics for a selected player."""
    current_data = st.session_state.data_ranges[st.session_state.current_range]
    player_data = current_data[current_data['Player'] == player]
    
    if player_data.empty:
        st.warning(f"No data available for {player}")
        return
        
    # Calculate player metrics
    player_metrics = analyze_player_trends(player_data)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fantasy Points per Game", f"{player_metrics['mean_fp']:.1f}")
    with col2:
        st.metric("Total Fantasy Points", f"{player_metrics['total_fp']:.0f}")
        
    # Display performance chart
    visualizer = DataVisualizer()
    st.plotly_chart(visualizer.plot_player_performance(player_data))
    
    # Display detailed statistics
    st.subheader("Detailed Statistics")
    stats_df = player_data.drop(['Player', 'Team', 'Status'], axis=1)
    st.dataframe(stats_df)

def display_player_analysis_page():
    """Display the player analysis interface."""
    st.header("Player Analysis")
    
    if not st.session_state.data_ranges:
        st.warning("Please upload player data files first.")
        return
        
    current_data = st.session_state.data_ranges[st.session_state.current_range]
    all_players = sorted(current_data['Player'].unique())
    
    selected_player = st.selectbox(
        "Select Player to Analyze",
        all_players,
        key="player_analysis_player"
    )
    
    if selected_player:
        display_player_stats(selected_player)

render_player_analysis_page()
