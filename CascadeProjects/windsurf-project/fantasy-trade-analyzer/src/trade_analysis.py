"""Trade analysis module for Fantasy Basketball Trade Analyzer."""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from debug import debug_manager
from data_loader import TEAM_MAPPINGS

def get_team_name(team_id: str) -> str:
    """Get full team name from team ID"""
    return TEAM_MAPPINGS.get(team_id, team_id)

def get_all_teams() -> List[str]:
    """Get a list of all teams from the data"""
    if not st.session_state.data_ranges:
        return []
    
    # Use the first available data range as reference for teams
    if st.session_state.current_range and st.session_state.current_range in st.session_state.data_ranges:
        data = st.session_state.data_ranges[st.session_state.current_range]
        if data is not None:
            return sorted(data['Status'].unique())
    
    # Fallback to first available range if current_range is not set
    for range_key, data in st.session_state.data_ranges.items():
        if data is not None:
            return sorted(data['Status'].unique())
    return []

def display_trade_analysis_page():
    """Display the trade analysis page"""
    st.write("## Trade Analysis")
    
    # Let user select the data range for analysis
    available_ranges = list(st.session_state.data_ranges.keys())
    selected_range = st.selectbox(
        "Select Data Range for Analysis",
        available_ranges,
        index=available_ranges.index(st.session_state.current_range) if st.session_state.current_range in available_ranges else 0
    )
    
    # Update current range and data
    st.session_state.current_range = selected_range
    st.session_state.data = st.session_state.data_ranges[selected_range]
    
    # Update trade analyzer with new data
    if st.session_state.trade_analyzer:
        st.session_state.trade_analyzer.update_data(st.session_state.data)
    
    # Setup and analyze trade
    trade_setup()
    
    # Display trade history
    st.write("## Trade Analysis History")
    if st.session_state.trade_analyzer:
        history = st.session_state.trade_analyzer.get_trade_history()
        for trade, blurb in history:
            st.write(blurb)
    else:
        st.write("No trade history available.")

def trade_setup():
    """Setup the trade with drag and drop interface"""
    debug_manager.log("Starting trade setup", level='debug')
    st.write("## Trade Setup")
    
    # Let users set weights for each time range
    st.write("### Time Range Weights")
    st.write("Set the importance of each time range in the analysis (weights should sum to 1.0)")
    
    time_range_weights = {}
    available_ranges = list(st.session_state.data_ranges.keys())
    
    # Create columns for weight inputs
    weight_cols = st.columns(len(available_ranges))
    
    # Calculate default weight with better precision
    default_weight = round(1.0 / len(available_ranges), 2)
    remaining_weight = 1.0
    
    for i, time_range in enumerate(available_ranges):
        with weight_cols[i]:
            # For the last range, use remaining weight as default
            if i == len(available_ranges) - 1:
                default = round(remaining_weight, 2)
            else:
                default = default_weight
                
            weight = st.number_input(
                f"{time_range} Weight",
                min_value=0.0,
                max_value=1.0,
                value=default,
                step=0.05,
                format="%.2f",
                key=f"weight_{time_range}"
            )
            time_range_weights[time_range] = weight
            remaining_weight -= weight
    
    # Calculate total with better precision
    total_weight = round(sum(time_range_weights.values()), 2)
    
    # Warning if weights don't sum to 1
    if total_weight != 1.0:
        st.warning(f"Weights sum to {total_weight:.2f}. Consider adjusting to sum to 1.0 for more accurate analysis.")
    
    # Display team legend with improved styling
    st.write("## Team Legend")
    team_legend = """
    <style>
    .team-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
    }
    .team-card {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 15px;
        text-align: center;
        width: 200px;
        transition: transform 0.2s, box-shadow 0.2s;
        color: #333;
    }
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    </style>
    <div class="team-legend">
    """
    for team_id, team_name in TEAM_MAPPINGS.items():
        team_legend += f"<div class='team-card'><strong>{team_id}</strong><br>{team_name}</div>"
    team_legend += "</div>"
    st.markdown(team_legend, unsafe_allow_html=True)

    st.write("## Analysis Settings")
    st.write("### Select Teams to Trade Between")
    selected_teams = st.multiselect(
        "Choose teams involved in the trade",
        options=get_all_teams(),
        help="Select the teams that will participate in the trade.",
        format_func=lambda x: get_team_name(x)
    )

    if not selected_teams:
        st.warning("Please select teams to begin trading.")
        return

    # Dictionary to store players involved in trade for each team
    trade_teams = {}
    
    # Create columns only for selected teams
    cols = st.columns(len(selected_teams))
    
    # Create drag and drop interface only for selected teams
    for i, team in enumerate(selected_teams):
        with cols[i]:
            team_name = get_team_name(team)
            st.write(f"### {team_name}")
            
            # Get available players for this team
            available_players = []
            for time_range, data in st.session_state.data_ranges.items():
                if data is not None:
                    data['Full Team Name'] = data['Status'].map(lambda x: get_team_name(x))
                    team_players = data[data['Full Team Name'] == team_name]['Player'].unique()
                    available_players.extend(team_players)
            
            available_players = list(set(available_players))
            available_players.sort()
            
            # Debug logging
            debug_manager.log(f"Available players for {team_name}: {len(available_players)}", level='debug')
            
            # Multi-select for players
            selected_players = st.multiselect(
                f"Select players from {team_name}",
                available_players,
                key=f"{team}_players"
            )
            
            trade_teams[team] = selected_players
            
            # Debug logging
            if selected_players:
                debug_manager.log(f"Selected players for {team_name}: {len(selected_players)}", level='debug')

    # Only show player assignment section if teams have selected players
    active_teams = {team: players for team, players in trade_teams.items() if players}
    
    # Debug logging
    debug_manager.log(f"Active teams: {len(active_teams)}", level='debug')
    
    if active_teams:
        st.write("### Assign Players to Teams")
        for team, players in active_teams.items():
            if players:  # Only show teams with selected players
                st.write(f"#### {get_team_name(team)}")
                for player in players:
                    destination_team = st.selectbox(
                        f"Select destination team for {player}",
                        options=[t for t in selected_teams if t != team],
                        key=f"{player}_destination"
                    )
                    # Store the destination team for each player
                    if isinstance(trade_teams[team], list):
                        trade_teams[team] = {}
                    trade_teams[team][player] = destination_team
        
        st.write("### Trade Summary")
        for team in selected_teams:
            players = trade_teams.get(team, {})
            if isinstance(players, dict) and players:  # Only show teams with assigned players
                st.write(f"**{get_team_name(team)}** will trade:")
                for player, dest in players.items():
                    st.write(f"- {player} to {get_team_name(dest)}")
        
        # Add analyze button
        if st.button("Analyze Trade", key="analyze_trade_button", help="Click to see detailed trade analysis"):
            # Debug logging
            debug_manager.log("Starting trade analysis", level='debug')
            debug_manager.log(f"Trade teams data: {trade_teams}", level='debug')
            
            analysis = st.session_state.trade_analyzer.evaluate_trade_fairness(trade_teams, time_range_weights)
            
            # Debug logging
            debug_manager.log("Analysis complete", level='debug')
            
            # Display analysis results
            st.write("## Trade Analysis Results")
            for team, result in analysis.items():
                st.write(f"### {get_team_name(team)}")
                
                # Display incoming players
                st.write("**Receiving:**")
                for player in result['incoming_players']:
                    st.write(f"- {player}")
                    # Show detailed FP/G breakdown
                    details = result['value_details']['incoming'][player]
                    for time_range, stats in details.items():
                        st.write(f"  - {time_range}: {stats['FP/G']:.1f} FP/G (weighted: {stats['weighted_value']:.1f})")
                
                # Display outgoing players
                st.write("**Trading Away:**")
                for player in result['outgoing_players']:
                    st.write(f"- {player}")
                    # Show detailed FP/G breakdown
                    details = result['value_details']['outgoing'][player]
                    for time_range, stats in details.items():
                        st.write(f"  - {time_range}: {stats['FP/G']:.1f} FP/G (weighted: {stats['weighted_value']:.1f})")
                
                # Display net value change
                net_value = result['value_change']
                value_color = "green" if net_value > 0 else "red" if net_value < 0 else "gray"
                st.markdown(f"**Net Value Change:** <span style='color: {value_color}'>{net_value:.1f}</span>", unsafe_allow_html=True)
                
            # Update trade history
            for team, result in analysis.items():
                incoming = ', '.join(result['incoming_players'])
                outgoing = ', '.join(result['outgoing_players'])
                net_value = result['value_change']
                blurb = (
                    f"Trade Impact for {team}:\n"
                    f"Receiving: {incoming}\n"
                    f"Trading Away: {outgoing}\n"
                    f"Net Value Change: {net_value:.1f}"
                )
                st.session_state.trade_analyzer.trade_history.append((team, blurb))
                if len(st.session_state.trade_analyzer.trade_history) > 25:
                    st.session_state.trade_analyzer.trade_history.pop(0)
