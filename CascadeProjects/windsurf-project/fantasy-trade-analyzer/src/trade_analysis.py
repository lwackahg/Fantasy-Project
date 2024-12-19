# trade_analysis.py
import streamlit as st
import pandas as pd

def display_trade_analysis(player_name, trade_details):
    """
    Display detailed analysis for potential trades.
    
    Args:
        player_name (str): Name of the player being offered
        trade_details (list): List of tuples containing (target_player, target_fpg)
    """
    st.title("Trade Analysis")
    
    # Remove emoji and arrow from player name for display
    clean_player_name = player_name.split(" ➜")[0].replace("📈 ", "")
    st.write(f"### Analysis for {clean_player_name}")
    
    if not trade_details:
        st.warning("No viable trade targets found.")
        return
        
    # Create DataFrame for trade targets
    trade_df = pd.DataFrame(trade_details, columns=['Player', 'FP/G'])
    trade_df = trade_df.sort_values('FP/G', ascending=False)
    
    # Style the DataFrame
    styled_df = trade_df.style\
        .highlight_max('FP/G', color='green')\
        .highlight_min('FP/G', color='red')\
        .format({'FP/G': '{:.2f}'})
    
    st.write("#### Potential Trade Targets")
    st.write("Players are ranked by Fantasy Points per Game (FP/G)")
    st.dataframe(styled_df)
    
    # Additional analysis and recommendations
    best_target = trade_df.iloc[0]
    worst_target = trade_df.iloc[-1]
    
    st.write("### Trade Recommendations")
    
    # Best trade recommendation
    st.write(f"🔥 **Best Trade Target:** {best_target['Player']}")
    st.write(f"- Average FP/G: {best_target['FP/G']:.2f}")
    
    # Show trade value comparison
    mid_value = trade_df['FP/G'].median()
    st.write("\n### Trade Value Distribution")
    st.write(f"- Median Trade Value: {mid_value:.2f} FP/G")
    st.write(f"- Value Range: {worst_target['FP/G']:.2f} - {best_target['FP/G']:.2f} FP/G")
    
    # Create a bar chart of trade values
    st.bar_chart(trade_df.set_index('Player')['FP/G'])

def calculate_trade_value(player_stats):
    """
    Calculate the trade value for a player based on their statistics.
    
    Args:
        player_stats (pd.DataFrame): DataFrame containing player statistics
        
    Returns:
        float: Calculated trade value
    """
    if player_stats.empty:
        return 0.0
        
    avg_fpg = player_stats['FP/G'].mean()
    std_dev = player_stats['FP/G'].std()
    
    # Simple trade value calculation
    # Could be enhanced with more sophisticated metrics
    trade_value = avg_fpg * (1 - (std_dev / avg_fpg if avg_fpg > 0 else 0))
    return max(0, trade_value)