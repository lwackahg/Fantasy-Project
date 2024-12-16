import streamlit as st
import pandas as pd

def display_player_data(data_ranges, combined_data):
    """Display the player data in a clean and searchable format."""
    st.subheader("Player Data Overview")
    
    if not combined_data.empty:
        search_query = st.text_input("Search Players", "").strip()  # Trim whitespace
        
        if search_query:
            # Filter combined_data to find players matching the search query
            filtered_data = combined_data[combined_data.index.str.contains(search_query, case=False, na=False)]
            
            if not filtered_data.empty:
                # Reset index for displaying
                filtered_data.reset_index(inplace=True)
                st.write(f"Data for **{search_query}**:")
                st.dataframe(filtered_data)  # Display filtered player's data
            else:
                st.warning("Player not found.")
        else:
            st.dataframe(combined_data)  # Display all player data if no search query
    else:
        st.warning("No data available to display.")

def display_metrics(data):
    """Display basic statistics as metrics in the Streamlit app."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Players", len(data))
    with col2:
        st.metric("Teams", data['Team'].nunique())
    with col3:
        st.metric("Avg FP/G", f"{data['FP/G'].mean():.1f}")
