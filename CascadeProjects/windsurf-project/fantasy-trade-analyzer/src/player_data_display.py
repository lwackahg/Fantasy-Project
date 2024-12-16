import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

def display_player_trends(player):
    """Display trend lines for the selected player."""
    all_data = []
    
    for key, data in st.session_state.data_ranges.items():
        player_data = data[data['Player'] == player]
        if not player_data.empty:
            player_data['Time Range'] = key  # tag the data with the time range we are plotting
            all_data.append(player_data)
    
    if all_data:
        combined_data = pd.concat(all_data)
        combined_data.sort_values(by='Timestamp', inplace=True)  # Use the new Timestamp column

        # Plotting metrics trend
        plt.figure(figsize=(12, 6))
        plt.plot(combined_data['Timestamp'], combined_data['FPts'], label='Fantasy Points')
        plt.title(f'{player} Fantasy Points Over Time')
        plt.xlabel('Time')
        plt.ylabel('Fantasy Points')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.warning(f"No historical data available for {player}.")