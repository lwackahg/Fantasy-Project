import streamlit as st
from schedule_display import load_schedule_data

def test_schedule_data_loading():
    """Test function to verify schedule data loading works correctly."""
    print("Testing schedule data loading...")
    
    # Load the schedule data
    schedule_df = load_schedule_data()
    
    if schedule_df is None:
        print("ERROR: Failed to load schedule data.")
        return False
    
    if schedule_df.empty:
        print("ERROR: Schedule data is empty.")
        return False
    
    # Check if we have the expected columns
    expected_columns = [
        "Scoring Period", "Date Range", "Team 1", "Score 1", 
        "Team 2", "Score 2", "Winner", "Period Number"
    ]
    
    for col in expected_columns:
        if col not in schedule_df.columns:
            print(f"ERROR: Missing expected column: {col}")
            return False
    
    # Check if we have data for all scoring periods
    num_periods = schedule_df["Period Number"].nunique()
    print(f"Found data for {num_periods} scoring periods.")
    
    # Count total matchups
    total_matchups = len(schedule_df)
    print(f"Total matchups: {total_matchups}")
    
    # Print some sample data
    print("\nSample data:")
    print(schedule_df.head(3))
    
    print("\nSchedule data loading test passed successfully!")
    return True

if __name__ == "__main__":
    test_schedule_data_loading()
