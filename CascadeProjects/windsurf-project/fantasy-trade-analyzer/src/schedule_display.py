import streamlit as st
from data_loader import load_schedule_data, calculate_team_stats
from schedule_analysis import swap_team_schedules, compare_team_stats, calculate_all_schedule_swaps
import plotly.express as px
import pandas as pd

def display_schedule_page():
    """
    Display the schedule data in a table format with filtering options.
    """
    # Initialize session state variables if they don't exist
    if 'schedule_view_type' not in st.session_state:
        st.session_state.schedule_view_type = "List View"
    if 'schedule_selected_period' not in st.session_state:
        st.session_state.schedule_selected_period = "All Periods"
    if 'schedule_selected_team' not in st.session_state:
        st.session_state.schedule_selected_team = "All Teams"
    if 'schedule_swap_team1' not in st.session_state:
        st.session_state.schedule_swap_team1 = None
    if 'schedule_swap_team2' not in st.session_state:
        st.session_state.schedule_swap_team2 = None
    if 'schedule_swap_performed' not in st.session_state:
        st.session_state.schedule_swap_performed = False
    if 'all_swaps_calculated' not in st.session_state:
        st.session_state.all_swaps_calculated = False
    if 'all_swaps_data' not in st.session_state:
        st.session_state.all_swaps_data = None
    if 'all_swaps_original_stats' not in st.session_state:
        st.session_state.all_swaps_original_stats = None
    if 'all_swaps_summary' not in st.session_state:
        st.session_state.all_swaps_summary = None
    
    st.title("Fantasy League Schedule")
    st.write("View the complete schedule and results for all fantasy matchups.")
    
    # Load the schedule data
    schedule_df = load_schedule_data()
    
    if schedule_df is None or schedule_df.empty:
        st.error("No schedule data available.")
        return
    
    # Add filters in an expander for cleaner UI
    with st.expander("Filter Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get unique periods for filtering
            periods = schedule_df["Scoring Period"].unique().tolist()
            
            # Safely get index for the selected period
            period_index = 0
            if st.session_state.schedule_selected_period != "All Periods" and st.session_state.schedule_selected_period in periods:
                period_index = periods.index(st.session_state.schedule_selected_period) + 1
            
            selected_period = st.selectbox(
                "Filter by Scoring Period",
                ["All Periods"] + periods,
                index=period_index
            )
            # Update session state
            st.session_state.schedule_selected_period = selected_period
        
        with col2:
            # Get unique teams for filtering
            all_teams = set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique())
            all_teams = sorted(list(all_teams))
            
            # Safely get index for the selected team
            team_index = 0
            if st.session_state.schedule_selected_team != "All Teams" and st.session_state.schedule_selected_team in all_teams:
                team_index = all_teams.index(st.session_state.schedule_selected_team) + 1
            
            selected_team = st.selectbox(
                "Filter by Team",
                ["All Teams"] + all_teams,
                index=team_index
            )
            # Update session state
            st.session_state.schedule_selected_team = selected_team
        
        with col3:
            # Add view type option
            view_index = 0 if st.session_state.schedule_view_type == "List View" else 1
            
            view_type = st.radio(
                "View Type",
                ["List View", "Table View"],
                index=view_index
            )
            # Update session state
            st.session_state.schedule_view_type = view_type
    
    # Apply filters
    filtered_df = schedule_df.copy()
    
    if selected_period != "All Periods":
        filtered_df = filtered_df[filtered_df["Scoring Period"] == selected_period]
    
    if selected_team != "All Teams":
        filtered_df = filtered_df[
            (filtered_df["Team 1"] == selected_team) | 
            (filtered_df["Team 2"] == selected_team)
        ]
    
    # Display the filtered schedule
    if not filtered_df.empty:
        # Create tabs for Schedule, Team Stats, and Schedule Swap
        tab1, tab2, tab3 = st.tabs(["Schedule", "Team Performance", "Schedule Swap"])
        
        with tab1:
            if view_type == "List View":
                display_list_view(filtered_df)
            else:  # Table View
                display_table_view(filtered_df)
            
            # Add a download button for the schedule data
            st.download_button(
                label="Download Schedule Data as CSV",
                data=filtered_df.to_csv(index=False),
                file_name="fantasy_schedule.csv",
                mime="text/csv",
            )
        
        with tab2:
            display_team_stats(schedule_df)
            
        with tab3:
            display_schedule_swap(schedule_df)
    else:
        st.info("No matchups found with the selected filters.")

def display_list_view(filtered_df):
    """
    Display schedule data in a list view grouped by scoring period.
    
    Args:
        filtered_df (pd.DataFrame): Filtered schedule data
    """
    # Group by scoring period and date range
    periods = filtered_df["Scoring Period"].unique()
    
    for period in periods:
        period_df = filtered_df[filtered_df["Scoring Period"] == period]
        date_range = period_df["Date Range"].iloc[0]
        
        # Display period header
        st.markdown(f"### {period}")
        st.markdown(f"*{date_range}*")
        
        # Display matchups
        for _, row in period_df.iterrows():
            team1 = row["Team 1"]
            score1 = row["Score 1 Display"]
            team2 = row["Team 2"]
            score2 = row["Score 2 Display"]
            winner = row["Winner"]
            
            # Format the matchup with color based on winner
            if winner == team1:
                st.markdown(f"**{team1}** - {score1} vs {team2} - {score2}")
            elif winner == team2:
                st.markdown(f"{team1} - {score1} vs **{team2}** - {score2}")
            else:
                st.markdown(f"{team1} - {score1} vs {team2} - {score2}")
        
        st.markdown("---")

def display_table_view(filtered_df):
    """
    Display schedule data in a table view.
    
    Args:
        filtered_df (pd.DataFrame): Filtered schedule data
    """
    # Prepare display columns for table view
    display_df = filtered_df[["Scoring Period", "Date Range", "Team 1", "Score 1 Display", "Team 2", "Score 2 Display", "Winner"]].copy()
    
    # Rename columns for better display
    display_df = display_df.rename(columns={
        "Team 1": "Home Team",
        "Team 2": "Away Team",
        "Score 1 Display": "Home Score",
        "Score 2 Display": "Away Score"
    })
    
    # Show the table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

def display_team_stats(schedule_df):
    """
    Display team performance statistics.
    
    Args:
        schedule_df (pd.DataFrame): The complete schedule data
    """
    st.subheader("Team Performance Summary")
    
    # Calculate team stats
    team_stats = calculate_team_stats(schedule_df)
    
    # Format the team stats table
    formatted_stats = team_stats.copy()
    
    # Add win-loss-tie record column
    formatted_stats["Record"] = formatted_stats.apply(
        lambda row: f"{int(row['Wins'])}-{int(row['Losses'])}-{int(row['Ties'])}",
        axis=1
    )
    
    # Reorder columns for better display
    display_columns = [
        "Record", "Win %", "Points For", "Points Against", 
        "Avg Points For", "Avg Points Against", "Total Matchups"
    ]
    
    formatted_stats = formatted_stats[display_columns]
    
    # Display the stats table
    st.dataframe(
        formatted_stats,
        use_container_width=True
    )

def display_schedule_swap(schedule_df):
    """
    Display the schedule swap tool UI.
    
    Args:
        schedule_df (pd.DataFrame): The complete schedule data
    """
    st.header("Schedule Swap Analysis")
    st.write("Simulate what would happen if two teams swapped their entire schedules.")
    
    # Get all unique teams
    teams = list(set(schedule_df["Team 1"].unique()) | set(schedule_df["Team 2"].unique()))
    # Remove any non-team entries (like scoring period headers)
    teams = [team for team in teams if not str(team).startswith("Scoring Period")]
    
    # Create two columns for team selection
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Select first team
        team1 = st.selectbox(
            "Select Team 1",
            options=teams,
            index=0 if st.session_state.schedule_swap_team1 is None else teams.index(st.session_state.schedule_swap_team1),
            key="team1_select"
        )
        st.session_state.schedule_swap_team1 = team1
    
    with col2:
        # Select second team
        # Filter out the first team from options
        team2_options = [team for team in teams if team != team1]
        
        team2 = st.selectbox(
            "Select Team 2",
            options=team2_options,
            index=0 if st.session_state.schedule_swap_team2 is None or st.session_state.schedule_swap_team2 == team1 else 
                  team2_options.index(st.session_state.schedule_swap_team2) if st.session_state.schedule_swap_team2 in team2_options else 0,
            key="team2_select"
        )
        st.session_state.schedule_swap_team2 = team2
    
    with col3:
        # Add analyze button
        analyze_button = st.button("Analyze Schedule Swap", use_container_width=True)
    
    # Add a button to calculate all possible swaps
    all_swaps_col1, all_swaps_col2 = st.columns([3, 1])
    
    with all_swaps_col2:
        calculate_all_button = st.button("Calculate All Swaps", use_container_width=True)
    
    with all_swaps_col1:
        st.write("Calculate all possible team schedule swaps to find the most impactful combinations.")
    
    # Handle the analyze button for a single swap
    if analyze_button:
        st.session_state.schedule_swap_performed = True
        
        with st.spinner("Analyzing schedule swap..."):
            # Perform the schedule swap
            swapped_df, original_stats, new_stats = swap_team_schedules(schedule_df, team1, team2)
            
            if swapped_df is not None:
                # Display the comparison
                st.subheader("Impact on Standings")
                
                # Get the comparison stats
                comparison = compare_team_stats(original_stats, new_stats)
                
                # Highlight the swapped teams
                st.markdown(f"### Teams Swapped: **{team1}** and **{team2}**")
                
                # Create columns for visualization
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Display the comparison table with conditional formatting
                    st.dataframe(
                        comparison.style.apply(
                            lambda x: ['background-color: #232300' if i in [team1, team2] else '' 
                                     for i in x.index],
                            axis=0
                        ).format({
                            "Original Win %": "{:.1f}%",
                            "New Win %": "{:.1f}%",
                            "Win % Change": "{:+.1f}%"
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    # Create a bar chart showing win percentage changes
                    top_changes = comparison.sort_values("Win % Change", ascending=False).head(6)
                    fig = {
                        "data": [
                            {
                                "x": top_changes.index,
                                "y": top_changes["Win % Change"],
                                "type": "bar",
                                "marker": {
                                    "color": ["#FFD700" if team in [team1, team2] else "#1E90FF" for team in top_changes.index]
                                }
                            }
                        ],
                        "layout": {
                            "title": "Top Win % Changes",
                            "xaxis": {"title": "Team"},
                            "yaxis": {"title": "Win % Change"},
                            "height": 300,
                            "margin": {"t": 40, "b": 30, "l": 40, "r": 10},
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                            "font": {"color": "#FFFFFF"}
                        }
                    }
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed impact for the swapped teams
                st.subheader("Detailed Impact on Swapped Teams")
                
                # Create columns for the two teams
                col1, col2 = st.columns(2)
                
                # Display detailed stats for team1
                with col1:
                    team_data = comparison.loc[team1]
                    st.markdown(f"### {team1}")
                    
                    # Calculate win difference
                    old_wins = int(team_data["Original Record"].split("-")[0])
                    new_wins = int(team_data["New Record"].split("-")[0])
                    win_diff = new_wins - old_wins
                    
                    # Calculate standings position change
                    original_standings = original_stats.sort_values("Win %", ascending=False).index.tolist()
                    new_standings = new_stats.sort_values("Win %", ascending=False).index.tolist()
                    
                    old_position = original_standings.index(team1) + 1
                    new_position = new_standings.index(team1) + 1
                    position_change = old_position - new_position
                    
                    # Create metrics for key stats
                    st.metric(
                        label="Record", 
                        value=team_data["New Record"], 
                        delta=f"{win_diff:+d} wins" if win_diff != 0 else "No change"
                    )
                    
                    # Show the change with an arrow indicator and color
                    win_pct_change = team_data["Win % Change"]
                    
                    st.metric(
                        label="Win %", 
                        value=f"{team_data['New Win %']:.1f}%", 
                        delta=f"{win_pct_change:+.1f}%"
                    )
                    
                    # Standings position
                    position_text = f"#{new_position}"
                    position_delta = None
                    delta_color = "off"  # Default to gray (no change)
                    
                    if position_change > 0:
                        position_delta = f"Up {position_change} spots"
                        delta_color = "normal"  # Green for improvement (lower number is better)
                    elif position_change < 0:
                        position_delta = f"Down {abs(position_change)} spots"
                        delta_color = "inverse"  # Red for decline
                    else:
                        position_delta = "No change"
                        
                    st.metric(
                        label="Standings Position", 
                        value=position_text,
                        delta=position_delta,
                        delta_color="normal" if position_change > 0 else "inverse" if position_change < 0 else "off"
                    )
                    
                    # Points change
                    pts_change = team_data["Pts For Change"]
                    st.metric(
                        label="Points For Change", 
                        value=f"{pts_change:+.1f}", 
                        delta=f"{pts_change:+.1f}",
                        delta_color="normal"
                    )
                    
                    st.markdown("---")
                
                # Display detailed stats for team2
                with col2:
                    team_data = comparison.loc[team2]
                    st.markdown(f"### {team2}")
                    
                    # Calculate win difference
                    old_wins = int(team_data["Original Record"].split("-")[0])
                    new_wins = int(team_data["New Record"].split("-")[0])
                    win_diff = new_wins - old_wins
                    
                    # Calculate standings position change
                    original_standings = original_stats.sort_values("Win %", ascending=False).index.tolist()
                    new_standings = new_stats.sort_values("Win %", ascending=False).index.tolist()
                    
                    old_position = original_standings.index(team2) + 1
                    new_position = new_standings.index(team2) + 1
                    position_change = old_position - new_position
                    
                    # Create metrics for key stats
                    st.metric(
                        label="Record", 
                        value=team_data["New Record"], 
                        delta=f"{win_diff:+d} wins" if win_diff != 0 else "No change"
                    )
                    
                    # Show the change with an arrow indicator and color
                    win_pct_change = team_data["Win % Change"]
                    
                    st.metric(
                        label="Win %", 
                        value=f"{team_data['New Win %']:.1f}%", 
                        delta=f"{win_pct_change:+.1f}%"
                    )
                    
                    # Standings position
                    position_text = f"#{new_position}"
                    position_delta = None
                    delta_color = "off"  # Default to gray (no change)
                    
                    if position_change > 0:
                        position_delta = f"Up {position_change} spots"
                        delta_color = "normal"  # Green for improvement (lower number is better)
                    elif position_change < 0:
                        position_delta = f"Down {abs(position_change)} spots"
                        delta_color = "inverse"  # Red for decline
                    else:
                        position_delta = "No change"
                        
                    st.metric(
                        label="Standings Position", 
                        value=position_text,
                        delta=position_delta,
                        delta_color="normal" if position_change > 0 else "inverse" if position_change < 0 else "off" 
                        
                    )
                    
                    # Points change
                    pts_change = team_data["Pts For Change"]
                    st.metric(
                        label="Points For Change", 
                        value=f"{pts_change:+.1f}", 
                        delta=f"{pts_change:+.1f}",
                        delta_color="normal"
                    )
                    
                    st.markdown("---")
                
                # Add a visualization of the swapped schedules
                st.subheader("Swapped Schedule Matchups")
                
                # Filter to show only the matchups involving the swapped teams
                team1_matchups = swapped_df[(swapped_df["Team 1"] == team1) | (swapped_df["Team 2"] == team1)]
                team2_matchups = swapped_df[(swapped_df["Team 1"] == team2) | (swapped_df["Team 2"] == team2)]
                
                # Create tabs for viewing the matchups
                tab1, tab2 = st.tabs([f"{team1}'s New Schedule", f"{team2}'s New Schedule"])
                
                with tab1:
                    # Display team1's new matchups
                    if not team1_matchups.empty:
                        # Clean up the display
                        display_df = team1_matchups.copy()
                        
                        # Skip scoring period rows
                        display_df = display_df[~display_df["Team 1"].astype(str).str.contains("Scoring Period")]
                        
                        # Include scoring period in the display
                        display_df = display_df[["Scoring Period", "Team 1", "Score 1", "Team 2", "Score 2", "Winner"]]
                        display_df.columns = ["Period", "Team", "Score", "Opponent", "Opp Score", "Winner"]
                        
                        # Extract period number for better display
                        display_df["Period"] = display_df["Period"].str.replace("Scoring Period ", "Period ")
                        
                        # Highlight rows where team1 won
                        st.dataframe(
                            display_df.style.apply(
                                lambda x: ['background-color: rgba(0, 255, 0, 0.2)' if x["Winner"] == team1 else 
                                          'background-color: rgba(255, 0, 0, 0.2)' if x["Winner"] != team1 and x["Winner"] != "Tie" else
                                          'background-color: rgba(255, 255, 0, 0.2)' for _ in x],
                                axis=1
                            ),
                            use_container_width=True
                        )
                
                with tab2:
                    # Display team2's new matchups
                    if not team2_matchups.empty:
                        # Clean up the display
                        display_df = team2_matchups.copy()
                        
                        # Skip scoring period rows
                        display_df = display_df[~display_df["Team 1"].astype(str).str.contains("Scoring Period")]
                        
                        # Include scoring period in the display
                        display_df = display_df[["Scoring Period", "Team 1", "Score 1", "Team 2", "Score 2", "Winner"]]
                        display_df.columns = ["Period", "Team", "Score", "Opponent", "Opp Score", "Winner"]
                        
                        # Extract period number for better display
                        display_df["Period"] = display_df["Period"].str.replace("Scoring Period ", "Period ")
                        
                        # Highlight rows where team2 won
                        st.dataframe(
                            display_df.style.apply(
                                lambda x: ['background-color: rgba(0, 255, 0, 0.2)' if x["Winner"] == team2 else 
                                          'background-color: rgba(255, 0, 0, 0.2)' if x["Winner"] != team2 and x["Winner"] != "Tie" else
                                          'background-color: rgba(255, 255, 0, 0.2)' for _ in x],
                                axis=1
                            ),
                            use_container_width=True
                        )
            else:
                st.error("Failed to perform schedule swap analysis.")
    
    # Handle the calculate all swaps button
    if calculate_all_button:
        with st.spinner("Calculating all possible schedule swaps... This may take a moment."):
            # Calculate all swaps
            all_swaps, original_stats, summary_df = calculate_all_schedule_swaps(schedule_df)
            
            # Store results in session state
            st.session_state.all_swaps_calculated = True
            st.session_state.all_swaps_data = all_swaps
            st.session_state.all_swaps_original_stats = original_stats
            st.session_state.all_swaps_summary = summary_df
    
    # Display all swaps summary if calculated
    if st.session_state.all_swaps_calculated and st.session_state.all_swaps_summary is not None:
        st.subheader("All Possible Schedule Swaps")
        
        # Create tabs for different views
        all_swaps_tab1, all_swaps_tab2 = st.tabs(["Most Impactful Swaps", "Filter by Team"])
        
        with all_swaps_tab1:
            # Display the most impactful swaps
            summary_df = st.session_state.all_swaps_summary
            
            # Create a more user-friendly display DataFrame
            display_df = pd.DataFrame()
            display_df["Team Pair"] = summary_df.apply(lambda row: f"{row['Team 1']} & {row['Team 2']}", axis=1)
            display_df["Team 1"] = summary_df["Team 1"]
            display_df["Team 1 Win % Change"] = summary_df["Team 1 Win % Change"].apply(lambda x: f"{x:+.1f}%")
            display_df["Team 1 Position Change"] = summary_df["Team 1 Position Change"].apply(
                lambda x: f"↑{abs(x)}" if x > 0 else (f"↓{abs(x)}" if x < 0 else "—")
            )
            display_df["Team 2"] = summary_df["Team 2"]
            display_df["Team 2 Win % Change"] = summary_df["Team 2 Win % Change"].apply(lambda x: f"{x:+.1f}%")
            display_df["Team 2 Position Change"] = summary_df["Team 2 Position Change"].apply(
                lambda x: f"↑{abs(x)}" if x > 0 else (f"↓{abs(x)}" if x < 0 else "—")
            )
            display_df["Total Impact"] = summary_df["Total Absolute Change"].apply(lambda x: f"{x:.1f}%")
            
            # Sort by total impact
            display_df = display_df.sort_values("Total Impact", ascending=False).reset_index(drop=True)
            
            # Apply styling
            def highlight_changes(val):
                if '↑' in str(val):
                    return 'color: green'
                elif '↓' in str(val):
                    return 'color: red'
                return ''
            
            def highlight_percentage(val):
                if '+' in str(val):
                    return 'color: green'
                elif '-' in str(val):
                    return 'color: red'
                return ''
            
            # Display the table with all swaps
            st.subheader("All Schedule Swaps (Sorted by Total Impact)")
            st.dataframe(
                display_df.style.applymap(
                    highlight_changes, 
                    subset=["Team 1 Position Change", "Team 2 Position Change"]
                ).applymap(
                    highlight_percentage,
                    subset=["Team 1 Win % Change", "Team 2 Win % Change"]
                )
            )
            
            # Create a bar chart of the top swaps
            top_pairs = summary_df.head(10)
            
            # Prepare data for the chart
            chart_data = []
            for _, row in top_pairs.iterrows():
                chart_data.append({
                    "Team": row["Team 1"],
                    "Win % Change": row["Team 1 Win % Change"],
                    "Pair": f"{row['Team 1']} & {row['Team 2']}"
                })
                chart_data.append({
                    "Team": row["Team 2"],
                    "Win % Change": row["Team 2 Win % Change"],
                    "Pair": f"{row['Team 1']} & {row['Team 2']}"
                })
            
            chart_df = pd.DataFrame(chart_data)
            
            # Create the chart
            fig = px.bar(
                chart_df, 
                x="Team", 
                y="Win % Change", 
                color="Pair",
                title="Win Percentage Changes for Top 10 Most Impactful Swaps",
                labels={"Win % Change": "Win % Change", "Team": "Team", "Pair": "Team Pair"},
                height=500
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Team",
                yaxis_title="Win % Change",
                legend_title="Team Pair",
                barmode="group"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with all_swaps_tab2:
            # Add a filter by team dropdown
            filter_team = st.selectbox(
                "Filter by Team",
                options=teams,
                index=0
            )
            
            # Filter the summary data to only show swaps involving the selected team
            filtered_summary = summary_df[
                (summary_df["Team 1"] == filter_team) | 
                (summary_df["Team 2"] == filter_team)
            ]
            
            # Create a display DataFrame
            filter_display_df = pd.DataFrame()
            filter_display_df["Swap Partner"] = filtered_summary.apply(
                lambda row: row["Team 2"] if row["Team 1"] == filter_team else row["Team 1"], 
                axis=1
            )
            
            # Get the selected team's win % change
            filter_display_df["Win % Change"] = filtered_summary.apply(
                lambda row: row["Team 1 Win % Change"] if row["Team 1"] == filter_team else row["Team 2 Win % Change"],
                axis=1
            )
            
            # Format win % change
            filter_display_df["Win % Change (formatted)"] = filter_display_df["Win % Change"].apply(lambda x: f"{x:+.1f}%")
            
            # Get position change
            filter_display_df["Position Change"] = filtered_summary.apply(
                lambda row: row["Team 1 Position Change"] if row["Team 1"] == filter_team else row["Team 2 Position Change"],
                axis=1
            )
            
            # Format position change
            filter_display_df["Position Change (formatted)"] = filter_display_df["Position Change"].apply(
                lambda x: f"↑{abs(x)}" if x > 0 else (f"↓{abs(x)}" if x < 0 else "—")
            )
            
            # Get partner win % change
            filter_display_df["Partner Win % Change"] = filtered_summary.apply(
                lambda row: row["Team 2 Win % Change"] if row["Team 1"] == filter_team else row["Team 1 Win % Change"],
                axis=1
            )
            
            # Format partner win % change
            filter_display_df["Partner Win % Change (formatted)"] = filter_display_df["Partner Win % Change"].apply(lambda x: f"{x:+.1f}%")
            
            # Sort by the selected team's win % change
            filter_display_df = filter_display_df.sort_values("Win % Change", ascending=False)
            
            # Apply styling
            def highlight_changes(val):
                if '↑' in str(val):
                    return 'color: green'
                elif '↓' in str(val):
                    return 'color: red'
                return ''
            
            def highlight_percentage(val):
                if '+' in str(val):
                    return 'color: green'
                elif '-' in str(val):
                    return 'color: red'
                return ''
            
            # Display the filtered data
            st.write(f"All possible schedule swaps for {filter_team}:")
            
            # Display columns we want to show
            display_columns = ["Swap Partner", "Win % Change (formatted)", "Position Change (formatted)", "Partner Win % Change (formatted)"]
            st.dataframe(
                filter_display_df[display_columns].style.applymap(
                    highlight_changes, 
                    subset=["Position Change (formatted)"]
                ).applymap(
                    highlight_percentage,
                    subset=["Win % Change (formatted)", "Partner Win % Change (formatted)"]
                ),
                use_container_width=True
            )
            
            # Create a bar chart showing all swaps for this team
            fig = px.bar(
                filter_display_df, 
                x="Swap Partner", 
                y="Win % Change",
                title=f"Win Percentage Changes for {filter_team} with Different Swap Partners",
                labels={"Win % Change": "Win % Change", "Swap Partner": "Swap Partner"},
                height=500,
                color="Win % Change",
                color_continuous_scale=["red", "white", "green"],
                range_color=[-max(abs(filter_display_df["Win % Change"])), max(abs(filter_display_df["Win % Change"]))]
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Swap Partner",
                yaxis_title="Win % Change"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a button to set these teams for detailed analysis
            selected_partner = filter_display_df.iloc[0]["Swap Partner"] if not filter_display_df.empty else None
            
            if selected_partner:
                if st.button(f"Analyze {filter_team} & {selected_partner} in Detail"):
                    st.session_state.schedule_swap_team1 = filter_team
                    st.session_state.schedule_swap_team2 = selected_partner
                    st.rerun()
