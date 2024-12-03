"""Module for data visualization functionality"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def plot_performance_trends(data, selected_metrics, title="Performance Trends"):
    """Create a performance trend plot with visible data points"""
    fig = go.Figure()
    time_ranges = ['60 Days', '30 Days', '14 Days', '7 Days']
    x_positions = list(range(len(time_ranges)))
    
    for metric in selected_metrics:
        values = []
        hover_texts = []
        
        for time_range in time_ranges:
            if time_range in data:
                value = data[time_range].get(metric)
                values.append(value)
                # Enhanced tooltip with more information
                hover_texts.append(
                    f"<b>{time_range}</b><br>" +
                    f"{metric}: {value:.2f}<br>" +
                    f"Time Period: Last {time_range}"
                )
            else:
                values.append(None)
                hover_texts.append(f"No data available for {time_range}")
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=values,
            name=metric,
            mode='lines',
            line=dict(width=2),
            showlegend=True
        ))
        
        # Add separate point trace for enhanced visibility
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=values,
            name=metric + ' (Points)',
            mode='markers',
            marker=dict(
                size=12,
                symbol='circle',
                line=dict(width=2, color='white'),
                color=fig.data[-1].line.color  # Match line color
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False  # Hide from legend since it's paired with line
        ))
    
    fig.update_layout(
        xaxis=dict(
            ticktext=time_ranges,
            tickvals=x_positions,
            title="Time Range",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title="Value",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        title=title,
        hovermode='closest',  # Show nearest point's data
        plot_bgcolor='white',  # White background
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig

def display_stats_table(stats_data, time_ranges, metrics, height=200):
    """Display a formatted statistics table"""
    if not stats_data:
        return
        
    df = pd.DataFrame(stats_data)
    
    # Format all numeric columns
    numeric_cols = [col for col in df.columns if col != 'Time Range']
    formatter = {col: '{:.1f}' for col in numeric_cols}
    
    return st.dataframe(
        df.style.format(formatter),
        hide_index=True,
        height=height
    )

def display_trade_summary(team, trade_details, other_teams):
    """Display trade summary for a team"""
    incoming = trade_details[team]['incoming']
    outgoing = trade_details[team]['outgoing']
    
    if incoming:
        incoming_details = []
        for player in incoming:
            value = calculate_player_value({
                k: v[v['Player'] == player] 
                for k, v in st.session_state.data_ranges.items()
            })
            # Find which team is giving this player
            from_team = next((t for t in other_teams if player in trade_details[t]['outgoing']), None)
            from_team_name = get_team_name(from_team) if from_team else "Unknown"
            incoming_details.append(f"{player} ({value:.1f}) from {from_team_name}")
        st.write("📥 Receiving:", ", ".join(incoming_details))
    
    if outgoing:
        outgoing_details = []
        for player in outgoing:
            value = calculate_player_value({
                k: v[v['Player'] == player] 
                for k, v in st.session_state.data_ranges.items()
            })
            # Find which team is getting this player
            to_team = next((t for t in other_teams if player in trade_details[t]['incoming']), None)
            to_team_name = get_team_name(to_team) if to_team else "Unknown"
            outgoing_details.append(f"{player} ({value:.1f}) to {to_team_name}")
        st.write("📤 Sending:", ", ".join(outgoing_details))

def display_fairness_score(team_name, score):
    """Display a team's fairness score with appropriate styling"""
    color = 'green' if score > 0.7 else 'orange' if score > 0.5 else 'red'
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='font-size: 16px;'>{team_name}</div>
        <div style='color: {color}; font-size: 20px;'>{score:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

def display_overall_fairness(min_fairness):
    """Display overall trade fairness with appropriate styling"""
    fairness_color = 'green' if min_fairness > 0.7 else 'orange' if min_fairness > 0.5 else 'red'
    fairness_text = 'Fair' if min_fairness > 0.7 else 'Questionable' if min_fairness > 0.5 else 'Unfair'
    
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='font-size: 16px;'>Overall</div>
        <div style='color: {fairness_color}; font-size: 20px;'>{min_fairness:.1%}</div>
        <div style='color: {fairness_color}; font-size: 16px;'>{fairness_text}</div>
    </div>
    """, unsafe_allow_html=True)
