"""
Similarity Visualization Module

Plotly-based visualization components for player similarity analysis.
Integrates with Streamlit UI patterns from trade_suggestions_ui_tab.py.

Based on 903_Euclidean_Distance_Implementation.md Section 12
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
from scipy import stats

from modules.trade_suggestions.player_similarity import (
    create_player_vector,
    euclidean_distance,
    weighted_euclidean_distance,
    cosine_similarity,
    player_similarity_score,
    roster_centroid,
    DIMENSION_NAMES,
    FPG_COLUMNS,
    _get_fpg_value,
    _get_fpg_column,
)
from modules.trade_suggestions.advanced_stats import (
    compute_pca_projection,
    get_pca_explained_variance,
)


# =============================================================================
# Color Scheme (matching existing UI)
# =============================================================================

COLORS = {
    'primary': '#4CAF50',      # Green - positive/you
    'secondary': '#2196F3',    # Blue - comparison
    'warning': '#FF9800',      # Orange - before/warning
    'danger': '#f44336',       # Red - negative
    'neutral': '#9E9E9E',      # Gray
}


# =============================================================================
# Player Comparison Radar Chart
# =============================================================================

def render_player_comparison_radar(
    player_a: pd.Series,
    player_b: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> go.Figure:
    """
    Radar chart comparing two players across all dimensions.
    
    Uses Plotly for interactive visualization.
    """
    dimensions = ['FP/G', 'Availability', 'Consistency', 'Value']
    
    def normalize_for_display(val, mean, std, invert=False):
        """Convert to 0-100 scale for display."""
        z = (val - mean) / std if std > 0 else 0
        # Convert z-score to 0-100 (z of -2 = 0, z of +2 = 100)
        normalized = 50 + (z * 25)
        if invert:
            normalized = 100 - normalized
        return max(0, min(100, normalized))
    
    values_a = [
        normalize_for_display(
            _get_fpg_value(player_a, 50), 
            league_means.get('Mean FPts', 70), 
            league_stds.get('Mean FPts', 25)
        ),
        normalize_for_display(
            player_a.get('AvailabilityRatio', 0.75) * 100, 
            75, 15
        ),
        normalize_for_display(
            player_a.get('CV%', 30), 
            league_means.get('CV%', 30), 
            league_stds.get('CV%', 10), 
            invert=True
        ),
        normalize_for_display(
            player_a.get('Value', 50), 
            league_means.get('Value', 50), 
            league_stds.get('Value', 20)
        ),
    ]
    
    values_b = [
        normalize_for_display(
            _get_fpg_value(player_b, 50), 
            league_means.get('Mean FPts', 70), 
            league_stds.get('Mean FPts', 25)
        ),
        normalize_for_display(
            player_b.get('AvailabilityRatio', 0.75) * 100, 
            75, 15
        ),
        normalize_for_display(
            player_b.get('CV%', 30), 
            league_means.get('CV%', 30), 
            league_stds.get('CV%', 10), 
            invert=True
        ),
        normalize_for_display(
            player_b.get('Value', 50), 
            league_means.get('Value', 50), 
            league_stds.get('Value', 20)
        ),
    ]
    
    fig = go.Figure()
    
    # Player A trace
    fig.add_trace(go.Scatterpolar(
        r=values_a + [values_a[0]],  # Close the polygon
        theta=dimensions + [dimensions[0]],
        fill='toself',
        name=str(player_a.get('Player', 'Player A')),
        line_color=COLORS['primary'],
        fillcolor='rgba(76, 175, 80, 0.3)',
    ))
    
    # Player B trace
    fig.add_trace(go.Scatterpolar(
        r=values_b + [values_b[0]],
        theta=dimensions + [dimensions[0]],
        fill='toself',
        name=str(player_b.get('Player', 'Player B')),
        line_color=COLORS['secondary'],
        fillcolor='rgba(33, 150, 243, 0.3)',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=True,
        title="Player Profile Comparison",
        height=400,
    )
    
    return fig


def render_multi_player_radar(
    players: List[pd.Series],
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    colors: List[str] = None,
) -> go.Figure:
    """
    Radar chart comparing multiple players.
    """
    dimensions = ['FP/G', 'Availability', 'Consistency', 'Value']
    
    if colors is None:
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['warning'], 
                  COLORS['danger'], COLORS['neutral']]
    
    def normalize_for_display(val, mean, std, invert=False):
        z = (val - mean) / std if std > 0 else 0
        normalized = 50 + (z * 25)
        if invert:
            normalized = 100 - normalized
        return max(0, min(100, normalized))
    
    fig = go.Figure()
    
    for i, player in enumerate(players):
        values = [
            normalize_for_display(
                _get_fpg_value(player, 50), 
                league_means.get('Mean FPts', 70), 
                league_stds.get('Mean FPts', 25)
            ),
            normalize_for_display(
                player.get('AvailabilityRatio', 0.75) * 100, 
                75, 15
            ),
            normalize_for_display(
                player.get('CV%', 30), 
                league_means.get('CV%', 30), 
                league_stds.get('CV%', 10), 
                invert=True
            ),
            normalize_for_display(
                player.get('Value', 50), 
                league_means.get('Value', 50), 
                league_stds.get('Value', 20)
            ),
        ]
        
        color = colors[i % len(colors)]
        rgba_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=dimensions + [dimensions[0]],
            fill='toself',
            name=str(player.get('Player', f'Player {i+1}')),
            line_color=color,
            fillcolor=rgba_color,
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Multi-Player Comparison",
        height=450,
    )
    
    return fig


# =============================================================================
# League Player Map (PCA Scatter)
# =============================================================================

def render_league_player_map(
    all_players_df: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
    highlight_players: List[str] = None,
    color_by: str = 'Position',
) -> go.Figure:
    """
    2D scatter plot of all players using PCA projection.
    
    Allows visual identification of similar players and team compositions.
    """
    # Compute PCA projection
    projected, explained_var, player_names, _ = compute_pca_projection(
        all_players_df, league_means, league_stds, n_components=2
    )
    
    # Build DataFrame for plotting
    fpg_col = _get_fpg_column(all_players_df)
    fpg_values = all_players_df[fpg_col].values if fpg_col in all_players_df.columns else [0] * len(player_names)
    # Ensure FP/G values are non-negative for marker size (Plotly requires size >= 0)
    fpg_values = [max(0, float(v)) if pd.notna(v) else 0 for v in fpg_values]
    
    plot_df = pd.DataFrame({
        'Player': player_names,
        'PC1': projected[:, 0],
        'PC2': projected[:, 1],
        'FP/G': fpg_values,
        'Position': all_players_df['Position'].values if 'Position' in all_players_df.columns else ['?'] * len(player_names),
        'Team': all_players_df['Status'].values if 'Status' in all_players_df.columns else ['?'] * len(player_names),
    })
    
    # Create base scatter
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color=color_by,
        size='FP/G',
        hover_name='Player',
        hover_data=['FP/G', 'Team', 'Position'],
        title='League Player Map (PCA Projection)',
    )
    
    # Highlight specific players
    if highlight_players:
        highlight_df = plot_df[plot_df['Player'].isin(highlight_players)]
        if not highlight_df.empty:
            fig.add_trace(go.Scatter(
                x=highlight_df['PC1'],
                y=highlight_df['PC2'],
                mode='markers+text',
                marker=dict(size=18, color='red', symbol='star', line=dict(width=2, color='white')),
                text=highlight_df['Player'],
                textposition='top center',
                name='Highlighted',
                hoverinfo='text',
            ))
    
    # Add explained variance to axis labels
    var_explained = get_pca_explained_variance(explained_var)
    fig.update_layout(
        xaxis_title=f"PC1 ({var_explained['PC1']*100:.1f}% variance)",
        yaxis_title=f"PC2 ({var_explained['PC2']*100:.1f}% variance)",
        height=600,
    )
    
    return fig


def render_team_on_map(
    all_players_df: pd.DataFrame,
    team_code: str,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> go.Figure:
    """
    Render league map with a specific team highlighted.
    """
    team_players = all_players_df[all_players_df['Status'] == team_code]['Player'].tolist()
    return render_league_player_map(
        all_players_df, league_means, league_stds,
        highlight_players=team_players,
        color_by='Position'
    )


# =============================================================================
# Trade Impact Visualizations
# =============================================================================

def render_trade_vector_change(
    your_team_before: pd.DataFrame,
    your_team_after: pd.DataFrame,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> go.Figure:
    """
    Visualize how a trade moves your team's centroid in vector space.
    """
    centroid_before = roster_centroid(your_team_before, league_means, league_stds)
    centroid_after = roster_centroid(your_team_after, league_means, league_stds)
    
    dimensions = ['FP/G', 'Availability', 'Consistency', 'Value']
    
    fig = go.Figure()
    
    # Before (baseline)
    fig.add_trace(go.Bar(
        name='Before Trade',
        x=dimensions,
        y=centroid_before,
        marker_color=COLORS['warning'],
        text=[f"{v:.2f}" for v in centroid_before],
        textposition='outside',
    ))
    
    # After
    fig.add_trace(go.Bar(
        name='After Trade',
        x=dimensions,
        y=centroid_after,
        marker_color=COLORS['primary'],
        text=[f"{v:.2f}" for v in centroid_after],
        textposition='outside',
    ))
    
    fig.update_layout(
        barmode='group',
        title='Team Profile Change (Z-Score)',
        yaxis_title='Z-Score (higher = better)',
        height=400,
    )
    
    return fig


def render_monte_carlo_distribution(
    before_stats: Dict[str, float],
    after_stats: Dict[str, float],
) -> go.Figure:
    """
    Visualize simulated weekly outcome distributions before/after trade.
    """
    fig = go.Figure()
    
    # Create distribution curves (approximated as normal)
    x_min = min(before_stats['p10'], after_stats['p10']) - 100
    x_max = max(before_stats['p90'], after_stats['p90']) + 100
    x_range = np.linspace(x_min, x_max, 200)
    
    before_curve = stats.norm.pdf(x_range, before_stats['mean'], before_stats['std'])
    after_curve = stats.norm.pdf(x_range, after_stats['mean'], after_stats['std'])
    
    # Normalize for display
    before_curve = before_curve / before_curve.max()
    after_curve = after_curve / after_curve.max()
    
    fig.add_trace(go.Scatter(
        x=x_range, y=before_curve,
        mode='lines', fill='tozeroy',
        name='Before Trade',
        line_color=COLORS['warning'],
        fillcolor='rgba(255, 152, 0, 0.3)',
    ))
    
    fig.add_trace(go.Scatter(
        x=x_range, y=after_curve,
        mode='lines', fill='tozeroy',
        name='After Trade',
        line_color=COLORS['primary'],
        fillcolor='rgba(76, 175, 80, 0.3)',
    ))
    
    # Add vertical lines for means
    fig.add_vline(x=before_stats['mean'], line_dash='dash', line_color=COLORS['warning'],
                  annotation_text=f"Before: {before_stats['mean']:.0f}")
    fig.add_vline(x=after_stats['mean'], line_dash='dash', line_color=COLORS['primary'],
                  annotation_text=f"After: {after_stats['mean']:.0f}")
    
    fig.update_layout(
        title='Projected Weekly FP Distribution',
        xaxis_title='Weekly Fantasy Points',
        yaxis_title='Relative Probability',
        height=400,
        showlegend=True,
    )
    
    return fig


def render_floor_ceiling_comparison(
    before_stats: Dict[str, float],
    after_stats: Dict[str, float],
) -> go.Figure:
    """
    Bar chart comparing floor (p10), median (p50), and ceiling (p90).
    """
    categories = ['Floor (10th)', 'Median (50th)', 'Ceiling (90th)']
    before_values = [before_stats['p10'], before_stats['p50'], before_stats['p90']]
    after_values = [after_stats['p10'], after_stats['p50'], after_stats['p90']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Before',
        x=categories,
        y=before_values,
        marker_color=COLORS['warning'],
        text=[f"{v:.0f}" for v in before_values],
        textposition='outside',
    ))
    
    fig.add_trace(go.Bar(
        name='After',
        x=categories,
        y=after_values,
        marker_color=COLORS['primary'],
        text=[f"{v:.0f}" for v in after_values],
        textposition='outside',
    ))
    
    fig.update_layout(
        barmode='group',
        title='Weekly FP Outcomes',
        yaxis_title='Fantasy Points',
        height=350,
    )
    
    return fig


# =============================================================================
# Team Composition Visualizations
# =============================================================================

def render_team_archetype_pie(
    composition: Dict[str, int],
    title: str = "Team Composition",
) -> go.Figure:
    """
    Pie chart of team archetype composition.
    """
    # Define colors for archetypes
    archetype_colors = {
        'Elite Studs': '#4CAF50',
        'Reliable Starters': '#2196F3',
        'Boom/Bust': '#FF9800',
        'Iron Men': '#9C27B0',
        'Streamers': '#9E9E9E',
        'Role Players': '#607D8B',
    }
    
    labels = list(composition.keys())
    values = list(composition.values())
    colors = [archetype_colors.get(label, '#607D8B') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+value',
        textposition='outside',
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True,
    )
    
    return fig


def render_archetype_comparison(
    your_composition: Dict[str, int],
    league_avg_composition: Dict[str, float],
) -> go.Figure:
    """
    Compare your team's archetype distribution vs league average.
    """
    all_archetypes = sorted(set(your_composition.keys()) | set(league_avg_composition.keys()))
    
    your_values = [your_composition.get(a, 0) for a in all_archetypes]
    league_values = [league_avg_composition.get(a, 0) for a in all_archetypes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Your Team',
        x=all_archetypes,
        y=your_values,
        marker_color=COLORS['primary'],
    ))
    
    fig.add_trace(go.Bar(
        name='League Avg',
        x=all_archetypes,
        y=league_values,
        marker_color=COLORS['neutral'],
    ))
    
    fig.update_layout(
        barmode='group',
        title='Team Composition vs League Average',
        yaxis_title='Player Count',
        height=400,
    )
    
    return fig


# =============================================================================
# Similarity Heatmap
# =============================================================================

def render_similarity_heatmap(
    players: List[pd.Series],
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> go.Figure:
    """
    Heatmap showing pairwise similarity between players.
    """
    n = len(players)
    player_names = [str(p.get('Player', f'Player {i}')) for i, p in enumerate(players)]
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        vec_i = create_player_vector(players[i], league_means, league_stds)
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 100
            else:
                vec_j = create_player_vector(players[j], league_means, league_stds)
                dist = weighted_euclidean_distance(vec_i, vec_j)
                similarity_matrix[i, j] = player_similarity_score(dist)
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=player_names,
        y=player_names,
        colorscale='RdYlGn',
        zmin=0,
        zmax=100,
        text=np.round(similarity_matrix, 1),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}: %{z:.1f}%<extra></extra>',
    ))
    
    fig.update_layout(
        title='Player Similarity Matrix',
        height=500,
        xaxis_title='',
        yaxis_title='',
    )
    
    return fig


# =============================================================================
# Dimension Breakdown
# =============================================================================

def render_dimension_breakdown(
    player: pd.Series,
    league_means: Dict[str, float],
    league_stds: Dict[str, float],
) -> go.Figure:
    """
    Horizontal bar chart showing player's z-score in each dimension.
    """
    vector = create_player_vector(player, league_means, league_stds)
    dimensions = ['FP/G', 'Availability', 'Consistency', 'Value']
    
    colors = [COLORS['primary'] if v >= 0 else COLORS['danger'] for v in vector]
    
    fig = go.Figure(go.Bar(
        x=vector,
        y=dimensions,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.2f}" for v in vector],
        textposition='outside',
    ))
    
    # Add zero line
    fig.add_vline(x=0, line_color='black', line_width=2)
    
    fig.update_layout(
        title=f"{player.get('Player', 'Player')} - Dimension Breakdown",
        xaxis_title='Z-Score (0 = league average)',
        height=300,
        xaxis=dict(range=[-3, 3]),
    )
    
    return fig
