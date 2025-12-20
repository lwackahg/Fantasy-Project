"""
Reusable UI components for trade suggestions display.
Provides cleaner, modular rendering functions for trade cards and analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from streamlit_compat import plotly_chart
from streamlit_compat import dataframe


# ============================================================================
# STYLING CONSTANTS
# ============================================================================

COLORS = {
    "positive": "#4CAF50",      # Green
    "negative": "#f44336",      # Red
    "neutral": "#FF9800",       # Orange
    "info": "#2196F3",          # Blue
    "muted": "#9e9e9e",         # Gray
    "background_dark": "#1e1e1e",
    "background_card": "#2d2d2d",
    "border": "#404040",
}

TIER_COLORS = {
    "excellent": "#4CAF50",
    "strong": "#8BC34A", 
    "decent": "#FFC107",
    "marginal": "#FF9800",
    "loss": "#f44336",
}


# ============================================================================
# TRADE CARD HEADER
# ============================================================================

def render_trade_header(
    suggestion: Dict,
    rank: int,
    expanded: bool = True,
) -> bool:
    """Render a compact trade card header with key metrics.
    
    Returns True if the card should be expanded (user clicked).
    """
    pattern = suggestion.get("pattern", "Trade")
    team = suggestion.get("team", "Unknown")
    value_gain = suggestion.get("value_gain", 0)
    opp_gain = suggestion.get("opp_core_gain", 0)
    
    # Determine trade quality tier
    if value_gain > 30:
        tier = "excellent"
        tier_label = "Excellent"
        tier_icon = "🟢"
    elif value_gain > 15:
        tier = "strong"
        tier_label = "Strong"
        tier_icon = "🟢"
    elif value_gain > 5:
        tier = "decent"
        tier_label = "Decent"
        tier_icon = "🟡"
    elif value_gain >= 0:
        tier = "marginal"
        tier_label = "Marginal"
        tier_icon = "🟡"
    else:
        tier = "loss"
        tier_label = "Loss"
        tier_icon = "🔴"
    
    # Win-win indicator
    if opp_gain > 0:
        win_win = "✅ Win-Win"
    elif opp_gain > -5:
        win_win = "⚖️ Balanced"
    else:
        win_win = "⚠️ Hard Sell"
    
    # Build header title
    header_title = f"#{rank} · {pattern} with **{team}** · {tier_icon} {tier_label} ({value_gain:+.1f} FP) · {win_win}"
    
    return st.expander(header_title, expanded=expanded)


def render_trade_summary_metrics(suggestion: Dict, key_prefix: str = ""):
    """Render the key trade metrics in a clean row."""
    value_gain = suggestion.get("value_gain", 0)
    opp_gain = suggestion.get("opp_core_gain", 0)
    
    your_fpts = suggestion.get("your_fpts", [])
    their_fpts = suggestion.get("their_fpts", [])
    your_cv = suggestion.get("your_cv", [])
    their_cv = suggestion.get("their_cv", [])
    
    your_avg_fpts = sum(your_fpts) / max(len(your_fpts), 1) if your_fpts else 0
    their_avg_fpts = sum(their_fpts) / max(len(their_fpts), 1) if their_fpts else 0
    your_avg_cv = sum(your_cv) / max(len(your_cv), 1) if your_cv else 0
    their_avg_cv = sum(their_cv) / max(len(their_cv), 1) if their_cv else 0
    
    fpts_diff = their_avg_fpts - your_avg_fpts
    cv_diff = their_avg_cv - your_avg_cv
    
    cols = st.columns(4)
    
    with cols[0]:
        delta_color = "normal" if value_gain >= 0 else "inverse"
        st.metric(
            "Your Core FP Gain",
            f"{value_gain:+.1f}",
            delta=f"weekly",
            delta_color=delta_color,
            help="Change to your top ~8 players' combined weekly fantasy points"
        )
    
    with cols[1]:
        opp_delta_color = "normal" if opp_gain >= 0 else "inverse"
        st.metric(
            "Opponent Core FP",
            f"{opp_gain:+.1f}",
            delta="win-win" if opp_gain > 0 else "they lose",
            delta_color=opp_delta_color,
            help="Change to opponent's top ~8 players' combined weekly FP"
        )
    
    with cols[2]:
        st.metric(
            "Package FP/G Δ",
            f"{fpts_diff:+.1f}",
            delta="you gain" if fpts_diff > 0 else "you give",
            delta_color="normal" if fpts_diff >= 0 else "inverse",
            help="Difference in average FP/G between packages"
        )
    
    with cols[3]:
        # Negative CV change is good (more consistent)
        cv_delta_color = "normal" if cv_diff <= 0 else "inverse"
        st.metric(
            "Consistency Δ",
            f"{cv_diff:+.1f}%",
            delta="more stable" if cv_diff < 0 else "more volatile",
            delta_color=cv_delta_color,
            help="Change in coefficient of variation (lower = more consistent), with availability/missed-games risk baked in"
        )


# ============================================================================
# PLAYER TABLES
# ============================================================================

def render_player_tables(
    suggestion: Dict,
    value_lookup: Dict[str, float],
    key_prefix: str = "",
):
    """Render the You Give / You Get player tables side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📤 You Give")
        give_df = _build_player_df(
            suggestion["you_give"],
            suggestion["your_fpts"],
            suggestion["your_cv"],
            value_lookup,
        )
        _style_player_table(give_df, is_giving=True)
        your_avg = sum(suggestion["your_fpts"]) / max(len(suggestion["your_fpts"]), 1)
        st.caption(f"Avg: **{your_avg:.1f}** FP/G")
    
    with col2:
        st.markdown("##### 📥 You Get")
        get_df = _build_player_df(
            suggestion["you_get"],
            suggestion["their_fpts"],
            suggestion["their_cv"],
            value_lookup,
        )
        _style_player_table(get_df, is_giving=False)
        their_avg = sum(suggestion["their_fpts"]) / max(len(suggestion["their_fpts"]), 1)
        st.caption(f"Avg: **{their_avg:.1f}** FP/G")


def _build_player_df(
    players: List[str],
    fpts: List[float],
    cv: List[float],
    value_lookup: Dict[str, float],
) -> pd.DataFrame:
    """Build a DataFrame for player display."""
    df = pd.DataFrame({
        "Player": players,
        "FP/G": [f"{f:.1f}" for f in fpts],
        "CV%": [f"{c:.1f}" for c in cv],
    })
    
    # Add value column if available
    values = [value_lookup.get(p) for p in players]
    if any(v is not None for v in values):
        df["$"] = [f"${v:.0f}" if v is not None else "-" for v in values]
    
    return df


def _style_player_table(df: pd.DataFrame, is_giving: bool = True):
    """Display a styled player table."""
    dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "FP/G": st.column_config.TextColumn("FP/G", width="small"),
            "CV%": st.column_config.TextColumn("CV%", width="small"),
            "$": st.column_config.TextColumn("Value", width="small"),
        }
    )


# ============================================================================
# IMPACT CHART
# ============================================================================

def render_impact_chart(suggestion: Dict, key_prefix: str = ""):
    """Render a compact horizontal bar chart showing trade impact."""
    value_gain = suggestion.get("value_gain", 0)
    opp_gain = suggestion.get("opp_core_gain", 0)
    
    fig = go.Figure()
    
    # Your gain bar
    fig.add_trace(go.Bar(
        y=["Your Core"],
        x=[value_gain],
        orientation='h',
        name="Your Change",
        marker_color=COLORS["positive"] if value_gain >= 0 else COLORS["negative"],
        text=f"{value_gain:+.1f}",
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))
    
    # Opponent gain bar
    fig.add_trace(go.Bar(
        y=["Opponent"],
        x=[opp_gain],
        orientation='h',
        name="Opponent Change",
        marker_color=COLORS["info"] if opp_gain >= 0 else COLORS["neutral"],
        text=f"{opp_gain:+.1f}",
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))
    
    fig.update_layout(
        height=120,
        margin=dict(l=0, r=60, t=10, b=10),
        showlegend=False,
        xaxis=dict(
            title="Weekly Core FP Change",
            zeroline=True,
            zerolinecolor="white",
            zerolinewidth=1,
        ),
        yaxis=dict(
            showticklabels=True,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    plotly_chart(fig, width="stretch", key=f"{key_prefix}_impact_chart")


# ============================================================================
# TRADE VERDICT
# ============================================================================

def render_trade_verdict(suggestion: Dict):
    """Render a clear verdict/recommendation for the trade."""
    value_gain = suggestion.get("value_gain", 0)
    opp_gain = suggestion.get("opp_core_gain", 0)
    
    your_cv = suggestion.get("your_cv", [])
    their_cv = suggestion.get("their_cv", [])
    your_avg_cv = sum(your_cv) / max(len(your_cv), 1) if your_cv else 0
    their_avg_cv = sum(their_cv) / max(len(their_cv), 1) if their_cv else 0
    cv_change = their_avg_cv - your_avg_cv
    
    # Build verdict
    if value_gain > 30:
        st.success("🎯 **Excellent Trade** — Major weekly core FP upgrade!")
    elif value_gain > 15:
        st.success("✅ **Strong Trade** — Solid weekly core FP gain")
    elif value_gain > 5:
        st.info("👍 **Decent Trade** — Modest improvement to your core")
    elif value_gain >= 0:
        if cv_change < -5:
            st.info("🛡️ **Consistency Play** — Small FP gain but much more stable roster")
        else:
            st.info("⚖️ **Marginal Trade** — Small weekly core FP gain")
    elif value_gain > -5:
        if cv_change < -5:
            st.info("🔄 **Trade-Off** — Slight FP loss for better consistency")
        else:
            st.warning("⚠️ **Slight Loss** — Small weekly core FP downgrade")
    else:
        st.error("❌ **Core FP Loss** — Only pursue if other factors outweigh the downgrade")
    
    # Opponent acceptance likelihood
    if opp_gain > 5:
        st.caption("✅ Opponent likely to accept — they also gain core FP")
    elif opp_gain > -5:
        st.caption("⚖️ Negotiable — opponent roughly breaks even")
    elif opp_gain > -15:
        st.caption("⚠️ Hard sell — opponent loses some core FP")
    else:
        st.caption("🚫 Very hard sell — opponent loses significant core FP")


# ============================================================================
# WHY THIS TRADE WORKS
# ============================================================================

def render_trade_reasoning(suggestion: Dict):
    """Render concise bullet points explaining why the trade works."""
    value_gain = suggestion.get("value_gain", 0)
    opp_gain = suggestion.get("opp_core_gain", 0)
    pattern = suggestion.get("pattern", "")
    
    your_fpts = suggestion.get("your_fpts", [])
    their_fpts = suggestion.get("their_fpts", [])
    your_cv = suggestion.get("your_cv", [])
    their_cv = suggestion.get("their_cv", [])
    
    your_avg_fpts = sum(your_fpts) / max(len(your_fpts), 1) if your_fpts else 0
    their_avg_fpts = sum(their_fpts) / max(len(their_fpts), 1) if their_fpts else 0
    your_avg_cv = sum(your_cv) / max(len(your_cv), 1) if your_cv else 0
    their_avg_cv = sum(their_cv) / max(len(their_cv), 1) if their_cv else 0
    
    fpts_diff = their_avg_fpts - your_avg_fpts
    cv_change = their_avg_cv - your_avg_cv
    
    reasons = []
    
    # Core value change
    if value_gain > 0:
        reasons.append(f"📈 **Core upgrade**: +{value_gain:.1f} weekly FP to your top 8")
    elif value_gain < -5:
        reasons.append(f"📉 **Core downgrade**: {value_gain:.1f} weekly FP loss")
    
    # Pattern-based reasoning
    if pattern in ("2-for-1", "3-for-1", "4-for-1"):
        reasons.append("📦 **Consolidation**: Trading depth for star power")
    elif pattern in ("1-for-2", "1-for-3", "1-for-4"):
        reasons.append("📊 **Depth play**: Turning one stud into multiple starters")
    
    # Package value
    if fpts_diff > 3:
        reasons.append(f"💰 **Package advantage**: +{fpts_diff:.1f} FP/G in what you receive")
    elif fpts_diff < -3:
        reasons.append(f"⚖️ **Package tax**: Paying {abs(fpts_diff):.1f} FP/G for roster fit")
    
    # Consistency
    if cv_change < -5:
        reasons.append("🛡️ **Risk reduction**: More consistent, reliable roster")
    elif cv_change > 5:
        reasons.append("🎲 **Higher variance**: More upside but shakier floor")
    
    # Win-win
    if opp_gain > 0:
        reasons.append("🤝 **Win-win**: Both teams improve their core")
    
    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.caption("Balanced value-based roster adjustment")


# ============================================================================
# TALKING POINTS
# ============================================================================

def render_talking_points(suggestion: Dict):
    """Render negotiation talking points for the opponent."""
    opp_gain = suggestion.get("opp_core_gain", 0)
    pattern = suggestion.get("pattern", "")
    
    your_fpts = suggestion.get("your_fpts", [])
    their_fpts = suggestion.get("their_fpts", [])
    your_cv = suggestion.get("your_cv", [])
    their_cv = suggestion.get("their_cv", [])
    
    your_avg_fpts = sum(your_fpts) / max(len(your_fpts), 1) if your_fpts else 0
    their_avg_fpts = sum(their_fpts) / max(len(their_fpts), 1) if their_fpts else 0
    your_avg_cv = sum(your_cv) / max(len(your_cv), 1) if your_cv else 0
    their_avg_cv = sum(their_cv) / max(len(their_cv), 1) if their_cv else 0
    
    opp_pkg_advantage = your_avg_fpts - their_avg_fpts
    opp_cv_change = your_avg_cv - their_avg_cv
    
    points = []
    
    if opp_gain > 0:
        points.append(f"✅ Their core **improves by +{opp_gain:.1f} weekly FP**")
    elif opp_gain > -5:
        points.append("⚖️ Frame as a **fit/consolidation move** — minimal core impact")
    
    if opp_pkg_advantage > 2:
        points.append(f"💰 They receive **+{opp_pkg_advantage:.1f} FP/G** in the package")
    
    if opp_cv_change < -3:
        points.append("🛡️ Their roster becomes **more consistent**")
    
    if pattern in ("1-for-2", "1-for-3"):
        points.append("📊 Sell as **depth play** — multiple playable pieces")
    elif pattern in ("2-for-1", "3-for-1"):
        points.append("📦 Sell as **consolidation** — clear core starter + roster spot")
    
    if points:
        st.markdown("**Pitch to opponent:**")
        for point in points:
            st.markdown(f"- {point}")
    else:
        st.caption("Lean on team fit and positional needs")
