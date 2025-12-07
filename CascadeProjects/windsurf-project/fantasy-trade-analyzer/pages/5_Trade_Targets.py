"""
Trade Target Finder

Practical tools for finding trade targets and gaining advantages:
- Find replacement players (similar production, different owner)
- Buy-low candidates (underperforming their minutes/role)
- Sell-high candidates (overperforming unsustainably)
- Monte Carlo trade impact simulation

Focused on actionable insights, not academic analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

try:
	from league_config import FANTRAX_DEFAULT_LEAGUE_ID
except ImportError:
	FANTRAX_DEFAULT_LEAGUE_ID = ""

from modules.trade_suggestions.player_similarity import (
    calculate_league_stats,
    find_similar_players,
    _get_fpg_value,
    _get_fpg_column,
)
from modules.trade_suggestions.advanced_stats import (
    simulate_weekly_outcome,
    simulate_head_to_head,
)
from modules.trade_analysis.consistency_integration import (
    load_player_consistency,
    build_league_consistency_index,
    enrich_roster_with_consistency,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Trade Targets",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Trade Target Finder")
st.caption("Find actionable trade targets to gain competitive advantages")


@st.cache_data(show_spinner=False)
def _get_league_stats_cached(df: pd.DataFrame):
	"""Cached wrapper around calculate_league_stats to avoid recomputing on every rerun."""
	return calculate_league_stats(df)


# =============================================================================
# Data Loading
# =============================================================================

def get_combined_data() -> pd.DataFrame:
    """Get combined player data, deduplicated."""
    if 'combined_data' in st.session_state and st.session_state.combined_data is not None:
        df = st.session_state.combined_data.reset_index()
        
        if 'Player' in df.columns:
            fpg_col = _get_fpg_column(df)
            if fpg_col in df.columns:
                df = df.sort_values(fpg_col, ascending=False)
                df = df.drop_duplicates(subset=['Player'], keep='first')
        
        return df
    return pd.DataFrame()


def get_league_id() -> str:
    """Get current league ID from session state."""
    return st.session_state.get('league_id') or FANTRAX_DEFAULT_LEAGUE_ID


def enrich_with_consistency(df: pd.DataFrame) -> pd.DataFrame:
	"""Add CV% and consistency data to DataFrame."""
	league_id = get_league_id()
	if not league_id or df.empty:
		return df
	
	with st.spinner("Loading consistency metrics for this league (one-time per league)..."):
		progress = st.progress(0)
		progress.progress(10)
		consistency_index = build_league_consistency_index(league_id)
		progress.progress(70)
		result_df = enrich_roster_with_consistency(df, league_id, consistency_index)
		progress.progress(100)
	
	return result_df


# =============================================================================
# Check Data
# =============================================================================

df = get_combined_data()

if df.empty:
    st.warning("⚠️ No player data available. Please load data from the main page first.")
    st.info("Go to **Trade Suggestions** or **Trade Analysis** and load your league data.")
    st.stop()

# Optional consistency enrichment (can be heavy on first load)
league_id = get_league_id()
include_consistency = st.checkbox(
	"Include consistency metrics (CV%, Boom/Bust, Min/G)",
	value=False,
	help="Enrich players with game-log-based consistency; may take longer on first load.",
)
if include_consistency and league_id:
	df = enrich_with_consistency(df)

league_means, league_stds = _get_league_stats_cached(df)
fpg_col = _get_fpg_column(df)


# =============================================================================
# Tabs
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 Find Trade Targets",
    "📊 Buy Low / Sell High",
    "🎲 Trade Impact Simulator"
])


# =============================================================================
# Tab 1: Find Trade Targets
# =============================================================================

with tab1:
    st.header("Find Trade Targets")
    
    # Info expander
    with st.expander("ℹ️ How This Works", expanded=False):
        st.markdown("""
        **Purpose:** Find players on OTHER teams who produce similar FP/G to your players.
        
        **Why it's useful:**
        - If you want to trade away Player A, find similar players you could target in return
        - Identify "replacement level" players - who else produces like your guy?
        - Find upgrade targets at similar production tiers
        
        **How similarity is calculated:**
        - **Primary:** FP/G (fantasy points per game) - the main currency
        - **Secondary:** Games played ratio (availability)
        - **Tertiary:** CV% (consistency) and composite value score
        
        Players with similar FP/G and availability are considered "similar" even if their 
        underlying stats differ (e.g., a high-scoring guard vs a triple-double forward).
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Your Player")
        
        # Get your team
        my_team = st.session_state.get('my_team_id', '')
        if my_team:
            my_players = df[df['Status'] == my_team]['Player'].tolist()
        else:
            my_players = df['Player'].tolist()
        
        if not my_players:
            st.warning("No players found")
            st.stop()
        
        target_player = st.selectbox("Player to find replacements for", my_players)
        
        st.markdown("---")
        st.subheader("Filters")
        
        n_results = st.slider("Number of results", 3, 15, 8)
        exclude_same_team = st.checkbox("Exclude same team", value=True)
        
        # Position filter
        positions = ['Any'] + sorted(df['Position'].dropna().unique().tolist())
        pos_filter = st.selectbox("Position", positions)
        pos_filter = None if pos_filter == 'Any' else pos_filter
        
        # FP/G range
        target_fpg = _get_fpg_value(df[df['Player'] == target_player].iloc[0], 50)
        fpg_range = st.slider(
            "FP/G range around target",
            0, 30, 15,
            help=f"Target player has {target_fpg:.1f} FP/G. This filters to ±{15} FP/G."
        )
        min_fpg = max(0, target_fpg - fpg_range)
    
    with col2:
        st.subheader(f"Players Similar to {target_player}")
        
        # Get target player info
        target_row = df[df['Player'] == target_player].iloc[0]
        target_team = target_row.get('Status', '')
        
        # Show target player stats
        st.markdown(f"**{target_player}** ({target_team})")
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        with tcol1:
            st.metric("FP/G", f"{_get_fpg_value(target_row, 0):.1f}")
        with tcol2:
            gp = target_row.get('GP', 0)
            st.metric("GP", f"{gp}")
        with tcol3:
            cv = target_row.get('CV%', None)
            st.metric("CV%", f"{cv:.1f}" if cv else "N/A")
        with tcol4:
            mins = target_row.get('Min', target_row.get('MPG', None))
            st.metric("Min/G", f"{mins:.1f}" if mins else "N/A")
        
        st.markdown("---")
        
        run_search = st.button("🔍 Find Similar Players", type="primary", key="find_trade_targets")
        similar = None
        effective_target_fpg = target_fpg
        
        if run_search:
            with st.spinner("Finding similar players..."):
                similar = find_similar_players(
                    target_player, df, league_means, league_stds,
                    n=n_results,
                    exclude_same_team=exclude_same_team,
                    position_filter=pos_filter,
                    min_fpg=min_fpg,
                )
            st.session_state["trade_target_finder_results"] = {
                "target_player": target_player,
                "target_fpg": target_fpg,
                "similar": similar,
            }
        else:
            stored = st.session_state.get("trade_target_finder_results")
            if stored and stored.get("target_player") == target_player:
                similar = stored.get("similar")
                effective_target_fpg = stored.get("target_fpg", target_fpg)
        
        if similar:
            results = []
            for name, score, row in similar:
                fpg = _get_fpg_value(row, 0)
                fpg_diff = fpg - effective_target_fpg
                cv = row.get('CV%', None)
                mins = row.get('Min', row.get('MPG', None))
                gp = row.get('GP', 0)
                
                # Trade angle assessment
                if fpg_diff > 5:
                    angle = "🔼 Upgrade"
                elif fpg_diff < -5:
                    angle = "🔽 Downgrade"
                else:
                    angle = "↔️ Lateral"
                
                results.append({
                    'Player': name,
                    'Team': row.get('Status', ''),
                    'Pos': row.get('Position', ''),
                    'FP/G': f"{fpg:.1f}",
                    'vs Target': f"{fpg_diff:+.1f}",
                    'GP': gp,
                    'CV%': f"{cv:.1f}" if cv else "-",
                    'Min/G': f"{mins:.1f}" if mins else "-",
                    'Similarity': f"{score:.0f}%",
                    'Trade Angle': angle,
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, hide_index=True, use_container_width=True)
            
            st.caption("**Similarity** = how close their production profile is. **Trade Angle** = upgrade/downgrade/lateral move.")
        elif run_search:
            st.info("No similar players found with current filters. Try widening the FP/G range.")
        else:
            st.info("Adjust filters and click 'Find Similar Players' to see suggested targets.")


# Tab 2: Buy Low / Sell High
# =============================================================================

with tab2:
    st.header("Buy Low / Sell High Opportunities")
    
    with st.expander("ℹ️ How This Works", expanded=False):
        st.markdown("""
        **Buy Low Candidates:**
        - Players with high minutes but lower FP/G than expected
        - Indicates they may be underperforming their role/opportunity
        - Good targets to acquire before they bounce back
        
        **Sell High Candidates:**
        - Players with lower minutes but higher FP/G than expected
        - May be overperforming unsustainably (hot streak, easy schedule)
        - Consider selling before regression
        
        **Key Metric: FP/G per Minute**
        - Measures efficiency: how much fantasy production per minute of playing time
        - League average is typically 1.5-2.0 FP per minute
        - Outliers (very high or low) suggest buy/sell opportunities
        """)
    
    # Check if we have minutes data; if not, try to enrich from DB-backed
    # consistency stats (game logs) before giving up.
    has_mins = 'Min' in df.columns or 'MPG' in df.columns
    mins_col = 'Min' if 'Min' in df.columns else 'MPG' if 'MPG' in df.columns else None
    
    if not has_mins and league_id:
        # This will pull mean_minutes out of the DB-backed consistency index
        # when available, and attach a 'Min' column to df.
        df = enrich_with_consistency(df)
        has_mins = 'Min' in df.columns or 'MPG' in df.columns
        mins_col = 'Min' if 'Min' in df.columns else 'MPG' if 'MPG' in df.columns else None
    
    if not has_mins:
        st.warning("⚠️ Minutes data not available. This analysis requires MPG (minutes per game) data.")
        st.info("Load player data that includes minutes to use this feature, or ensure game logs/DB stats are available for your league.")
    else:
        # Calculate efficiency metrics
        analysis_df = df.copy()
        analysis_df['FP/G'] = analysis_df.apply(lambda r: _get_fpg_value(r, 0), axis=1)
        analysis_df['Minutes'] = analysis_df[mins_col].fillna(0)
        analysis_df['FP_per_Min'] = np.where(
            analysis_df['Minutes'] > 0,
            analysis_df['FP/G'] / analysis_df['Minutes'],
            0
        )
        
        # Filter to meaningful players (min 15 MPG, min 40 FP/G)
        analysis_df = analysis_df[
            (analysis_df['Minutes'] >= 15) & 
            (analysis_df['FP/G'] >= 40)
        ].copy()
        
        if analysis_df.empty:
            st.warning("Not enough players meet the minimum thresholds (15+ MPG, 40+ FP/G)")
        else:
            # Calculate league averages
            avg_fp_per_min = analysis_df['FP_per_Min'].mean()
            std_fp_per_min = analysis_df['FP_per_Min'].std()
            
            st.markdown(f"**League Average:** {avg_fp_per_min:.2f} FP per minute (σ = {std_fp_per_min:.2f})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔻 Buy Low Candidates")
                st.caption("High minutes, low efficiency → may bounce back")
                
                # Buy low: high minutes, below average efficiency
                buy_low = analysis_df[
                    (analysis_df['Minutes'] >= 28) &  # High minutes
                    (analysis_df['FP_per_Min'] < avg_fp_per_min - 0.3)  # Below average efficiency
                ].sort_values('FP_per_Min', ascending=True).head(10)
                
                if buy_low.empty:
                    st.info("No clear buy-low candidates found")
                else:
                    buy_results = []
                    for _, row in buy_low.iterrows():
                        expected_fpg = row['Minutes'] * avg_fp_per_min
                        upside = expected_fpg - row['FP/G']
                        buy_results.append({
                            'Player': row['Player'],
                            'Team': row.get('Status', ''),
                            'FP/G': f"{row['FP/G']:.1f}",
                            'Min/G': f"{row['Minutes']:.1f}",
                            'FP/Min': f"{row['FP_per_Min']:.2f}",
                            'Expected FP/G': f"{expected_fpg:.1f}",
                            'Upside': f"+{upside:.1f}",
                        })
                    st.dataframe(pd.DataFrame(buy_results), hide_index=True, use_container_width=True)
                    st.caption("**Upside** = potential FP/G gain if they played at league-average efficiency")
            
            with col2:
                st.subheader("🔺 Sell High Candidates")
                st.caption("Lower minutes, high efficiency → may regress")
                
                # Sell high: moderate minutes, above average efficiency
                sell_high = analysis_df[
                    (analysis_df['Minutes'] <= 32) &  # Not elite minutes
                    (analysis_df['FP_per_Min'] > avg_fp_per_min + 0.4)  # Above average efficiency
                ].sort_values('FP_per_Min', ascending=False).head(10)
                
                if sell_high.empty:
                    st.info("No clear sell-high candidates found")
                else:
                    sell_results = []
                    for _, row in sell_high.iterrows():
                        expected_fpg = row['Minutes'] * avg_fp_per_min
                        risk = row['FP/G'] - expected_fpg
                        sell_results.append({
                            'Player': row['Player'],
                            'Team': row.get('Status', ''),
                            'FP/G': f"{row['FP/G']:.1f}",
                            'Min/G': f"{row['Minutes']:.1f}",
                            'FP/Min': f"{row['FP_per_Min']:.2f}",
                            'Expected FP/G': f"{expected_fpg:.1f}",
                            'Risk': f"-{risk:.1f}",
                        })
                    st.dataframe(pd.DataFrame(sell_results), hide_index=True, use_container_width=True)
                    st.caption("**Risk** = potential FP/G loss if they regress to league-average efficiency")


# =============================================================================
# Tab 3: Trade Impact Simulator
# =============================================================================

with tab3:
    st.header("Trade Impact Simulator")
    
    with st.expander("ℹ️ How Monte Carlo Simulation Works", expanded=False):
        st.markdown("""
        **What it does:**
        - Simulates 1000+ weekly outcomes for your team
        - Uses each player's FP/G as the expected value
        - Uses CV% (coefficient of variation) to model game-to-game variance
        - Calculates floor (10th percentile), ceiling (90th percentile), and expected value
        
        **Why it matters:**
        - A team with high-floor players is more consistent week-to-week
        - A team with boom/bust players has higher variance
        - Helps you understand if a trade makes you more or less consistent
        
        **How to use:**
        1. Select your team to see current projections
        2. Compare against other teams to see win probability
        3. Use this to evaluate if a trade improves your floor or ceiling
        """)
    
    team_options = sorted(df['Status'].dropna().unique().tolist())
    
    if not team_options:
        st.warning("No teams found in data")
    else:
        st.subheader("📈 Team Weekly Projection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            sim_team = st.selectbox("Select team", team_options, key='sim_team')
            n_sims = st.slider("Simulations", 500, 5000, 1000, step=500)
            games_target = st.slider("Games per week", 20, 30, 25)
            
            run_sim = st.button("🎲 Run Simulation", type="primary")
        
        with col2:
            if run_sim:
                team_df = df[df['Status'] == sim_team].copy()
                
                if team_df.empty:
                    st.warning("No players found for this team")
                else:
                    # Ensure CV% exists
                    if 'CV%' not in team_df.columns:
                        team_df['CV%'] = 30.0  # Default CV%
                    
                    with st.spinner(f"Running {n_sims} simulations..."):
                        results = simulate_weekly_outcome(team_df, n_sims, games_target)
                    
                    # Display results
                    st.success("Simulation complete!")
                    
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    with mcol1:
                        st.metric("Expected", f"{results['mean']:.0f} FP")
                    with mcol2:
                        st.metric("Floor (10%)", f"{results['p10']:.0f} FP")
                    with mcol3:
                        st.metric("Median", f"{results['p50']:.0f} FP")
                    with mcol4:
                        st.metric("Ceiling (90%)", f"{results['p90']:.0f} FP")
                    
                    # Interpretation
                    spread = results['p90'] - results['p10']
                    if spread < 150:
                        st.success("✅ **Consistent team** - Low variance, predictable weekly output")
                    elif spread < 250:
                        st.info("ℹ️ **Moderate variance** - Some week-to-week fluctuation")
                    else:
                        st.warning("⚠️ **High variance** - Boom/bust potential, unpredictable weeks")
        
        st.markdown("---")
        st.subheader("⚔️ Head-to-Head Comparison")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            team_a = st.selectbox("Team A", team_options, key='h2h_a')
        with col2:
            remaining = [t for t in team_options if t != team_a]
            team_b = st.selectbox("Team B", remaining if remaining else team_options, key='h2h_b')
        with col3:
            h2h_sims = st.slider("Simulations", 500, 5000, 1000, step=500, key='h2h_sims')
        
        if st.button("⚔️ Simulate Matchup", type="secondary"):
            team_a_df = df[df['Status'] == team_a].copy()
            team_b_df = df[df['Status'] == team_b].copy()
            
            if team_a_df.empty or team_b_df.empty:
                st.warning("One or both teams have no players")
            else:
                # Ensure CV% exists
                if 'CV%' not in team_a_df.columns:
                    team_a_df['CV%'] = 30.0
                if 'CV%' not in team_b_df.columns:
                    team_b_df['CV%'] = 30.0
                
                with st.spinner("Simulating head-to-head matchups..."):
                    h2h = simulate_head_to_head(team_a_df, team_b_df, n_simulations=h2h_sims)
                
                rcol1, rcol2, rcol3 = st.columns(3)
                with rcol1:
                    pct = h2h['team_a_win_pct']
                    color = "🟢" if pct > 55 else "🔴" if pct < 45 else "🟡"
                    st.metric(f"{team_a} Wins", f"{color} {pct:.1f}%")
                with rcol2:
                    st.metric("Ties", f"{h2h['tie_pct']:.1f}%")
                with rcol3:
                    pct = h2h['team_b_win_pct']
                    color = "🟢" if pct > 55 else "🔴" if pct < 45 else "🟡"
                    st.metric(f"{team_b} Wins", f"{color} {pct:.1f}%")
                
                # Interpretation
                diff = h2h['team_a_win_pct'] - h2h['team_b_win_pct']
                if abs(diff) < 10:
                    st.info("📊 **Close matchup** - Either team could win on any given week")
                elif diff > 0:
                    st.success(f"✅ **{team_a} favored** - {diff:.0f}% edge in win probability")
                else:
                    st.success(f"✅ **{team_b} favored** - {-diff:.0f}% edge in win probability")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption("Trade Target Finder | Use these insights to identify actionable trade opportunities")
