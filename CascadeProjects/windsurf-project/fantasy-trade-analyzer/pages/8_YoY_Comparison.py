"""
Year-over-Year Player Comparison
Compare player performance across multiple seasons
"""

import streamlit as st
import os
import pandas as pd
import re
from pathlib import Path
from modules.historical_ytd_downloader.logic import load_and_compare_seasons, DOWNLOAD_DIR

try:
	from league_config import FANTRAX_LEAGUE_IDS as CONFIG_LEAGUE_IDS, FANTRAX_LEAGUE_NAMES as CONFIG_LEAGUE_NAMES
except ImportError:
	CONFIG_LEAGUE_IDS = []
	CONFIG_LEAGUE_NAMES = []

st.set_page_config(page_title="YoY Comparison", page_icon="📊", layout="wide")

st.title("📊 Year-over-Year Player Comparison")

st.markdown("""
Compare player performance across multiple seasons to identify trends, breakouts, and regression candidates.
""")

# League selection
def _parse_env_list(key: str) -> list[str]:
	return [item.strip() for item in os.getenv(key, '').split(',') if item.strip()]

env_league_ids = _parse_env_list('FANTRAX_LEAGUE_IDS')
env_league_names = _parse_env_list('FANTRAX_LEAGUE_NAMES')

if env_league_ids:
	league_ids = env_league_ids
	league_names = env_league_names if env_league_names else [lid for lid in env_league_ids]
elif CONFIG_LEAGUE_IDS:
	league_ids = list(CONFIG_LEAGUE_IDS)
	league_names = list(CONFIG_LEAGUE_NAMES)
else:
	league_ids = []
	league_names = []

if not league_ids:
	st.error("No league IDs configured. Update league_config.py or set FANTRAX_LEAGUE_IDS.")
	st.stop()

# Ensure every ID has a display name (fallback to ID itself)
league_options = {}
for idx, league_id in enumerate(league_ids):
	name = league_names[idx] if idx < len(league_names) else league_id
	name = name.strip()
	league_options[name] = league_id.strip()

league_options = {name: lid for name, lid in league_options.items() if name and lid}

if not league_options:
	st.error("No valid league configurations found")
	st.stop()

selected_league_name = st.selectbox(
	"Select League",
	options=list(league_options.keys()),
	help="Choose which league to analyze"
)

# Get sanitized league name for file lookup
def sanitize_name(name):
	return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip().replace(" ", "_"))

league_name_sanitized = sanitize_name(selected_league_name)

# Find available seasons for this league
available_files = list(DOWNLOAD_DIR.glob(f"Fantrax-Players-{league_name_sanitized}-YTD-*.csv"))

if not available_files:
	st.warning(f"No historical data found for {selected_league_name}. Please download data first in Admin Tools → Historical YTD.")
	st.stop()

# Extract season labels from filenames
available_seasons = []
for file in available_files:
	# Extract season from filename: Fantrax-Players-{League}-YTD-{Season}.csv
	parts = file.stem.split('-YTD-')
	if len(parts) == 2:
		available_seasons.append(parts[1])

available_seasons = sorted(available_seasons, reverse=True)  # Most recent first

if len(available_seasons) < 2:
	st.warning("At least 2 seasons are required for comparison. Please download more historical data in Admin Tools.")
	st.stop()

# Always select all seasons
selected_seasons = available_seasons

st.info(f"📊 Comparing all available seasons: {', '.join(selected_seasons)}")

# Load and compare data
with st.spinner("Loading and comparing data..."):
	comparison_df = load_and_compare_seasons(league_name_sanitized, selected_seasons)

if comparison_df is None or comparison_df.empty:
	st.error("Failed to load comparison data. Please ensure the files exist and contain valid data.")
	st.stop()

# Display comparison
st.markdown("---")
st.markdown(f"### 📊 Year-over-Year Comparison")

# Filter options
col1, col2, col3 = st.columns(3)

with col1:
	min_fpg = st.number_input(
		"Min FP/G (most recent season)",
		min_value=0.0,
		value=30.0,
		step=5.0,
		help="Filter to players with at least this FP/G in the most recent season"
	)

with col2:
	show_only_improvers = st.checkbox(
		"Show only improvers",
		value=False,
		help="Show only players who improved YoY"
	)

with col3:
	min_yoy_change = st.number_input(
		"Min YoY % Change",
		min_value=-100.0,
		value=-100.0,
		step=5.0,
		help="Filter to players with at least this % change"
	)

# Apply filters
filtered_df = comparison_df.copy()

# Filter by min FP/G
most_recent_col = f'FP/G_{selected_seasons[0]}'
if most_recent_col in filtered_df.columns:
	filtered_df = filtered_df[filtered_df[most_recent_col] >= min_fpg]

# Filter by YoY change if applicable
if len(selected_seasons) >= 2:
	pct_col = f'YoY_Pct_{selected_seasons[0]}_vs_{selected_seasons[1]}'
	if pct_col in filtered_df.columns:
		if show_only_improvers:
			filtered_df = filtered_df[filtered_df[pct_col] > 0]
		else:
			filtered_df = filtered_df[filtered_df[pct_col] >= min_yoy_change]

# Display metrics
st.markdown("#### Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
	st.metric("Total Players", len(filtered_df))

with col2:
	if len(selected_seasons) >= 2:
		pct_col = f'YoY_Pct_{selected_seasons[0]}_vs_{selected_seasons[1]}'
		if pct_col in filtered_df.columns:
			improvers = len(filtered_df[filtered_df[pct_col] > 0])
			st.metric("Improvers", improvers)

with col3:
	if len(selected_seasons) >= 2:
		pct_col = f'YoY_Pct_{selected_seasons[0]}_vs_{selected_seasons[1]}'
		if pct_col in filtered_df.columns:
			decliners = len(filtered_df[filtered_df[pct_col] < 0])
			st.metric("Decliners", decliners)

with col4:
	if len(selected_seasons) >= 2:
		pct_col = f'YoY_Pct_{selected_seasons[0]}_vs_{selected_seasons[1]}'
		if pct_col in filtered_df.columns:
			breakouts = len(filtered_df[filtered_df[pct_col] > 20])
			st.metric("Breakouts (>20%)", breakouts)

# Display dataframe
st.markdown("#### Player Comparison")

# Format the dataframe for display
display_df = filtered_df.copy()

# Round numeric columns
for col in display_df.columns:
	if col != 'Player' and display_df[col].dtype in ['float64', 'float32']:
		display_df[col] = display_df[col].round(2)

# Configure column display
column_config = {
	"Player": st.column_config.TextColumn("Player", width="medium"),
}

# Add config for FP/G columns
for season in selected_seasons:
	col_name = f'FP/G_{season}'
	if col_name in display_df.columns:
		column_config[col_name] = st.column_config.NumberColumn(
			f"FP/G {season}",
			format="%.2f"
		)

# Add config for change columns
for i in range(len(selected_seasons) - 1):
	change_col = f'YoY_Change_{selected_seasons[i]}_vs_{selected_seasons[i+1]}'
	pct_col = f'YoY_Pct_{selected_seasons[i]}_vs_{selected_seasons[i+1]}'
	
	if change_col in display_df.columns:
		column_config[change_col] = st.column_config.NumberColumn(
			f"Δ {selected_seasons[i]} vs {selected_seasons[i+1]}",
			format="%.2f"
		)
	
	if pct_col in display_df.columns:
		column_config[pct_col] = st.column_config.NumberColumn(
			f"% {selected_seasons[i]} vs {selected_seasons[i+1]}",
			format="%.1f%%"
		)

st.dataframe(
	display_df,
	hide_index=True,
	use_container_width=True,
	column_config=column_config,
	height=600
)

# Download button
csv = display_df.to_csv(index=False)
st.download_button(
	label="📥 Download Comparison as CSV",
	data=csv,
	file_name=f"YoY_Comparison_{'-'.join(selected_seasons)}.csv",
	mime="text/csv"
)

# Usage tips
with st.expander("💡 How to Use This Tool", expanded=False):
	st.markdown("""
	### Finding Trade Targets
	
	**Breakout Candidates (Buy Targets):**
	- Set "Min YoY % Change" to 20%
	- Set "Show only improvers" to ✓
	- Look for players with consistent improvement across multiple seasons
	
	**Sell-High Candidates:**
	- Set "Min YoY % Change" to 30%
	- Look for players with unsustainable spikes (likely to regress)
	- Cross-reference with age and opportunity changes
	
	**Buy-Low Candidates:**
	- Set "Min YoY % Change" to -20%
	- Look for established players having down years
	- Check if decline is due to injury or temporary situation
	
	**Consistent Performers:**
	- Set "Min YoY % Change" to -10% and max to 10%
	- Find players with stable production year-over-year
	- Great for playoffs when you need reliability
	
	### Understanding the Data
	
	- **FP/G {Season}**: Fantasy points per game for that season
	- **Δ (Delta)**: Absolute change in FP/G between seasons
	- **% (Percent)**: Percentage change between seasons
	- **Improvers**: Players with positive YoY change
	- **Decliners**: Players with negative YoY change
	- **Breakouts**: Players with >20% improvement
	
	### Tips
	
	- Players without data for a season (rookies, injuries) will show as blank
	- COVID seasons (2019-20, 2020-21) had unusual circumstances
	- Consider age curves: young players improve, veterans decline
	- Look for 2-3 year trends, not just single-season changes
	""")
