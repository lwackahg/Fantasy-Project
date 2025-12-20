"""
UI for Historical YTD Downloader
"""

import streamlit as st
from pathlib import Path
from streamlit_compat import dataframe
from .logic import (
	download_all_historical_seasons,
	get_available_seasons,
	load_and_compare_seasons,
	DOWNLOAD_DIR
)


def display_historical_ytd_ui():
	"""Display the historical YTD downloader interface."""
	st.subheader("📅 Historical YTD Downloader")
	
	st.markdown("""
	Download year-to-date (YTD) stats from past NBA seasons for year-over-year analysis.
	This allows you to compare current season performance against previous seasons.
	
	**Use Cases:**
	- Track player improvement/decline year-over-year
	- Identify breakout players vs. regression candidates
	- Compare rookie seasons to sophomore seasons
	- Analyze career trajectories
	
	**Note:** Not all players will have data for all seasons (rookies, injuries, etc.)
	""")
	
	# League selection
	import os
	league_ids = os.getenv('FANTRAX_LEAGUE_IDS', '').split(',')
	league_names = os.getenv('FANTRAX_LEAGUE_NAMES', '').split(',')
	
	if not league_ids or not league_ids[0]:
		st.error("No league IDs configured in fantrax.env")
		return
	
	league_options = {name.strip(): id.strip() for name, id in zip(league_names, league_ids) if name.strip() and id.strip()}
	
	if not league_options:
		st.error("No valid league configurations found")
		return
	
	selected_league_name = st.selectbox(
		"Select League",
		options=list(league_options.keys()),
		help="Choose which league to download historical data for"
	)
	
	league_id = league_options[selected_league_name]
	
	# Season selection
	st.markdown("---")
	st.markdown("### Select Seasons to Download")
	
	available_seasons = get_available_seasons()
	
	col1, col2 = st.columns([3, 1])
	
	with col1:
		selected_seasons = st.multiselect(
			"Choose Seasons",
			options=available_seasons,
			default=[available_seasons[0], available_seasons[1]],  # Default to current and last season
			help="Select which seasons to download. Current season is always included."
		)
	
	with col2:
		st.markdown("**Quick Select:**")
		if st.button("Last 3 Seasons", width="stretch"):
			selected_seasons = available_seasons[:3]
			st.rerun()
		if st.button("All Seasons", width="stretch"):
			selected_seasons = available_seasons
			st.rerun()
	
	if not selected_seasons:
		st.warning("Please select at least one season to download")
		return
	
	# Display selected seasons info
	st.info(f"📊 Selected {len(selected_seasons)} season(s): {', '.join(selected_seasons)}")
	
	# Download button
	st.markdown("---")
	
	if st.button("🚀 Download Historical YTD Data", type="primary", width="stretch"):
		progress_bar = st.progress(0)
		status_text = st.empty()
		
		def progress_callback(current, total, season_label):
			progress = current / total
			progress_bar.progress(progress)
			status_text.text(f"Downloading {season_label}... ({current}/{total})")
		
		with st.spinner("Initializing download..."):
			messages = download_all_historical_seasons(
				league_id,
				seasons_to_download=selected_seasons,
				progress_callback=progress_callback
			)
		
		progress_bar.progress(1.0)
		status_text.text("Download complete!")
		
		# Display results
		st.markdown("### 📋 Download Results")
		
		success_count = sum(1 for msg in messages if 'Success' in msg)
		fail_count = len(messages) - success_count
		
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Total Downloads", len(messages))
		with col2:
			st.metric("Successful", success_count)
		with col3:
			st.metric("Failed", fail_count)
		
		# Show detailed messages
		with st.expander("📝 Detailed Results", expanded=True):
			for msg in messages:
				if 'Success' in msg:
					st.success(msg)
				else:
					st.error(msg)
	
	# Show existing files
	st.markdown("---")
	st.markdown("### 📁 Downloaded Files")
	
	if DOWNLOAD_DIR.exists():
		files = sorted(DOWNLOAD_DIR.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
		
		if files:
			st.markdown(f"**Location:** `{DOWNLOAD_DIR}`")
			
			file_data = []
			for file in files:
				size_kb = file.stat().st_size / 1024
				modified = file.stat().st_mtime
				from datetime import datetime
				mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
				
				file_data.append({
					"Filename": file.name,
					"Size (KB)": f"{size_kb:.1f}",
					"Modified": mod_time
				})
			
			dataframe(
				file_data,
				hide_index=True,
				width="stretch",
				column_config={
					"Filename": st.column_config.TextColumn("Filename", width="large"),
					"Size (KB)": st.column_config.TextColumn("Size (KB)", width="small"),
					"Modified": st.column_config.TextColumn("Modified", width="medium")
				}
			)
		else:
			st.info("No historical YTD files downloaded yet")
	else:
		st.info("Download directory not created yet")
	
	# Usage instructions
	with st.expander("ℹ️ How to Use Historical Data", expanded=False):
		st.markdown("""
		### Using Historical YTD Data
		
		**File Naming Convention:**
		- Format: `Fantrax-Players-{LeagueName}-YTD-{Season}.csv`
		- Example: `Fantrax-Players-MyLeague-YTD-2023-24.csv`
		
		**Data Analysis Tips:**
		
		1. **Year-over-Year Comparison:**
		   - Compare current season FP/G to last season FP/G
		   - Calculate improvement percentage: `(Current - Previous) / Previous * 100`
		
		2. **Trend Analysis:**
		   - Look at 3+ seasons to identify consistent improvers
		   - Flag players with declining trends
		
		3. **Breakout Detection:**
		   - Identify players with >20% YoY improvement
		   - Cross-reference with age and opportunity changes
		
		4. **Regression Candidates:**
		   - Players with unsustainable spikes (>30% improvement)
		   - Veterans with declining trends
		
		5. **Rookie Analysis:**
		   - Compare rookie season to sophomore season
		   - Historical "sophomore slump" patterns
		
		**Integration with Trade Analysis:**
		- Use historical data to project future performance
		- Weight recent seasons more heavily
		- Consider age curves and career stage
		
		**Data Limitations:**
		- Players may not have played in all seasons (injuries, rookies)
		- Different team contexts affect production
		- League scoring settings may have changed
		- COVID seasons (2019-20, 2020-21) had unusual circumstances
		
		**Note:** To compare seasons, use the "YoY Comparison" page in the main navigation.
		""")
