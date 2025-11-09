"""
Test script for Excel export functionality
Run this to verify the Excel export works with cached data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from modules.standings_adjuster.excel_export import (
	generate_comprehensive_excel,
	generate_period_excel,
	load_all_cached_data
)

def test_excel_export():
	"""Test Excel export with available cached data"""
	
	print("=" * 60)
	print("Testing Excel Export Functionality")
	print("=" * 60)
	
	# Test league ID from sample data
	test_league_id = "6zeydg0cm03y4myx"
	test_league_name = "Test League"
	
	print(f"\n1. Loading cached data for league: {test_league_id}")
	cached_data = load_all_cached_data(test_league_id)
	
	if not cached_data:
		print("   ❌ No cached data found!")
		print("   Please run the Weekly Standings Analyzer first to cache data.")
		return False
	
	print(f"   ✅ Found {len(cached_data)} cached period(s): {sorted(cached_data.keys())}")
	
	# Test comprehensive export
	print(f"\n2. Testing comprehensive export (all periods)...")
	try:
		filepath = generate_comprehensive_excel(test_league_id, test_league_name)
		print(f"   ✅ Success! File created: {filepath}")
		print(f"   📁 File size: {filepath.stat().st_size:,} bytes")
	except Exception as e:
		print(f"   ❌ Error: {e}")
		return False
	
	# Test single period export
	if cached_data:
		first_period = sorted(cached_data.keys())[0]
		print(f"\n3. Testing single period export (Period {first_period})...")
		try:
			filepath = generate_period_excel(test_league_id, first_period, test_league_name)
			print(f"   ✅ Success! File created: {filepath}")
			print(f"   📁 File size: {filepath.stat().st_size:,} bytes")
		except Exception as e:
			print(f"   ❌ Error: {e}")
			return False
	
	print("\n" + "=" * 60)
	print("✅ All tests passed!")
	print("=" * 60)
	print("\nGenerated files are in: data/standings_exports/")
	
	return True

if __name__ == "__main__":
	try:
		success = test_excel_export()
		sys.exit(0 if success else 1)
	except Exception as e:
		print(f"\n❌ Test failed with error: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
