"""
Enhanced Auction Draft Value Analysis - 2025-26 Season
Accounts for games played, availability, and total season value vs per-game efficiency
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DRAFT_FILE = DATA_DIR / "Fantrax-Draft-Results-Mr Squidward s Gay Layup Line.csv"
YTD_FILE = DATA_DIR / "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(YTD).csv"

# League constants (adjust based on your league)
EXPECTED_GAMES_PER_WEEK = 3.5
SEASON_WEEKS = 20  # Approximate fantasy season length
MAX_POSSIBLE_GAMES = SEASON_WEEKS * 4  # ~80 games in a season

def load_data():
    """Load draft results and YTD player stats"""
    draft = pd.read_csv(DRAFT_FILE)
    ytd = pd.read_csv(YTD_FILE)
    
    draft.columns = draft.columns.str.strip()
    ytd.columns = ytd.columns.str.strip()
    
    return draft, ytd

def merge_draft_with_stats(draft, ytd):
    """Merge draft results with actual YTD performance"""
    merged = draft.merge(
        ytd[['ID', 'Player', 'FP/G', 'GP', 'FPts']],
        left_on='Player ID',
        right_on='ID',
        how='left',
        suffixes=('_draft', '_ytd')
    )
    
    merged['FP/G'] = merged['FP/G'].astype(str).str.replace(',', '').astype(float)
    merged['FPts'] = merged['FPts'].astype(str).str.replace(',', '').astype(float)
    merged['Bid'] = pd.to_numeric(merged['Bid'], errors='coerce')
    merged['GP'] = pd.to_numeric(merged['GP'], errors='coerce')
    
    return merged

def calculate_availability_metrics(df):
    """Calculate games played, games missed, and availability rate"""
    df['Games_Missed'] = MAX_POSSIBLE_GAMES - df['GP']
    df['Availability_Rate'] = (df['GP'] / MAX_POSSIBLE_GAMES) * 100
    
    # Classify availability
    def classify_availability(rate):
        if pd.isna(rate):
            return 'Unknown'
        elif rate >= 90:
            return 'Elite (90%+)'
        elif rate >= 75:
            return 'Good (75-90%)'
        elif rate >= 60:
            return 'Moderate (60-75%)'
        elif rate >= 45:
            return 'Poor (45-60%)'
        else:
            return 'Injury Prone (<45%)'
    
    df['Availability_Tier'] = df['Availability_Rate'].apply(classify_availability)
    
    return df

def calculate_value_metrics(df):
    """Calculate comprehensive value metrics"""
    # Per-game efficiency
    df['$/FP/G'] = df['Bid'] / df['FP/G']
    df['$/FP/G'] = df['$/FP/G'].replace([np.inf, -np.inf], np.nan)
    
    # Total season value
    df['Total_FP'] = df['FP/G'] * df['GP']
    df['$/Total_FP'] = df['Bid'] / df['Total_FP']
    df['$/Total_FP'] = df['$/Total_FP'].replace([np.inf, -np.inf], np.nan)
    
    # FP per dollar (inverse - higher is better)
    df['FP/G_per_Dollar'] = df['FP/G'] / df['Bid']
    df['Total_FP_per_Dollar'] = df['Total_FP'] / df['Bid']
    df['FP/G_per_Dollar'] = df['FP/G_per_Dollar'].replace([np.inf, -np.inf], np.nan)
    df['Total_FP_per_Dollar'] = df['Total_FP_per_Dollar'].replace([np.inf, -np.inf], np.nan)
    
    # Availability-adjusted value (penalize missed games)
    df['Availability_Adjusted_Value'] = df['Total_FP_per_Dollar'] * (df['Availability_Rate'] / 100)
    
    # Value tiers
    def classify_value(ratio):
        if pd.isna(ratio):
            return 'Unknown'
        elif ratio >= 1.50:
            return 'Severe Overpay'
        elif ratio >= 1.00:
            return 'Premium Price'
        elif ratio >= 0.70:
            return 'Fair Market'
        elif ratio >= 0.50:
            return 'Good Value'
        elif ratio >= 0.30:
            return 'Strong Value'
        else:
            return 'Steal'
    
    df['Value_Tier_PerGame'] = df['$/FP/G'].apply(classify_value)
    df['Value_Tier_Total'] = df['$/Total_FP'].apply(classify_value)
    
    # FP/G tier
    def classify_fpg_tier(fpg):
        if pd.isna(fpg):
            return 'Unknown'
        elif fpg >= 100:
            return 'S (100+)'
        elif fpg >= 80:
            return 'A (80-100)'
        elif fpg >= 70:
            return 'B (70-80)'
        elif fpg >= 60:
            return 'C (60-70)'
        elif fpg >= 50:
            return 'D (50-60)'
        elif fpg >= 40:
            return 'E (40-50)'
        else:
            return 'F (<40)'
    
    df['FPG_Tier'] = df['FP/G'].apply(classify_fpg_tier)
    
    return df

def identify_injury_busts(df):
    """Find players who were expensive but missed significant time"""
    injury_busts = df[
        (df['Bid'] >= 30) & 
        (df['Availability_Rate'] < 70)
    ].copy()
    
    injury_busts['Value_Lost'] = injury_busts['Bid'] * (1 - injury_busts['Availability_Rate'] / 100)
    
    return injury_busts.sort_values('Value_Lost', ascending=False)

def identify_ironman_values(df):
    """Find players with elite availability who delivered value"""
    ironmen = df[
        (df['Availability_Rate'] >= 90) &
        (df['$/FP/G'] < 0.50)
    ].copy()
    
    return ironmen.sort_values('Total_FP_per_Dollar', ascending=False)

def analyze_by_team(df):
    """Analyze draft efficiency by fantasy team with availability context"""
    team_stats = df.groupby('Fantasy Team').agg({
        'Bid': 'sum',
        'FP/G': 'mean',
        '$/FP/G': 'mean',
        'GP': 'mean',
        'Availability_Rate': 'mean',
        'Total_FP': 'sum',
        'Total_FP_per_Dollar': 'mean',
        'Availability_Adjusted_Value': 'mean',
        'Player_ytd': 'count'
    }).round(2)
    
    team_stats.columns = [
        'Total_Spent', 'Avg_FPG', 'Avg_$/FPG', 'Avg_GP', 
        'Avg_Availability%', 'Total_FP_Season', 'Avg_FP_per_$', 
        'Availability_Adj_Value', 'Players_Drafted'
    ]
    
    # Sort by availability-adjusted value (best overall metric)
    team_stats = team_stats.sort_values('Availability_Adj_Value', ascending=False)
    
    return team_stats

def main():
    """Run enhanced draft value analysis"""
    print("=" * 100)
    print("ENHANCED AUCTION DRAFT VALUE ANALYSIS - 2025-26 SEASON")
    print("(Accounting for Games Played, Availability, and Total Season Value)")
    print("=" * 100)
    
    # Load data
    print("\n📊 Loading data...")
    draft, ytd = load_data()
    
    # Merge and calculate
    print("🔗 Merging draft with YTD stats...")
    df = merge_draft_with_stats(draft, ytd)
    df = calculate_availability_metrics(df)
    df = calculate_value_metrics(df)
    
    # Overall stats
    print("\n" + "=" * 100)
    print("LEAGUE-WIDE SUMMARY")
    print("=" * 100)
    print(f"Total $ Spent: ${df['Bid'].sum():,.0f}")
    print(f"Average FP/G (all drafted): {df['FP/G'].mean():.2f}")
    print(f"Average GP: {df['GP'].mean():.1f} / {MAX_POSSIBLE_GAMES}")
    print(f"Average Availability: {df['Availability_Rate'].mean():.1f}%")
    print(f"Median $/FP/G (per-game value): ${df['$/FP/G'].median():.3f}")
    print(f"Median $/Total_FP (season value): ${df['$/Total_FP'].median():.5f}")
    
    # Availability distribution
    print("\n" + "-" * 100)
    print("AVAILABILITY TIER DISTRIBUTION")
    print("-" * 100)
    avail_dist = df['Availability_Tier'].value_counts().sort_index()
    for tier, count in avail_dist.items():
        pct = (count / len(df)) * 100
        print(f"{tier:25s}: {count:3d} picks ({pct:5.1f}%)")
    
    # Injury busts
    print("\n" + "=" * 100)
    print("TOP 10 INJURY BUSTS (Expensive + Missed Games)")
    print("=" * 100)
    injury_busts = identify_injury_busts(df)
    if len(injury_busts) > 0:
        bust_cols = ['Player_ytd', 'Fantasy Team', 'Bid', 'FP/G', 'GP', 'Availability_Rate', 'Value_Lost']
        print(injury_busts[bust_cols].head(10).to_string(index=False))
    else:
        print("No significant injury busts found.")
    
    # Ironman values
    print("\n" + "=" * 100)
    print("TOP 15 IRONMAN VALUES (90%+ Availability + Great Value)")
    print("=" * 100)
    ironmen = identify_ironman_values(df)
    if len(ironmen) > 0:
        iron_cols = ['Player_ytd', 'Fantasy Team', 'Bid', 'FP/G', 'GP', 'Availability_Rate', '$/FP/G']
        print(ironmen[iron_cols].head(15).to_string(index=False))
    else:
        print("No ironman values found.")
    
    # Best total season value (accounting for GP)
    print("\n" + "=" * 100)
    print("TOP 15 TOTAL SEASON VALUE (FP/G × GP / Price)")
    print("=" * 100)
    best_total = df[df['Total_FP_per_Dollar'].notna()].nlargest(15, 'Total_FP_per_Dollar')
    total_cols = ['Player_ytd', 'Fantasy Team', 'Bid', 'FP/G', 'GP', 'Total_FP', 'Total_FP_per_Dollar']
    print(best_total[total_cols].to_string(index=False))
    
    # Worst total season value
    print("\n" + "=" * 100)
    print("WORST 15 TOTAL SEASON VALUE (Overpaid + Underperformed/Injured)")
    print("=" * 100)
    worst_total = df[(df['Total_FP_per_Dollar'].notna()) & (df['Bid'] > 10)].nsmallest(15, 'Total_FP_per_Dollar')
    print(worst_total[total_cols].to_string(index=False))
    
    # Team analysis
    print("\n" + "=" * 100)
    print("TEAM DRAFT EFFICIENCY RANKINGS (Availability-Adjusted)")
    print("=" * 100)
    team_stats = analyze_by_team(df)
    print(team_stats.to_string())
    
    # Per-game vs total value comparison
    print("\n" + "=" * 100)
    print("PER-GAME VALUE vs TOTAL SEASON VALUE (Top 20 by Bid)")
    print("=" * 100)
    top_bids = df.nlargest(20, 'Bid')
    comparison_cols = ['Player_ytd', 'Bid', 'FP/G', 'GP', '$/FP/G', '$/Total_FP', 'Availability_Rate']
    print(top_bids[comparison_cols].to_string(index=False))
    
    # Export
    output_file = Path(__file__).parent / "enhanced_draft_results.csv"
    export_cols = [
        'Pick', 'Player_ytd', 'Fantasy Team', 'Bid', 'FP/G', 'GP', 'Games_Missed',
        'Availability_Rate', 'Availability_Tier', 'Total_FP', '$/FP/G', '$/Total_FP',
        'FP/G_per_Dollar', 'Total_FP_per_Dollar', 'Availability_Adjusted_Value',
        'Value_Tier_PerGame', 'Value_Tier_Total', 'FPG_Tier'
    ]
    df[export_cols].to_csv(output_file, index=False)
    print(f"\n✅ Enhanced results exported to: {output_file}")
    
    return df, team_stats

if __name__ == "__main__":
    df, team_stats = main()
