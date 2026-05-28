"""
Auction Draft Value Analysis
Analyzes actual draft results against auction value theory from 904_Auction_Draft_Value_Theory.md
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DRAFT_FILE = DATA_DIR / "Fantrax-Draft-Results-Mr Squidward s Gay Layup Line.csv"
YTD_FILE = DATA_DIR / "Fantrax-Players-Mr_Squidward_s_Gay_Layup_Line-(YTD).csv"

def load_data():
    """Load draft results and YTD player stats"""
    draft = pd.read_csv(DRAFT_FILE)
    ytd = pd.read_csv(YTD_FILE)
    
    # Clean column names
    draft.columns = draft.columns.str.strip()
    ytd.columns = ytd.columns.str.strip()
    
    return draft, ytd

def merge_draft_with_stats(draft, ytd):
    """Merge draft results with actual YTD performance"""
    # Merge on Player ID
    merged = draft.merge(
        ytd[['ID', 'Player', 'FP/G', 'GP', 'FPts']],
        left_on='Player ID',
        right_on='ID',
        how='left',
        suffixes=('_draft', '_ytd')
    )
    
    # Clean FP/G - remove commas and convert to float
    merged['FP/G'] = merged['FP/G'].astype(str).str.replace(',', '').astype(float)
    merged['Bid'] = pd.to_numeric(merged['Bid'], errors='coerce')
    merged['GP'] = pd.to_numeric(merged['GP'], errors='coerce')
    
    return merged

def calculate_value_metrics(df):
    """Calculate auction value efficiency metrics"""
    # $/FP/G ratio
    df['$/FP/G'] = df['Bid'] / df['FP/G']
    df['$/FP/G'] = df['$/FP/G'].replace([np.inf, -np.inf], np.nan)
    
    # Value tier based on $/FP/G
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
    
    df['Value_Tier'] = df['$/FP/G'].apply(classify_value)
    
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
    
    # Total value generated (FP/G * GP vs cost)
    df['Total_FP'] = df['FP/G'] * df['GP']
    df['FP_per_Dollar'] = df['Total_FP'] / df['Bid']
    df['FP_per_Dollar'] = df['FP_per_Dollar'].replace([np.inf, -np.inf], np.nan)
    
    return df

def analyze_by_team(df):
    """Analyze draft efficiency by fantasy team"""
    team_stats = df.groupby('Fantasy Team').agg({
        'Bid': 'sum',
        'FP/G': 'mean',
        '$/FP/G': 'mean',
        'Player_ytd': 'count',
        'Total_FP': 'sum'
    }).round(2)
    
    team_stats.columns = ['Total_Spent', 'Avg_FPG', 'Avg_$/FPG', 'Players_Drafted', 'Total_FP_Season']
    team_stats['FP_per_Dollar'] = (team_stats['Total_FP_Season'] / team_stats['Total_Spent']).round(2)
    
    # Sort by draft efficiency (lowest $/FP/G is best)
    team_stats = team_stats.sort_values('Avg_$/FPG')
    
    return team_stats

def find_best_worst_picks(df, n=10):
    """Find the best and worst value picks"""
    valid = df[df['$/FP/G'].notna() & (df['Bid'] > 0)].copy()
    
    best = valid.nsmallest(n, '$/FP/G')[['Player_ytd', 'Fantasy Team', 'Bid', 'FP/G', '$/FP/G', 'Value_Tier']]
    worst = valid.nlargest(n, '$/FP/G')[['Player_ytd', 'Fantasy Team', 'Bid', 'FP/G', '$/FP/G', 'Value_Tier']]
    
    return best, worst

def analyze_price_tiers(df):
    """Analyze value efficiency across different price ranges"""
    bins = [0, 10, 20, 30, 40, 50, 75, 100, 200]
    labels = ['$1-10', '$11-20', '$21-30', '$31-40', '$41-50', '$51-75', '$76-100', '$100+']
    
    df['Price_Tier'] = pd.cut(df['Bid'], bins=bins, labels=labels, include_lowest=True)
    
    tier_analysis = df.groupby('Price_Tier', observed=True).agg({
        'FP/G': ['mean', 'median', 'min', 'max'],
        '$/FP/G': ['mean', 'median'],
        'Player_ytd': 'count'
    }).round(2)
    
    return tier_analysis

def main():
    """Run complete draft value analysis"""
    print("=" * 80)
    print("AUCTION DRAFT VALUE ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    draft, ytd = load_data()
    print(f"   Draft picks: {len(draft)}")
    print(f"   YTD players: {len(ytd)}")
    
    # Merge and calculate
    print("\n🔗 Merging draft with YTD stats...")
    df = merge_draft_with_stats(draft, ytd)
    df = calculate_value_metrics(df)
    
    # Overall stats
    print("\n" + "=" * 80)
    print("LEAGUE-WIDE SUMMARY")
    print("=" * 80)
    total_spent = df['Bid'].sum()
    avg_fpg = df['FP/G'].mean()
    median_price_per_fpg = df['$/FP/G'].median()
    
    print(f"Total $ Spent: ${total_spent:,.0f}")
    print(f"Average FP/G (all drafted): {avg_fpg:.2f}")
    print(f"Median $/FP/G: ${median_price_per_fpg:.2f}")
    
    # Value tier distribution
    print("\n" + "-" * 80)
    print("VALUE TIER DISTRIBUTION")
    print("-" * 80)
    value_dist = df['Value_Tier'].value_counts().sort_index()
    for tier, count in value_dist.items():
        pct = (count / len(df)) * 100
        print(f"{tier:20s}: {count:3d} picks ({pct:5.1f}%)")
    
    # Best/Worst picks
    print("\n" + "=" * 80)
    print("TOP 15 VALUE PICKS (Lowest $/FP/G)")
    print("=" * 80)
    best, worst = find_best_worst_picks(df, n=15)
    print(best.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("TOP 15 OVERPAYS (Highest $/FP/G)")
    print("=" * 80)
    print(worst.to_string(index=False))
    
    # Team analysis
    print("\n" + "=" * 80)
    print("TEAM DRAFT EFFICIENCY RANKINGS")
    print("=" * 80)
    team_stats = analyze_by_team(df)
    print(team_stats.to_string())
    
    # Price tier analysis
    print("\n" + "=" * 80)
    print("VALUE BY PRICE TIER")
    print("=" * 80)
    price_tiers = analyze_price_tiers(df)
    print(price_tiers.to_string())
    
    # Export detailed results
    output_file = DATA_DIR.parent / "analysis" / "draft_value_results.csv"
    output_file.parent.mkdir(exist_ok=True)
    
    export_cols = [
        'Pick', 'Player_ytd', 'Fantasy Team', 'Bid', 'FP/G', 'GP', 
        '$/FP/G', 'Value_Tier', 'FPG_Tier', 'Total_FP', 'FP_per_Dollar'
    ]
    df[export_cols].to_csv(output_file, index=False)
    print(f"\n✅ Detailed results exported to: {output_file}")
    
    return df, team_stats

if __name__ == "__main__":
    df, team_stats = main()
