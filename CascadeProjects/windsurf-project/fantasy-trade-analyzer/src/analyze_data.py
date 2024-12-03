import os
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_import import DataImporter

def main():
    # Initialize the data importer
    importer = DataImporter()
    
    # Get the absolute path to your data file
    project_root = os.path.dirname(current_dir)
    data_file = os.path.join(project_root, "data", "Fantrax-Players-Mr Squidward s 69 (60).csv")
    
    # Import the data
    data = importer.import_csv(data_file)
    
    # Print data preview
    print("\n=== Data Preview ===")
    importer.preview_data()
    
    # Print some basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Total Players: {len(data)}")
    print(f"Average Fantasy Points: {data['FPts'].mean():.2f}")
    print(f"Average Fantasy Points per Game: {data['FP/G'].mean():.2f}")
    
    # Position distribution
    print("\n=== Position Distribution ===")
    position_counts = {}
    for pos_list in data['Position'].str.split(','):
        for pos in pos_list:
            position_counts[pos] = position_counts.get(pos, 0) + 1
    
    for pos, count in sorted(position_counts.items()):
        print(f"{pos}: {count} players")
    
    # Top 10 players by Fantasy Points
    print("\n=== Top 10 Players by Fantasy Points ===")
    top_players = data.nlargest(10, 'FPts')[['Player', 'Position', 'Status', 'FPts', 'FP/G']]
    print(top_players.to_string(index=False))

if __name__ == "__main__":
    main()
