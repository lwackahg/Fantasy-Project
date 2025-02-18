import pandas as pd
from typing import Tuple, List


class ScheduleManager:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.schedule_df = self.load_schedule_data(file_path)

    def load_schedule_data(self, file_path: str) -> pd.DataFrame:
        """Load and clean schedule data from CSV."""
        # Read the first two rows for period and date range
        with open(file_path, 'r') as f:
            period = f.readline().strip().split(',')[0]
            date_range = f.readline().strip().strip('"')
        
        # Read the actual data starting from row 3
        df = pd.read_csv(file_path, skiprows=2, header=None)
        
        # Extract relevant columns
        df = df.iloc[:, :4]
        df.columns = ['Away', 'Away_Score', 'Home', 'Home_Score']
        
        # Filter for valid team names
        valid_teams = self.get_teams(df)
        df = df[df['Away'].isin(valid_teams) & df['Home'].isin(valid_teams)]
        
        # Add Period and Date Range columns
        df['Period'] = period
        df['Date_Range'] = date_range
        
        # Convert scores to numeric
        df['Away_Score'] = pd.to_numeric(df['Away_Score'].str.replace(',', ''), errors='coerce')
        df['Home_Score'] = pd.to_numeric(df['Home_Score'].str.replace(',', ''), errors='coerce')
        
        return df

    def calculate_standings(self) -> pd.DataFrame:
        """Calculate standings based on schedule results."""
        teams = pd.concat([self.schedule_df['Away'], self.schedule_df['Home']]).unique()
        
        standings = []
        for team in teams:
            team_games = self.schedule_df[(self.schedule_df['Away'] == team) | (self.schedule_df['Home'] == team)]
            
            wins = ((team_games['Away'] == team) & (team_games['Away_Score'] > team_games['Home_Score'])) | \
                   ((team_games['Home'] == team) & (team_games['Home_Score'] > team_games['Away_Score']))
            
            losses = ((team_games['Away'] == team) & (team_games['Away_Score'] < team_games['Home_Score'])) | \
                    ((team_games['Home'] == team) & (team_games['Home_Score'] < team_games['Away_Score']))
            
            points_for = team_games[team_games['Away'] == team]['Away_Score'].sum() + \
                        team_games[team_games['Home'] == team]['Home_Score'].sum()
            
            points_against = team_games[team_games['Away'] == team]['Home_Score'].sum() + \
                            team_games[team_games['Home'] == team]['Away_Score'].sum()
            
            standings.append({
                'Team': team,
                'Wins': wins.sum(),
                'Losses': losses.sum(),
                'Points_For': points_for,
                'Points_Against': points_against,
                'Total_Points': wins.sum() * 2  # 2 points per win
            })
        
        return pd.DataFrame(standings).sort_values('Total_Points', ascending=False)

    def swap_schedules(self, team1: str, team2: str) -> pd.DataFrame:
        """Swap schedules between two teams and return updated standings."""
        if team1 == team2:
            return self.calculate_standings()
        
        new_schedule = self.schedule_df.copy()
        
        # Find head-to-head matchups between team1 and team2
        head_to_head = (
            ((new_schedule['Away'] == team1) & (new_schedule['Home'] == team2)) |
            ((new_schedule['Away'] == team2) & (new_schedule['Home'] == team1))
        )
        
        # Non head-to-head matches
        non_h2h = ~head_to_head
        
        # Swap team names and scores
        new_schedule.loc[non_h2h & (new_schedule['Away'] == team1), 'Away'] = team2
        new_schedule.loc[non_h2h & (new_schedule['Home'] == team1), 'Home'] = team2
        new_schedule.loc[non_h2h & (new_schedule['Away'] == team2), 'Away'] = team1
        new_schedule.loc[non_h2h & (new_schedule['Home'] == team2), 'Home'] = team1

        # Calculate updated standings with the new matchups
        updated_standings = self.calculate_standings()
        
        return updated_standings

    def validate_schedule_swap(self, team1: str, team2: str) -> Tuple[bool, str]:
        """Validate if schedule swap is possible."""
        if team1 not in self.schedule_df['Away'].values and team1 not in self.schedule_df['Home'].values:
            return False, f"Team {team1} not found in schedule"
        if team2 not in self.schedule_df['Away'].values and team2 not in self.schedule_df['Home'].values:
            return False, f"Team {team2} not found in schedule"
        return True, ""

    def get_team_schedule(self, team: str) -> pd.DataFrame:
        """Get schedule for a specific team."""
        return self.schedule_df[(self.schedule_df['Away'] == team) | (self.schedule_df['Home'] == team)]

    def get_teams(self, df: pd.DataFrame = None) -> List[str]:
        """Get list of unique team names."""
        if df is None:
            df = self.schedule_df
        
        # Convert all values to strings and filter out metadata rows
        valid_teams = pd.concat([df['Away'].astype(str), df['Home'].astype(str)])
        valid_teams = valid_teams[~valid_teams.str.contains(',,,', na=False)]
        valid_teams = valid_teams[~valid_teams.str.contains('Scoring Period', na=False)]
        valid_teams = valid_teams[~valid_teams.str.contains('FPtsAdjTotal', na=False)]
        valid_teams = valid_teams[valid_teams != '']
        
        return valid_teams.unique().tolist()
