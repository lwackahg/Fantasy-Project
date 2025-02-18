import pandas as pd
from typing import List


class StandingsManager:
    def __init__(self, schedule_df: pd.DataFrame):
        self.schedule_df = schedule_df

    def calculate_standings(self, selected_teams: List[str]) -> pd.DataFrame:
        """Calculate standings for selected teams."""
        # Filter schedule for selected teams
        selected_games = self.schedule_df[
            self.schedule_df['Away'].isin(selected_teams) |
            self.schedule_df['Home'].isin(selected_teams)
        ]

        # Calculate standings
        standings = pd.DataFrame(columns=['Team', 'Wins', 'Losses', 'Points', 'Period'])

        for team in selected_teams:
            team_games = selected_games[
                (selected_games['Away'] == team) |
                (selected_games['Home'] == team)
            ]

            wins = 0
            losses = 0
            points = 0
            period = team_games['Period'].iloc[0]

            for _, game in team_games.iterrows():
                if game['Away'] == team:
                    if game['Away_Score'] > game['Home_Score']:
                        wins += 1
                        points += game['Away_Score']
                    else:
                        losses += 1
                else:
                    if game['Home_Score'] > game['Away_Score']:
                        wins += 1
                        points += game['Home_Score']
                    else:
                        losses += 1

            standings = pd.concat([
                standings,
                pd.DataFrame({
                    'Team': [team],
                    'Wins': [wins],
                    'Losses': [losses],
                    'Points': [points],
                    'Period': [period]
                })
            ], ignore_index=True)

        return standings.sort_values(by=['Wins', 'Points'], ascending=False)
