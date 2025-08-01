import pandas as pd

class SmartAuctionBot:
    """
    A strategic auction bot that recommends nominations to maximize budget drain
    from opponents.
    """
    def __init__(self, my_team_name: str, teams: dict, available_players: pd.DataFrame, draft_history: list, budget: int, roster_composition: dict):
        """
        Initializes the bot with the current state of the draft.

        Args:
            my_team_name (str): The name of the user's team.
            teams (dict): A dictionary containing all teams, their budgets, and drafted players.
            available_players (pd.DataFrame): A DataFrame of players still available.
            draft_history (list): A list of dictionaries, where each dict represents a drafted player.
            budget (int): The starting budget for each team.
            roster_composition (dict): A dictionary defining the required number of players for each position.
        """
        self.my_team_name = my_team_name
        self.teams = teams
        self.available_players = available_players
        self.draft_history = draft_history
        self.budget = budget
        self.roster_composition = roster_composition
        self.opponent_teams = {name: data for name, data in teams.items() if name != my_team_name}

    def _calculate_opponent_needs(self) -> dict:
        """
        Calculates the remaining roster needs for each opponent.
        """
        needs = {}
        for team_name, data in self.opponent_teams.items():
            needs[team_name] = self.roster_composition.copy()
            # Create a deep copy for manipulation
            team_needs = {k: v for k, v in self.roster_composition.items()}

            for player in data.get('players', []):
                pos = player.get('Position') # Corrected from 'AssignedPosition'
                if pos in team_needs and team_needs[pos] > 0:
                    team_needs[pos] -= 1
                # Handle flex positions if a primary position is full
                elif 'Flx' in team_needs and team_needs['Flx'] > 0:
                    team_needs['Flx'] -= 1
            needs[team_name] = team_needs
        return needs

    def _calculate_budget_pressure_score(self, player: pd.Series, opponent_needs: dict) -> float:
        """Calculates the financial pressure a player's nomination would cause."""
        opponents_who_need_player = []
        for team_name, needs in opponent_needs.items():
            for pos in player['Position'].split('/'):
                if needs.get(pos, 0) > 0:
                    opponents_who_need_player.append(team_name)
                    break
        
        if not opponents_who_need_player:
            return 0.0

        avg_budget_of_needy_opponents = sum(self.opponent_teams[name]['budget'] for name in opponents_who_need_player) / len(opponents_who_need_player)
        
        if avg_budget_of_needy_opponents == 0:
            return 1.0 # Max pressure if they have no money but need the spot

        pressure_score = player['AdjValue'] / avg_budget_of_needy_opponents
        return min(pressure_score, 1.0) # Cap score at 1.0

    def _calculate_positional_scarcity_score(self, player: pd.Series) -> float:
        """Calculates the scarcity of a player's position among top-tier available players."""
        # Filter for Tier 1 players only
        top_tier_players = self.available_players[self.available_players['Tier'] == 1]
        if top_tier_players.empty:
            return 0.5  # Return a neutral score if no elite players are left

        # Get the positions the player is eligible for
        player_positions = player['Position'].split('/')
        
        final_scarcity_scores = []

        # Calculate scarcity for each of the player's positions
        for position_to_check in player_positions:
            count_at_pos = 0
            # Count how many top-tier players are eligible for this position
            for _, other_player in top_tier_players.iterrows():
                if position_to_check in other_player['Position'].split('/'):
                    count_at_pos += 1
            
            # The score is inversely proportional to the count
            scarcity_score = 1 / (1 + count_at_pos)
            final_scarcity_scores.append(scarcity_score)

        # For multi-position players, use the highest scarcity score
        return max(final_scarcity_scores)

    def _calculate_value_inflation_score(self, player: pd.Series) -> float:
        """Calculates the potential for a player to be overpaid, driving inflation."""
        if player['BaseValue'] == 0:
            return 0.0
        
        inflation_ratio = (player['AdjValue'] - player['BaseValue']) / player['BaseValue']
        # Normalize to a 0-1 scale, assuming max inflation of 100% (ratio=1.0)
        return min(max(inflation_ratio, 0), 1)

    def get_nomination_recommendation(self) -> dict:
        """
        Calculates a NominationScore for each available player and returns the best one.

        The score is based on:
        1. Budget Pressure: How much a player will drain opponents' budgets.
        2. Positional Scarcity: How rare the player's position is among remaining top-tier players.
        3. Value Inflation: How overpriced a player is compared to their base value.

        Returns:
            dict: A dictionary containing the recommended player and the reason.
        """
        if self.available_players.empty:
            return {'player': None, 'reason': "The draft is complete."}

        opponent_needs = self._calculate_opponent_needs()

        # Tunable weights for each strategic component
        weights = {
            'budget_pressure': 0.5,
            'positional_scarcity': 0.3,
            'value_inflation': 0.2
        }

        recommendations = []
        for _, player in self.available_players.iterrows():
            budget_score = self._calculate_budget_pressure_score(player, opponent_needs)
            scarcity_score = self._calculate_positional_scarcity_score(player)
            inflation_score = self._calculate_value_inflation_score(player)

            nomination_score = (
                weights['budget_pressure'] * budget_score +
                weights['positional_scarcity'] * scarcity_score +
                weights['value_inflation'] * inflation_score
            )

            recommendations.append({
                'player': player['PlayerName'],
                'nomination_score': nomination_score,
                'budget_score': budget_score,
                'scarcity_score': scarcity_score,
                'inflation_score': inflation_score
            })
        
        if not recommendations:
            return {'player': None, 'reason': "Could not generate a recommendation."}

        # Select the player with the highest nomination score
        best_nominee = max(recommendations, key=lambda x: x['nomination_score'])

        # Construct the reason string
        reason = f"**Nomination Score: {best_nominee['nomination_score']:.2f}**\n\n"
        reason += f"- **Budget Pressure ({weights['budget_pressure'] * 100:.0f}%):** This player is projected to cost a significant portion of the remaining budget for opponents who need their position (Score: {best_nominee['budget_score']:.2f}).\n"
        reason += f"- **Positional Scarcity ({weights['positional_scarcity'] * 100:.0f}%):** They play a position where top-tier talent is becoming scarce, increasing their strategic value (Score: {best_nominee['scarcity_score']:.2f}).\n"
        reason += f"- **Value Inflation ({weights['value_inflation'] * 100:.0f}%):** Their adjusted value is high compared to their base value, making them a prime candidate to drive up market prices (Score: {best_nominee['inflation_score']:.2f})."

        return {
            'player': best_nominee['player'],
            'reason': reason
        }
