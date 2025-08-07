import streamlit as st
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

        # Non-linear scoring to heavily reward high inflation
        if inflation_ratio > 0.75:
            return 1.5  # Strong signal for a money pit
        elif inflation_ratio > 0.5:
            return 1.0
        else:
            return max(inflation_ratio, 0) # Linear for lower values

    def get_nomination_recommendation(self, weights=None):
        """Generate the best player nomination recommendation."""
        if self.available_players.empty:
            return [{'player': None, 'reason': "The draft is complete."}]

        opponent_needs = self._calculate_opponent_needs()
        my_team_needs = self._calculate_opponent_needs().get(self.my_team_name, {})

        weights = weights or {'budget_pressure': 0.5, 'positional_scarcity': 0.3, 'value_inflation': 0.2}

        recommendations = []
        for _, player in self.available_players.iterrows():
            if player['AdjValue'] <= 0:
                continue

            # Exclude players that are a good value for our team
            if player['AdjValue'] > player['MarketValue'] * 1.15:
                continue

            budget_score = self._calculate_budget_pressure_score(player, opponent_needs)
            scarcity_score = self._calculate_positional_scarcity_score(player)
            inflation_score = self._calculate_value_inflation_score(player)

            nomination_score = (
                weights['budget_pressure'] * budget_score +
                weights['positional_scarcity'] * scarcity_score +
                weights['value_inflation'] * inflation_score
            )

            adj_value = player['AdjValue']
            target_nom_price = max(adj_value, player['MarketValue']) * 1.10

            recommendations.append({
                'player': player.name,
                'nomination_score': nomination_score,
                'reason': f"Budget pressure {budget_score:.2f}, scarcity {scarcity_score:.2f}, inflation {inflation_score:.2f}",
                'adj_value': adj_value,
                'target_nom_price': target_nom_price
            })
        
        if not recommendations:
            return [{'player': None, 'reason': "Could not generate a recommendation."}]

        top_five = sorted(recommendations, key=lambda x: x['nomination_score'], reverse=True)[:5]
        return top_five
