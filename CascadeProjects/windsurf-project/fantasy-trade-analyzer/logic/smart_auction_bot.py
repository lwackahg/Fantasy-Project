import pandas as pd
import numpy as np
from logic.team_optimizer import EnhancedTeamOptimizer
from logic.optimizer_strategies import STRATEGIES

class SmartAuctionBot:
    """Provides intelligent auction recommendations based on team strategy and auction dynamics."""

    def __init__(self, available_players, my_team, budget, roster_composition, team_strategy_name='balanced'):
        self.initial_players = available_players.copy()
        self.available_players = available_players.copy()

        # Ensure PlayerName is the index for consistent lookups
        if 'PlayerName' in self.available_players.columns:
            self.available_players.set_index('PlayerName', inplace=True)
        
        # Ensure 'AdjValue' column exists, falling back to 'BaseValue' if it doesn't.
        if 'AdjValue' not in self.available_players.columns:
            if 'BaseValue' in self.available_players.columns:
                self.available_players['AdjValue'] = self.available_players['BaseValue']
            else:
                # As a last resort, create a default value to prevent crashes.
                self.available_players['AdjValue'] = 1
        if isinstance(my_team, pd.DataFrame):
            self.my_team = my_team
        elif my_team:
            self.my_team = pd.DataFrame(my_team)
        else:
            self.my_team = pd.DataFrame(columns=['PlayerName', 'Position', 'DraftPrice'])
        self.budget = budget
        self.roster_composition = roster_composition
        self.team_strategy = STRATEGIES[team_strategy_name]
        self.optimizer = EnhancedTeamOptimizer()

    def analyze_situation(self):
        """Analyzes the current state of the auction to determine needs and resources."""
        spent = self.my_team['DraftPrice'].sum() if not self.my_team.empty else 0
        remaining_budget = self.budget - spent

        remaining_roster = self.roster_composition.copy()
        if not self.my_team.empty:
            current_counts = self.my_team['Position'].value_counts()
            for pos, count in current_counts.items():
                if pos in remaining_roster:
                    remaining_roster[pos] -= count
        
        needs = {pos: count for pos, count in remaining_roster.items() if count > 0}
        remaining_spots = sum(needs.values())

        return {
            "remaining_budget": remaining_budget,
            "remaining_spots": remaining_spots,
            "needs": needs
        }

    def get_simple_nomination_recommendation(self):
        """Recommends a player to nominate based on the highest value player that fills a need."""
        situation = self.analyze_situation()
        if not situation['needs']:
            return {"player": None, "reason": "Your roster is full."}

        candidate_players = self.available_players[self.available_players['Position'].isin(situation['needs'].keys())].copy()
        
        if candidate_players.empty:
            return {"player": None, "reason": "No available players match your team's needs."}

        # Ensure 'AdjValue' is numeric and handle NaNs
        candidate_players['AdjValue'] = pd.to_numeric(candidate_players['AdjValue'], errors='coerce')
        candidate_players.dropna(subset=['AdjValue'], inplace=True)

        if candidate_players.empty:
            return {"player": None, "reason": "Could not determine value for available players."}

        best_player = candidate_players.loc[candidate_players['AdjValue'].idxmax()]
        
        player_name = best_player.name
        reason = f"Nominate the highest-valued player needed, {player_name}, projected at ${best_player['AdjValue']:.0f}."
        
        return {"player": player_name, "reason": reason}

    def get_nomination_recommendation(self):
        """Recommends a player to nominate based on multiple strategies."""
        situation = self.analyze_situation()
        if not situation['needs']:
            return {"player": None, "reason": "Your roster is full."}

        needed_positions = list(situation['needs'].keys())
        candidate_players = self.available_players[self.available_players['Position'].isin(needed_positions)]
        if candidate_players.empty:
            return {"player": None, "reason": "No available players match your team's needs."}

        # Strategy 1: Target player with highest marginal value for my team
        top_candidates = candidate_players.sort_values('AdjValue', ascending=False).head(10)
        top_candidates['marginal_value'] = top_candidates.index.to_series().apply(self._calculate_marginal_value)
        top_candidates.dropna(subset=['marginal_value'], inplace=True)

        if not top_candidates.empty:
            best_fit_player_series = top_candidates.sort_values('marginal_value', ascending=False).iloc[0]
            best_fit_player = best_fit_player_series.name
            max_marginal_value = best_fit_player_series['marginal_value']
        else:
            best_fit_player = None
            max_marginal_value = -1

        if best_fit_player and max_marginal_value > 5:
            reason = f"This player has the highest marginal value for your team, adding an estimated ${max_marginal_value:.2f} in value. Securing them would be a significant boost."
            return {"player": best_fit_player, "reason": reason}

        # Strategy 2: Pressure opponents by nominating a high-value player you DON'T need
        my_positions = [pos for pos, count in self.roster_composition.items() if pos not in needed_positions]
        if my_positions:
            pressure_candidates = self.available_players[~self.available_players['Position'].isin(needed_positions)]
            pressure_candidates = pressure_candidates[pressure_candidates['Tier'] == 1]
            if not pressure_candidates.empty:
                player_to_nominate = pressure_candidates.sort_values('AdjValue', ascending=False).index[0]
                reason = f"Nominate a top-tier player like {player_to_nominate} that you don't need. This pressures opponents to spend their budget, giving you an advantage later."
                return {"player": player_to_nominate, "reason": reason}

        # Strategy 3: Find a bargain - high VORP to AdjValue ratio
        # Create a copy to avoid SettingWithCopyWarning
        candidate_players_copy = candidate_players.copy()
        candidate_players_copy['ValueRatio'] = candidate_players_copy['VORP'] / candidate_players_copy['AdjValue'].replace(0, 1)
        best_bargain = candidate_players_copy.sort_values('ValueRatio', ascending=False).iloc[0]
        if best_bargain['ValueRatio'] > 1.5: # If VORP is 50% higher than cost
            reason = f"This player is a potential bargain. With a VORP of {best_bargain['VORP']:.2f} and a projected cost of ${best_bargain['AdjValue']:.0f}, they offer excellent value."
            return {"player": best_bargain.name, "reason": reason}

        # Strategy 4: Fallback to highest value player if no other strategy fits
        best_overall_player = candidate_players.sort_values('AdjValue', ascending=False).iloc[0]
        reason = f"As a safe bet, nominate the highest-valued player available that fits your team's needs. This player has a solid projected value of ${best_overall_player['AdjValue']:.0f}."
        return {"player": best_overall_player.name, "reason": reason}

    def _calculate_marginal_value(self, player_name):
        """Calculates the marginal value of a player by running optimizer simulations."""
        situation = self.analyze_situation()
        player_series = self.available_players.loc[player_name]

        # 1. Get baseline score without the player
        baseline_results = self.optimizer.run_optimized_ga(
            self.available_players.drop(player_name),
            situation['remaining_budget'],
            situation['needs'],
            self.team_strategy.name,
            time_limit=1.0, # Quick simulation
            generations=15
        )
        baseline_score = baseline_results[0]['VORP'].sum() if baseline_results else 0

        # 2. Simulate acquiring the player for their base value
        sim_budget = situation['remaining_budget'] - player_series['AdjValue']
        if sim_budget < 0:
            return 0 # Cannot afford

        sim_needs = situation['needs'].copy()
        sim_needs[player_series['Position']] -= 1
        sim_needs = {pos: count for pos, count in sim_needs.items() if count > 0}
        if not sim_needs: # This player fills the last spot
            return player_series['VORP'] # Marginal value is just their VORP

        # 3. Get score of remaining team
        remaining_team_results = self.optimizer.run_optimized_ga(
            self.available_players.drop(player_name),
            sim_budget,
            sim_needs,
            self.team_strategy.name,
            time_limit=1.0,
            generations=15
        )
        remaining_team_score = remaining_team_results[0]['VORP'].sum() if remaining_team_results else 0

        # 4. Calculate marginal value
        total_value_with_player = player_series['VORP'] + remaining_team_score
        marginal_value = total_value_with_player - baseline_score
        return marginal_value

    def get_bidding_advice(self, player_name):
        """Provides bidding advice for a player currently on the block."""
        try:
            if player_name in self.available_players.index:
                player_series = self.available_players.loc[player_name]
            elif self.initial_players is not None and player_name in self.initial_players.index:
                player_series = self.initial_players.loc[player_name]
            else:
                return {"max_bid": 0, "reason": f"Player {player_name} not found in our database."}
        except (KeyError, AttributeError):
            return {"max_bid": 0, "reason": f"Could not retrieve data for player {player_name}."}

        max_bid = int(player_series['AdjValue'] * 1.1)  # Allow bidding up to 10% over projected value

        situation = self.analyze_situation()
        if max_bid > situation['remaining_budget']:
            max_bid = situation['remaining_budget']

        reason = f"Based on our projection of \\${player_series['AdjValue']:.0f}, we recommend a max bid of \\${max_bid:.0f}. "

        return {"max_bid": max_bid, "reason": reason}
