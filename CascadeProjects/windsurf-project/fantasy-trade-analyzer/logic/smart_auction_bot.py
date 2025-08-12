import streamlit as st
import pandas as pd
import math

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

    def _split_positions(self, pos_str: str) -> list:
        """Split a position string on commas or slashes and trim whitespace.

        Examples:
        - "G,Flx" -> ["G", "Flx"]
        - "F/C"   -> ["F", "C"]
        - "G,F,Flx" -> ["G", "F", "Flx"]
        """
        if not isinstance(pos_str, str):
            return []
        tokens = []
        for part in pos_str.replace('/', ',').split(','):
            p = part.strip()
            if p:
                tokens.append(p)
        return tokens

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

    def _calculate_budget_pressure_score(self, player: pd.Series, opponent_needs: dict) -> tuple[float, dict]:
        """Calculates the financial pressure a player's nomination would cause.

        Returns a (score, details) tuple where details include needy_count and avg_budget.
        """
        opponents_who_need_player = []
        player_positions = self._split_positions(player['Position'])
        for team_name, needs in opponent_needs.items():
            for pos in player_positions:
                # A team can still bid via Flex even if the primary slot is full
                if needs.get(pos, 0) > 0 or needs.get('Flx', 0) > 0:
                    opponents_who_need_player.append(team_name)
                    break

        needy_count = len(opponents_who_need_player)
        if needy_count == 0:
            return 0.0, {'needy_count': 0, 'avg_budget': 0}

        avg_budget_of_needy_opponents = sum(self.opponent_teams[name]['budget'] for name in opponents_who_need_player) / needy_count

        if avg_budget_of_needy_opponents == 0:
            return 1.0, {'needy_count': needy_count, 'avg_budget': 0}

        pressure_score = player['AdjValue'] / avg_budget_of_needy_opponents
        return min(pressure_score, 1.0), {'needy_count': needy_count, 'avg_budget': avg_budget_of_needy_opponents}

    def _calculate_positional_scarcity_score(self, player: pd.Series) -> tuple[float, dict]:
        """Calculates the scarcity of a player's position among available players in the SAME tier as the player.

        Returns a (score, details) tuple with the position used and remaining count at that position among the player's tier.
        """
        # Filter for players in the same tier as the candidate
        player_tier = int(player.get('Tier', 1))
        same_tier_players = self.available_players[self.available_players['Tier'] == player_tier]
        if same_tier_players.empty:
            return 0.5, {'position': None, 'same_tier_count': 0, 'tier': player_tier, 'global_same_tier_total': 0}

        # Get the positions the player is eligible for (normalize commas/slashes)
        player_positions = self._split_positions(player['Position'])
        
        final_scarcity_scores = []
        pos_counts = []

        # Calculate scarcity for each of the player's positions
        for position_to_check in player_positions:
            count_at_pos = 0
            # Count how many top-tier players are eligible for this position
            for _, other_player in same_tier_players.iterrows():
                other_positions = self._split_positions(other_player['Position'])
                if position_to_check in other_positions:
                    count_at_pos += 1
            
            # The score is inversely proportional to the count
            scarcity_score = 1 / (1 + count_at_pos)
            final_scarcity_scores.append(scarcity_score)
            pos_counts.append((position_to_check, count_at_pos))

        # For multi-position players, use the highest scarcity score
        best_idx = max(range(len(final_scarcity_scores)), key=lambda i: final_scarcity_scores[i])
        best_pos, best_count = pos_counts[best_idx]
        return final_scarcity_scores[best_idx], {
            'position': best_pos,
            'same_tier_count': best_count,
            'tier': player_tier,
            'global_same_tier_total': int(len(same_tier_players))
        }

    def _calculate_value_inflation_score(self, player: pd.Series) -> tuple[float, dict]:
        """Calculates the potential for a player to be overpaid, driving inflation.

        Returns a (score, details) tuple with raw inflation_ratio.
        """
        if player['BaseValue'] == 0:
            return 0.0, {'inflation_ratio': 0.0}
        
        inflation_ratio = (player['AdjValue'] - player['BaseValue']) / player['BaseValue']

        # Non-linear scoring to heavily reward high inflation
        if inflation_ratio > 0.75:
            return 1.5, {'inflation_ratio': inflation_ratio}  # Strong signal for a money pit
        elif inflation_ratio > 0.5:
            return 1.0, {'inflation_ratio': inflation_ratio}
        else:
            return max(inflation_ratio, 0), {'inflation_ratio': inflation_ratio} # Linear for lower values

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

            budget_score, bp_details = self._calculate_budget_pressure_score(player, opponent_needs)
            scarcity_score, sc_details = self._calculate_positional_scarcity_score(player)
            inflation_score, inf_details = self._calculate_value_inflation_score(player)

            nomination_score = (
                weights['budget_pressure'] * budget_score +
                weights['positional_scarcity'] * scarcity_score +
                weights['value_inflation'] * inflation_score
            )

            adj_value = player['AdjValue']
            market_value = player.get('MarketValue', 0)
            raw_target = max(adj_value, market_value) * 1.10
            target_nom_price = math.ceil(raw_target)

            # Friendlier scarcity sentence that uses the player's own tier
            st_pos_count = sc_details.get('same_tier_count', 0)
            st_global = sc_details.get('global_same_tier_total', 0)
            tier_num = sc_details.get('tier')
            pos_label = sc_details.get('position') or 'this position'
            if st_pos_count == 0 and st_global > 0:
                scarcity_line = f"- Scarcity at {pos_label}: plenty of Tier {tier_num} options remain overall; this spot isn't scarce yet.\n"
            elif st_pos_count <= 2:
                scarcity_line = f"- Scarcity at {pos_label}: only {st_pos_count} Tier {tier_num} option(s) remain, so bids can run up.\n"
            else:
                scarcity_line = f"- Scarcity at {pos_label}: about {st_pos_count} Tier {tier_num} options remain.\n"

            explain = (
                f"- About {bp_details['needy_count']} team(s) still need this position; they have roughly ${bp_details['avg_budget']:.0f} each on average.\n"
                f"{scarcity_line}"
                f"- Managers may overpay here: Adjusted vs Base suggests ~{inf_details['inflation_ratio']:.0%} inflation.\n"
                f"- Target ${target_nom_price:,}: we take the higher of Adjusted (${adj_value:.0f}) or Market (${market_value:.0f}), add ~10%, then round up."
            )

            recommendations.append({
                'player': player.name,
                'nomination_score': nomination_score,
                'reason': f"Budget pressure {budget_score:.2f}, scarcity {scarcity_score:.2f}, inflation {inflation_score:.2f}",
                'adj_value': adj_value,
                'market_value': market_value,
                'target_nom_price': target_nom_price,
                'explain': explain,
                'details': {
                    'budget_pressure': bp_details,
                    'positional_scarcity': sc_details,
                    'value_inflation': inf_details,
                },
                'scores': {
                    'budget_pressure': budget_score,
                    'positional_scarcity': scarcity_score,
                    'value_inflation': inflation_score,
                },
                'weights': weights
            })
        
        if not recommendations:
            return [{'player': None, 'reason': "Could not generate a recommendation."}]

        top_five = sorted(recommendations, key=lambda x: x['nomination_score'], reverse=True)[:5]
        return top_five
