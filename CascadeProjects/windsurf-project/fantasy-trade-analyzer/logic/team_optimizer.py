import random
import time
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from logic.optimizer_strategies import STRATEGIES

class EnhancedTeamOptimizer:
    """Enhanced optimizer that works both pre-auction and during the auction."""
    def __init__(self):
        self.strategies = STRATEGIES

    def run_optimized_ga(self, available_players, budget, roster_composition, strategy='balanced', time_limit=5.0, generations=50):
        """Time-bounded genetic algorithm for real-time recommendations."""
        start_time = time.time()

        # Ensure required columns exist
        if 'AdjValue' not in available_players.columns:
            if 'BaseValue' in available_players.columns:
                available_players['AdjValue'] = available_players['BaseValue']
            else:
                return [] # Not enough data to run
        if 'Tier' not in available_players.columns or 'VORP' not in available_players.columns:
            return [] # Missing critical data

        # Setup GA
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        player_indices = list(available_players.index)
        num_roster_spots = sum(roster_composition.values())

        def biased_init():
            sorted_players = available_players.sort_values('AdjValue', ascending=False)
            weights = np.linspace(2.0, 0.5, len(sorted_players))
            return np.random.choice(sorted_players.index, size=num_roster_spots, replace=False, p=weights/weights.sum()).tolist()

        toolbox.register("individual", tools.initIterate, creator.Individual, biased_init)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        current_strategy = self.strategies[strategy]

        def evaluate(individual):
            team_df = available_players.loc[individual]
            total_cost = team_df['AdjValue'].sum()

            if total_cost > budget:
                return -1000,

            position_counts = team_df['Position'].value_counts()
            for pos, required in roster_composition.items():
                if position_counts.get(pos, 0) != required:
                    return -1000,

            fitness = team_df['VORP'].sum() * 10
            if all(cat in team_df.columns for cat in current_strategy.category_weights):
                for cat, weight in current_strategy.category_weights.items():
                    fitness += team_df[cat].sum() * weight
            
            tier_counts = team_df['Tier'].value_counts()
            for tier, target_pct in current_strategy.tier_distribution.items():
                actual_pct = tier_counts.get(tier, 0) / len(team_df)
                fitness += (1 - abs(actual_pct - target_pct)) * 50

            return fitness,

        def position_aware_crossover(ind1, ind2):
            team1_df = available_players.loc[ind1]
            team2_df = available_players.loc[ind2]
            child1, child2 = [], []
            for pos in roster_composition.keys():
                pos1_players = team1_df[team1_df['Position'] == pos].index.tolist()
                pos2_players = team2_df[team2_df['Position'] == pos].index.tolist()
                if random.random() < 0.5:
                    child1.extend(pos2_players)
                    child2.extend(pos1_players)
                else:
                    child1.extend(pos1_players)
                    child2.extend(pos2_players)
            return creator.Individual(child1), creator.Individual(child2)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", position_aware_crossover)
        toolbox.register("mutate", self.smart_mutate, available_players=available_players, roster_composition=roster_composition)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(3)

        gen = 0
        while (time.time() - start_time) < time_limit and gen < generations:
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, k=len(pop))
            hof.update(pop)
            gen += 1

        results = [available_players.loc[ind].sort_values('AdjValue', ascending=False) for ind in hof]
        return results

    def smart_mutate(self, individual, available_players, roster_composition):
        if random.random() < 0.3:
            team_df = available_players.loc[individual]
            position = random.choice(list(roster_composition.keys()))
            pos_players = team_df[team_df['Position'] == position]
            if not pos_players.empty:
                player_to_remove = random.choice(pos_players.index.tolist())
                all_pos_players = available_players[available_players['Position'] == position]
                available_replacements = all_pos_players[~all_pos_players.index.isin(individual)]
                if not available_replacements.empty:
                    current_tier = available_players.loc[player_to_remove, 'Tier']
                    tier_weights = 1 / (abs(available_replacements['Tier'] - current_tier) + 1)
                    replacement = np.random.choice(available_replacements.index, p=tier_weights/tier_weights.sum())
                    idx = individual.index(player_to_remove)
                    individual[idx] = replacement
        return individual,

def run_team_optimizer(available_players, budget, roster_composition, strategy='balanced', generations=50, population_size=100):
    """Wrapper function to run the enhanced team optimizer."""
    optimizer = EnhancedTeamOptimizer()
    # The new optimizer returns a list of top teams. We'll return the best one for compatibility.
    optimal_teams = optimizer.run_optimized_ga(available_players, budget, roster_composition, strategy=strategy, generations=generations)
    return optimal_teams[0] if optimal_teams else pd.DataFrame()
