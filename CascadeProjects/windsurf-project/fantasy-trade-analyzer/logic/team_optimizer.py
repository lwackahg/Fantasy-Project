import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

def run_team_optimizer(available_players, budget, roster_composition, generations=50, population_size=100):
    """
    Runs a genetic algorithm to find the optimal team composition.

    :param available_players: DataFrame of available players with their stats and values.
    :param budget: The total budget for the team.
    :param roster_composition: A dictionary defining the required number of players for each position.
    :param generations: The number of generations to run the algorithm.
    :param population_size: The size of the population in each generation.
    :return: The best team (as a DataFrame) found by the algorithm.
    """
    if available_players.empty:
        return pd.DataFrame()

    # --- 1. Problem Definition ---
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # --- 2. Initialization ---
    player_indices = list(available_players.index)
    num_roster_spots = sum(roster_composition.values())

    toolbox.register("player_indices", random.sample, player_indices, num_roster_spots)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.player_indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --- 3. Genetic Operators ---
    def evaluate(individual):
        team_df = available_players.loc[individual]
        total_cost = team_df['AdjValue'].sum()
        total_value = team_df['VORP'].sum()

        # Constraint 1: Budget
        if total_cost > budget:
            return -1, # Severe penalty for exceeding budget

        # Constraint 2: Roster Composition
        position_counts = team_df['Position'].value_counts()
        for pos, required_count in roster_composition.items():
            if position_counts.get(pos, 0) != required_count:
                return -1, # Severe penalty for incorrect roster structure

        return total_value,

    def custom_mutate(individual):
        # Mutate by swapping a player with another valid one from the pool
        current_player_idx = random.choice(individual)
        
        # Find a replacement candidate that is not already in the team
        candidates = [p_idx for p_idx in player_indices if p_idx not in individual]
        if not candidates:
            return individual, # No possible mutation

        new_player_idx = random.choice(candidates)
        
        # Swap players
        swap_index = individual.index(current_player_idx)
        individual[swap_index] = new_player_idx
        return individual,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- 4. Algorithm Execution ---
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1) # Store the single best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Run the algorithm
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, 
                        stats=stats, halloffame=hof, verbose=False)

    # --- 5. Return Best Result ---
    if hof:
        best_individual = hof[0]
        best_team_df = available_players.loc[best_individual].sort_values(by='AdjValue', ascending=False)
        return best_team_df
    else:
        return pd.DataFrame() # Return empty if no solution found
