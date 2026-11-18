from typing import List
import ioh
import numpy as np
import itertools
from GA import create_problem, studentnumber1_studentnumber2_GA

# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import ProblemClass, get_problem, logger

budget = 100_000
DIM_F18 = 50
DIM_F23 = 49

# configurations, evaluations per run, runs per problem
SCHEDULE = [(20, 500, 1), (6, 1500, 1), (3, 5000, 2)]

# Make results reproducible
np.random.seed(42)

# Hyperparameters
hyperparameter_space = {
    "population_size": [10, 20, 30, 50],
    "mutation_rate": [0.01, 0.02, 0.03, 0.04],
    "crossover_rate": [0.5, 0.7, 0.9],
    "n_elites": [1, 2, 3],
}

def sample_configs(amount):
    keys = list(hyperparameter_space.keys())
    all_configs = []
    for config in (itertools.product(*hyperparameter_space.values())):
        all_configs.append(dict(zip(keys, config)))
    idx = np.random.choice(len(all_configs), size=amount, replace=False)
    configs = [all_configs[i] for i in idx]
    return configs

def one_run(problem, dimension, params, run_budget, seed):
    np.random.seed(seed)
    best = studentnumber1_studentnumber2_GA(problem, dimension, budget=run_budget, **params)
    used = problem.state.evaluations
    problem.reset()
    return best, used

def rank_score(scores):
    ranks = []
    for s in scores:
        better = 0
        equal = 0
        for t in scores:
            if t > s:
                better += 1
            elif t == s:
                equal += 1
        rang = better + (equal+1) / 2
        ranks.append(rang)
    return np.array(ranks)

def successive_halving(configs, F18, F23):
    total_used = 0
    for rung, (n_config, run_budget, n_runs) in enumerate(SCHEDULE):
        configs = configs[:n_config]
        F18_scores = []
        F23_scores = []
        for params in configs:
            score_F18 = 0
            score_F23 = 0
            for r in range(n_runs):
                best_F18, used_F18 = one_run(F18, dimension=DIM_F18, params=params, run_budget=run_budget, seed=1000*rung+r)
                best_F23, used_F23 = one_run(F23, dimension=DIM_F23, params=params, run_budget=run_budget, seed=1000*rung+r)
                total_used += used_F18 + used_F23
                score_F18+= best_F18
                score_F23 += best_F23
            avg_F18 = score_F18/n_runs
            avg_F23 = score_F23/n_runs
            F18_scores.append(avg_F18)
            F23_scores.append(avg_F23)
        combined = (rank_score(F18_scores) + rank_score(F23_scores)) / 2  # lower is better
        order = np.argsort(combined)
        print(f"Round {rung}: {total_used} evaluations used.") 
        for i in order:
            print(configs[i], F18_scores[i], F23_scores[i], combined[i])
        
        configs = [configs[i] for i in order]
    assert total_used <= budget    
    winner = configs[0]
    return winner

def tune_hyperparameters():
    F18 = get_problem(18, dimension=50, instance=1, problem_class=ProblemClass.PBO)
    F23 = get_problem(23, dimension=49, instance=1, problem_class=ProblemClass.PBO)
    winner = successive_halving(sample_configs(SCHEDULE[0][0]), F18, F23)
    return winner["population_size"], winner["mutation_rate"], winner["crossover_rate"], winner["n_elites"]
    
if __name__ == "__main__":
    population_size, mutation_rate, crossover_rate, n_elites = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)
    print(n_elites)
