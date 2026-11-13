from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000

# To make your results reproducible (not required by the assignment), you could set the random seed by
np.random.seed(42)

pop_size = 50
mutation_rate = 0.01
crossover_rate = 0.5
n_elites = 3


def crossover(
        parent_1: np.ndarray,
        parent_2: np.ndarray,
        dimension
) -> Tuple[np.ndarray, np.ndarray]:
    """Create two offspring using uniform crossover."""
    # `crossover_probability` is handled in the main GA loop. Inside
    # uniform crossover, each bit position is swapped with rate 0.5.
    child_1, child_2 = np.zeros(dimension), np.zeros(dimension)
    for i_1, bit_1 in enumerate(child_1):
        pbit1, pbit2 = parent_1[i_1], parent_2[i_1]
        chosen_bit = np.random.choice([pbit1, pbit2])
        child_1[i_1] = chosen_bit

    for i_2, bit_2 in enumerate(child_2):
        pbit1, pbit2 = parent_1[i_2], parent_2[i_2]
        chosen_bit = np.random.choice([pbit1, pbit2])
        child_2[i_2] = chosen_bit

    return child_1, child_2


def mutation(individual: np.ndarray, rate: float) -> np.ndarray:
    """Apply standard bit mutation and return a new individual."""
    new_individual = individual.copy()
    decisions = [True, False]
    for i, bit in enumerate(new_individual):
        p = [rate, 1 - rate]
        flip = np.random.choice(decisions, p=p)
        if flip:
            if new_individual[i] == 0:
                new_individual[i] = 1
            else:
                new_individual[i] = 0

    return new_individual


def mating_selection(
        parents: list[np.ndarray],
        parent_fitness: list[float],
) -> list[np.ndarray]:
    min_f = min(parent_fitness)
    shifted_fitness = [f - min_f for f in parent_fitness]
    total_f = sum(shifted_fitness)

    n_parents = len(parents)
    if total_f == 0:
        p = [1.0 / n_parents] * n_parents
    else:
        p = [f / total_f for f in shifted_fitness]

    # print(p)

    selected_indices = np.random.choice(n_parents, size=n_parents, p=p)
    return [parents[i].copy() for i in selected_indices]


def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO, dimension) -> None:
    parents = [
            np.random.randint(0, 2, size=dimension, dtype=int)
            for _ in range(pop_size)
        ]

    # Evaluate the initial population and record its best individual.
    parent_fitness: list[float] = []
    best_fitness = -np.inf
    best_individual = None

    for individual in parents:
        fitness = float(problem(individual))
        # print(f'Fitness: {fitness}')
        parent_fitness.append(fitness)

        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual.copy()

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        mating_pool = mating_selection(parents, parent_fitness)
        if len(mating_pool) != len(parents):
            raise ValueError("mating_selection must preserve the population size.")

        # The last generation may be smaller when little budget remains.
        n_offspring = min(pop_size - n_elites, budget - problem.state.evaluations)
        offspring: list[np.ndarray] = []
        pair_index = 0

        while len(offspring) < n_offspring:
            first = (2 * pair_index) % pop_size
            second = (first + 1) % pop_size
            parent_1 = np.asarray(mating_pool[first], dtype=int).copy()
            parent_2 = np.asarray(mating_pool[second], dtype=int).copy()

            if np.random.random() < crossover_rate:
                child_1, child_2 = crossover(parent_1, parent_2, dimension)
            else:
                child_1, child_2 = parent_1.copy(), parent_2.copy()

            child_1 = mutation(np.asarray(child_1, dtype=int), mutation_rate)
            child_2 = mutation(np.asarray(child_2, dtype=int), mutation_rate)
            offspring.append(np.asarray(child_1, dtype=int))
            if len(offspring) < n_offspring:
                offspring.append(np.asarray(child_2, dtype=int))

            pair_index += 1

        offspring_fitness: list[float] = []
        for individual in offspring:
            fitness = float(problem(individual))
            offspring_fitness.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()

        # if problem.state.evaluations % 1000 == 0:
        #     print(f'---- Eval {problem.state.evaluations}: best fitness {best_fitness} ----')

        # print(f'Offspring fitness: {offspring_fitness}')

        # Generational replacement. A shortened final generation is safe
        # because the evaluation budget is exhausted immediately afterward.

        elite_indices = np.argsort(parent_fitness)[-n_elites:]
        elites = [parents[i].copy() for i in elite_indices]
        elite_fitness = [parent_fitness[i] for i in elite_indices]

        parents = elites + offspring
        parent_fitness = elite_fitness + offspring_fitness


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    dim_f18 = 50
    F18, _logger = create_problem(dimension=dim_f18, fid=18)
    print(f'--- F18 ---')
    for run in range(20):
        print(f'Run: {run}')
        studentnumber1_studentnumber2_GA(F18, dim_f18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    dim_f23 = 49
    F23, _logger = create_problem(dimension=dim_f23, fid=23)
    print(f'--- F23 ---')
    for run in range(20):
        print(f'Run: {run}')
        studentnumber1_studentnumber2_GA(F23, dim_f23)
        F23.reset()
    _logger.close()