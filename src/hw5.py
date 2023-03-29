# HW5 - Genetic Algorithm
# Jaxon Jones
# Version 1.0

import numpy as np
import settings
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from obj_func import obj_func
from con_func import con_func
from meshing import x2mesh, x0_cube, x0_hyperboloid
from subprocess import run, call
from util import *
from random import shuffle
import time

# Settings
verbose = True
plot = False
tolerance = 1e-3
starting_population = 1000
function_span = (0, 1)

# Global Variables
mutations = 0

# GA FUNCTIONS
def generate_population(n_individuals, n_variables, span=(0, 1)):
    """Generate a random latin hypercube population of size n.
    
    Args:
        n_population (int): Number of individuals in the population.
        n_variables (int, optional): Number of variables in the population.
        
    Returns:
        ndarray: ndarray of the population.
    """
    # create a mxn matrix of random numbers between 0 and 1
    population = np.random.uniform(low=span[0], high=span[1], size=(n_individuals, n_variables))
    return population

def elitism(population, fitness, k=None, percent=90, mate="elite"):
    """Selects the top k individuals from the population (results in an even population size). 
    If k is not specified, it will select the top 90% of the population. 
    Function then mates strongest individuals and returns next generation.

    Args:
        population (ndarray): ndarray of individuals to select from.
        fitness (ndarray): ndarray of fitness for each individual.
        k (int, optional): Number of individual to select. Defaults to None (percentage based).
        percent (int, optional): Percentage of population to select. Defaults to 90.
        mate (str, optional): Mating function to use. Defaults to "random". Options: "random", "elite"

    Returns:
        ndarray: ndarray of the top k individuals.
    """
    if k is None:
        k = int(len(population)*(percent/100))
    if k % 2 != 0:
        k += 1
    sorted_population = population[np.argsort(-fitness)][:k]
    next_generation = np.zeros(sorted_population.shape)

    if mate == "elite":
        for i in range(0, len(sorted_population)-1, 2):
            # Mate strongest individuals
            a = population[i]
            b = population[i+1]

            # create children
            for j in range(len(sorted_population[0])):
                next_generation[i, j] = (a[j] + b[j])/2
                next_generation[i+1, j] = (a[j] + b[j])/2
        return next_generation
    else:
        shuffle(sorted_population)
        for i in range(0, len(sorted_population)-1, 2):
            # Mate strongest individuals
            a = population[i]
            b = population[i+1]

            # create children
            for j in range(len(sorted_population[0])):
                next_generation[i, j] = (a[j] + b[j])/2
                next_generation[i+1, j] = (a[j] + b[j])/2
        return next_generation

def tournament(population, fitness, con_func=None, mate="random"):
    """Creates new generation using tournament selection.
    Function then mates random individuals and returns next generation.
    *Note: This function allows constraints.

    Args:
        population (ndarray): ndarray of individuals to select from.
        fitness (ndarray): ndarray of fitness for each individual.
        con_func (function, optional): Constraint function. Defaults to None.
        mate (str, optional): Mating function to use. Defaults to "random". Options: "random", "elite"

    Returns:
        ndarray: ndarray of the top k individuals for the next generation
    """
    if len(population) % 2 != 0:
        raise ValueError("Population size must be even.")
    next_gen_parents = np.zeros(population.shape)
    next_generation = np.zeros(population.shape)
    remaining = np.copy(population)

    h = 0
    if con_func is not None: # constraints
        for i in range(2): # 2x to create entire pool
            remaining = np.copy(population)
            for j in range(len(population)):
                # Select 2 random individuals
                index_a = np.random.randint(len(remaining))
                index_b = np.random.randint(len(remaining))
                while index_a == index_b:
                    index_b = np.random.randint(len(remaining))
                a = remaining[index_a]
                b = remaining[index_b]

                fitness_a = fitness[np.where((remaining == a).all(axis=1))[0]]
                fitness_b = fitness[np.where((remaining == b).all(axis=1))[0]]
                # check for duplicates
                if len(fitness_a) > 1:
                    fitness_a = fitness_a[0]
                if len(fitness_b) > 1:
                    fitness_b = fitness_b[0]

                # check constraints
                a_con = False
                b_con = False
                if con_func(a) < 0: a_con = True
                if con_func(b) < 0: b_con = True

                # battle
                if a_con and b_con: # both are feasible -> compare fitness
                    index_a = np.where((remaining == a).all(axis=1))[0]
                    if len(index_a) > 1:
                        index_a = index_a[0]

                    index_b = np.where((remaining == b).all(axis=1))[0]
                    if len(index_b) > 1:
                        index_b = index_b[0]

                    if fitness[index_a] < fitness[index_b]:
                        next_gen_parents[h] = a
                    else:
                        next_gen_parents[h] = b
                        
                elif a_con or b_con: # one is feasible -> choose feasible
                    if a_con:
                        next_gen_parents[h] = a
                    else:
                        next_gen_parents[h] = b

                else: # both are infeasible -> find which one is less infeasible
                    if con_func(a) < con_func(b):
                        next_gen_parents[h] = a
                    else:
                        next_gen_parents[h] = b

                # Remove individuals from remaining population
                remaining = np.delete(remaining, index_a, axis=0)
                index_b = np.where((remaining == b).all(axis=1))[0]
                if len(index_b) > 1:
                    index_b = index_b[0]
                remaining = np.delete(remaining, index_b, axis=0)
                h += 1
                if remaining.size == 0:
                    break
    
    else: # no constraints
        for i in range(2): # 2x to create entire pool
            remaining = np.copy(population)
            for j in range(len(population)):
                # Select 2 random individuals
                index_a = np.random.randint(len(remaining))
                index_b = np.random.randint(len(remaining))
                while index_a == index_b:
                    index_b = np.random.randint(len(remaining))
                a = remaining[index_a]
                b = remaining[index_b]

                fitness_a = fitness[np.where((remaining == a).all(axis=1))[0]]
                fitness_b = fitness[np.where((remaining == b).all(axis=1))[0]]
                # check for duplicates
                if len(fitness_a) > 1:
                    fitness_a = fitness_a[0]
                if len(fitness_b) > 1:
                    fitness_b = fitness_b[0]

                if fitness_a < fitness_b:
                    next_gen_parents[h] = a
                else:
                    next_gen_parents[h] = b
                
                # Remove individuals from remaining population
                index_a = np.where((remaining == a).all(axis=1))[0]
                if len(index_a) > 1:
                    index_a = index_a[0]
                remaining = np.delete(remaining, index_a, axis=0)

                index_b = np.where((remaining == b).all(axis=1))[0]
                if len(index_b) > 1:
                    index_b = index_b[0]
                
                remaining = np.delete(remaining, index_b, axis=0)
                h += 1
                if remaining.size == 0:
                    break
                
                

    # Mate individuals
    if mate == "random":
        for i in range(len(population) - 1):
            # Mate strongest individuals
            a = next_gen_parents[i]
            b = next_gen_parents[i+1]

            # create children
            for j in range(len(population[0])):
                next_generation[i, j] = (a[j] + b[j])/2
                next_generation[i+1, j] = (a[j] + b[j])/2
            i += 2
        
    return next_generation
        
def mutate(population, chance=0.2, k=None, percent=15):
    """Mutate an individual.

    Args:
        population (ndarray): ndarray of individuals to mutate.
        chance (float, optional): Chance of mutation. Defaults to 0.2.
        k (int, optional): Number of possible mutations. Defaults to None (percentage based).
        percent (int, optional): Percentage of population to mutate. Defaults to 15.

    Returns:
        ndarray: ndarray of the individual after mutation.
    """
    if k is None:
        k = int(len(population)*(percent/100))
    if k % 2 != 0:
        k -= 1

    mutated_population = population.copy()
    for i in range(k):
        if np.random.rand() < chance:
            gene = np.random.randint(0, len(population[0]))
            a = np.random.randint(0, len(population))
            mutated_population[a, gene] = np.random.uniform(function_span[0], function_span[1], 1)
            # vprnt("Mutated gene", gene, "on individual", a, log=True)
            global mutations
            mutations += 1
        else:
            # vprnt("No mutation", log=True)
            pass
    return mutated_population

# hw5 main
def hw5_2():
    # Set up the population
    nv = settings.resolution[0]*settings.resolution[1]*settings.resolution[2]
    population  =   generate_population(starting_population, nv, span=function_span)
    fitness = np.zeros(starting_population)
    i = 0

    while(i < settings.maxiter):
        vprnt("\nGeneration", i)
        print("Generation", i+1, "of", settings.maxiter)
        vprnt("Population size:", len(population))
        vprnt("Population:", population)

        # Set up the fitness function
        for k in range(len(population)):
            fitness[k] = obj_func(population[k])
        vprnt("Fitness:", fitness)

        # Selection & crossover
        population  =   tournament(population, fitness, con_func=con_func)
        
        # Mutation
        population  =    mutate(population)

        i+=1 # increment generation

    # Save the x vector to file (FOR DEBUGGING)
    np.savetxt(f"cube_optimized.txt", population)

    # Print results
    print("\n\n--- RESULTS ---")
    from obj_func import f_calls
    from con_func import g_calls
    print(f"Number of function calls: {f_calls}")
    print(f"Number of constraint calls: {g_calls}")
    print("\nTotal Generations:", i)
    print("Starting Population Size:", starting_population)
    print("Final Population Size:", len(population))
    print("Final Population:", population)
    print("Minimum:", min(fitness))
    print("x*:", population[0])
    print("Mutations:", mutations)
    print("--- -- -- -- -- -- -- -- ---\n\n")

    x2mesh(population[0], "cube_optimized", dim=settings.resolution, out_format="vtk")

    # Visualize the optimized voxelization
    # run(["sfepy-view", "cube_optimized.vtk"])



if __name__ == "__main__":
    start_time = time.time()
    hw5_2()
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")