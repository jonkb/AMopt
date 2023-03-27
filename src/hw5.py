# HW5 - Genetic Algorithm
# Jaxon Jones
# Version 1.0

import numpy as np
import time
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


# Settings
verbose = False
plot = True
tolerance = 1e-5
starting_population = 1000
function_span = (-10, 10)
maxiter = 30

# Global Variables
mutations = 0


# Utility functions
def setup():
    f = open("Genetic Algorithm Log.txt", 'r+')
    f.truncate(0) # need '0' when using r+
    with open("Genetic Algorithm Log.txt", "a") as f:
            print("Genetic Algorithm Results", file=f)

def vprnt(*args, log=False, verbose=verbose):
    """ Thin wrapper around the standard print function
    Only prints if verbose == True
    """
    if verbose:
        print(*args)
    if log:
        return
        #print to log file
        with open("Genetic Algorithm Log.txt", "a") as f:
            print(*args, file=f)


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

def sort_and_select(population, fitness, k=None, percent=90):
    """Selects the top k individuals from the population (results in an even population size). If k is not specified, it will select the top 90% of the population.

    Args:
        population (ndarray): ndarray of individuals to select from.
        fitness (ndarray): ndarray of fitness for each individual.
        k (int, optional): Number of individual to select. Defaults to None (percentage based).
        percent (int, optional): Percentage of population to select. Defaults to 90.

    Returns:
        ndarray: ndarray of the top k individuals.
    """
    if k is None:
        k = int(len(population)*(percent/100))
    if k % 2 != 0:
        k += 1
    sorted_population = population[np.argsort(fitness)][:k]
    return sorted_population

def random_selection(population, k=None, percent=90):
    """Selects k random individuals from the population (results in an even population size). If k is not specified, it will randomly select 90% of the population.

    Args:
        population (ndarray): ndarray of individuals to select from.
        fitness (ndarray): ndarray of fitness for each individual.
        k (int, optional): Number of individual to select. Defaults to None (percentage based).
        percent (int, optional): Percentage of population to select. Defaults to 90.

    Returns:
        ndarray: ndarray of the top k individuals.
    """
    if k is None:
        k = int(len(population)*(percent/100))
    if k % 2 != 0:
        k += 1

    random_population = population[np.random.choice(len(population), k)]
    return random_population

def roulette_selection(population, fitness, k=None, percent=90):
    """Selects k individuals from the population using roulette selection (results in an even population size). If k is not specified, it will randomly select 90% of the population.

    Args:
        population (ndarray): ndarray of individuals to select from.
        fitness (ndarray): ndarray of fitness for each individual.
        k (int, optional): Number of individual to select (times the "wheel" is spun). Defaults to None (percentage based).
        percent (int, optional): Percentage of population to select. Defaults to 90.

    Returns:
        ndarray: ndarray of the top k individuals.
    """
    if k is None:
        k = int(len(population)*(percent/100))
    if k % 2 != 0:
        k += 1

    roulette_population = np.zeros((k, len(population[0])))
    Fmax = np.max(fitness)
    Fmin = np.min(fitness)
    deltaF = 1.1*Fmax - 0.1*Fmin
    S = np.zeros(len(population))
    for i in range(len(population)):
        F = (-fitness[i] + deltaF)/max(1, deltaF-Fmin)
        S[i] = F/np.sum(fitness)
    
    vprnt("S:", S, verbose=True)

    i = 0
    for j in range(k):
        r = np.random.rand()
        if j == 0:
            if r < S[j]:
                roulette_population[i] = population[j]
        else:
            if r < S[j] and r > S[j-1]:
                roulette_population[j] = population[j]
        i += 1
    return roulette_population

def mate(population):
    """mate all individuals in a population.

    Args:
        population (ndarray): population.

    Returns:
        ndarray: ndarray.
    """
    new_population = np.zeros(population.shape)
    for i in range(len(population) - 1):
        # Mate strongest individuals
        a = population[i]
        b = population[i+1]

        # create children
        for j in range(len(population[0])):
            new_population[i, j] = (a[j] + b[j])/2
            new_population[i+1, j] = (a[j] + b[j])/2
        i += 2
    return new_population

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
            vprnt("Mutated gene", gene, "on individual", a, log=True)
            global mutations
            mutations += 1
        else:
            vprnt("No mutation", log=True)
    return mutated_population

def plot_population(population, fitness, title=""):
    """Plot the population over the fitness function.

    Args:
        population (ndarray): ndarray of individuals.
        fitness (ndarray): ndarray of fitness for each individual.
        title (str, optional): Title of the plot. Defaults to "".
    """
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(function_span[0], function_span[1])
    plt.ylim(function_span[0], function_span[1])
    plt.scatter(population[:,0], population[:,1], c=fitness, cmap='viridis')
    plt.colorbar()
    plt.show()

# Fitness Function
def eggshell_fitness_function(x):
    fitness = np.zeros(x.shape[0])
    # find fitness for each individual
    for i in range(x.shape[0]):
        fitness[i] = 0.1*x[i,0]**2 + 0.1*x[i,1]**2 - np.cos(3*x[i,0]) - np.cos(3*x[i,1])
    return fitness

# hw5 main
def hw5():
    setup()
    population = generate_population(starting_population, 2, span=function_span)
    i = 0
    while(i < maxiter):
        vprnt("\nGeneration", i, log=True)
        vprnt("Population size:", len(population), log=True)
        vprnt("Population:", population, log=False)

        # Evaluate fitness
        fitness     =    eggshell_fitness_function(population)
        vprnt("Fitness:", fitness)

        # Selection
        # population  =    sort_and_select(population, fitness) # Elitism
        # population  =    random_selection(population) # Random
        population  =    roulette_selection(population, fitness) # Roulette

        # Crossover
        population  =    mate(population) # next generation

        # Mutation
        population  =    mutate(population)

        i+=1 # increment generation

        # plot ever 10 generations
        if plot and i % 10 == 0:
            plot_population(population, fitness, i)


    # print results
    vprnt("\nTotal Generations:", i, verbose=True, log=True)
    vprnt("Starting Population Size:", starting_population, verbose=True, log=True)
    vprnt("Final Population Size:", len(population), verbose=True, log=True)
    vprnt("Final Population:", population, log=True)
    vprnt("Final Fitness:", eggshell_fitness_function(population), log=True)
    vprnt("Minimum:", min(eggshell_fitness_function(population)), log=True, verbose=True)
    vprnt("x*:", population[0], verbose=True, log=True)
    vprnt("Mutations:", mutations, verbose=True, log=True)



if __name__ == "__main__":
    start_time = time.time()
    hw5()
    print("Time:", round(time.time() - start_time), "seconds")

    if plot:
        # graph eggshell function on 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = 0.1*X**2 + 0.1*Y**2 - np.cos(3*X) - np.cos(3*Y)
        ax.plot_surface(X, Y, Z, linewidth=1, antialiased=False)
        plt.show()





'''
def crossover(population, chance=0.2, k=2):
    """Crossover two individuals.

    Args:
        population (ndarray): ndarray of individuals to crossover.
        chance (float, optional): Chance of crossover. Defaults to 0.2.
        k (int, optional): Number of possible crossovers. Defaults to 2.

    Returns:
        ndarray: ndarray of the two individuals after crossover.
    """
    crossed_population = population.copy()
    for i in range(k):
        if np.random.rand() < chance:
            gene = np.random.randint(0, len(population[0]))
            a = np.random.randint(0, len(population))
            b = np.random.randint(0, len(population))
            crossed_population[a, gene] = population[b, gene]
            crossed_population[b, gene] = population[a, gene]
            vprnt("Crossover gene", gene, "with individuals", a, "&", b, log=True)
        else:
            vprnt("No crossover", log=True)
    return crossed_population

    

    #     #calculate change in fitness
    #     new_min = min(fitness)
    #     change = abs(new_min - last_min)
    #     if change < tolerance:
    #         msg = "Success - Change in fitness is less than tolerance"
    #         break
    #     else:
    #         old_min = new_min

    # if i == maxiter:
    #     msg = "Failed - Maximum iterations reached"

    # vprnt(msg, log=True, verbose=True)
'''