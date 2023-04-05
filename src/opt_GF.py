""" Gradient-free optimization

Currently implemented:
* Genetic Algorithm, GA

Jon Black
2023-03-29
"""

import settings
import numpy as np
from optsol import optsol
if settings.legacy:
    from pyDOE import lhs
else:
    from scipy.stats import qmc # Not compatible with Python 3.6

# Initialize random number generator
rng = np.random.default_rng()

## Methods for custom real-encoded genetic algorithm

def breed(p1, p2, f1, f2, fbest=0, fscale=1., mutation1=0.05, 
    mutation2=0.40):
    """ Generate a child from p1 & p2

    Used by GA.

    TODO: Doesn't work very well if the x dimensions are not well-scaled to
        each other. E.g. if the bounds on one variable are -1000 to 1000 and
        the bounds on another are -1 to 1. Could make fscale a vector that
        is scaled by bounds.

    p1, p2: parent points (np arrays)
    f1, f2: corresponding function values.
    fbest: best (lowest) f so far.
    fscale (float): approximate scaling factor btw dx & df.
        E.g. if f range is btw 0 and 1000, while x is btw 0 and 1, this should
        be 1000. Think of it as a sensitivity or unit conversion.
    mutation1: Variance factor for mutation #1, corresponding to shifting
        the parents before combining. A value of 1 means that the variance of
        this distribution would be equal to ||p2 - p1|| if 
        (f1-fbest) / fscale = 1
    mutation2: Variance factor for mutation #2, corresponding to the
        distribution of points along the line btw p1p & p2p. A value of 1 means
        that the variance of this distribution is equal to the distance btw 
        p1 & p2.
        High values of mutation2 encourage more exploration along the line btw
        parents and slow the removal of diversity. However, it increases the 
        convergence time, and if mutation2 > ~1.5, it becomes unstable (more
        likely to grow than shrink). That'd be an interesting probability
        question to figure out.
    """
    assert p2.size == p1.size, "p1 & p2 must be of the same size"
    N_x = p1.size
    # inverse fitness: how bad is the point. fi >= 0
    fi1 = f1 - fbest
    fi2 = f2 - fbest
    ## Generate mutated parents (p1' & p2')
    # Variances are proportional to the distance btw parents as well as how
    #   much worse that parent is than the best
    # 2-norm / sqrt(N) is a rough approximation of N-norm that doesn't blow up
    #   to infinity for large N.
    dist = np.linalg.norm(p2-p1, 2) / np.sqrt(N_x)
    # dist = np.linalg.norm(p2-p1, N_x)
    # dist = np.abs(p2-p1)
    v1 = fi1 / fscale * dist * mutation1 / N_x
    v2 = fi2 / fscale * dist * mutation1 / N_x
    assert np.all(v1 >= 0) and np.all(v2 >= 0), (f"Negative variance. fi1={fi1}"
        f", fi2={fi2}, dist={dist}")
    p1p = rng.normal(p1, v1)
    p2p = rng.normal(p2, v2)
    # DEBUGGING
    # print(55, p1, p2, v1, v2, mutation1, dist)
    # print(60, fbest, f1, f2, fi1, fi2)
    # print(f"p1p: {p1p}, p2p: {p2p}")
    ## Linear crossover
    # r0: center of child distr. closer to better one
    if fi1 == 0:
        r0 = 0
    elif fi2 == 0:
        r0 = 1
    else:
        # Weighted avg btw 0 & 1, weighted by 1/fi
        r0 = (1/fi2) / (1/fi1 + 1/fi2)
    r2 = rng.normal(r0, mutation2)
    # Starting at slide some random amount along the line btw p1p & p2p
    child = p1p + r2*(p2p - p1p)
    return child

def find_best(population, popf, popg=None):
    """Find best point in a population.
    
    Used by GA.
    """
    if popg is not None:# If constrained
        # Worst constraint for each point
        maxg = np.max(popg, axis=1)
        # TODO: This may fail if >1 constraint. np.all?
        i_feasible = np.where(maxg <= 0)[0]
        isfeasible = False
        if i_feasible.size == 0:
            # If none feasible, return closest to feasible
            g_star = popg[ maxg == np.min(maxg) ]
            i_star = np.where(popg == g_star)[0][0]
            f_star = popf[i_star]
            x_star = population[i_star]
        else:
            isfeasible = True
            # Best of the feasible points
            f_feasible = popf[i_feasible]
            g_feasible = popg[i_feasible]
            x_feasible = population[i_feasible]
            f_star = np.min(f_feasible)
            i_star = np.where(f_feasible == f_star)[0][0]
            g_star = g_feasible[i_star]
            x_star = x_feasible[i_star]
        return x_star, f_star, g_star, isfeasible
    else:
        # Unconstrained: pick the point with lowest f
        f_star = np.min(popf)
        i_star = np.where(popf == f_star)[0][0]
        x_star = population[i_star]
        return x_star, f_star, None, None

def tournament(population, popf, popg=None):
    """ Perform tournament selection.
    Return an array of winner indices

    Used by GA
    """
    N_pop = population.shape[0]
    N_pairs_tourn = int(np.ceil(N_pop/2))
    N_pairs_breed = int(np.ceil(N_pairs_tourn/2))
    i_list = np.arange(N_pop)
    rng.shuffle(i_list) # Shuffles in-place
    # Create random pairs, repeating one if N_pop is odd
    pairs = np.resize(i_list, (N_pairs_tourn,2))
    winners = np.zeros(N_pairs_tourn, dtype=int)
    # NOTE: This could likely be vectorized, but it's ok for now
    for i_pair, pair in enumerate(pairs):
        if popg is not None:
            # Follow logic from book 7.6.3
            gmax_p0 = np.max(popg[pair[0]])
            gmax_p1 = np.max(popg[pair[1]])
            if gmax_p0 > 0 and gmax_p1 <= 0:
                # p1 is feasible, p0 isn't --> prefer p1
                winners[i_pair] = pair[1]
            elif gmax_p1 > 0 and gmax_p0 <= 0:
                # p0 is feasible, p1 isn't --> prefer p0
                winners[i_pair] = pair[0]
            elif gmax_p0 > 0 and gmax_p1 > 0:
                # Neither is feasible --> prefer the one that has lower g
                winners[i_pair] = pair[0] if gmax_p0 < gmax_p1 else pair[1]
            else:
                # Both are feasible --> prefer the one that has lower f
                f_p0 = popf[pair[0]]
                f_p1 = popf[pair[1]]
                winners[i_pair] = pair[0] if f_p0 < f_p1 else pair[1]
        else:
            # Unconstrained --> prefer the one that has lower f
            f_p0 = popf[pair[0]]
            f_p1 = popf[pair[1]]
            winners[i_pair] = pair[0] if f_p0 < f_p1 else pair[1]
    # Create parent pairs, repeating one if N_pairs_tourn is odd
    parents = np.resize(winners, (N_pairs_breed,2))
    return winners, parents

def GA(f, bounds, pop_size=15, constraints=(), it_max=100, xtol=1e-8, 
    mutation1=0.05, mutation2=0.40, elitist=True, figax=None, verbose=False,
    callback=None, warm_start=None):
    """ Genetic Algorithm Optimization

    Sampling: LHS
    Selection: Tournament
    Crossover: Custom gaussian crossover method.
        Generate mutated parents by sampling from a gaussian ball around the
        parents, then perform linear crossover by sampling from a 1d gaussian,
        centered around a weighted average of the parents.
    Mutation: Incorporated in custom crossover process. The mutation variance
        is proportional to (f - fbest), so the best point isn't mutated, though
        its children will not be identical, due to crossover.

    Not written for vectorized functions.

    f (callable): Function to minimize
        f: x -> y, x \in Real^N_x, y \in Real
    bounds (N_x x 2): Design variable bounds
        E.g [(lb1, ub1), (lb2, ub2), ...]
    pop_size (int): Size of population. N_pop = pop_size * N_x
    constraints (tuple of callable): Tuple of constraint functions g_j.
        Each g: x -> y, x \in Real^N_x, y \in Real
        The constraint is that each g_j(x) < 0
    it_max (int): Max iterations
    figax (None or 2-tuple): If provided, then figax[0] = pyplot figure and 
        figas[1] = pyplot axis on that figure. The population will be displayed 
        on that figure using scatter.
    elitist (bool): Whether to forcibly keep the best point.
    verbose (bool): Whether to print a message every iteration.
    callback (function): A function to be called after every iteration.
    warm_start (dict): Warm start population. Dictionary with the following:
        "population" (N_pop x N_x): initial population
        "popf" (N_pop x 1): objective function evaluated at population
        "popf" (N_pop x N_g): constraint function evaluated at population
    """

    ## Generate initial population (LHS)
    bounds = np.array(bounds)
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    N_x = bounds.shape[0]
    N_pop = pop_size * N_x
    if warm_start is not None:
        population = warm_start["population"]
        # Use the given population as the starting population
        assert population.shape == (N_pop, N_x), ("Population passed with"
            " warm_start does not match expected dimensions"
            f", ({N_pop},{N_x})")
    elif settings.legacy:
        sample = lhs(N_x, N_pop) # Uniform btw 0 and 1
        population = sample * (ubounds-lbounds) + lbounds
    else:
        sampler = qmc.LatinHypercube(d=N_x)
        sample = sampler.random(n=N_pop)
        population = qmc.scale(sample, lbounds, ubounds)
    # Other setup
    N_g = len(constraints)
    it = 0
    nfev = 0
    ngev = 0
    maxvar = np.max(np.var(population, axis=0))

    ## Plot population (if ax provided)
    if figax is not None:
        hscat = figax[1].scatter(population[:,0], population[:,1])
        figax[0].canvas.draw()
        input("Press any key to continue")

    ## Evaluate f & g for initial population
    if (warm_start is not None) and ("popf" in warm_start) and (warm_start["popf"] is not None):
        popf = warm_start["popf"]
    else:
        popf = np.array([f(x) for x in population])
        nfev += N_pop
    if (warm_start is not None) and ("popg" in warm_start) and (warm_start["popg"] is not None):
        popg = warm_start["popg"].reshape(N_pop, N_g)
    else:
        if N_g > 0:
            popg = np.array([[g(x) for g in constraints] for x in population])
            ngev += N_pop * N_g
        else:
            popg = None # Still initialize to not break things later
    
    # Print status after initial sample (Iteration 0)
    if (callback is not None) or verbose:
        xbest, fbest, gbest, isfeasible = find_best(population, popf, popg)
        if verbose:
            print(f"Iteration {it} (initial sample) complete."
                f"\n\tBest x: {xbest}"
                f"\n\tBest f: {fbest}"
                f"\n\tBest g: {gbest}"
                f" {'(feasible)' if isfeasible else '(not feasible)'}"
                f"\n\tMax x-variance: {maxvar}")
        if (callback is not None):
            # Pack the data relevant to the current state of the GA
            #   into this optsol object
            data = optsol(xbest, fbest, gbest, it=it)
            data.maxvar = maxvar
            data.isfeasible = isfeasible
            data.population = population
            data.popf = popf
            data.popg = popg
            callback(data)

    while (it < it_max) and (maxvar > xtol):
        ## Tournament selection
        # Perform 2 tournaments and pull 2 children from each parent pair
        winners1, parents1 = tournament(population, popf, popg)
        winners2, parents2 = tournament(population, popf, popg)
        winners = np.concatenate([winners1, winners2])
        parents = np.concatenate([parents1, parents2])
        # Lowest function call among winners. Not necessarily feasible.
        fbest = np.min(popf[winners])

        ## Create next generation
        new_pop = np.zeros_like(population)
        for i_np in range(N_pop-1 if elitist else N_pop):
            # If elitist, leave the last entry in new_pop for the best from the 
            #   previous generation.
            # This next line makes it so each parent couple makes two children.
            #   We may not use every couple twice, due to odd numbers.
            couple = parents[int(np.floor(i_np/2))]
            p0 = population[couple[0]]
            p1 = population[couple[1]]
            f0 = popf[couple[0]]
            f1 = popf[couple[1]]
            new_pop[i_np] = breed(p0, p1, f0, f1, fbest=fbest, 
                mutation1=mutation1, mutation2=mutation2)
        if elitist:
            # Not necessarily the same x as fbest from above
            xbest, *_ = find_best(population, popf, popg)
            new_pop[-1,:] = xbest
        # Update population
        population = new_pop
        # Clip to bounds
        population = np.clip(population, lbounds, ubounds)
        
        ## Update figure axis, if passed
        if figax is not None:
            hscat.set_offsets(population)
            figax[0].canvas.draw()
            input("Press any key to continue")

        ## Evaluate f & g for population
        popf = np.array([f(x) for x in population])
        nfev += N_pop
        if N_g > 0:
            popg = np.array([[g(x) for g in constraints] for x in population])
            ngev += N_pop * N_g
        
        ## x-variance stopping tolerance. If all points are very close to each
        ##  other, then stop
        maxvar = np.max(np.var(population, axis=0))

        it += 1
        if (callback is not None) or verbose:
            xbest, fbest, gbest, isfeasible = find_best(population, popf, popg)
            if verbose:
                print(f"Iteration {it} complete."
                    f"\n\tBest x: {xbest}"
                    f"\n\tBest f: {fbest}"
                    f"\n\tBest g: {gbest}"
                    f" {'(feasible)' if isfeasible else '(not feasible)'}"
                    f"\n\tMax x-variance: {maxvar}")
            if (callback is not None):
                # Pack the data relevant to the current state of the GA
                #   into this optsol object
                data = optsol(xbest, fbest, gbest, it=it)
                data.maxvar = maxvar
                data.isfeasible = isfeasible
                data.population = population
                data.popf = popf
                data.popg = popg
                callback(data)
    
    if it == it_max:
        msg = "Max iterations reached. "
    else:
        msg = "Population variance < xtol. "
    
    # Find & return best point
    xbest, fbest, gbest, isfeasible = find_best(population, popf, popg)
    if N_g > 0:
        msg += ("Feasible point found. " if isfeasible else 
            "No feasible point found. ")
    return optsol(xbest, fbest, g_star=gbest, it=it, msg=msg, nfev=nfev, 
        ngev=ngev)

def hyperprm_opt(opt=False, test=False):
    """ Try optimizing hyperparameters for GA

    Find the min popsize s.t. min. accuracy >= 0.5
    """

    from example_functions import eggshell
    b = 5
    bounds = ((-b, b), (-b, b))
    N_outer = 1000
    right_tol = .50
    correct = np.array([0,0])

    def test(pop_size=15, mutation1=0.15, mutation2=0.75, xtol=1e-3):
        N_right = 0
        sumit = 0
        sumnfev = 0
        for i in range(N_outer):
            sol = GA(eggshell, bounds, pop_size=int(pop_size), xtol=xtol,
                mutation1=mutation1, mutation2=mutation2)
            if np.linalg.norm(sol.x_star - correct) < right_tol:
                N_right += 1
            sumit += sol.it
            sumnfev += sol.nfev
        acc = N_right / N_outer
        avgit = sumit / N_outer
        avgnfev = sumnfev / N_outer
        return acc, avgit, avgnfev
    
    ## Test a list of popsizes
    if test:
        popsizes = np.arange(10, 1, -1)
        accuracies = []
        for pop_size in popsizes:
            acc, avgit, avgnfev = test(pop_size)
            accuracies.append(acc)
            print(f"With pop_size = {pop_size}, accuracy is {acc*100:.2f}%"
                f", avg nfev = {avgnfev:.2f}")
    
    ## Optimize
    if opt:
        # def hpo_f(x):
        #     # x: (pop_size, mutation1, mutation2, xtol) - OLD
        #     # x: (mutation1, mutation2)
        #     # x[0] = np.round(x[0])
        #     acc, avgit, avgnfev = test(mutation1=x[0], mutation2=x[1])
        #     # Max. accuracy, min. nfev
        #     return avgnfev * acc**(-2)
        
        fcall_data = {}
        def hpo_f(x):
            # x: (pop_size, mutation1, mutation2, xtol) - OLD
            # x: (mutation1, mutation2)
            # x[0] = np.round(x[0])
            sx = str(x)
            if sx in fcall_data:
                return fcall_data[sx][2]
            else:
                acc, avgit, avgnfev = test(mutation1=x[0], mutation2=x[1])
                # print(315, x, acc, avgnfev)
                fcall_data[sx] = (acc, avgit, avgnfev)
                return avgnfev
        def hpo_g(x):
            min_acc = 0.90
            # x: (mutation1, mutation2)
            # x[0] = np.round(x[0])
            sx = str(x)
            if sx in fcall_data:
                # acc > min_acc --> min_acc - acc < 0
                return min_acc - fcall_data[sx][0]
            else:
                acc, avgit, avgnfev = test(mutation1=x[0], mutation2=x[1])
                fcall_data[sx] = (acc, avgit, avgnfev)
                # acc > min_acc --> min_acc - acc < 0
                return min_acc - acc
        
        # hpo_constraints = (hpo_g,)
        # hpo_bounds = ((1, 25), (0, 0.2), (0, 1.25), (0, 1e-1))
        hpo_bounds = ((0, 0.2), (0, 1.25))
        # sol = GA(hpo_f, hpo_bounds, pop_size=20, xtol=1e-3, verbose=True, 
        #     elitist=False, constraints=hpo_constraints, mutation1=0.10, 
        #     mutation2=0.60)
        # sol.printall()
        from scipy.optimize import differential_evolution, NonlinearConstraint
        hpo_constraints = NonlinearConstraint(hpo_g, -np.inf, 0.0)
        res = differential_evolution(hpo_f, bounds=hpo_bounds, polish=False,
            constraints=hpo_constraints, tol=5e-2, disp=True, maxiter=4, 
            popsize=50)
        # res = differential_evolution(hpo_f, bounds=hpo_bounds, polish=False,
        #     tol=0.01, disp=True, maxiter=4)
        print(res)

if __name__ == "__main__":

    # hyperprm_opt(test=True)
    # quit()

    b = 5
    bounds = ((-b, b), (-b, b))

    from example_functions import eggshell
    import matplotlib.pyplot as plt

    con_on = False
    if con_on:
        g = lambda x: 1.0 - (x[0]+0.9)**2 - x[1]**2
        constraints = (g,)
    else:
        constraints = ()

    # Contour plot
    n1 = 100
    n2 = 100
    x1 = np.linspace(bounds[0][0],bounds[0][1],n1)
    x2 = np.linspace(bounds[1][0],bounds[1][1],n2)
    fun = np.zeros([n1,n2])
    con = np.zeros([n1,n2])

    # function evaluations for the contour plot
    for i in range(n1):
        for j in range(n2):
            fun[i,j] = eggshell([x1[i],x2[j]])
            if con_on:
                con[i,j] = -g([x1[i],x2[j]])

    fig = plt.figure()
    ax = fig.subplots()
    hc = ax.contour(x1, x2, np.transpose(fun), 16)
    if con_on:
        ax.contourf(x1, x2, np.transpose(con),[-1000,0], alpha=0.25, 
            colors='red')
    fig.colorbar(hc)
    fig.show()
    input("ENTER to continue")
    # sol = GA(eggshell, bounds, pop_size=20, xtol=1e-3, figax=[fig, ax], 
    #     constraints=constraints, mutation1=0.10, mutation2=0.60)
    # sol = GA(eggshell, bounds, pop_size=15, xtol=1e-3, figax=[fig, ax], 
    #     constraints=constraints, mutation1=0.034, mutation2=0.245)
    sol = GA(eggshell, bounds, pop_size=8, xtol=1e-3, figax=[fig, ax], 
        constraints=constraints, mutation1=0.15, mutation2=0.75)
    sol.printall()
    
    input("ENTER")


    # Testing
    # f = lambda x: x[0]**2 + x[1]**2
    # # x[0] < -1
    # g = lambda x: x[0] + 1
    # bounds = ((-4, 4), (-4, 4))
    # sol = GA(eggshell, bounds, constraints=(g,), pop_size=30)
