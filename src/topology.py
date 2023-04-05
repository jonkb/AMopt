import settings # Be sure to import settings before importing numpy
from obj_func import obj_func
from con_func import con_func
from meshing import x2mesh, x0_cube, x0_hyperboloid
from util import *
from subprocess import run
from scipy.optimize import (NonlinearConstraint, Bounds, minimize, 
    differential_evolution)
import numpy as np

## SETUP
# start timer
times = tic()

# Initial guess (for local methods)
x0 = None
# x0 = x0_hyperboloid()
# x0 = x0_cube()

# Load initial population for warm start (for global methods)
if settings.warm_pop is None:
    warm_start = {
        "population": None,
        "popf": None,
        "popg": None
    }
else:
    pop_init = np.loadtxt(settings.warm_pop, dtype=float)
    popf_init = (np.loadtxt(settings.warm_popf, dtype=float) 
        if settings.warm_popf is not None else None)
    popg_init = (np.loadtxt(settings.warm_popg, dtype=float) 
        if settings.warm_popg is not None else None)
    warm_start = {
        "population": pop_init,
        "popf": popf_init,
        "popg": popg_init
    }

# For SciPy methods
theConstraints = NonlinearConstraint(con_func, -np.inf, 0.0)#, finite_diff_rel_step=[1e8])
theBounds = [(0, 1)] * settings.nx

if settings.method == "spmin":
    ## SciPy minimize
    # Scipy.optimize.minimize settings
    theOptions = {'maxiter':settings.maxiter}#, 'finite_diff_rel_step':[1e6]}
    optimality = []
    def callback(xk, res):
        """
        callback function for minimize
        """
        optimality.append(res.optimality)
        print(f"optimality: {res.optimality}")
    # Run minimize
    res = minimize(obj_func, x0, constraints=theConstraints, method='trust-constr', 
    bounds=theBounds, tol=1e-5, options=theOptions, callback=callback)

if settings.method == "spDE":
    ## SciPy differential evolution
    def callback(xk, convergence):
        # Delete unneeded temporary files after every iteration
        file_cleanup(["stl", "msh", "mesh", "vtk"])
        # Print the time for this iteration
        toc(times, f"DE iteration")
        print(f"convergence: {convergence}")

    # res = differential_evolution(obj_func, bounds=theBounds, constraints=theConstraints,
    #     tol=5e-2, disp=settings.verbose, maxiter=settings.maxiter, polish=False)
    toc(times) # Start timing DE
    res = differential_evolution(obj_func, bounds=theBounds, tol=settings.xtol,
        popsize=settings.pop_size, constraints=theConstraints, disp=True,
        maxiter=settings.maxiter, polish=settings.polish, callback=callback, 
        init=warm_start["population"])

if settings.method == "jGA":
    ## Jon's GA
    from opt_GF import GA

    def GAcb(data):
        # Delete unneeded temporary files after every iteration
        file_cleanup(["stl", "msh", "mesh", "vtk"])
        # Print the time for this iteration
        toc(times, f"Iteration {data.it}")
        # Save the population to file
        np.savetxt(f"population_it{data.it}.txt", data.population)
        np.savetxt(f"popf_it{data.it}.txt", data.popf)
        np.savetxt(f"popg_it{data.it}.txt", data.popg)

    print("Running GA")
    bounds = [(0, 1)] * settings.nx
    constraints = (lambda x: con_func(x)[0],)
    toc(times) # Start timing GA
    res = GA(obj_func, bounds, constraints=constraints, xtol=settings.xtol, 
        it_max=settings.maxiter, pop_size=settings.pop_size, verbose=True,
        mutation1=settings.mutation1, mutation2=settings.mutation2, 
        callback=GAcb, warm_start=warm_start)
    res.printall()
    # Tacky fix for compatability
    res.x = res.x_star

if settings.method == "jIP":
    ## Jon's IP constrained optimization
    import opt_constr
    print(f"Running Interior Penalty Optimization")
    res = opt_constr.ip_min(obj_func, con_func, x0, maxit=settings.maxiter,
        bounds=Bounds(0,1))


## Inspect results
# Save the x vector to file
np.savetxt(f"x_optimized.txt", res.x)
# Save optimized voxelization here
x2mesh(res.x, "x_optimized", dim=settings.resolution, out_format="mesh")
# Calculate stress in x_optimized
run(["sfepy-run", "cube_traction.py", "-d", f"tag='x_optimized'"], 
    stdout=settings.terminal_output, stderr=settings.terminal_output)

# Print results
print("\n\n--- RESULTS ---")
print(res)
from obj_func import f_calls # NOTE: Must be imported at this time to work
from con_func import g_calls
print(f"Number of function calls: {f_calls}")
print(f"Number of constraint calls: {g_calls}")
# print(f"Optimality: {optimality}")
toc(times, msg=f"\n\nTotal optimization time for {settings.maxiter} Iterations:", total=True)
print("--- -- -- -- -- -- -- -- ---\n\n")

# Visualize the optimized voxelization
#run(["sfepy-view", "x_optimized.vtk"], 
#    stdout=settings.terminal_output, stderr=settings.terminal_output)
