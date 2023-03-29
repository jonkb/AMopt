import settings # Be sure to import settings before importing numpy
from obj_func import obj_func
from con_func import con_func
from meshing import x2mesh, x0_cube, x0_hyperboloid
from util import *
from subprocess import run
from scipy.optimize import (NonlinearConstraint, Bounds, minimize, 
    differential_evolution)
import numpy as np

# start timer
times = tic()

## Initial guess
# x0 = x0_hyperboloid()
# x0 = x0_cube()

## SciPy minimize
# Scipy.optimize.minimize settings
theConstraints = NonlinearConstraint(con_func, -np.inf, 0.0)#, finite_diff_rel_step=[1e8])
theBounds = [(0, 1)] * settings.nx
theOptions = {'maxiter':settings.maxiter}#, 'finite_diff_rel_step':[1e6]}
optimality = []
def callback(xk, res):
    """
    callback function for minimize
    """
    optimality.append(res.optimality)
    print(f"optimality: {res.optimality}")
#res = minimize(obj_func, x0, constraints=theConstraints, method='trust-constr', 
#    bounds=theBounds, tol=1e-5, options=theOptions, callback=callback)

<<<<<<< HEAD
# SciPy differential evolution
res = differential_evolution(obj_func, bounds=theBounds, constraints=theConstraints,
    tol=5e-2, disp=settings.verbose, maxiter=settings.maxiter, polish=False)

# ## Jon's IP constrained optimization
=======
## SciPy differential evolution
# res = differential_evolution(obj_func, bounds=theBounds, constraints=theConstraints,
#     tol=5e-2, disp=settings.verbose, maxiter=settings.maxiter, polish=False)
#res = differential_evolution(obj_func, bounds=theBounds, popsize=50,
#    constraints=theConstraints, tol=5e-2, disp=settings.verbose, 
#    maxiter=settings.maxiter, polish=True)

## Jon's GA
from opt_GF import GA

def GAcb(data):
    # Delete unneeded temporary files after every iteration
    file_cleanup(["stl", "msh", "mesh"])
    # Print the time for this iteration
    toc(times, f"Iteration {data.it}")

print("Running GA")
constraints = (con_func,)
toc(times) # Start timing GA
res = GA(obj_func, theBounds, constraints=constraints, it_max=settings.maxiter,
    pop_size=15, xtol=1e-6, mutation1=0.075, mutation2=0.400, 
    verbose=settings.verbose, callback=GAcb)
res.printall()
# Tacky fix for compatability
res.x = res.x_star

## Jon's IP constrained optimization
>>>>>>> main
# import opt_constr
# print(f"Running Interior Point Optimization")
# res = opt_constr.ip_min(obj_func, con_func, x0, maxit=settings.maxiter,
#     bounds=Bounds(0,1))

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
print(f"Optimality: {optimality}")
toc(times, msg=f"\n\nTotal optimization time for {settings.maxiter} Iterations:", total=True)
print("--- -- -- -- -- -- -- -- ---\n\n")

# Visualize the optimized voxelization
#run(["sfepy-view", "x_optimized.vtk"], 
#    stdout=settings.terminal_output, stderr=settings.terminal_output)
