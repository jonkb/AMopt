from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt
from obj_func import obj_func, f_calls
from con_func import con_func, g_calls
from meshing import x2mesh, x0_cube, x0_hyperboloid
from subprocess import run, call
from util import *

# start timer
times = tic()

# Scipy.optimize.minimize settings
x0 = x0_hyperboloid() # initial guess
# x0 = x0_cube() # initial guess
lb = -np.inf
ub = 0.0 
theConstraints = NonlinearConstraint(con_func, lb, ub)#, finite_diff_rel_step=[1e8])
theBounds = [(0, 1)]
theOptions = {'maxiter':settings.maxiter}#, 'finite_diff_rel_step':[1e6]}
optimality = []
def callback(xk, res):
    """
    callback function for minimize
    """
    optimality.append(res.optimality)
    print(f"optimality: {res.optimality}")

# Run SciPy minimize
res = minimize(obj_func, x0, constraints=theConstraints, method='trust-constr', 
    bounds=theBounds, tol=1e-5, options=theOptions, callback=callback)

# Save the x vector to file (FOR DEBUGGING)
np.savetxt(f"cube_optimized.txt", res.x)
# Save optimized voxelization here
x2mesh(res.x, "cube_optimized", dim=settings.resolution, out_format="vtk")

# Print results
print("\n\n--- RESULTS ---")
print(res)
from obj_func import f_calls
from con_func import g_calls
print(f"Number of function calls: {f_calls}")
print(f"Number of constraint calls: {g_calls}")
print(f"Optimality: {optimality}")
toc(times, msg=f"\n\nTotal optimization time for {settings.maxiter} Iterations:", total=True)
print("--- -- -- -- -- -- -- -- ---\n\n")


# Visualize the optimized voxelization
run(["sfepy-view", "cube_optimized.vtk"])