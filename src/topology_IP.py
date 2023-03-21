
import os
# Set environment variables to limit multithreading
# This must be done before importing np
threads = "16"
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["OMP_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

from scipy.optimize import minimize, NonlinearConstraint, Bounds
import numpy as np
import matplotlib.pyplot as plt
from obj_func import obj_func, f_calls
from con_func import con_func, g_calls
from meshing import x2mesh, x0_cube, x0_hyperboloid
from subprocess import run, call
from util import *

import opt_constr # Jon's custom constrained optimization

# start timer
times = tic()

#x0 = x0_hyperboloid() # initial guess
x0 = x0_cube() # initial guess

# Run IP optimization
print(f"Running Interior Point Optimization")
res = opt_constr.ip_min(obj_func, con_func, x0, maxit=settings.maxiter,
    bounds=Bounds(0,1))

# Save the x vector to file (FOR DEBUGGING)
np.savetxt(f"x_optimized.txt", res.x)
# Save optimized voxelization here
x2mesh(res.x, "x_optimized", dim=settings.resolution, out_format="mesh")

# Print results
print("\n\n--- RESULTS ---")
print(res)
from obj_func import f_calls # NOTE: Is reimporting this necessary?
from con_func import g_calls
print(f"Number of function calls: {f_calls}")
print(f"Number of constraint calls: {g_calls}")
# print(f"Optimality: {optimality}")
toc(times, msg=f"\n\nTotal optimization time for {settings.maxiter} Iterations:", total=True)
print("--- -- -- -- -- -- -- -- ---\n\n")


# Visualize the optimized voxelization
run(["sfepy-view", "x_optimized.mesh"])
