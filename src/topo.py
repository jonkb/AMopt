from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt
from obj_func import obj_func, f_calls
from con_func import con_func, g_calls
from exteriorpenalty import exterior_penalty_method
from meshing import x2mesh, x0_cube, x0_hyperboloid
from subprocess import run, call
from util import *

# start timer
times = tic() 

def penalty_function(x, penalty_coeff):
    """Define the penalty function"""
    penalty = 0.0
    for c in con_func(x):
        penalty += max(0.0, c)**2
    return obj_func(x) + penalty_coeff * penalty

# Scipy.optimize.minimize settings
x0 = x0_hyperboloid() # initial guess
penalty_coeff = 1.0
res = exterior_penalty_method(x0, penalty_coeff)

lb = -np.inf
ub = 0.0 
theConstraints = NonlinearConstraint(con_func, lb, ub)#, finite_diff_rel_step=[1e8])
theBounds = [(0, 1)]

# theOptions = {'maxiter':settings.maxiter}#, 'finite_diff_rel_step':[1e6]}
# optimality = []
# def callback(xk, res):
#     """
#     callback function for minimize
#     """
#     optimality.append(res.optimality)
#     print(f"optimality: {res.optimality}")



# res = minimize(obj_func, x0, constraints=theConstraints, method='trust-constr', 
#     bounds=theBounds, tol=1e-5, options=theOptions, callback=callback)

# Save optimized voxelization here
x2mesh(res, "cube-optimized", out_format="vtk")

# Print results
# print("\n\n--- RESULTS ---")
# print(res)
# from obj_func import f_calls
# from con_func import g_calls
# print(f"Number of function calls: {f_calls}")
# print(f"Number of constraint calls: {g_calls}")
# print(f"Optimality: {optimality}")
# toc(times, msg=f"\n\nTotal optimization time for {settings.maxiter} Iterations:", total=True)
# print("--- -- -- -- -- -- -- -- ---\n\n")


# Visualize the optimized voxelization
run(["sfepy-view", "cube-optimized.vtk"])