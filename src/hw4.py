# hw4 - Exterior penalty method
# Jaxon Jones
# Version: 1.0

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

# Variables
PRINT = True
# PRINT = False

# Utility functions
def bprnt(*args):
    if PRINT:
        print(*args)

# Functions
def f(x):
    return obj_func(x)

def g(x):
    return con_func(x)

def interior_penalty(f, g, x, mu=3, p=.8, tol=1e-5, maxiter=10):
    """Interior Penalty Method

    Args:
        f (Callable): Objective function
        g (Callable): Constraint function
        x (ndarray): Initial guess
        mu (int, optional): Initial penalty parameter. Defaults to 3.
        p (int, optional): Penalty decrease factor. Defaults to .8.
    """

   # Scipy.optimize.minimize settings
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
    
    k = 0 # iteration counter
    ext_msg = "Maximum number of iterations reached"
    if g(x) > tol:
        return bprnt("Initial guess is not feasible")

    
    while k < maxiter: # max iterations
        bprnt("k =", k) 
        bprnt("mu =", mu)
        bprnt("x =", x)
        bprnt("f(x) =", f(x))
        bprnt("g(x) =", g(x))
        bprnt("")
        # Solve the subproblem
        objfun = lambda x: f(x) - mu * np.log(-g(x))
        # Run SciPy minimize
        res = minimize(obj_func, x0, constraints=theConstraints, method='trust-constr', 
            bounds=theBounds, tol=1e-5, options=theOptions, callback=callback)
        x = res.x # Update x
        mu = mu * p # Update mu
        k += 1 # increment counter

        # Check if the constraints are satisfied
        if np.all(g(x) <= tol):
            ext_msg = "Sucess"
            break

    # Save the x vector to file (FOR DEBUGGING)
    np.savetxt(f"cube_optimized.txt", res.x)
    # Save optimized voxelization here
    x2mesh(res.x, "cube_optimized", dim=settings.resolution, out_format="vtk")

    # Visualize the optimized voxelization
    run(["sfepy-view", "cube_optimized.vtk"])

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
    print(ext_msg)
    print("Optimal solution found:", x)
    print("Optimal objective value:", f(x))



if __name__ == "__main__":
    # initial guess
    x0 = x0_hyperboloid() # initial guess
    interior_penalty(f, g, x0)