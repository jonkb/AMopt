from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt
from obj_func import obj_func
from con_func import con_func
from meshing import x2mesh, x0_cube, x0_hyperboloid
from subprocess import run, call
from util import *

# Variables
PLOT = True

x0 = x0_hyperboloid() # initial guess
lb = -np.inf
ub = 0.0 
theConstraints = NonlinearConstraint(con_func, lb, ub)#, finite_diff_rel_step=[1e8])
theBounds = [(0, 1)]
theOptions = {'maxiter':10}#, 'finite_diff_rel_step':[1e6]}
optimality = []
def callback(xk, res):
    optimality.append(res.optimality)
    print(f"optimality: {res.optimality}")

res = minimize(obj_func, x0, constraints=theConstraints, method='trust-constr', 
    bounds=theBounds, tol=1e-5, options=theOptions, callback=callback)
print(res)

# Save optimized voxelization here
x2mesh(res.x, "cube-optimized", out_format="mesh")

print(optimality)
