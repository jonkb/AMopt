from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt
from obj_func import obj_func
from con_func import con_func
from meshing import x2mesh, x0_cube
from subprocess import run, call
from util import *

# Variables
PLOT = True

x0 = x0_cube() # initial guess
lb = -np.inf
ub = 0.0 
theConstraints = NonlinearConstraint(con_func, lb, ub)
theBounds = [(0, 1)]
theOptions = {'maxiter':1000}
optimality = []
def callback(xk, res):
    # optimality.append(res.optimality)
    # print(res.optimality)
    pass
res = minimize(obj_func, x0, constraints=theConstraints, method='trust-constr', bounds=theBounds, tol=1e-5, options=theOptions, callback=callback)
print(res)

# Save optimized voxelization here
x2mesh(res.x, "cube-optimized", out_format="mesh")

# Visualize the results
run(["sfepy-view", "cube-optimized.vtk"])


# print(optimality)