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

x0 = x0_cube((10,10,10)) # initial guess
lb = 60 # MPa (yield strength of PLA)
ub = np.inf # MPa 
theConstraints = NonlinearConstraint(con_func, lb, ub)
theBounds = [(0, np.inf)]
theOptions = {'maxiter':1000}
optimality = []
def callback(xk, res):
    optimality.append(res.optimality)
    # print(res.optimality)
    pass
res = minimize(obj_func, x0, constraints=theConstraints, method='trust-constr', bounds=theBounds, tol=1e-5, options=theOptions, callback=callback)
print(res)


print(optimality)