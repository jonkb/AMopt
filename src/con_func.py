import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cube_traction as ct 
import obj_func as of
from meshing import x2mesh, x0_cube
from subprocess import run, call 
from util import *
import settings


def con_func(x):
    """
    Constraint function
    """
    stress_limit = settings.stress_limit

    # Create the mesh
    x2mesh(x, "cube", out_format="mesh")

    # Evaluate mesh in FEA
    results = run(["sfepy-run", "cube_traction.py"], capture_output=False, text=True)
    vprnt(results.stdout)

    # Pull max stress from max_stress.txt
    max_stress = np.loadtxt('max_stress.txt', dtype=float)
    
    g0 = np.zeros(1)
    g0[0] = max_stress - stress_limit

    return g0