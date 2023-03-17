import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cube_traction as ct 
from meshing import x2mesh, x0_cube
from subprocess import run, call 
from util import *

# Define the objective function
def obj_func(x):
    """
    Objective function
    """
    times = tic()
    # Create the mesh
    x2mesh(x0, "cube", out_format="mesh")

    # evaluate mesh in FEA
    results = run(["sfepy-run", "cube_traction.py"], capture_output=False, text=True)
    print(results.stdout)
    
    # Visualize the results
    # run(["sfepy-view", "cube.vtk"])
    toc(times, f"Total obj_func eval time", total=True)

    # return max_stress


if __name__ == "__main__":
    # initial guess
    x0 = x0_cube((20,20,20))
    obj_func(x0)