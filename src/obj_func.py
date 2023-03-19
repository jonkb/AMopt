import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cube_traction as ct 
from meshing import x2mesh, x0_cube, x0_hyperboloid
from subprocess import run, call 
from util import *
import settings

# Define the objective function
def obj_func(x):
    """
    Objective function
    """
    times = tic()

    # The following is for the stress -- the constraints
    # # Create the mesh
    # x2mesh(x, "cube", out_format="mesh")

    # # Evaluate mesh in FEA
    # results = run(["sfepy-run", "cube_traction.py"], capture_output=False, text=True)
    # vprnt(results.stdout)
    
    # Calculate mass from x
    rho = settings.rho
    num_voxels = np.prod(settings.resolution) # Nx*Ny*Nz
    volume_voxel = np.prod(settings.voxel_dim) # Volume per voxel = hx*hy*hz
    mass = np.sum(rho*volume_voxel*num_voxels)
    vprnt(f"Volume of voxel: {volume_voxel}")
    vprnt(f"Number of voxels: {num_voxels}")
    vprnt(f"Mass: {mass} grams")

    toc(times, f"Total obj_func eval time", total=True)

    return mass


if __name__ == "__main__":
    x0 = x0_cube() # initial guess
    obj_func(x0)
    # run(["sfepy-view", "cube.vtk"]) # Visualize the results