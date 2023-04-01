import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cube_traction as ct 
from meshing import x2mesh, x0_cube, x0_hyperboloid
from subprocess import run, call 
from util import *
import settings

f_calls = 0

# Define the objective function
def obj_func(x):
    """
    Objective function
    """
    global f_calls
    f_calls += 1
    #vprnt(f"f_calls: {f_calls}")
    # Calculate mass from x
    rho = settings.rho
    # num_voxels = np.prod(settings.resolution) # Nx*Ny*Nz
    volume_voxel = np.prod(settings.voxel_dim) # Volume per voxel = hx*hy*hz
    mass = np.sum(x * rho * volume_voxel)
    # mass = np.sum(rho*volume_voxel*num_voxels*x)
    vprnt(f"Volume per voxel: {volume_voxel}")
    # vprnt(f"Number of voxels: {num_voxels}")
    print(f"f_calls: {f_calls}, Mass: {mass} grams")

    return mass


if __name__ == "__main__":
    x0 = x0_cube() # initial guess
    obj_func(x0)
    # run(["sfepy-view", "cube.vtk"]) # Visualize the results
