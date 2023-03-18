import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cube_traction as ct 
from meshing import x2mesh, x0_cube, x0_hyperboloid
from subprocess import run, call 
from util import *

# Define the objective function
def obj_func(x):
    """
    Objective function
    """
    times = tic()
    # Create the mesh
    x2mesh(x, "cube", out_format="mesh")

    # Evaluate mesh in FEA
    results = run(["sfepy-run", "cube_traction.py"], capture_output=False, text=True)
    print(results.stdout)
    
    # Calculate mass from x
    rho = .0014 #g/mm3
    #note: these are hardcoded for now but should be passed in somewhere for consistency
    xres = 10
    yres = 10
    zres = 10
    # spacing = (2, 2, 2)
    x_lims = (-10, 10)
    y_lims = (-10, 10)
    z_lims = (-10, 10)

    Lx = max(x_lims) - min(x_lims)
    Ly = max(x_lims) - min(x_lims)
    Lz = max(x_lims) - min(x_lims)

    volume_voxel = (Lx*Ly*Lz)/ (xres*yres*zres)
    num_voxels = Lx*Ly*Lz/volume_voxel
    mass = rho*volume_voxel*num_voxels
    # print(f"Volume of voxel: {volume_voxel}")
    # print(f"Number of voxels: {num_voxels}")
    mass = np.sum(mass)
    # print(f"Mass: {mass} grams")

    toc(times, f"Total obj_func eval time", total=True)

    return mass


if __name__ == "__main__":
    x0 = x0_cube((10,10,10)) # initial guess
    obj_func(x0)
    run(["sfepy-view", "cube.vtk"]) # Visualize the results