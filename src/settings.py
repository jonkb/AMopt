""" Settings.py
This file is intended to be imported as a module by the other python files so
they all have access to the same global settings.

Should be imported before importing numpy
"""
from subprocess import DEVNULL

## General settings
# Whether to print to console when calling the util.vprnt function
verbose = False
terminal_output = DEVNULL
# Whether to delete mesh files after every function call
clean_up_each = True

## Limit multithreading (useful when running on a large, public computer)
max_threads = "16"
import os
# Set environment variables to limit multithreading
# This must be done before importing numpy
os.environ["OPENBLAS_NUM_THREADS"] = max_threads
os.environ["OMP_NUM_THREADS"] = max_threads
os.environ["MKL_NUM_THREADS"] = max_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = max_threads
os.environ["NUMEXPR_NUM_THREADS"] = max_threads

# Whether to use legacy LHS package (pyDOE). Needed for Python 3.6
legacy = False

# TODO: Temporary folder path

## Optimization options
# Methods: ["spmin" (SciPy minimize), "spDE" (SciPy differential_evolution), 
#   "jGA" (Jon's Genetic Algorithm), "jIP" (Jon's Interior Penalty)]
method = "jGA"
# Max iterations for optimization
maxiter = 8
# Warm start initial population txt file (from numpy.savetxt)
# warm_pop = None # LHS sample
warm_pop = None #"population_init.txt"
warm_popf = "popf_init.txt" # obj_fun(warm_pop)
warm_popg = "popg_init.txt" # con_fun(warm_pop)
# GA options
pop_size = 16 # N_pop = pop_size * N_x
mutation1 = 0.10
mutation2 = 0.40
xtol = 1e-6

## Cube dimensions & Spacing
#   See meshing.XYZ_grid() for density function XYZ grid generation
import numpy as np
# Cube side lengths (doesn't need to be an actual cube)
side_lengths = np.array((20, 20, 20))
# Number of voxels in each direction (doesn't need to be equal)
resolution = np.array((8, 8, 8))
# Number of design variables
nx = np.prod(resolution)
# voxel dimensions
voxel_dim = side_lengths / resolution
# face_thickness (int): Thickness of the top and bottom faces to be added, in
#   number of voxels. Minimum is 1. The integer limitation is because the 
#   marching cubes algorithm doesn't support variable element spacing.
face_thickness=1

# Material density (PLA)
rho = .0014 #g/mm3
stress_limit = 50.0 # MPa - The average compressive stress limit I could find was 40-60 MPa

# Applied load
applied_traction = 8.0 # MPa
