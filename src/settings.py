""" Settings.py
This file is intended to be imported as a module by the other python files so
they all have access to the same global settings.
"""
import numpy as np
from subprocess import DEVNULL

## General settings
# Whether to print to console when calling the util.vprnt function
verbose = True
terminal_output = DEVNULL

# TODO: Temporary folder path

# Max iterations for optimization
maxiter = 10

## Cube dimensions & Spacing
#   See meshing.XYZ_grid() for density function XYZ grid generation
# Cube side lengths (doesn't need to be an actual cube)
side_lengths = np.array((20, 20, 20))
# Number of voxels in each direction (doesn't need to be equal)
resolution = np.array((3, 3, 3))
# voxel dimensions
voxel_dim = side_lengths / resolution
# face_thickness (int): Thickness of the top and bottom faces to be added, in
#   number of voxels. Minimum is 1. The integer limitation is because the 
#   marching cubes algorithm doesn't support variable element spacing.
face_thickness=1

# Material density (PLA)
rho = .0014 #g/mm3
stress_limit = 50.0 # MPa - The average compressive stress limit I could find was 40-60 MPa

