import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cube_traction as ct 
import obj_func as of
from meshing import x2mesh, x0_cube
from subprocess import run, call 
from util import *


def con_func(x):
    """
    Constraint function
    """
    stress_limit = 40.0 # MPa - The average compressive stress limit I could find was 40-60 MPa

    # Pull max stress from obj_func
    max_stress = of.obj_func(x)
    
    g0 = np.zeros(1)
    g0[0] = max_stress - stress_limit

    return g0