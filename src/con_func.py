import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cube_traction as ct 
import obj_func as of
from meshing import x2mesh, x0_cube
from subprocess import run, call 
from util import *
import settings

g_calls = 0

def con_func(x):
    """
    Constraint function
    """
    stress_limit = settings.stress_limit

    # Count call numbers to make unique filenames
    global g_calls
    tag = f"x{g_calls:06d}"
    g_calls += 1
    vprnt(f"g_calls: {g_calls}")

    # Save the x vector to file (FOR DEBUGGING)
    np.savetxt(f"{tag}.txt", x)

    # Create the mesh
    x2mesh(x, tag, dim=settings.resolution, out_format="mesh")

    # Evaluate mesh in FEA
    results = run(["sfepy-run", "cube_traction.py", "-d", f"tag='{tag}'"], stdout=settings.terminal_output, stderr=settings.terminal_output)
    # vprnt(results.stdout)

    # Pull max stress from max_stress.txt
    max_stress = np.loadtxt(f'{tag}_max_stress.txt', dtype=float)
    vprnt(f"Max stress: {max_stress}")
    
    g0 = np.zeros(1)
    g0[0] = max_stress - stress_limit

    return g0