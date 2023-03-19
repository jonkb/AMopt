import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cube_traction as ct 
import obj_func as of
from meshing import x2mesh, x0_cube
from subprocess import run, call 
from util import *
import settings

call_num = 0

def con_func(x):
    """
    Constraint function
    """
    stress_limit = settings.stress_limit

    # Count call numbers to make unique filenames
    global call_num
    tag = f"x{call_num:06d}"
    call_num += 1

    # Save the x vector to file (FOR DEBUGGING)
    np.savetxt(f"{tag}.txt", x)

    # Create the mesh
    x2mesh(x, tag, out_format="mesh")

    # Evaluate mesh in FEA
    results = run(["sfepy-run", "cube_traction.py", "-d", f"tag='{tag}'"], capture_output=False, text=True)
    # vprnt(results.stdout)

    # Pull max stress from max_stress.txt
    max_stress = np.loadtxt(f'{tag}_max_stress.txt', dtype=float)
    
    g0 = np.zeros(1)
    g0[0] = max_stress - stress_limit

    return g0