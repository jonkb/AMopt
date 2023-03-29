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

    # Pull max stress from max_stress.txt
    try:
        max_stress = np.loadtxt(f'{tag}_max_stress.txt', dtype=float)
        vprnt(f"Max stress: {max_stress}")
    except:
        max_stress = settings.stress_limit*300
        vprnt(f"Error: unable to load max stress")
    
    g0 = np.zeros(1)
    g0[0] = max_stress - settings.stress_limit

    return g0
    
if __name__ == "__main__":
    x0 = x0_cube()
    g = con_func(x0)
    print(f"g = {g}")
    run(["sfepy-view", "x000000.vtk"]) # Visualize the results
