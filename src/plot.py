""" Read and plot max_stress files
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import settings

stress_list = []

for filename in os.listdir():
    if "max_stress.txt" in filename:
        sigma = np.loadtxt(filename, dtype=float)
        stress_list.append(sigma)	

fig, ax = plt.subplots()
xl = np.arange(len(stress_list))
ax.scatter(xl, stress_list)
ax.set_ylim(0, 4*settings.stress_limit)
ax.set_xlabel("g_call")
ax.set_ylabel("max stress (MPa)")
ax.axhline(settings.stress_limit, ls="--")
fig.savefig("max_stress_plot.png")


