""" Read and plot max_stress files
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import settings

def it_progress():
    """ Make whisker plots of f & g at each iteration

    TODO: Could annotate with location of best point, 
        using opt_GF.find_best
    """
    f_vals = [] #(iteration, pt)
    g_vals = []
    # Search directory for population files
    for filename in os.listdir():
        if "popf_it" in filename:
            #print(17, filename)
            popf = np.loadtxt(filename, dtype=float)
            f_vals.append(popf)	
        elif "popg_it" in filename:
            #print(21, filename)
            popg = np.loadtxt(filename, dtype=float)
            g_vals.append(popg)	
    #f_vals = np.array(f_vals)
    #g_vals = np.array(g_vals)
    #print(24, f_vals.shape)
    #print(24, g_vals.shape)
    fig1, ax1 = plt.subplots()
    il = np.arange(len(f_vals))
    labels = [f"it {i}" for i in il]
    #print(31, len(f_vals), len(g_vals), len(labels))
    ax1.boxplot(f_vals, labels=labels)
    fig1.savefig("f_it.png")
    fig2, ax2 = plt.subplots()
    ax2.boxplot(g_vals, labels=labels)
    ax2.set_ylim(-settings.stress_limit, 4*settings.stress_limit)
    fig2.savefig("g_it.png")

if __name__ == "__main__":
    it_progress()
    

def max_stress():
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

if __name__ == "__main__":
    it_progress()
