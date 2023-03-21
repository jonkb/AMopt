""" Constrained optimization

Currently implemented: Interior penalty (logarithmic barrier)

Jon Black, 2023-03-20
"""

import numpy as np
import scipy.optimize as spo

def f_hat_ip(f, g, x, mu):
    # Add a logarithmic barrier penalty to the objective function
    fi = f(x)
    gi = g(x) # May be a vector
    if np.any(gi > 0):
        penalty = 1e3
    else:
        penalty = -mu*np.sum(np.log(-gi))
    return fi + penalty

def ip_min(f, g, x0, bounds=None, maxit=8, xtol=1e-3, rho=0.25, mu0=1):
    # Interior penalty method
    
    mu = mu0
    xi = x0
    it = 0
    dx = xtol * 2
    # TODO: xtol is a bad convergence metric for this.
    # Constrained optimality would be better.
    while it < maxit and dx > xtol:
        # Solve subproblem
        f_hat = lambda x: f_hat_ip(f, g, x, mu)
        res = spo.minimize(f_hat, xi, bounds=bounds, tol=1e-2)
        print(f"Iteration {it}: \n{res}")
        # Set up for next iteration
        dx = np.linalg.norm(xi - res.x)
        xi = res.x
        mu *= rho
        it += 1
    print(f"Optimal point: {xi}")
    print(f"#it: {it}")
    return res


if __name__ == "__main__":
    f = lambda x: x[0] + x[1] + x[26]
    g = lambda x: x[0]**2 + x[1]**2 - 16
    x0 = np.ones(27)

    # f_hat = lambda x: f_hat_ip(f, g, x, 1)
    # res = spo.minimize(f_hat, x0)
    # print(25, res)

    ip_min(f, g, x0)
    
