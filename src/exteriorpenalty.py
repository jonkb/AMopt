import numpy as np
from scipy.optimize import minimize

def objective(x):
    """Define the objective function to be optimized"""
    
    return x[0]**2 + x[1]**2

def constraints(x):
    """Define the constraint functions"""

    return [x[0] + x[1] - 1, x[0]**2 + x[1]**2 - 2]

def penalty_function(x, penalty_coeff):
    """Define the penalty function"""
    penalty = 0.0
    for c in constraints(x):
        penalty += max(0.0, c)**2
    return objective(x) + penalty_coeff * penalty

def exterior_penalty_method(x0, penalty_coeff, max_iter=1000, tol=1e-6):
    """
    Implement the exterior penalty method optimization algorithm.
    x0: initial guess for the solution
    penalty_coeff: the coefficient for the penalty function
    max_iter: maximum number of iterations
    tol: tolerance for the optimization
    """
    x = x0
    for i in range(max_iter):
        # Define the objective function as the penalty function
        obj = lambda x: penalty_function(x, penalty_coeff)
        # Use the L-BFGS-B method to minimize the penalty function
        res = minimize(obj, x, method='L-BFGS-B')
        # Update x with the solution from the optimization
        x = res.x
        # Check if all constraints are satisfied within the given tolerance
        constraints_satisfied = all(abs(c) < tol for c in constraints(x))
        if constraints_satisfied:
            break
        # Increase the penalty coefficient for the next iteration
        penalty_coeff *= 10
    return x

# # Test the algorithm with an initial guess of (0, 0) and penalty coefficient of 1
# x0 = np.array([1.0, 1.0])
# penalty_coeff = 1.0
# x = exterior_penalty_method(x0, penalty_coeff)
# print(x)  # prints the optimal solution