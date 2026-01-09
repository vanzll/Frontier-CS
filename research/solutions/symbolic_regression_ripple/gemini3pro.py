import numpy as np
import pandas as pd
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySR.
        """
        # Ensure input data is float64 for numerical stability
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Configure the PySRRegressor
        # Optimized for the 8 vCPU environment with reasonable iteration counts and population sizes
        # We include '^' in binary_operators to allow for polynomial discovery (mapped to **)
        model = PySRRegressor(
            niterations=60,              # Sufficient iterations for convergence on ripple functions
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=16,              # 2 populations per vCPU
            population_size=50,          # Standard population size
            ncycles_per_iteration=500,   # Evolution steps per iteration
            maxsize=35,                  # Allow enough complexity for nested trig/poly functions
            parsimony=0.001,             # Mild penalty for complexity
            model_selection="best",      # Select best model based on score/complexity
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,                     # Use all 8 vCPUs
            multiprocessing=True,
            timeout_in_seconds=300,      # 5 minute timeout safety
            temp_equation_file=True      # Use temporary files for equation storage
        )

        try:
            # Fit the symbolic regression model
            model.fit(X, y, variable_names=["x1", "x2"])

            # Retrieve the best symbolic expression as a sympy object
            best_expr = model.sympy()
            
            # Convert sympy expression to a Python-compatible string
            # Sympy automatically handles conversion of Power to **, etc.
            expression = str(best_expr)

            # Generate predictions using the fitted model
            # Note: predict() uses the discovered equation
            predictions = model.predict(X)
            
            # Handle any potential numerical issues (NaNs/Infs) in predictions
            # This can happen if the found function has domain limitations (e.g. log(negative))
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception:
            # Robust fallback to Linear Regression if symbolic regression fails
            # or if PySR encounters an environment issue
            x1 = X[:, 0]
            x2 = X[:, 1]
            # Construct design matrix for linear fit: y = a*x1 + b*x2 + c
            A = np.column_stack([x1, x2, np.ones_like(x1)])
            res = np.linalg.lstsq(A, y, rcond=None)
            coeffs = res[0]
            a, b, c = coeffs
            
            expression = f"({a})*x1 + ({b})*x2 + ({c})"
            predictions = a * x1 + b * x2 + c

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }