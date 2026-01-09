import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem using PySR.
        Targeting a ripple-like function with polynomial amplitude and trig oscillations.
        """
        # Configure PySRRegressor
        # Using 8 vCPUs (procs=8) with parallel populations
        # Including '^' for polynomials/roots, and trig functions for oscillations
        # Excluding 'log' to minimize domain error risks (e.g. log of negative)
        model = PySRRegressor(
            niterations=80,             # Sufficient depth for 8 vCPUs
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "exp"],
            populations=16,             # Multiple of vCPUs
            population_size=50,
            ncycles_per_iteration=500,
            maxsize=45,                 # Allow enough complexity for modulated waves
            parsimony=0.001,            # Penalty for complexity
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,
            multithreading=False,       # Multiprocessing is better for PySR
            model_selection="best",     # Select best model by score
            temp_equation_file=True,
            timeout_in_seconds=550      # Safety timeout
        )

        # Fit the model to the data
        # variable_names ensures the output string uses 'x1' and 'x2'
        model.fit(X, y, variable_names=["x1", "x2"])

        # Retrieve the best symbolic expression
        try:
            # model.sympy() returns the SymPy representation of the best equation
            best_expr = model.sympy()
        except Exception:
            # Fallback if PySR fails to converge or return a model
            if hasattr(model, "equations_") and not model.equations_.empty:
                # Use the last equation found (usually best score in standard sorting)
                best_expr = model.equations_.iloc[-1]["sympy_format"]
            else:
                # Linear fallback
                best_expr = sympy.sympify("0.0")

        # Generate predictions
        try:
            predictions = model.predict(X)
        except Exception:
            # Fallback prediction using the expression directly
            try:
                f_lamb = sympy.lambdify(["x1", "x2"], best_expr, modules=["numpy"])
                predictions = f_lamb(X[:, 0], X[:, 1])
            except Exception:
                predictions = np.zeros(len(X))

        return {
            "expression": str(best_expr),
            "predictions": predictions.tolist(),
            "details": {}
        }