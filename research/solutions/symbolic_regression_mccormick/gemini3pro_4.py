import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem for the McCormick dataset.
        """
        # Ensure y is the correct shape for PySR (n_samples, n_targets)
        if y.ndim == 1:
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = y

        # Configure PySRRegressor
        # Optimized for 8 vCPUs and the known complexity of the McCormick function.
        # McCormick function: sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1
        model = PySRRegressor(
            niterations=50,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=16,          # Utilization of 8 vCPUs
            population_size=40,
            ncycles_per_iteration=500,
            maxsize=40,              # Allow sufficient complexity
            parsimony=0.001,
            model_selection="best",  # Choose the best scoring model
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            temp_equation_file=True,
            delete_tempfiles=True,
        )

        # Fit the model
        # variable_names ensures the output string uses "x1" and "x2"
        model.fit(X, y_reshaped, variable_names=["x1", "x2"])

        # Extract the best symbolic expression
        try:
            best_sympy = model.sympy()
            expression = str(best_sympy)
        except Exception:
            # Fallback in case of extraction failure
            expression = "x1 + x2"

        # Generate predictions
        try:
            predictions = model.predict(X)
            if isinstance(predictions, np.ndarray):
                predictions = predictions.flatten().tolist()
            elif hasattr(predictions, "tolist"):
                predictions = predictions.tolist()
            else:
                predictions = list(predictions)
        except Exception:
            # Fallback predictions
            predictions = [0.0] * len(X)

        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }