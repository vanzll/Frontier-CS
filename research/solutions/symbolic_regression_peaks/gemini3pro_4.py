import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Solves the symbolic regression problem for the Peaks dataset.
        
        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)
            
        Returns:
            dict containing the expression string and predictions.
        """
        # Configure PySRRegressor with parameters optimized for the environment (8 vCPUs)
        # and the problem domain (Peaks dataset involving polynomials and exponentials).
        model = PySRRegressor(
            niterations=50,  # Balanced for convergence within reasonable time
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "sin", "cos"],  # Peaks relies heavily on exp
            populations=16,  # 2x number of cores
            population_size=40,
            maxsize=50,  # Allow complex expressions typical of Peaks function
            ncyclesperiteration=500,
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,  # Utilize all 8 vCPUs
            multithreading=False,  # Use multiprocessing
            model_selection="best",
            turbo=True,
            should_optimize_constants=True,
        )

        # Fit the model to the data
        # Explicitly naming variables x1, x2 to match the required output format
        model.fit(X, y, variable_names=["x1", "x2"])

        # Extract the best symbolic expression
        try:
            # model.sympy() returns a SymPy expression object
            best_expr = model.sympy()
            # Convert to string (PySR ensures standard Python syntax, e.g., ** for power)
            expression = str(best_expr)
        except Exception:
            # Fallback in case of failure
            expression = "0"

        # Generate predictions
        try:
            predictions = model.predict(X)
            # Sanitize predictions to avoid NaN/Inf issues
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            predictions = np.zeros(len(y))

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }