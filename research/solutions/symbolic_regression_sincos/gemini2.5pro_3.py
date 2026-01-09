import numpy as np
from pysr import PySRRegressor
import pandas as pd
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits a symbolic expression to the given data using PySR.

        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            A dictionary with the symbolic expression and predictions.
        """

        model = PySRRegressor(
            niterations=40,
            populations=16,
            population_size=50,
            maxsize=25,
            procs=8,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos"],
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
            },
            timeout_in_seconds=480,
            verbosity=0,
            progress=False,
            random_state=42,
            # Use a slightly more robust loss function against outliers
            loss="L1DistLoss()",
        )

        model.fit(X, y, variable_names=["x1", "x2"])

        if not hasattr(model, 'equations_') or model.equations_.shape[0] == 0:
            # Fallback to a simple constant if PySR fails or finds no equations
            mean_y = np.mean(y)
            expression = f"{mean_y:.8f}"
            predictions = np.full_like(y, mean_y)
        else:
            # Get the best expression
            best_expr_sympy = model.sympy()
            expression = str(best_expr_sympy)
            
            # Use PySR's predict method for robustness
            predictions = model.predict(X)

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }