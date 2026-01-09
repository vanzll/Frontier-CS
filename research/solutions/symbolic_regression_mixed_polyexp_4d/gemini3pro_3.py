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
        """
        # Configure PySRRegressor
        # Using 8 processes matches the 8 vCPUs available
        # Unary operators match the allowed functions
        # Nested constraints prevent overly complex nested transcendental functions
        model = PySRRegressor(
            niterations=200,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=20,
            population_size=50,
            maxsize=50,
            verbosity=0,
            progress=False,
            random_state=42,
            nested_constraints={
                "sin": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                "cos": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                "exp": {"exp": 0, "sin": 0, "cos": 0, "log": 0},
                "log": {"exp": 0, "log": 0, "sin": 0, "cos": 0}
            },
            model_selection="best",
            procs=8,
            multithreading=False,
            deterministic=True,
            temp_equation_file=None,
            equation_file=None
        )

        # Fit the model
        # variable_names are essential for the output string to match the required format (x1, x2, x3, x4)
        model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])

        # Extract the best expression found
        # model.sympy() returns a sympy expression object
        best_expr = model.sympy()
        expression_str = str(best_expr)

        # Generate predictions using the fitted model
        predictions = model.predict(X)

        return {
            "expression": expression_str,
            "predictions": predictions.tolist(),
            "details": {}
        }