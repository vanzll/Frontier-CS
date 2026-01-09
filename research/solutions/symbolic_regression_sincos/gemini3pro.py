import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fits a symbolic regression model to predict y from X using PySR.
        """
        # Ensure inputs are contiguous arrays
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)

        # Configure PySRRegressor
        # Optimized for 8 vCPUs with multiprocessing
        # Includes timeout and iteration limits appropriate for the environment
        model = PySRRegressor(
            niterations=50,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=40,
            maxsize=30,
            verbosity=0,
            progress=False,
            random_state=42,
            procs=8,
            multiprocessing=True,
            model_selection="best",
            temp_equation_file=True,
            delete_tempfiles=True,
            early_stop_condition=None
        )

        try:
            # Fit the model specifying variable names to match output requirements
            model.fit(X, y, variable_names=["x1", "x2"])

            # Retrieve the best expression as a sympy object and convert to string
            best_expr = model.sympy()
            expression = str(best_expr)
            
            # Generate predictions using the fitted model
            predictions = model.predict(X)
            
        except Exception:
            # Fallback strategy: Linear Regression
            # This handles cases where symbolic regression might fail to initialize or converge
            
            # Create design matrix with intercept: [x1, x2, 1]
            A = np.column_stack([X[:, 0], X[:, 1], np.ones(len(X))])
            
            # Solve using Least Squares
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            
            # Construct a valid Python expression string
            # coeffs = [w_x1, w_x2, bias]
            expression = f"({coeffs[0]}) * x1 + ({coeffs[1]}) * x2 + ({coeffs[2]})"
            
            # Compute predictions
            predictions = A @ coeffs

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }